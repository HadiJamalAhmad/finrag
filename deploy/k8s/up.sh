#!/usr/bin/env bash
# Run finrag on a local Kubernetes cluster (kind), end to end:
#   cluster -> image -> services -> secret -> corpus -> model -> ingestion -> smoke test.
#
#   deploy/k8s/up.sh                 # corpus: data/ if it holds a manifest, else the sample
#   CORPUS=sample deploy/k8s/up.sh   # force the one-filing sample (what CI runs)
#   CORPUS=fr deploy/k8s/up.sh       # French documents (data/docs, `finrag fetch-docs`), overlay fr
#   WITH_LLM=0 deploy/k8s/up.sh      # no Ollama (low-memory machine): retrieval and refusals only
#   CONTEXT=docker-desktop deploy/k8s/up.sh   # Docker Desktop's own Kubernetes instead of kind
#   deploy/k8s/down.sh               # delete the cluster
#
# Needs docker, kubectl, and kind for the default target. Idempotent: re-running updates
# what changed.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Git Bash on Windows rewrites arguments such as /data into C:/Program Files/Git/data
# before kubectl sees them. No effect on Linux or macOS.
export MSYS_NO_PATHCONV=1

CLUSTER=${CLUSTER:-finrag}
NS=finrag
WITH_LLM=${WITH_LLM:-1}
KIND_NODE=${KIND_NODE:-kindest/node:v1.32.2}
LOADER_IMAGE=busybox:1.36.1
# The cluster: a kind cluster created here (default), or any existing context, such as
# docker-desktop, whose nodes share this machine's Docker images.
CONTEXT=${CONTEXT:-kind-$CLUSTER}
KIND=0; [ "$CONTEXT" = "kind-$CLUSTER" ] && KIND=1
k() { kubectl --context "$CONTEXT" -n "$NS" "$@"; }
step() { printf '\n== %s\n' "$*"; }

# ------------------------------------------------------------------ corpus to load
OVERLAY=local
if [ "${CORPUS:-}" = fr ] || [ "${CORPUS:-}" = fr-sample ]; then
  OVERLAY=fr
  # fr: the 45 sheets in data/docs. fr-sample: one sheet, committed (what CI runs).
  SRC=data/docs; [ "$CORPUS" = fr-sample ] && SRC=deploy/k8s/sample-fr
  [ -n "$(ls "$SRC"/*.json 2>/dev/null)" ] || { echo "no $SRC: run uv run finrag fetch-docs" >&2; exit 1; }
  STAGE=$(mktemp -d)
  mkdir -p "$STAGE/docs" "$STAGE/raw"  # raw/ stays empty: documents mode reads docs/
  cp "$SRC"/*.json "$SRC/SOURCE.md" "$STAGE/docs/"
elif [ "${CORPUS:-}" = sample ] || { [ -z "${CORPUS:-}" ] && [ ! -f data/manifest.json ]; }; then
  CORPUS=sample
  STAGE=$(mktemp -d)
  mkdir -p "$STAGE/raw"
  cp deploy/k8s/sample/manifest.json "$STAGE/"
  cp tests/fixtures/company_tickers.json "$STAGE/"
  cp tests/fixtures/acme_10k.html "$STAGE/raw/ACME_10-K_2025-09-27.html"
else
  CORPUS=${CORPUS:-data}
  STAGE=$(mktemp -d)
  # The documents and their metadata only: never the local index (data/qdrant).
  cp -r "$CORPUS/raw" "$CORPUS/manifest.json" "$CORPUS/company_tickers.json" "$STAGE/"
  [ -d "$CORPUS/facts" ] && cp -r "$CORPUS/facts" "$STAGE/"
fi
trap 'rm -rf "$STAGE"' EXIT
if [ "$OVERLAY" = fr ]; then
  echo "corpus: $CORPUS ($(ls "$STAGE/docs"/*.json | wc -l) service-public.fr sheets), overlay $OVERLAY"
else
  echo "corpus: $CORPUS ($(ls "$STAGE/raw" | wc -l) filings), overlay $OVERLAY"
fi

# ------------------------------------------------------------------ cluster and image
step "cluster $CONTEXT"
if [ "$KIND" = 1 ]; then
  kind get clusters 2>/dev/null | grep -qx "$CLUSTER" \
    || kind create cluster --name "$CLUSTER" --image "$KIND_NODE" --wait 180s
fi
kubectl --context "$CONTEXT" get nodes

step "images"
docker build -q -t finrag:local .
# Third-party images: pulled once on this machine, so the nodes never pull (faster, no
# registry rate limit, and the same path as an air-gapped site).
EXTERNAL="$LOADER_IMAGE qdrant/qdrant:v1.20.0"
[ "$WITH_LLM" != 0 ] && EXTERNAL="$EXTERNAL ollama/ollama:0.12.6"
for img in $EXTERNAL; do
  docker image inspect "$img" >/dev/null 2>&1 || docker pull -q --platform linux/amd64 "$img"
done
if [ "$KIND" = 1 ]; then
  kind load docker-image finrag:local --name "$CLUSTER"
  for img in $EXTERNAL; do
    # Not `kind load docker-image`: with Docker Desktop's containerd image store it
    # imports --all-platforms and fails on the layers of other architectures
    # ("content digest ... not found"). Import the node's platform only.
    docker save "$img" | docker exec -i "$CLUSTER-control-plane" ctr --namespace=k8s.io \
      images import --digests --snapshotter=overlayfs --platform linux/amd64 - >/dev/null
    echo "loaded $img"
  done
else
  echo "context $CONTEXT: its nodes use this machine's Docker images, nothing to load"
fi

# ------------------------------------------------------------------ services
step "services (kubectl apply -k deploy/k8s/overlays/$OVERLAY)"
kubectl --context "$CONTEXT" apply -k deploy/k8s/overlays/$OVERLAY
if ! k get secret finrag-secrets >/dev/null 2>&1; then
  # A random key, sent on stdin: never in a file, the repository, or a command line
  # that `ps` could show.
  random_hex() { head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
  printf 'apiVersion: v1\nkind: Secret\nmetadata: {name: finrag-secrets}\nstringData:\n  %s: %s\n' \
    FINRAG_API_KEY "$(random_hex)" | k apply -f - >/dev/null
  echo "created Secret finrag-secrets (a random API key, kept only in the cluster)"
fi
if [ "$WITH_LLM" = 0 ]; then
  k scale deployment/ollama --replicas=0
fi
k rollout status statefulset/qdrant --timeout=300s

# ------------------------------------------------------------------ corpus volume
step "corpus -> volume finrag-corpus"
k delete pod corpus-loader --ignore-not-found --wait
k apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: corpus-loader
spec:
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    seccompProfile: {type: RuntimeDefault}
  containers:
    - name: loader
      image: $LOADER_IMAGE
      imagePullPolicy: IfNotPresent
      command: [sleep, "3600"]
      volumeMounts: [{name: corpus, mountPath: /data}]
      securityContext:
        allowPrivilegeEscalation: false
        capabilities: {drop: [ALL]}
  volumes:
    - name: corpus
      persistentVolumeClaim: {claimName: finrag-corpus}
EOF
k wait --for=condition=Ready pod/corpus-loader --timeout=180s
k exec corpus-loader -- sh -c 'rm -rf /data/* && mkdir -p /data'
# tar through the exec stream: no Windows drive letters for `kubectl cp` to misread.
tar -C "$STAGE" -cf - . | k exec -i corpus-loader -- tar -xf - -C /data
# The files only: /data itself belongs to root and the loader is not root.
k exec corpus-loader -- sh -c 'chmod -R a+rX /data/* && ls -R /data | head -20'
k delete pod corpus-loader --wait

# ------------------------------------------------------------------ jobs
run_job() {  # name label timeout
  k delete job "$1" --ignore-not-found --wait
  kubectl --context "$CONTEXT" apply -k deploy/k8s/overlays/local/jobs -l "finrag/job=$2"
  if ! k wait --for=condition=complete "job/$1" --timeout="$3"; then
    k logs "job/$1" --tail=50 || true
    echo "job $1 did not complete" >&2
    exit 1
  fi
  k logs "job/$1" --tail=8
}
if [ "$WITH_LLM" != 0 ]; then
  k rollout status deployment/ollama --timeout=300s
  step "model (job finrag-pull-model, digest checked)"
  run_job finrag-pull-model pull-model 30m
fi
step "ingestion (job finrag-ingest)"
run_job finrag-ingest ingest 20m

# ------------------------------------------------------------------ ready, then test
step "API rollout"
k rollout restart deployment/finrag-api  # a fresh scope from the new manifest
k rollout status deployment/finrag-api --timeout=300s
k get pods -o wide

step "smoke test"
CONTEXT=$CONTEXT WITH_LLM=$WITH_LLM CORPUS=$CORPUS deploy/k8s/smoke.sh
