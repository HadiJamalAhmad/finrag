#!/usr/bin/env bash
# The sovereign-LLM proof on a Kubernetes cluster, end to end (what CI runs):
#   weights in (checksums) -> vLLM + Prometheus -> KEDA autoscaling -> sovereignty checks
#   -> load test with the replica count sampled every 5 s -> report (deploy/k8s/vllm/report.py).
#
#   WEIGHTS=/path/to/qwen2.5-0.5b-instruct CONTEXT=kind-llm deploy/k8s/vllm/run.sh
#
# WEIGHTS holds the files listed in weights.lock (the workflow downloads them at a pinned
# revision). Needs kubectl, helm and python3 on this machine. Writes $OUT (default
# eval/reports/k8s-vllm.json) and exits non-zero if any check fails.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MSYS_NO_PATHCONV=1

: "${WEIGHTS:?set WEIGHTS to the directory holding the model files}"
CONTEXT=${CONTEXT:-kind-llm}
NS=finrag-llm
OUT=${OUT:-eval/reports/k8s-vllm.json}
CONC=${CONC:-24}          # parallel clients in the load test
SECONDS_LOAD=${SECONDS_LOAD:-300}
MAX_TOKENS=${MAX_TOKENS:-64}
KEDA_VERSION=2.21.1
CLIENT_IMAGE=python:3.12.8-slim-bookworm
D=deploy/k8s/vllm
WORK=$(mktemp -d)
k() { kubectl --context "$CONTEXT" -n "$NS" "$@"; }
step() { printf '\n== %s\n' "$*"; }
PF=""
SAMPLER=""
cleanup() { [ -n "$PF" ] && kill "$PF" 2>/dev/null; [ -n "$SAMPLER" ] && kill "$SAMPLER" 2>/dev/null; true; }
trap cleanup EXIT

# ------------------------------------------------------------------ weights
step "weights -> volume llm-weights (checked against weights.lock inside the cluster)"
kubectl --context "$CONTEXT" apply -f $D/namespace.yaml
k apply -f $D/weights.yaml  # the volume first: vLLM starts only once the weights are in
k delete pod weights-loader --ignore-not-found --wait
k apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: weights-loader
spec:
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    seccompProfile: {type: RuntimeDefault}
  containers:
    - name: loader
      image: busybox:1.36.1
      command: [sleep, "3600"]
      volumeMounts: [{name: weights, mountPath: /models}]
      securityContext:
        allowPrivilegeEscalation: false
        capabilities: {drop: [ALL]}
  volumes:
    - name: weights
      persistentVolumeClaim: {claimName: llm-weights}
EOF
k wait --for=condition=Ready pod/weights-loader --timeout=300s
k exec weights-loader -- sh -c 'rm -rf /models/qwen2.5-0.5b-instruct && mkdir -p /models/qwen2.5-0.5b-instruct'
tar -C "$WEIGHTS" -cf - . | k exec -i weights-loader -- tar -xf - -C /models/qwen2.5-0.5b-instruct
k exec -i weights-loader -- sh -c 'cat > /tmp/weights.lock' < $D/weights.lock
k exec weights-loader -- sh -c 'cd /models/qwen2.5-0.5b-instruct && sha256sum -c /tmp/weights.lock \
  && chmod -R a+rX /models/qwen2.5-0.5b-instruct' | tee "$WORK/weights-check.txt"
k delete pod weights-loader --wait

# ------------------------------------------------------------------ services
step "vLLM and Prometheus (kubectl apply -k $D)"
kubectl --context "$CONTEXT" apply -k $D
k rollout status deployment/prometheus --timeout=300s
k rollout status deployment/vllm --timeout=900s
k get pods -o wide

step "KEDA $KEDA_VERSION, then the ScaledObject"
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update kedacore >/dev/null
helm --kube-context "$CONTEXT" upgrade --install keda kedacore/keda --version "$KEDA_VERSION" \
  --namespace keda --create-namespace --wait --timeout 10m
kubectl --context "$CONTEXT" apply -f $D/autoscale/scaledobject.yaml
for _ in $(seq 60); do
  [ "$(k get scaledobject vllm -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')" = True ] && break
  sleep 5
done
k get scaledobject vllm
k get hpa

# ------------------------------------------------------------------ client pod
step "client pod (label finrag-llm/client: the only pods allowed to call vLLM)"
k delete pod llm-client --ignore-not-found --wait
k apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: llm-client
  labels: {finrag-llm/client: "true"}
spec:
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    seccompProfile: {type: RuntimeDefault}
  containers:
    - name: client
      image: $CLIENT_IMAGE
      command: [sleep, "7200"]
      resources:
        requests: {cpu: 100m, memory: 128Mi}
      securityContext:
        allowPrivilegeEscalation: false
        capabilities: {drop: [ALL]}
EOF
k wait --for=condition=Ready pod/llm-client --timeout=300s

# ------------------------------------------------------------------ sovereignty
step "sovereignty: the model server is cut off from the Internet and still answers"
k exec -i deploy/vllm -- python3 - < $D/netcheck.py | tail -1 | tee "$WORK/net-vllm.json"
k exec -i llm-client -- python - < $D/netcheck.py | tail -1 | tee "$WORK/net-control.json"
k exec deploy/vllm -- printenv HF_HUB_OFFLINE TRANSFORMERS_OFFLINE | tee "$WORK/offline-env.txt"
k exec -i llm-client -- python - <<'EOF' | tail -1 | tee "$WORK/answer.json"
import json, time, urllib.request
body = json.dumps({"model": "qwen2.5-0.5b", "temperature": 0, "max_tokens": 16, "messages": [
    {"role": "user", "content": "What is the capital of France? Answer with one word."}]}).encode()
t = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://vllm:8000/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json"}), timeout=300))
print(json.dumps({"question": "What is the capital of France? Answer with one word.",
                  "answer": r["choices"][0]["message"]["content"],
                  "model": r["model"], "seconds": round(time.time() - t, 2)}))
EOF
head -c 600 "$WEIGHTS/LICENSE" | tr -s ' \n' ' ' | head -c 80 > "$WORK/license.txt"; echo

# ------------------------------------------------------------------ load test
step "load test: $CONC clients for ${SECONDS_LOAD}s, replicas sampled every 5 s"
k port-forward svc/prometheus 19090:9090 >/dev/null 2>&1 &
PF=$!
for _ in $(seq 30); do curl -s -o /dev/null http://127.0.0.1:19090/-/ready && break; sleep 1; done
promq() {  # one instant query -> its value (0 when empty)
  curl -s --get http://127.0.0.1:19090/api/v1/query --data-urlencode "query=$1" \
    | python3 -c "import sys,json;r=json.load(sys.stdin)['data']['result'];print(r[0]['value'][1] if r else 0)"
}
T0=$(date +%s)
(
  while true; do
    ready=$(k get deploy vllm -o jsonpath='{.status.readyReplicas}')
    desired=$(k get deploy vllm -o jsonpath='{.spec.replicas}')
    printf '%s %s %s %s %s\n' "$(( $(date +%s) - T0 ))" "${ready:-0}" "${desired:-0}" \
      "$(promq 'sum(vllm:num_requests_waiting)')" "$(promq 'sum(vllm:num_requests_running)')"
    sleep 5
  done
) > "$WORK/samples.txt" 2>/dev/null &
SAMPLER=$!
sleep 15  # a quiet baseline before the burst
k exec -i llm-client -- python - http://vllm:8000 "$CONC" "$SECONDS_LOAD" "$MAX_TOKENS" \
  < $D/loadtest.py | grep '^RESULT ' | sed 's/^RESULT //' > "$WORK/load.json"
sleep 20  # the queue after the burst
kill "$SAMPLER"; SAMPLER=""
cat "$WORK/samples.txt"

# Per replica: requests completed and tokens generated (counters since each pod started).
curl -s --get http://127.0.0.1:19090/api/v1/query \
  --data-urlencode 'query=sum by (instance) (vllm:request_success_total)' > "$WORK/per-pod.json"
k get pods -o wide

# ------------------------------------------------------------------ report
step "report -> $OUT"
python3 $D/report.py "$WORK" "$OUT" "$T0"
