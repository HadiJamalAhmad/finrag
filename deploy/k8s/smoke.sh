#!/usr/bin/env bash
# Black-box checks of a running finrag cluster, over HTTP through the API Service.
# Every check prints PASS or FAIL; the script exits non-zero on any FAIL.
#
#   CONTEXT=kind-finrag WITH_LLM=1 CORPUS=sample deploy/k8s/smoke.sh
set -uo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash: keep /paths as they are (see up.sh)

CLUSTER=${CLUSTER:-finrag}
WITH_LLM=${WITH_LLM:-1}
CORPUS=${CORPUS:-sample}
PORT=${PORT:-18080}
CONTEXT=${CONTEXT:-kind-$CLUSTER}
k() { kubectl --context "$CONTEXT" -n finrag "$@"; }
PY=$(command -v python3 || command -v python)

if [ "$CORPUS" = sample ]; then
  IN_Q="Who supplies Acme's precision gears?"
else
  IN_Q="What supply chain risks does Apple describe?"
fi
OUT_Q="What was Tesla's revenue in fiscal 2025?"  # TSLA is a SEC company outside the corpus

KEY=$(k get secret finrag-secrets -o jsonpath='{.data.FINRAG_API_KEY}' | base64 -d)
k port-forward svc/finrag-api "$PORT:80" >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null' EXIT
for _ in $(seq 30); do curl -s -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 1; done

URL="http://127.0.0.1:$PORT"
FAILS=0
check() {  # description, then a command that must succeed
  local what=$1; shift
  if "$@"; then echo "PASS  $what"; else echo "FAIL  $what"; FAILS=$((FAILS + 1)); fi
}
status() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
# A field of a JSON body; empty (not a traceback) when the body is an error page.
field() {
  "$PY" -c "import json,sys
try: print(json.load(sys.stdin).get('$1', ''))
except ValueError: print('')"
}
ask() {
  curl -s --max-time 900 -H "x-api-key: $KEY" -H 'content-type: application/json' \
    -d "{\"question\": \"$1\"}" "$URL/query"
}

check "liveness /health answers without a key" \
  test "$(status "$URL/health")" = 200
check "readiness /ready: index reachable, filled, manifest present" \
  test "$(status "$URL/ready")" = 200
echo "      $(curl -s "$URL/ready")"
check "no x-api-key: 401" \
  test "$(status "$URL/models")" = 401
check "wrong x-api-key: 401" \
  test "$(status -H 'x-api-key: wrong' "$URL/models")" = 401
check "right x-api-key: 200" \
  test "$(status -H "x-api-key: $KEY" "$URL/models")" = 200

OUT=$(ask "$OUT_Q")
check "a company outside the corpus is refused at the scope stage" \
  test "$(echo "$OUT" | field stage)" = scope

ENDPOINTS=$(k get endpoints finrag-api -o jsonpath='{range .subsets[*].addresses[*]}{.ip} {end}')
check "two API replicas behind the Service ($ENDPOINTS)" \
  test "$(echo "$ENDPOINTS" | wc -w)" -ge 2

if [ "$WITH_LLM" != 0 ]; then
  echo "      asking (CPU inference, can take a few minutes): $IN_Q"
  START=$(date +%s)
  IN=$(ask "$IN_Q")
  STAGE=$(echo "$IN" | field stage)
  echo "      $(( $(date +%s) - START )) s  stage=${STAGE:-none}"
  if [ -z "$STAGE" ]; then
    echo "      not an answer: $(echo "$IN" | head -c 200)"
  else
    echo "      answer: $(echo "$IN" | field answer | head -c 400)"
    echo "      sources: $(echo "$IN" | "$PY" -c 'import json,sys
print("; ".join(s.get("cite") or s.get("section_title", "?") for s in json.load(sys.stdin)["sources"]))' | head -c 300)"
  fi
  check "an in-scope question goes through retrieval and the in-cluster LLM: answered" \
    test "$STAGE" = answered
  SOURCES=$(echo "$IN" | "$PY" -c 'import json,sys
try: print(len(json.load(sys.stdin)["sources"]))
except (ValueError, KeyError): print(0)')
  check "the answer cites its sources ($SOURCES)" test "$SOURCES" -gt 0
else
  echo "SKIP  answer through the LLM (WITH_LLM=0)"
fi

echo
if [ "$FAILS" -eq 0 ]; then echo "smoke test: all checks passed"; else echo "smoke test: $FAILS failed"; fi
exit "$FAILS"
