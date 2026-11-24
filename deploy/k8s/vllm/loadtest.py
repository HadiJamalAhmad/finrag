"""Load test for the vLLM Service, run INSIDE the cluster (a client pod), standard library only.

    python loadtest.py http://vllm:8000 <concurrency> <seconds> <max_tokens>

Each worker sends chat completions back to back until the time is up; requests already
sent are awaited. Prints one line, `RESULT {json}`: every request's start, latency,
status and generated tokens. Through the Service, kube-proxy spreads the requests over
whatever replicas are ready at that moment.
"""

import json
import sys
import threading
import time
import urllib.request

URL, CONC, SECONDS, MAX_TOKENS = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])

TOPICS = ["a bank's liquidity ratio", "inflation", "a 10-K filing", "interest rates",
          "credit risk", "a balance sheet", "cash flow", "bond yields", "diversification",
          "operating margin", "a central bank", "market risk"]

results = []
lock = threading.Lock()
t0 = time.time()


def worker(i):
    n = 0
    while time.time() - t0 < SECONDS:
        topic = TOPICS[(i + n) % len(TOPICS)]
        n += 1
        body = json.dumps({
            "model": "qwen2.5-0.5b",
            "messages": [{"role": "user",
                          "content": f"In three sentences, explain {topic} to a new analyst."}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        start = time.time()
        rec = {"start": round(start - t0, 3)}
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.load(r)
            rec.update(ok=True, status=200,
                       tokens=out.get("usage", {}).get("completion_tokens", 0))
        except Exception as e:  # recorded, never hidden: the report counts failures
            rec.update(ok=False, status=getattr(e, "code", None), error=type(e).__name__)
        rec["latency"] = round(time.time() - start, 3)
        with lock:
            results.append(rec)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONC)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("RESULT " + json.dumps({
    "concurrency": CONC, "seconds": SECONDS, "max_tokens": MAX_TOKENS,
    "wall": round(time.time() - t0, 3), "requests": results,
}))
