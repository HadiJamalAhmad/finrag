"""Turn run.sh's raw outputs into eval/reports/k8s-vllm.json, and check them.

    python3 deploy/k8s/vllm/report.py <work dir> <out.json> <sampler start epoch>

Every check prints PASS or FAIL; the exit code is non-zero if any fails. Nothing here
loosens a check to make it pass: a FAIL is a finding to fix in the deployment.
"""

import json
import platform
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

work, out = Path(sys.argv[1]), Path(sys.argv[2])


def read(name):
    return (work / name).read_text(encoding="utf-8").strip()


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    i = min(len(values) - 1, max(0, round(q / 100 * (len(values) - 1))))
    return values[i]


def cpu():
    try:
        info = Path("/proc/cpuinfo").read_text()
    except OSError:
        return {"machine": platform.machine()}
    model = next((ln.split(":", 1)[1].strip() for ln in info.splitlines()
                  if ln.startswith("model name")), "?")
    flags = next((ln.split(":", 1)[1].split() for ln in info.splitlines()
                  if ln.startswith("flags")), [])
    return {"model": model, "vcpus": info.count("processor\t:"),
            "avx512f": "avx512f" in flags, "avx2": "avx2" in flags}


# ------------------------------------------------------------------ sovereignty
weights_lines = [ln for ln in read("weights-check.txt").splitlines() if ln.strip()]
net_vllm = json.loads(read("net-vllm.json"))
net_control = json.loads(read("net-control.json"))
offline = read("offline-env.txt").split()
answer = json.loads(read("answer.json"))
license_head = read("license.txt")

# ------------------------------------------------------------------ load test
load = json.loads(read("load.json"))
reqs = load["requests"]
ok = [r for r in reqs if r["ok"]]
failed = [r for r in reqs if not r["ok"]]
tokens = sum(r.get("tokens", 0) for r in ok)
lat = [r["latency"] for r in ok]
window = 30
buckets = {}
for r in ok:
    b = int((r["start"] + r["latency"]) // window)
    buckets[b] = buckets.get(b, 0) + r.get("tokens", 0)
throughput = [{"from_s": b * window, "tokens_per_s": round(v / window, 1)}
              for b, v in sorted(buckets.items())]

samples = []
for ln in read("samples.txt").splitlines():
    parts = ln.split()
    if len(parts) == 5:
        t, ready, desired, waiting, running = parts
        samples.append({"t": int(t), "ready": int(ready), "desired": int(desired),
                        "waiting": float(waiting), "running": float(running)})
# The burst starts when run.sh launches the load test (recorded, not guessed from a gauge).
burst = int(read("load-start.txt"))
gaps = [b["t"] - a["t"] for a, b in zip(samples, samples[1:], strict=False)]
sample_period = round(statistics.median(gaps), 1) if gaps else None


def first(pred):
    s = next((s for s in samples if pred(s)), None)
    return None if s is None else s["t"] - burst


per_pod_raw = json.loads(read("per-pod.json"))["data"]["result"]
per_pod = {r["metric"].get("instance", "?"): int(float(r["value"][1])) for r in per_pod_raw}
max_ready = max((s["ready"] for s in samples), default=0)

checks = {
    "weights_verified": len(weights_lines) == 8 and all(ln.endswith(": OK")
                                                        for ln in weights_lines),
    "vllm_pod_cut_off_from_internet": not any(v["reached"] for v in net_vllm.values()),
    "control_pod_reaches_internet": net_control["https_huggingface"]["reached"]
    and net_control["tcp_1.1.1.1"]["reached"],
    "answers_while_cut_off": "paris" in answer["answer"].lower(),
    "offline_mode_set": offline == ["1", "1"],
    "no_failed_request": len(ok) > 0 and not failed,
    "scaled_to_3_ready_replicas": max_ready == 3,
    "load_spread_over_replicas": sum(1 for v in per_pod.values() if v > 0) >= 2,
    "queue_drained_after_burst": bool(samples) and samples[-1]["waiting"] == 0,
}

report = {
    "what": "vLLM on Kubernetes (kind), open weights from a volume, egress blocked, "
            "KEDA autoscaling on vLLM's queue, Prometheus metrics, in-cluster load test",
    "date": datetime.now(UTC).strftime("%Y-%m-%d"),
    "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct",
              "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
              "license": license_head, "dtype": "bfloat16", "device": "CPU"},
    "versions": {"vllm_image": "vllm/vllm-openai-cpu:v0.34.0-x86_64",
                 "keda": "2.21.1", "prometheus": "v3.15.1"},
    "machine": cpu(),
    "sovereignty": {"weights_sha256": weights_lines, "vllm_pod_network": net_vllm,
                    "control_pod_network": net_control,
                    "HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE": offline, "answer": answer},
    "load": {
        "concurrency": load["concurrency"], "seconds": load["seconds"],
        "max_tokens": load["max_tokens"], "requests": len(reqs), "ok": len(ok),
        "failed": len(failed), "tokens": tokens,
        "tokens_per_s": round(tokens / load["wall"], 1) if load["wall"] else None,
        "latency_s": {"p50": pct(lat, 50), "p95": pct(lat, 95),
                      "mean": round(statistics.mean(lat), 3) if lat else None},
        "throughput_by_30s": throughput,
    },
    "autoscaling": {
        "max_ready_replicas": max_ready,
        "burst_starts_at_s": burst,
        "sample_period_s": sample_period,  # the times below are known to this precision
        "seconds_from_burst_to_scale_decision": first(
            lambda s: s["t"] >= burst and s["desired"] > 1),
        "seconds_from_burst_to_3_ready": first(lambda s: s["t"] >= burst and s["ready"] >= 3),
        "requests_completed_per_replica": per_pod,
        "timeline": samples,
    },
    "note": "One runner, 4 shared vCPUs: extra replicas prove the operations (scheduling, "
            "readiness, load spreading, queue-driven scaling), not more throughput. The "
            "GPU throughput is measured separately (vllm-throughput.json, Kaggle T4).",
    "checks": checks,
    "passed": all(checks.values()),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

for name, good in checks.items():
    print(("PASS " if good else "FAIL ") + name)
lo = report["load"]
print(f"requests {lo['ok']}/{lo['requests']} ok, p50 {lo['latency_s']['p50']} s, "
      f"p95 {lo['latency_s']['p95']} s, {lo['tokens_per_s']} tok/s, "
      f"replicas max {max_ready}, per replica {per_pod}")
sys.exit(0 if report["passed"] else 1)
