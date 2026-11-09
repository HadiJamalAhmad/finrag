"""Image build step: pull every model in models.lock and fail on a digest mismatch.

A tag is a moving pointer; the digest is the weights. If Ollama now serves different
weights under a pinned tag, the build stops here instead of silently changing the
model behind every eval number.

    python pull_models.py /deploy/models.lock
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434"


def read_lock(path: str) -> dict[str, str]:
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            tag, digest = line.split()
            out[tag] = digest
    return out


def main(lock_path: str) -> None:
    lock = read_lock(lock_path)
    server = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=1)
                break
            except OSError:
                time.sleep(1)
        for tag in lock:
            subprocess.run(["ollama", "pull", tag], check=True)
        tags = json.load(urllib.request.urlopen(f"{OLLAMA}/api/tags"))["models"]
        pulled = {m["name"]: m["digest"] for m in tags}
        bad = [f"{t}: pinned {d[:12]}, pulled {pulled.get(t, 'nothing')[:12]}"
               for t, d in lock.items() if pulled.get(t) != d]
        if bad:
            sys.exit("digest mismatch (update models.lock AND src/finrag/models.py "
                     "together, then re-run the eval):\n  " + "\n  ".join(bad))
        print("all models match models.lock:", ", ".join(lock))
    finally:
        server.terminate()


if __name__ == "__main__":
    main(sys.argv[1])
