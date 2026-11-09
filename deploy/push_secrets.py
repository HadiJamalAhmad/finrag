"""Push the hosted-model and tracing keys from .env to the Modal secret `finrag-providers`.

    uv run python deploy/push_secrets.py

Reads only these names from .env and never prints a value. A key missing from .env is
left out: its model stays unavailable (503) or tracing stays off. The SEC User-Agent is
NOT pushed: Modal never calls SEC (the company facts are cached in the Volume).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

NAMES = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
         "LANGFUSE_HOST"]
ENV = Path(__file__).resolve().parents[1] / ".env"


def read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main() -> None:
    env = read_env(ENV)
    found = {k: env[k] for k in NAMES if env.get(k)}
    if not found:
        sys.exit(f"none of {', '.join(NAMES)} is set in .env: nothing pushed")
    args = ["modal", "secret", "create", "finrag-providers", "--force",
            *[f"{k}={v}" for k, v in found.items()]]
    # Output suppressed: the CLI may echo the command line, values included.
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("modal secret create failed (output hidden: it may contain the keys)")
    print("finrag-providers updated with:", ", ".join(found), "(values not shown)")
    print("redeploy for running containers to pick it up: uv run modal deploy deploy/modal_app.py")


if __name__ == "__main__":
    main()
