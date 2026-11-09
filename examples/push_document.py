"""Reference connector: push a document from an information system into finrag.

What a GED export, a CRM job or an RPA bot does, in a few lines:

    FINRAG_URL=https://<backend> FINRAG_INGEST_KEY=... \\
        uv run python examples/push_document.py policy.txt --id pol-7 \\
        --title "Counterparty risk policy" --groups risques --source ged

The document is searchable at once, and ONLY by users in one of its groups (the SSO
gateway gives each user their groups). Pushing the same --id again replaces it.
The key is read from the environment, never written in code or logs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx


def push(url: str, key: str, doc: dict) -> dict:
    r = httpx.post(f"{url.rstrip('/')}/documents", json=doc,
                   headers={"x-ingest-key": key}, timeout=120)
    if r.status_code >= 400:
        sys.exit(f"{r.status_code}: {r.text[:300]}")
    return r.json()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("file", help="UTF-8 text of the document")
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--groups", nargs="+", required=True, help="who may read it")
    p.add_argument("--source", default="connector", help="ged, crm, erp, rpa...")
    p.add_argument("--doc-url", default="", help="link back to the document in its system")
    a = p.parse_args()
    url, key = os.environ.get("FINRAG_URL", ""), os.environ.get("FINRAG_INGEST_KEY", "")
    if not url or not key:
        sys.exit("set FINRAG_URL and FINRAG_INGEST_KEY")
    doc = {"id": a.id, "title": a.title, "text": Path(a.file).read_text(encoding="utf-8"),
           "groups": a.groups, "source": a.source, "url": a.doc_url}
    print(json.dumps(push(url, key, doc)))


if __name__ == "__main__":
    main()
