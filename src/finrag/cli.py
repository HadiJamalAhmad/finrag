"""Command line: fetch -> ingest -> ask / eval.

  uv run finrag fetch AAPL MSFT NVDA JPM --forms 10-K 10-Q --limit 2
  uv run finrag ingest
  uv run finrag ask "What supply chain risks does Apple describe?"
  uv run finrag eval eval/suites/seed.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from finrag.config import get_settings


def _scope(settings):
    from finrag import edgar, scope

    names = (json.loads(settings.sec_names_path.read_text(encoding="utf-8"))
             if settings.sec_names_path.exists() else None)
    return scope.Scope.build(edgar.load_manifest(settings.manifest_path), names)


def cmd_fetch(args, s) -> None:
    from finrag import edgar

    client = edgar.EdgarClient(s.sec_user_agent)
    filings = edgar.fetch(client, args.tickers, tuple(args.forms), args.limit,
                          s.raw_dir, s.manifest_path, s.sec_names_path)
    for f in filings:
        print(f"{f.ticker:5} {f.form:5} period {f.period} filed {f.filed}")


def cmd_ingest(args, s) -> None:
    from finrag import edgar, parse
    from finrag.chunk import chunk_filing
    from finrag.store import Store, embed

    filings = edgar.load_manifest(s.manifest_path)
    if not filings:
        sys.exit("No filings: run `finrag fetch` first.")
    chunks = []
    for f in filings:
        html = (s.raw_dir / f"{f.doc_id}.html").read_text(encoding="utf-8")
        sections = parse.split_sections(parse.html_to_text(html), f.form)
        found = ", ".join(sec.key for sec in sections) or "NONE: check the parser"
        print(f"{f.doc_id}: {len(sections)} sections ({found})")
        chunks += chunk_filing(f, sections)
    store = Store(s)
    store.reset(len(embed(["dimension probe"], s.embed_model)[0]))
    print(f"indexed {store.add(chunks)} chunks into {s.collection}")


def cmd_ask(args, s) -> None:
    from finrag.answer import answer
    from finrag.store import Store

    a = answer(args.question, _scope(s), Store(s), s)
    print(a.answer)
    print(f"\n[{a.stage}] {a.reason}")
    for src in a.sources:
        print(f"  [{src['n']}] {src['ticker']} {src['form']} {src['period']} "
              f"Item {src['section']} {src['section_title']} (score {src['score']})")
    if a.ungrounded_numbers:
        print("  ungrounded figures:", ", ".join(a.ungrounded_numbers))


def cmd_eval(args, s) -> None:
    from finrag import evaluate
    from finrag.store import Store

    report = evaluate.run(evaluate.load_suite(Path(args.suite)), _scope(s), Store(s), s,
                          with_llm=not args.retrieval_only)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, g in report["gates"].items():
        print(f"{'PASS' if g['pass'] else 'FAIL'}  {name} = {g['value']}  (gate {g['threshold']})")
    print(f"report: {out}")
    sys.exit(0 if all(g["pass"] for g in report["gates"].values()) else 1)


def main() -> None:
    p = argparse.ArgumentParser(prog="finrag")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download filings from SEC EDGAR")
    f.add_argument("tickers", nargs="+")
    f.add_argument("--forms", nargs="+", default=["10-K"])
    f.add_argument("--limit", type=int, default=1, help="filings per form and company")
    sub.add_parser("ingest", help="parse, chunk, embed and index the downloaded filings")
    a = sub.add_parser("ask")
    a.add_argument("question")
    e = sub.add_parser("eval")
    e.add_argument("suite")
    e.add_argument("--out", default="eval/reports/latest.json")
    e.add_argument("--retrieval-only", action="store_true", help="skip the LLM (fast)")
    args = p.parse_args()
    {"fetch": cmd_fetch, "ingest": cmd_ingest, "ask": cmd_ask, "eval": cmd_eval}[args.cmd](
        args, get_settings())


if __name__ == "__main__":
    main()
