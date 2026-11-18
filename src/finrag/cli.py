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


def cmd_pull_model(args, s) -> None:
    """Load the model into the Ollama server at OLLAMA_URL, digest checked (deploy/k8s)."""
    from finrag.models import pull_verified

    try:
        digest = pull_verified(s.preset, s.ollama_url)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"{s.preset.model_id} ready at {s.ollama_url}, digest {digest[:12]} matches the pin")


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


def cmd_fetch_docs(args, s) -> None:
    """service-public.fr open dataset -> the Interior ministry's sheets in data/docs."""
    import httpx

    from finrag import documents

    zip_path = s.data_dir / "sp" / "vosdroits.zip"
    if not zip_path.exists() or args.refresh:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", documents.DATASET_URL, timeout=300.0,
                          follow_redirects=True) as r:
            r.raise_for_status()
            with zip_path.open("wb") as f:
                for part in r.iter_bytes():
                    f.write(part)
    docs = documents.interior_sheets(zip_path)
    documents.save(docs, s.docs_dir)
    print(f"{len(docs)} sheets -> {s.docs_dir}  ({documents.LICENCE})")


def _ingest_documents(s) -> None:
    from finrag import documents
    from finrag.store import Store, embed

    docs = documents.load(s.docs_dir)
    if not docs:
        sys.exit(f"No documents in {s.docs_dir}: run `finrag fetch-docs` first.")
    chunks = documents.to_chunks(docs)
    for d in docs:
        print(f"{d.id}: {d.title} ({len(d.sections)} sections)")
    n = Store(s).rebuild(chunks, len(embed(["dimension probe"], s.embed_model)[0]))
    print(f"indexed {n} chunks from {len(docs)} documents into {s.collection}")


def cmd_ingest(args, s) -> None:
    from finrag import edgar, parse
    from finrag.chunk import chunk_filing
    from finrag.store import Store, embed

    if s.domain == "documents":
        return _ingest_documents(s)
    filings = edgar.load_manifest(s.manifest_path)
    if not filings:
        sys.exit("No filings: run `finrag fetch` first.")
    chunks = []
    for f in filings:
        html = (s.raw_dir / f"{f.doc_id}.html").read_text(encoding="utf-8")
        sections = parse.split_sections(parse.html_to_text(html), f.form)
        # Word counts make a broken parse visible: a pointer-only or page-sized
        # Item shows up as a handful of words.
        found = "  ".join(f"{sec.key}:{len(sec.text.split())}" for sec in sections)
        print(f"{f.doc_id}: {len(sections)} sections (Item:words)\n  "
              f"{found or 'NONE: check the parser'}")
        chunks += chunk_filing(f, sections)
    store = Store(s)
    n = store.rebuild(chunks, len(embed(["dimension probe"], s.embed_model)[0]))
    print(f"indexed {n} chunks into {s.collection}")


def cmd_ask(args, s) -> None:
    from finrag.answer import answer
    from finrag.store import Store
    from finrag.tracing import Tracer

    tracer = Tracer.from_settings(s, channel="cli")
    a = answer(args.question, _scope(s), Store(s), s, tracer=tracer)
    tracer.flush()  # a short-lived process: send the trace before exiting
    print(a.answer)
    print(f"\n[{a.stage}] {a.reason}")
    _print_usage(a.usage)
    for src in a.sources:
        print(f"  [{src['n']}] {src['ticker']} {src['form']} {src['period']} "
              f"Item {src['section']} {src['section_title']} (score {src['score']})")
    if a.ungrounded_numbers:
        print("  ungrounded figures:", ", ".join(a.ungrounded_numbers))


def cmd_agent(args, s) -> None:
    from finrag import agent
    from finrag.store import Store
    from finrag.tracing import Tracer

    sc = _scope(s)
    guard, model = agent.from_settings(s, Store(s), sc, args.max_calls)
    tracer = Tracer.from_settings(s, channel="cli")
    a = agent.traced_run(args.question, sc, guard, model, s, tracer)
    tracer.flush()
    print(a.answer)
    print(f"\n[{a.stage}] {a.reason}")
    _print_usage(a.usage)
    print(f"tool calls ({len(a.steps)}/{args.max_calls}):")
    for st in a.steps:
        outcome = "ok" if st["ok"] else f"REFUSED: {st['error']}"
        print(f"  {st['tool']}({st['args']}) -> {outcome}  {st['ms']} ms")
    for src in a.sources:
        print(f"  [{src['n']}] {src['tool']}: {src['ticker']} {src['form']} {src['period']} "
              f"{src['section_title']}")


def cmd_eval(args, s) -> None:
    from finrag import evaluate
    from finrag.store import Store
    from finrag.tracing import Tracer

    # Retrieval-only runs never call the model: nothing worth a trace.
    tracer = (Tracer.from_settings(s, environment="eval", channel="eval")
              if not args.retrieval_only else None)
    report = evaluate.run(evaluate.load_suite(Path(args.suite)), _scope(s), Store(s), s,
                          with_llm=not args.retrieval_only, tracer=tracer)
    if tracer:
        tracer.flush()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if "model" in report:
        m, lat, u = report["model"], report["latency_s"], report["usage"]
        where = f"@{m['digest'][:12]} on {m['gpu']}" if m["digest"] else f"via {m['provider']}"
        print(f"model {m['preset']} ({m['model_id']} {where})")
        print(f"latency mean {lat['mean']} s, p50 {lat['p50']} s, p95 {lat['p95']} s; "
              f"est. GPU cost {u['est_cost_usd_total']} USD for {u['llm_answers']} answers")
    for name, g in report["gates"].items():
        print(f"{'PASS' if g['pass'] else 'FAIL'}  {name} = {g['value']}  (gate {g['threshold']})")
    print(f"report: {out}")
    sys.exit(0 if all(g["pass"] for g in report["gates"].values()) else 1)


def _extract_all(ids: list[str], s) -> dict:
    from finrag import documents, extract

    docs = {d.id: d for d in documents.load(s.docs_dir)}
    if not docs:
        sys.exit(f"No documents in {s.docs_dir}: run `finrag fetch-docs` first.")
    missing = [i for i in ids if i not in docs]
    if missing:
        sys.exit(f"not in {s.docs_dir}: {', '.join(missing)}")
    return {i: extract.extract(docs[i], s) for i in (ids or sorted(docs))}


def _extract_cached(ids: list[str], s, cache: str | None, pause: float) -> dict:
    """Like _extract_all, but each finished sheet is saved to `cache`, so an interrupted run
    resumes where it stopped, and `pause` seconds separate model calls (laptop GPU)."""
    if not cache and not pause:
        return _extract_all(ids, s)
    import time

    from finrag import documents, extract

    docs = {d.id: d for d in documents.load(s.docs_dir)}
    missing = [i for i in ids if i not in docs]
    if missing:
        sys.exit(f"not in {s.docs_dir}: {', '.join(missing)}")
    folder = Path(cache) if cache else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
    results = {}
    for n, i in enumerate(ids, 1):
        path = folder / f"{i}.json" if folder else None
        if path and path.exists():
            results[i] = extract.from_json(path.read_text(encoding="utf-8"))
            print(f"[{n}/{len(ids)}] {i}: cached", flush=True)
            continue
        if results and pause:
            time.sleep(pause)
        results[i] = extract.extract(docs[i], s, pause=pause)
        if path:
            path.write_text(extract.to_json(results[i]), encoding="utf-8")
        print(f"[{n}/{len(ids)}] {i}: done", flush=True)
    return results


def cmd_extract(args, s) -> None:
    """Sheets -> data/structured/<id>.json, every field with its verified source sentence."""
    from finrag import extract

    out = s.data_dir / "structured"
    out.mkdir(parents=True, exist_ok=True)
    for i, e in _extract_all(args.ids, s).items():
        (out / f"{i}.json").write_text(extract.to_json(e), encoding="utf-8")
        kept = {n: f["value"] for n, f in e.fields.items() if n != "couts"}
        costs = ", ".join(f"{c['montant']} ({c['condition']})" for c in e.fields["couts"])
        print(f"{i}: {kept} couts: [{costs}] rejected: {len(e.rejected)}")
    print(f"-> {out}")


def cmd_eval_extract(args, s) -> None:
    from finrag import extract

    gold = [json.loads(ln) for ln in Path(args.suite).read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    results = _extract_cached([g["id"] for g in gold], s, args.cache, args.pause)
    report = extract.score(results, gold) | {
        "model": s.preset.model_id, "digest": s.preset.digest,
        "sheets": {i: {"fields": e.fields, "rejected": e.rejected} for i, e in results.items()}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("field_accuracy", "amount_recall", "amount_precision",
              "unsupported_values_kept", "rejected_by_verification"):
        print(f"{k}: {report[k]}")
    print(f"report: {out}")


def _print_usage(usage: dict | None) -> None:
    if usage:
        print(f"model {usage['model']}: {usage['prompt_tokens']}+{usage['completion_tokens']} "
              f"tokens, {usage['llm_seconds']} s, est. {usage['est_cost_usd']} USD "
              f"on {usage['gpu']}")


def main() -> None:
    from finrag.models import MODEL_REGISTRY

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", choices=list(MODEL_REGISTRY),
                        help="model preset (default: FINRAG_MODEL, then llama-3b)")
    p = argparse.ArgumentParser(prog="finrag")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download filings from SEC EDGAR")
    f.add_argument("tickers", nargs="+")
    f.add_argument("--forms", nargs="+", default=["10-K"])
    f.add_argument("--limit", type=int, default=1, help="filings per form and company")
    sub.add_parser("ingest", help="parse, chunk, embed and index the downloaded filings "
                                  "(FINRAG_DOMAIN=documents: the documents in data/docs)")
    fd = sub.add_parser("fetch-docs",
                        help="service-public.fr sheets of the Interior ministry -> data/docs")
    fd.add_argument("--refresh", action="store_true", help="download the dataset again")
    a = sub.add_parser("ask", parents=[common])
    a.add_argument("question")
    g = sub.add_parser("agent", parents=[common],
                       help="answer with tools (search, reported figures, calculator)")
    g.add_argument("question")
    g.add_argument("--max-calls", type=int, default=6, help="tool call budget")
    e = sub.add_parser("eval", parents=[common])
    e.add_argument("suite")
    e.add_argument("--out", default="eval/reports/latest.json")
    e.add_argument("--retrieval-only", action="store_true", help="skip the LLM (fast)")
    sub.add_parser("pull-model", parents=[common],
                   help="load the model into the Ollama server at OLLAMA_URL, digest checked")
    x = sub.add_parser("extract", parents=[common],
                       help="sheets in data/docs -> structured fields, each with its proof")
    x.add_argument("ids", nargs="*", help="sheet ids (default: all)")
    xe = sub.add_parser("eval-extract", parents=[common],
                        help="score the extraction against gold read off the sheets")
    xe.add_argument("suite")
    xe.add_argument("--out", default="eval/reports/extraction.json")
    xe.add_argument("--cache", help="folder keeping each finished sheet: a rerun resumes")
    xe.add_argument("--pause", type=float, default=0.0,
                    help="seconds between model calls, to let a laptop GPU cool down")
    args = p.parse_args()
    {"fetch": cmd_fetch, "ingest": cmd_ingest, "ask": cmd_ask, "agent": cmd_agent,
     "eval": cmd_eval, "pull-model": cmd_pull_model,
     "fetch-docs": cmd_fetch_docs, "extract": cmd_extract,
     "eval-extract": cmd_eval_extract}[args.cmd](args, _settings(args))


def _settings(args):
    from dataclasses import replace

    s = get_settings()
    return replace(s, model=args.model) if getattr(args, "model", None) else s


if __name__ == "__main__":
    main()
