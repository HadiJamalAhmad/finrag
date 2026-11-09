"""Langfuse traces: one per question, one span per stage, the LLM call as a generation.

    trace "query"  tags: [model, stage]           input: question   output: answer / refusal
      span  scope            in scope? tickers, years
      span  retrieve         top-k passages with scores
      span  evidence_gate    best score vs min_score         (WARNING when it refuses)
      gen   llm              model, prompt, answer, tokens, cost
      span  numbers          grounded? ungrounded figures    (WARNING when it refuses)

Plain HTTP to Langfuse's public ingestion API (POST /api/public/ingestion, basic auth
with the project's public and secret keys), like the rest of finrag: no SDK. Tracing
must never break an answer: without keys it is a no-op, and a failed send is logged and
dropped. Traces hold questions and passages of public SEC filings; they go to Langfuse
Cloud, EU region (the owner's choice).
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

import httpx

TIMEOUT = 5.0


def _iso(t: float | None = None) -> str:
    return datetime.fromtimestamp(t or time.time(), UTC).isoformat(timespec="milliseconds")


class Span:
    """What a stage reports; filled inside `with trace.span(...) as sp:`."""

    def __init__(self):
        self.output = None
        self.metadata: dict = {}
        self.warning: str | None = None  # set when the stage refuses


class Trace:
    def __init__(self, tracer: Tracer, name: str, input, tags: list[str], metadata: dict):
        self.tracer, self.id, self.events = tracer, str(uuid.uuid4()), []
        self.name, self.input, self.tags, self.metadata = name, input, list(tags), metadata
        self.start = time.time()

    def _event(self, kind: str, body: dict) -> None:
        self.events.append({"id": str(uuid.uuid4()), "timestamp": _iso(), "type": kind,
                            "body": {"id": str(uuid.uuid4()), "traceId": self.id, **body}})

    @contextmanager
    def span(self, name: str, input=None):
        sp, t0 = Span(), time.time()
        try:
            yield sp
        finally:
            body = {"name": name, "startTime": _iso(t0), "endTime": _iso(), "input": input,
                    "output": sp.output, "metadata": sp.metadata or None}
            if sp.warning:
                body |= {"level": "WARNING", "statusMessage": sp.warning}
            self._event("span-create", body)

    def generation(self, name: str, start: float, model: str, input, output, usage: dict | None,
                   metadata: dict | None = None) -> None:
        body = {"name": name, "startTime": _iso(start), "endTime": _iso(), "model": model,
                "input": input, "output": output, "metadata": metadata}
        if usage:
            body["usageDetails"] = {"input": usage["prompt_tokens"],
                                    "output": usage["completion_tokens"]}
            body["costDetails"] = {"total": usage["est_cost_usd"]}
        self._event("generation-create", body)

    def end(self, output, tags: list[str] = (), metadata: dict | None = None) -> None:
        self.events.insert(0, {"id": str(uuid.uuid4()), "timestamp": _iso(self.start),
                               "type": "trace-create", "body": {
                                   "id": self.id, "name": self.name, "timestamp": _iso(self.start),
                                   "input": self.input, "output": output,
                                   "tags": sorted({*self.tags, *tags}),
                                   "metadata": {**self.metadata, **(metadata or {})}}})
        self.tracer.send(self.events)


class NullTrace:
    """The no-op trace: same calls, records nothing (tracing off, or in tests)."""

    id = None

    @contextmanager
    def span(self, name: str, input=None):
        yield Span()

    def generation(self, *args, **kwargs) -> None:
        pass

    def end(self, *args, **kwargs) -> None:
        pass


NULL = NullTrace()


class Tracer:
    def __init__(self, public_key: str, secret_key: str, host: str,
                 transport: httpx.BaseTransport | None = None):
        self.enabled = bool(public_key and secret_key)
        self.url = f"{host.rstrip('/')}/api/public/ingestion"
        self._auth = (public_key, secret_key)
        self._transport = transport
        self.errors: list[str] = []  # kept for tests and /health-style checks

    @classmethod
    def from_settings(cls, s) -> Tracer:
        return cls(s.langfuse_public_key, s.langfuse_secret_key, s.langfuse_host)

    def trace(self, name: str, input, tags: list[str] = (), metadata: dict | None = None):
        if not self.enabled:
            return NULL
        return Trace(self, name, input, list(tags), metadata or {})

    def send(self, events: list[dict]) -> None:
        try:
            with httpx.Client(transport=self._transport, timeout=TIMEOUT) as c:
                r = c.post(self.url, json={"batch": events}, auth=self._auth)
            # 207 Multi-Status: the batch was read; per-event errors are listed.
            problems = r.json().get("errors") if r.status_code in (200, 207) else None
            if r.status_code not in (200, 207) or problems:
                self._warn(f"langfuse ingestion {r.status_code}: {problems or r.text[:200]}")
        except Exception as exc:  # never let tracing break an answer
            self._warn(f"langfuse send failed: {type(exc).__name__}: {exc}")

    def _warn(self, msg: str) -> None:
        self.errors.append(msg)
        print(msg)
