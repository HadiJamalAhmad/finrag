"""Langfuse tracing through the official Python SDK (langfuse 5.x, OpenTelemetry based).

Why the SDK: Langfuse Cloud stops accepting traces on the legacy /api/public/ingestion
endpoint on 2026-11-16 (it keeps only scores). The SDK exports OTLP spans in the
background, batches, retries and flushes; finrag's first tracer (raw HTTP, 2026-11-09)
targeted the endpoint that goes away.

One trace per unit of work, names verb-first and stable (they are an API: evaluators,
dashboards and saved filters key on them), each observation typed:

    answer-question       chain       input: question       output: answer or refusal
      check-scope         guardrail   in scope? tickers, years       (WARNING if refused)
      retrieve-passages   retriever   the passages the model will see, with scores
      check-evidence      guardrail   best score vs min_score        (WARNING if refused)
      generate-answer     generation  messages, model, parameters, usage, cost, reasoning
      check-figures       guardrail   grounded? ungrounded figures   (WARNING if refused)

    answer-with-tools     agent       the agent: one generation per model turn, each
      check-scope         guardrail   tool call a `tool` sibling of the generation that
      generate-response   generation  asked for it, with its tool_call_id
      search-filings      tool
      ...
      verify-answer       guardrail

Tags are set at creation (they are immutable): finrag, the feature (pipeline | agent)
and the model preset. What is only known at the end (which stage decided) is a score:
`outcome` (categorical) and `refused` (boolean), plus `correct` on eval traces.
Environment separates development, production (Modal) and eval runs.

Without keys this is a no-op (NullObservation): no client is built, nothing is sent.
SDK errors are caught and logged by the SDK; tracing never breaks an answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from finrag import __version__


class Observation:
    """One Langfuse observation (span, generation, tool...). Thin: SDK names pass through."""

    def __init__(self, obs):
        self._obs = obs

    @contextmanager
    def child(self, name: str, as_type: str = "span", **fields) -> Iterator[Observation]:
        with self._obs.start_as_current_observation(name=name, as_type=as_type,
                                                    **fields) as obs:
            yield Observation(obs)

    def update(self, **fields) -> None:
        self._obs.update(**fields)

    def warn(self, message: str) -> None:
        """Mark the stage that refused: WARNING level, filterable in the UI."""
        self._obs.update(level="WARNING", status_message=message)

    def score(self, name: str, value, data_type: str, comment: str | None = None) -> None:
        """A score on the whole trace (labels known only after the fact, never tags)."""
        self._obs.score_trace(name=name, value=value, data_type=data_type, comment=comment)

    @property
    def trace_id(self) -> str | None:
        return getattr(self._obs, "trace_id", None)


class NullObservation:
    """Same calls, records nothing: tracing off (no keys) or not wanted."""

    trace_id = None

    @contextmanager
    def child(self, name: str, as_type: str = "span", **fields) -> Iterator[NullObservation]:
        yield self

    def update(self, **fields) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    def score(self, *args, **kwargs) -> None:
        pass


NULL = NullObservation()


class Tracer:
    def __init__(self, client=None, environment: str = "development",
                 channel: str | None = None):
        # channel: where requests come from (api | cli | eval), a tag on every trace.
        self.client, self.environment, self.channel = client, environment, channel

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @classmethod
    def from_settings(cls, s, environment: str | None = None, channel: str | None = None,
                      **sdk_options) -> Tracer:
        """The SDK client from Settings; a disabled Tracer when the keys are absent.

        sdk_options go to langfuse.Langfuse (tests pass an in-memory span_exporter, an
        isolated tracer_provider and a mocked httpx_client: nothing leaves the machine).
        """
        env = environment or s.environment
        if not (s.langfuse_public_key and s.langfuse_secret_key):
            return cls(None, env, channel)
        import os

        from langfuse import Langfuse

        # Otherwise OpenTelemetry reports "unknown_service:python" on every observation.
        os.environ.setdefault("OTEL_SERVICE_NAME", "finrag")
        client = Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key,
                          base_url=s.langfuse_base_url, environment=env,
                          release=f"finrag-{__version__}", **sdk_options)
        return cls(client, env, channel)

    @contextmanager
    def root(self, name: str, as_type: str, input, tags: list[str],
             metadata: dict | None = None,
             environment: str | None = None) -> Iterator[Observation | NullObservation]:
        """Open a trace: tags and environment propagate to every child observation."""
        if not self.enabled:
            yield NULL
            return
        from langfuse import propagate_attributes
        from opentelemetry import context as otel_context

        channel = [self.channel] if self.channel else []
        # An empty OpenTelemetry context: finrag's root has no parent and starts its own
        # trace. Seen in production (2026-11-09): with a foreign span active (runtime or a
        # library), roots nested under it; the export filter dropped that span, leaving
        # each root orphaned in a borrowed trace id.
        token = otel_context.attach(otel_context.Context())
        try:
            # Order kept, duplicates dropped (the eval channel and eval context both say "eval").
            unique = list(dict.fromkeys(["finrag", *channel, *tags]))
            with propagate_attributes(tags=unique, trace_name=name,
                                      environment=environment or self.environment):
                with self.client.start_as_current_observation(
                        name=name, as_type=as_type, input=input, metadata=metadata) as obs:
                    yield Observation(obs)
        finally:
            otel_context.detach(token)

    def score_trace(self, trace_id: str, name: str, value, data_type: str,
                    comment: str | None = None) -> None:
        """Score a finished trace by id (the eval labels each case after the answer)."""
        if self.client is not None and trace_id:
            self.client.create_score(trace_id=trace_id, name=name, value=value,
                                     data_type=data_type, comment=comment)

    def flush(self) -> None:
        """Send what is buffered. Call before a short-lived process (CLI, eval) exits."""
        if self.client is not None:
            self.client.flush()

    def shutdown(self) -> None:
        """Flush and stop the background exporter (a server's shutdown hook)."""
        if self.client is not None:
            self.client.shutdown()
