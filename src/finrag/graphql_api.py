"""GraphQL API (Strawberry): the same answer path as REST, typed and introspectable.

    POST /graphql   { models { name label provider available } }
                    { answer(question: "...", model: "mistral-24b", mode: AGENT) {
                        answer stage reason sources { ticker section } usage { estCostUsd } } }

Not a second implementation: every resolver calls what POST /query and POST /agent call
(finrag.api), so the scope check, the evidence gate, numeric grounding, the ToolGuard,
the cost estimate and the Langfuse trace are identical. Input is validated by the same
pydantic model as REST (question 3-500 characters, a known model preset).

Queries go by POST only (allow_queries_via_get=False): a cross-site GET must never be
able to start a paid model call. On Modal the same x-api-key middleware guards /graphql.
"""

from __future__ import annotations

from enum import Enum

import strawberry
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON


@strawberry.enum
class Mode(Enum):
    PIPELINE = "pipeline"  # scope -> retrieve -> evidence gate -> model -> figure check
    AGENT = "agent"  # tool-calling agent under the ToolGuard


@strawberry.type
class Model:
    name: str
    label: str
    model_id: str
    provider: str
    digest: str | None
    available: bool


@strawberry.type
class Source:
    n: str  # "1".."5" in the pipeline, "S1".. for the agent's evidence
    ticker: str | None = None
    form: str | None = None
    period: str | None = None
    section: str | None = None
    section_title: str | None = None
    url: str | None = None
    score: float | None = None
    tool: str | None = None


@strawberry.type
class Usage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int
    llm_seconds: float
    gpu: str | None
    pricing: str
    est_cost_usd: float


@strawberry.type
class Answer:
    question: str
    answer: str
    refused: bool
    stage: str  # scope | evidence | model | numbers | limit | answered
    reason: str
    sources: list[Source]
    ungrounded_numbers: list[str]
    usage: Usage | None
    trace_id: str | None
    steps: JSON | None  # the agent's ToolGuard audit log (agent mode only)


_SOURCE_FIELDS = set(Source.__annotations__)


def _to_answer(d: dict) -> Answer:
    """From what the REST handlers return (dataclasses.asdict of Answer / AgentAnswer)."""
    sources = [Source(**{k: (str(v) if k == "n" else v) for k, v in s.items()
                         if k in _SOURCE_FIELDS}) for s in d["sources"]]
    usage = Usage(**d["usage"]) if d.get("usage") else None
    return Answer(question=d["question"], answer=d["answer"], refused=d["refused"],
                  stage=d["stage"], reason=d["reason"], sources=sources,
                  ungrounded_numbers=d["ungrounded_numbers"], usage=usage,
                  trace_id=d.get("trace_id"), steps=d.get("steps"))


def _validated(question: str, model: str | None):
    from finrag.api import Query

    try:
        return Query(question=question, model=model)
    except ValidationError as exc:  # same rules and messages as the REST endpoints
        raise ValueError("; ".join(e["msg"] for e in exc.errors())) from None


def _run(q, mode: Mode) -> Answer:
    from finrag import api

    return _to_answer(api.agent_query(q) if mode is Mode.AGENT else api.query(q))


@strawberry.type
class Query:
    @strawberry.field(description="The model presets and whether each is usable now.")
    def models(self) -> list[Model]:
        from finrag import api

        return [Model(**m) for m in api.models()["models"]]

    @strawberry.field(description="Answer from the filings, or a refusal naming its stage.")
    async def answer(self, question: str, model: str | None = None,
                     mode: Mode = Mode.PIPELINE) -> Answer:
        q = _validated(question, model)
        # The model call blocks for seconds: off the event loop, like FastAPI's sync routes.
        return await run_in_threadpool(_run, q, mode)


schema = strawberry.Schema(query=Query)
router = GraphQLRouter(schema, graphql_ide="graphiql", allow_queries_via_get=False)
