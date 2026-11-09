"""Embeddings (bge-small-en-v1.5 via fastembed) and Qdrant in local mode.

Local mode stores vectors on disk under data/qdrant, with no server to run.
Moving to a Qdrant server later is one line: QdrantClient(url=...).
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from finrag.chunk import Chunk
from finrag.config import Settings

# bge models are trained with this instruction on the QUERY side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def _embedder(model: str):
    from fastembed import TextEmbedding  # heavy import: only when actually embedding

    return TextEmbedding(model_name=model)


def embed(texts: list[str], model: str) -> list[list[float]]:
    return [v.tolist() for v in _embedder(model).embed(texts)]


@lru_cache(maxsize=2)
def _client(path: str):
    import atexit

    from qdrant_client import QdrantClient

    client = QdrantClient(path=path)
    # Close while the interpreter is still alive; left to __del__, local mode
    # raises a noisy ImportError during shutdown.
    atexit.register(client.close)
    return client


class Store:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = _client(str(settings.qdrant_path))

    def reset(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.s.collection):
            self.client.delete_collection(self.s.collection)
        self.client.create_collection(
            self.s.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    def add(self, chunks: list[Chunk], batch: int = 64) -> int:
        from qdrant_client.models import PointStruct

        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            vectors = embed([c.text for c in part], self.s.embed_model)
            self.client.upsert(
                self.s.collection,
                points=[
                    PointStruct(id=_uuid(c.id), vector=v, payload=asdict(c))
                    for c, v in zip(part, vectors, strict=True)
                ],
            )
        return len(chunks)

    def search(
        self, query: str, k: int, tickers: frozenset[str] = frozenset(),
        years: frozenset[int] = frozenset(), groups: frozenset[str] = frozenset(),
    ) -> list[tuple[float, dict]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        must = [permission_filter(groups)]
        if tickers:
            must.append(FieldCondition(key="ticker", match=MatchAny(any=sorted(tickers))))
        if years:
            # A 10-K for fiscal year Y also reports Y-1 and Y-2, so a question about
            # 2024 may be answered by the FY2025 or FY2026 annual report.
            span = sorted({y + d for y in years for d in (0, 1, 2)})
            must.append(FieldCondition(key="fiscal_year", match=MatchAny(any=span)))
        vector = embed([QUERY_PREFIX + query], self.s.embed_model)[0]
        hits = self.client.query_points(
            self.s.collection, query=vector, limit=k, query_filter=Filter(must=must),
        ).points
        return [(h.score, h.payload) for h in hits]


def permission_filter(groups: frozenset[str]):
    """Default deny: a chunk is visible if it is public (no `groups`) or shares a group
    with the user. No identity = public documents only. Enforced in the vector query
    itself, so a document a user may not read never reaches the model."""
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        IsEmptyCondition,
        MatchAny,
        PayloadField,
    )

    public = IsEmptyCondition(is_empty=PayloadField(key="groups"))
    if not groups:
        return Filter(must=[public])
    return Filter(should=[public, FieldCondition(key="groups",
                                                 match=MatchAny(any=sorted(groups)))])


def _uuid(hex_digest: str) -> str:
    h = hex_digest[:32]
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
