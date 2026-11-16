"""Embeddings (bge-small-en-v1.5 via fastembed) and Qdrant in local mode.

Local mode stores vectors on disk under data/qdrant, with no server to run. With
FINRAG_QDRANT_URL set, the same code talks to a Qdrant server instead (deploy/k8s):
several API replicas can then share one index, which local mode cannot (it locks
its directory to one process).
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


def embed(texts: list[str], model: str, batch_size: int = 0) -> list[list[float]]:
    """*batch_size* 0: fastembed's default. A small one caps memory: 64 passages of 512
    tokens need ~1 GB of attention buffers per layer (OOM at 2 GiB in Kubernetes)."""
    kw = {"batch_size": batch_size} if batch_size else {}
    return [v.tolist() for v in _embedder(model).embed(texts, **kw)]


def _location(settings: Settings) -> str:
    """Where the vectors live: the server URL when one is configured, else the folder."""
    return settings.qdrant_url or str(settings.qdrant_path)


@lru_cache(maxsize=2)
def _client(location: str):
    import atexit

    from qdrant_client import QdrantClient

    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location, timeout=10)
    client = QdrantClient(path=location)
    # Close while the interpreter is still alive; left to __del__, local mode
    # raises a noisy ImportError during shutdown.
    atexit.register(client.close)
    return client


class Store:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = _client(_location(settings))

    def count(self) -> int:
        """Indexed chunks; 0 when the collection does not exist yet (before ingestion)."""
        # The name is usually an alias (rebuild); do not rely on collection_exists
        # resolving aliases the same way in local mode and on a server.
        if self._alias_target() is None and not self.client.collection_exists(self.s.collection):
            return 0
        return self.client.count(self.s.collection, exact=False).count

    def rebuild(self, chunks: list[Chunk], dim: int, batch: int = 64) -> int:
        """Replace the whole index with *chunks*, without ever exposing a partial one.

        The chunks go into a fresh collection; only when all of them are in does the
        name `settings.collection` (an alias) switch to it, in one atomic operation,
        and the previous collection is dropped. Until then searches keep reading the
        old index. Deleting and refilling in place (the earlier reset + add) let the
        API answer from a half-built index, and /ready reported it ready: seen on the
        first Kubernetes deployment, 2026-11-16.
        """
        import uuid

        from qdrant_client.models import (
            CreateAlias,
            CreateAliasOperation,
            DeleteAlias,
            DeleteAliasOperation,
            Distance,
            VectorParams,
        )

        name = self.s.collection
        current = self._alias_target()
        # Leftovers of an interrupted rebuild: never aliased, safe to drop.
        for c in self.client.get_collections().collections:
            if c.name.startswith(f"{name}--") and c.name != current:
                self.client.delete_collection(c.name)
        new = f"{name}--{uuid.uuid4().hex[:12]}"
        self.client.create_collection(
            new, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        self._upsert(new, chunks, batch)
        if current is None and self.client.collection_exists(name):
            # A plain collection from before aliases existed: it must go before the
            # alias can take its name. A one-time gap, on the first rebuild only.
            self.client.delete_collection(name)
        ops = [DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=name))] if current else []
        ops.append(CreateAliasOperation(create_alias=CreateAlias(collection_name=new,
                                                                 alias_name=name)))
        self.client.update_collection_aliases(change_aliases_operations=ops)
        if current:
            self.client.delete_collection(current)
        return len(chunks)

    def _alias_target(self) -> str | None:
        for a in self.client.get_aliases().aliases:
            if a.alias_name == self.s.collection:
                return a.collection_name
        return None

    def add(self, chunks: list[Chunk], batch: int = 64) -> int:
        """Add or replace chunks in the live index (connector documents, POST /documents)."""
        self._upsert(self.s.collection, chunks, batch)
        return len(chunks)

    def _upsert(self, collection: str, chunks: list[Chunk], batch: int) -> None:
        from qdrant_client.models import PointStruct

        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            vectors = embed([c.text for c in part], self.s.embed_model, self.s.embed_batch)
            self.client.upsert(
                collection,
                points=[
                    PointStruct(id=_uuid(c.id), vector=v, payload=asdict(c))
                    for c, v in zip(part, vectors, strict=True)
                ],
            )

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
