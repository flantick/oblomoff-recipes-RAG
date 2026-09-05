"""A wrapper around Qdrant: a collection with dense+sparse vectors, idempotent
upserts, dense and hybrid (RRF) search (Steps 3-4)."""
from __future__ import annotations

import uuid
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient, models

from src.config import DENSE_VECTOR_SIZE, QDRANT_COLLECTION, QDRANT_URL
from src.index.embedder import Embedding

# a fixed namespace -> a deterministic point id derived from chunk_id
# (re-running build upserts the same points instead of duplicating them)
_NS = uuid.uuid5(uuid.NAMESPACE_URL, "oblomoff-rag")

DENSE = "dense"
LEXICAL = "lexical"

_PAYLOAD_INDEXES = [
    ("video_id", models.PayloadSchemaType.KEYWORD),
    ("chunk_index", models.PayloadSchemaType.INTEGER),   # filtered on in fetch_chunks
    ("section", models.PayloadSchemaType.KEYWORD),
    ("has_ingredients", models.PayloadSchemaType.BOOL),
    ("has_steps", models.PayloadSchemaType.BOOL),
    ("playlist_titles", models.PayloadSchemaType.KEYWORD),
    ("title", models.PayloadSchemaType.TEXT),
]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class VectorStore:
    def __init__(
        self,
        *,
        url: str = QDRANT_URL,
        collection: str = QDRANT_COLLECTION,
        path: str | None = None,
    ) -> None:
        self.client = (
            QdrantClient(path=path)
            if path
            else QdrantClient(url=url, timeout=60, check_compatibility=False)
        )
        self.collection = collection

    # --- schema ------------------------------------------------
    def ensure_collection(self, *, dense_size: int = DENSE_VECTOR_SIZE, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config={
                    DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={LEXICAL: models.SparseVectorParams()},
            )
            logger.info("Коллекция {} создана", self.collection)
        self.ensure_payload_indexes()

    def ensure_payload_indexes(self) -> None:
        """Creates any payload indexes missing from a live collection.

        Indexes used to be created only on the creation path, so a newly added
        one arrived exclusively with --recreate: on an existing collection
        ensure_collection returned early via `if exists: return`.
        """
        try:
            # `or {}` is not redundant with the except below: a collection with
            # no payload schema yet is the normal state of a fresh collection,
            # and letting that fall into the handler would log a false warning.
            have = set((self.client.get_collection(self.collection).payload_schema or {}).keys())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось прочитать схему payload: {}", exc)
            have = set()
        for field, schema in _PAYLOAD_INDEXES:
            if field in have:
                continue
            try:
                self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)
                logger.info("Payload-индекс создан: {}", field)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Payload-индекс {} не создан: {}", field, exc)

    # --- writing ----------------------------------------------
    def upsert(self, chunks: list[dict[str, Any]], embs: list[Embedding], *, batch: int = 128) -> int:
        points: list[models.PointStruct] = []
        for ch, e in zip(chunks, embs):
            points.append(
                models.PointStruct(
                    id=point_id(ch["chunk_id"]),
                    vector={
                        DENSE: e.dense,
                        LEXICAL: models.SparseVector(indices=e.sparse_indices, values=e.sparse_values),
                    },
                    payload=ch,
                )
            )
        for i in range(0, len(points), batch):
            self.client.upsert(self.collection, points=points[i:i + batch], wait=True)
        return len(points)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def fetch_chunks(self, video_id: str, indices: list[int]) -> list[dict[str, Any]]:
        """Payloads of one video's chunks by their chunk_index (to assemble neighbours).

        video_id is indexed, so the filter first narrows down to a single video
        and the MatchAny over chunk_index then runs inside it.
        """
        if not indices:
            return []
        res, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id)),
                    models.FieldCondition(key="chunk_index", match=models.MatchAny(any=sorted(indices))),
                ]
            ),
            limit=len(indices),
            with_payload=True,
            with_vectors=False,
        )
        return [p.payload or {} for p in res]

    # --- search -----------------------------------------------
    def search(
        self,
        emb: Embedding,
        *,
        k: int = 5,
        mode: str = "hybrid",             # "hybrid" | "dense" | "sparse"
        query_filter: models.Filter | None = None,
        overfetch: int = 5,
    ) -> list[models.ScoredPoint]:
        sparse = models.SparseVector(indices=emb.sparse_indices, values=emb.sparse_values)
        if mode == "dense":
            res = self.client.query_points(
                self.collection, query=emb.dense, using=DENSE,
                limit=k, query_filter=query_filter, with_payload=True,
            )
        elif mode == "sparse":
            res = self.client.query_points(
                self.collection, query=sparse, using=LEXICAL,
                limit=k, query_filter=query_filter, with_payload=True,
            )
        else:
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=emb.dense, using=DENSE, limit=k * overfetch),
                    models.Prefetch(query=sparse, using=LEXICAL, limit=k * overfetch),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k, query_filter=query_filter, with_payload=True,
            )
        return res.points
