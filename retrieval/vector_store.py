from __future__ import annotations

import logging
from typing import Any

from config import get_settings
from models import Chunk, RetrievedChunk

LOGGER = logging.getLogger(__name__)


def _distance_from_name(name: str):
    from qdrant_client.http import models as qmodels

    lowered = name.lower()
    if lowered == "dot":
        return qmodels.Distance.DOT
    if lowered == "euclid":
        return qmodels.Distance.EUCLID
    return qmodels.Distance.COSINE


class QdrantVectorStore:
    """Qdrant wrapper for collection lifecycle and dense retrieval."""

    def __init__(self, url: str, collection_name: str, vector_size: int, distance_metric: str) -> None:
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = _distance_from_name(distance_metric)

    def create_collection(self) -> None:
        from qdrant_client.http import models as qmodels

        collections = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in collections:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=self.vector_size, distance=self.distance),
        )

    def delete_collection(self) -> None:
        self.client.delete_collection(collection_name=self.collection_name)

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client.http import models as qmodels

        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must have matching lengths")

        points: list[qmodels.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "chunk_id": chunk.chunk_id,
                "source_doc": chunk.source_doc,
                "title": chunk.title,
                "page": chunk.page,
                "section": chunk.section,
                "doc_type": chunk.doc_type,
                "date": chunk.date,
                "text": chunk.text,
            }
            points.append(qmodels.PointStruct(id=chunk.chunk_id, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def dense_search(self, query_vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        qfilter = self._build_filter(metadata_filter)
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qfilter,
            limit=top_k,
        )

        results: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk = Chunk(
                chunk_id=str(payload.get("chunk_id", hit.id)),
                source_doc=str(payload.get("source_doc", "unknown")),
                title=str(payload.get("title", "untitled")),
                page=payload.get("page"),
                section=payload.get("section"),
                doc_type=str(payload.get("doc_type", "unknown")),
                date=payload.get("date"),
                text=str(payload.get("text", "")),
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(hit.score), source="dense"))

        return results

    def _build_filter(self, metadata_filter: dict[str, Any] | None):
        from qdrant_client.http import models as qmodels

        if not metadata_filter:
            return None

        clauses = [
            qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
            for key, value in metadata_filter.items()
        ]
        return qmodels.Filter(must=clauses)


def build_default_vector_store(vector_size: int | None = None) -> QdrantVectorStore:
    settings = get_settings()
    return QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        vector_size=vector_size or settings.vector_size,
        distance_metric=settings.distance_metric,
    )
