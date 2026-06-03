from __future__ import annotations

import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from models import Chunk, RetrievedChunk

LOGGER = logging.getLogger(__name__)


class BM25Index:
    """Sparse BM25 retrieval with persistence utilities."""

    def __init__(self) -> None:
        self._chunks_by_id: dict[str, Chunk] = {}
        self._tokenized_corpus: list[list[str]] = []
        self._chunk_ids: list[str] = []
        self._index: BM25Okapi | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        changed = False
        for chunk in chunks:
            if chunk.chunk_id in self._chunks_by_id:
                continue
            self._chunks_by_id[chunk.chunk_id] = chunk
            self._chunk_ids.append(chunk.chunk_id)
            self._tokenized_corpus.append(self._tokenize(chunk.text))
            changed = True

        if changed:
            self._index = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._index is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks_by_id[chunk_id]
            results.append(RetrievedChunk(chunk=chunk, score=float(score), source="sparse"))
        return results

    def save(self, path: str | Path) -> None:
        data = {
            "chunks_by_id": self._chunks_by_id,
            "chunk_ids": self._chunk_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with Path(path).open("wb") as handle:
            pickle.dump(data, handle)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as handle:
            data = pickle.load(handle)

        obj = cls()
        obj._chunks_by_id = data["chunks_by_id"]
        obj._chunk_ids = data["chunk_ids"]
        obj._tokenized_corpus = data["tokenized_corpus"]
        obj._index = BM25Okapi(obj._tokenized_corpus) if obj._tokenized_corpus else None
        return obj

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()
