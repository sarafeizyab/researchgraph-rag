from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from models import RetrievedChunk

LOGGER = logging.getLogger(__name__)


@dataclass
class CrossEncoderReranker:
    """Cross-encoder reranker for precision over hybrid candidates."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [[query, item.chunk.text] for item in candidates]
        scores = self._model.predict(pairs)

        rescored = [RetrievedChunk(chunk=item.chunk, score=float(score), source="reranked") for item, score in zip(candidates, scores)]
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]
