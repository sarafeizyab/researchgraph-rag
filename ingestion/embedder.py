from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

LOGGER = logging.getLogger(__name__)


@dataclass
class EmbeddingClient:
    """Embedding abstraction for OpenAI and local sentence transformers."""

    provider: str
    model_name: str
    openai_api_key: str | None = None

    def __post_init__(self) -> None:
        self._openai_client: Any | None = None
        self._local_model: Any | None = None

        if self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self.model_name)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []

        if self.provider == "openai":
            assert self._openai_client is not None
            vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                resp = self._openai_client.embeddings.create(model=self.model_name, input=batch)
                vectors.extend([item.embedding for item in resp.data])
            return vectors

        assert self._local_model is not None
        vectors_arr = self._local_model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return [row.tolist() for row in vectors_arr]

    def embedding_dimension(self) -> int:
        probe = self.embed_texts(["dimension probe"])
        if not probe:
            raise RuntimeError("Failed to generate embedding dimension probe")
        return len(probe[0])


def build_default_embedder() -> EmbeddingClient:
    settings = get_settings()
    return EmbeddingClient(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
