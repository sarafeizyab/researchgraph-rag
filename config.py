from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai", "huggingface", "ollama"] = Field(default="openai", alias="LLM_PROVIDER")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_endpoint_url: str | None = Field(default=None, alias="HF_ENDPOINT_URL")
    hf_endpoint_mode: Literal["text-generation", "chat-completions"] = Field(
        default="text-generation",
        alias="HF_ENDPOINT_MODE",
    )
    hf_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", alias="HF_MODEL")
    hf_max_new_tokens: int = Field(default=768, alias="HF_MAX_NEW_TOKENS")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")

    embedding_provider: Literal["openai", "local"] = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="researchgraph_chunks", alias="QDRANT_COLLECTION")
    bm25_index_path: str = Field(default="artifacts/bm25_index.pkl", alias="BM25_INDEX_PATH")
    vector_size: int = Field(default=1024, alias="VECTOR_SIZE")
    distance_metric: Literal["cosine", "dot", "euclid"] = Field(default="cosine", alias="DISTANCE_METRIC")

    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    dense_top_k: int = Field(default=20, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(default=20, alias="SPARSE_TOP_K")
    hybrid_top_k: int = Field(default=20, alias="HYBRID_TOP_K")
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")

    max_hops: int = Field(default=4, alias="MAX_HOPS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached app settings instance."""

    return Settings()
