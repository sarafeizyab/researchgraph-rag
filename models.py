from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Chunk:
    """Represents one retrievable text chunk with metadata."""

    chunk_id: str
    source_doc: str
    title: str
    page: int | None
    section: str | None
    doc_type: str
    date: str | None
    text: str


@dataclass
class RetrievedChunk:
    """Represents one retrieved chunk with ranking score."""

    chunk: Chunk
    score: float
    source: str


@dataclass
class Citation:
    """Citation payload returned to API consumers."""

    chunk_id: str
    source_doc: str
    page: int | None
    section: str | None
    excerpt: str


@dataclass
class QueryMetrics:
    """Operational metrics per user query."""

    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    reranking_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    hops: int = 0
    retrieved_chunks: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Full result returned by the multi-hop QA pipeline."""

    answer: str
    citations: list[Citation]
    reasoning_trace: list[dict[str, Any]]
    metrics: QueryMetrics
    sub_queries: list[str]
    used_chunk_ids: list[str]
    generated_at: datetime
