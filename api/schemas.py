from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class RootResponse(BaseModel):
    service: str
    status: str
    docs_url: str
    endpoints: list[str]


class IngestResponse(BaseModel):
    source_doc: str
    chunks_created: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=5)


class CitationResponse(BaseModel):
    chunk_id: str
    source_doc: str
    page: int | None = None
    section: str | None = None
    excerpt: str


class QueryMetricsResponse(BaseModel):
    total_latency_ms: float
    retrieval_latency_ms: float
    reranking_latency_ms: float
    llm_latency_ms: float
    hops: int
    retrieved_chunks: int
    token_usage: dict[str, int]


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    reasoning_trace: list[dict[str, Any]]
    metrics: QueryMetricsResponse
    sub_queries: list[str]
    generated_at: datetime


class StreamEvent(BaseModel):
    event: str
    data: dict[str, Any]
