from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from api.streaming import iter_reasoning_trace, sse_event
from config import get_settings
from logging_utils import setup_logging

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="ResearchGraph-RAG", version="0.1.0")


class AppServices:
    """Lazy dependency container for API runtime services."""

    def __init__(self) -> None:
        self.initialized = False
        self.pipeline: Any | None = None
        self.agent: Any | None = None

    def initialize(self) -> None:
        if self.initialized:
            return

        from agent.graph import build_default_agent
        from ingestion.chunker import build_default_chunker
        from ingestion.embedder import build_default_embedder
        from ingestion.loader import DocumentLoader
        from ingestion.pipeline import IngestionPipeline
        from retrieval.bm25_index import BM25Index
        from retrieval.hybrid import HybridRetriever
        from retrieval.reranker import CrossEncoderReranker
        from retrieval.vector_store import build_default_vector_store

        settings = get_settings()
        setup_logging(settings.log_level)

        embedder = build_default_embedder()
        try:
            inferred_vector_size = embedder.embedding_dimension()
        except Exception as exc:  # pragma: no cover - provider dependent
            LOGGER.warning("Failed to infer embedding dimension; falling back to VECTOR_SIZE: %s", exc)
            inferred_vector_size = settings.vector_size

        vector_store = build_default_vector_store(vector_size=inferred_vector_size)
        vector_store.create_collection()

        if Path(settings.bm25_index_path).exists():
            bm25_index = BM25Index.load(settings.bm25_index_path)
        else:
            bm25_index = BM25Index()
        reranker: CrossEncoderReranker | None
        try:
            reranker = CrossEncoderReranker()
        except Exception as exc:  # pragma: no cover - model download dependent
            LOGGER.warning("Failed to initialize reranker model: %s", exc)
            reranker = None

        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedder=embedder,
            reranker=reranker,
        )
        self.agent = build_default_agent(retriever)
        self.pipeline = IngestionPipeline(
            loader=DocumentLoader(),
            chunker=build_default_chunker(),
            embedder=embedder,
            vector_store=vector_store,
            bm25_index=bm25_index,
            bm25_index_path=settings.bm25_index_path,
        )
        self.initialized = True


SERVICES = AppServices()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    arxiv_id: str | None = Form(default=None),
) -> IngestResponse:
    SERVICES.initialize()
    assert SERVICES.pipeline is not None

    choices = [file is not None, bool(url), bool(arxiv_id)]
    if sum(1 for c in choices if c) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one of file, url, or arxiv_id")

    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        file_bytes = await file.read()
        if suffix == ".pdf":
            result = SERVICES.pipeline.ingest_pdf(file_bytes=file_bytes, source_name=file.filename or "uploaded.pdf")
        elif suffix == ".docx":
            result = SERVICES.pipeline.ingest_docx(file_bytes=file_bytes, source_name=file.filename or "uploaded.docx")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .pdf or .docx")
        return IngestResponse(source_doc=result.source_doc, chunks_created=result.chunks_created)

    if url:
        result = SERVICES.pipeline.ingest_url(url)
        return IngestResponse(source_doc=result.source_doc, chunks_created=result.chunks_created)

    if arxiv_id:
        result = SERVICES.pipeline.ingest_arxiv(arxiv_id)
        return IngestResponse(source_doc=result.source_doc, chunks_created=result.chunks_created)

    raise HTTPException(status_code=400, detail="No ingestion source provided")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    SERVICES.initialize()
    if SERVICES.agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    result = SERVICES.agent.run(request.question)
    return QueryResponse(
        answer=result.answer,
        citations=[
            {
                "chunk_id": c.chunk_id,
                "source_doc": c.source_doc,
                "page": c.page,
                "section": c.section,
                "excerpt": c.excerpt,
            }
            for c in result.citations
        ],
        reasoning_trace=result.reasoning_trace,
        metrics={
            "total_latency_ms": result.metrics.total_latency_ms,
            "retrieval_latency_ms": result.metrics.retrieval_latency_ms,
            "reranking_latency_ms": result.metrics.reranking_latency_ms,
            "llm_latency_ms": result.metrics.llm_latency_ms,
            "hops": result.metrics.hops,
            "retrieved_chunks": result.metrics.retrieved_chunks,
            "token_usage": result.metrics.token_usage,
        },
        sub_queries=result.sub_queries,
        generated_at=result.generated_at,
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    SERVICES.initialize()
    if SERVICES.agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    def event_generator() -> Generator[str, None, None]:
        yield sse_event("start", {"question": request.question})
        result = SERVICES.agent.run(request.question)
        for event in iter_reasoning_trace(result.reasoning_trace):
            yield event
        yield sse_event(
            "final_answer",
            {
                "answer": result.answer,
                "citations": [
                    {
                        "chunk_id": c.chunk_id,
                        "source_doc": c.source_doc,
                        "page": c.page,
                        "section": c.section,
                        "excerpt": c.excerpt,
                    }
                    for c in result.citations
                ],
                "metrics": {
                    "total_latency_ms": result.metrics.total_latency_ms,
                    "retrieval_latency_ms": result.metrics.retrieval_latency_ms,
                    "reranking_latency_ms": result.metrics.reranking_latency_ms,
                    "llm_latency_ms": result.metrics.llm_latency_ms,
                    "hops": result.metrics.hops,
                    "retrieved_chunks": result.metrics.retrieved_chunks,
                    "token_usage": result.metrics.token_usage,
                },
            },
        )
        yield sse_event("done", {"ok": True})

    return StreamingResponse(event_generator(), media_type="text/event-stream")
