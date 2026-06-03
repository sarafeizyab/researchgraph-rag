from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
import time

from config import get_settings
from ingestion.embedder import EmbeddingClient
from models import RetrievedChunk
from retrieval.bm25_index import BM25Index
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import QdrantVectorStore

LOGGER = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever with dense+sparse fusion and optional reranking."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
        embedder: EmbeddingClient,
        reranker: CrossEncoderReranker | None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.reranker = reranker
        self.settings = get_settings()

    def retrieve(
        self,
        query: str,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        hybrid_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        results, _ = self.retrieve_with_diagnostics(
            query=query,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            hybrid_top_k=hybrid_top_k,
            rerank_top_k=rerank_top_k,
        )
        return results

    def retrieve_with_diagnostics(
        self,
        query: str,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        hybrid_top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        dense_k = dense_top_k or self.settings.dense_top_k
        sparse_k = sparse_top_k or self.settings.sparse_top_k
        hybrid_k = hybrid_top_k or self.settings.hybrid_top_k
        rerank_k = rerank_top_k or self.settings.rerank_top_k

        query_vector = self.embedder.embed_texts([query])[0]

        start_retrieval = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self.vector_store.dense_search, query_vector, dense_k)
            sparse_future = pool.submit(self.bm25_index.search, query, sparse_k)
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0

        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self.settings.rrf_k)
        fused_top = fused[:hybrid_k]

        if self.reranker is None:
            return fused_top[:rerank_k], {"retrieval_ms": retrieval_ms, "reranking_ms": 0.0}

        start_rerank = time.perf_counter()
        reranked = self.reranker.rerank(query=query, candidates=fused_top, top_k=rerank_k)
        reranking_ms = (time.perf_counter() - start_rerank) * 1000.0
        return reranked, {"retrieval_ms": retrieval_ms, "reranking_ms": reranking_ms}


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Merge ranked lists with Reciprocal Rank Fusion."""

    fused_scores: dict[str, float] = {}
    exemplar_by_id: dict[str, RetrievedChunk] = {}

    for ranked_list in rankings:
        for rank, item in enumerate(ranked_list, start=1):
            cid = item.chunk.chunk_id
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)
            exemplar_by_id[cid] = item

    output = [RetrievedChunk(chunk=exemplar_by_id[cid].chunk, score=score, source="hybrid") for cid, score in fused_scores.items()]
    output.sort(key=lambda x: x.score, reverse=True)
    return output
