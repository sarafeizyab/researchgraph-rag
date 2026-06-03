import pytest

pytest.importorskip("tenacity")

from models import Chunk, RetrievedChunk
from retrieval.hybrid import reciprocal_rank_fusion


def _r(chunk_id: str, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            source_doc="doc",
            title="title",
            page=1,
            section="results",
            doc_type="pdf",
            date="2026-01-01",
            text=f"text for {chunk_id}",
        ),
        score=score,
        source=source,
    )


def test_rrf_prefers_consensus_documents() -> None:
    dense = [_r("a", 0.9, "dense"), _r("b", 0.8, "dense"), _r("c", 0.7, "dense")]
    sparse = [_r("x", 11.0, "sparse"), _r("b", 9.0, "sparse"), _r("a", 8.0, "sparse")]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ranked_ids = [item.chunk.chunk_id for item in fused[:3]]

    assert ranked_ids[0] in {"a", "b"}
    assert "x" in ranked_ids
