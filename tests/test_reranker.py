from models import Chunk, RetrievedChunk
from retrieval.reranker import CrossEncoderReranker


class _DummyCrossEncoder:
    def predict(self, pairs):
        return [0.1, 0.9, 0.4]


def _candidate(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=cid,
            source_doc="doc",
            title="t",
            page=1,
            section="s",
            doc_type="pdf",
            date="2026-01-01",
            text=text,
        ),
        score=0.0,
        source="hybrid",
    )


def test_reranker_orders_by_cross_encoder_score() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "dummy"
    reranker._model = _DummyCrossEncoder()

    candidates = [_candidate("a", "one"), _candidate("b", "two"), _candidate("c", "three")]
    ranked = reranker.rerank("query", candidates, top_k=2)

    assert [r.chunk.chunk_id for r in ranked] == ["b", "c"]
