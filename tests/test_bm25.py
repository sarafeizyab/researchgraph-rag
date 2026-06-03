import pytest

pytest.importorskip("rank_bm25")

from models import Chunk
from retrieval.bm25_index import BM25Index


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_doc="doc",
        title="title",
        page=1,
        section="intro",
        doc_type="pdf",
        date="2026-01-01",
        text=text,
    )


def test_bm25_returns_relevant_chunk_first() -> None:
    index = BM25Index()
    index.add_chunks(
        [
            _chunk("c1", "transformer model for retinal disease classification"),
            _chunk("c2", "genomics variant prioritization and pathogenicity"),
            _chunk("c3", "cat images and unrelated topic"),
        ]
    )

    results = index.search("retinal disease transformer", top_k=2)
    assert results
    assert results[0].chunk.chunk_id == "c1"


def test_bm25_save_and_load_round_trip(tmp_path) -> None:
    index = BM25Index()
    index.add_chunks([_chunk("c1", "foundation models for computational biology")])

    path = tmp_path / "bm25.pkl"
    index.save(path)

    loaded = BM25Index.load(path)
    results = loaded.search("computational biology", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
