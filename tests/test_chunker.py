import pytest

pytest.importorskip("httpx")
pytest.importorskip("bs4")

from ingestion.chunker import TextChunker
from ingestion.loader import LoadedDocument, RawSegment


def test_chunker_overlap_and_metadata() -> None:
    text = " ".join([f"tok{i}" for i in range(250)])
    doc = LoadedDocument(
        source_doc="unit_test_doc",
        title="Unit Test",
        doc_type="pdf",
        date="2026-01-01",
        segments=[RawSegment(text=text, page=2, section="Methods")],
    )

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 4
    assert chunks[0].source_doc == "unit_test_doc"
    assert chunks[0].page == 2
    assert chunks[0].section == "Methods"

    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-20:] == second_tokens[:20]
