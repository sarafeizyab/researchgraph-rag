from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ingestion.chunker import TextChunker, build_default_chunker
from ingestion.embedder import EmbeddingClient, build_default_embedder
from ingestion.loader import DocumentLoader, LoadedDocument
from models import Chunk
from retrieval.bm25_index import BM25Index
from retrieval.vector_store import QdrantVectorStore

LOGGER = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result metadata for a single ingestion action."""

    source_doc: str
    chunks_created: int


class IngestionPipeline:
    """End-to-end ingestion pipeline from source to indexes."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedder: EmbeddingClient,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
        bm25_index_path: str | None = None,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.bm25_index_path = bm25_index_path

    def ingest_loaded_document(self, loaded: LoadedDocument) -> IngestResult:
        chunks = self.chunker.chunk_document(loaded)
        if not chunks:
            LOGGER.warning("No chunks were produced for source %s", loaded.source_doc)
            return IngestResult(source_doc=loaded.source_doc, chunks_created=0)

        vectors = self.embedder.embed_texts([c.text for c in chunks])
        self.vector_store.upsert_chunks(chunks=chunks, vectors=vectors)
        self.bm25_index.add_chunks(chunks)
        self._persist_bm25_index()

        return IngestResult(source_doc=loaded.source_doc, chunks_created=len(chunks))

    def ingest_pdf(self, file_bytes: bytes, source_name: str) -> IngestResult:
        return self.ingest_loaded_document(self.loader.load_pdf(file_bytes=file_bytes, source_name=source_name))

    def ingest_docx(self, file_bytes: bytes, source_name: str) -> IngestResult:
        return self.ingest_loaded_document(self.loader.load_docx(file_bytes=file_bytes, source_name=source_name))

    def ingest_url(self, url: str) -> IngestResult:
        return self.ingest_loaded_document(self.loader.load_url(url=url))

    def ingest_arxiv(self, arxiv_id: str) -> IngestResult:
        return self.ingest_loaded_document(self.loader.load_arxiv(arxiv_id=arxiv_id))

    def _persist_bm25_index(self) -> None:
        if not self.bm25_index_path:
            return

        path = Path(self.bm25_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25_index.save(path)


class InMemoryCorpus:
    """Utility registry of all chunks for debugging and evaluation."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


def build_default_pipeline(
    vector_store: QdrantVectorStore,
    bm25_index: BM25Index,
    bm25_index_path: str | None = None,
) -> IngestionPipeline:
    return IngestionPipeline(
        loader=DocumentLoader(),
        chunker=build_default_chunker(),
        embedder=build_default_embedder(),
        vector_store=vector_store,
        bm25_index=bm25_index,
        bm25_index_path=bm25_index_path,
    )
