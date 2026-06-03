from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from config import get_settings
from ingestion.loader import LoadedDocument, RawSegment
from models import Chunk

LOGGER = logging.getLogger(__name__)


@dataclass
class TextChunker:
    """Token-based chunker with overlap for retrieval quality."""

    chunk_size: int
    chunk_overlap: int

    def chunk_document(self, loaded_doc: LoadedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for segment in loaded_doc.segments:
            chunks.extend(self._chunk_segment(segment, loaded_doc))
        return chunks

    def _chunk_segment(self, segment: RawSegment, loaded_doc: LoadedDocument) -> list[Chunk]:
        tokens = self._simple_tokenize(segment.text)
        if not tokens:
            return []

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        stride = self.chunk_size - self.chunk_overlap
        output: list[Chunk] = []

        for start in range(0, len(tokens), stride):
            end = start + self.chunk_size
            token_slice = tokens[start:end]
            if not token_slice:
                continue
            text = " ".join(token_slice).strip()
            if not text:
                continue

            output.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    source_doc=loaded_doc.source_doc,
                    title=loaded_doc.title,
                    page=segment.page,
                    section=segment.section,
                    doc_type=loaded_doc.doc_type,
                    date=loaded_doc.date,
                    text=text,
                )
            )

            if end >= len(tokens):
                break

        return output

    def _simple_tokenize(self, text: str) -> list[str]:
        # Lightweight tokenization fallback keeps package constraints simple.
        return text.split()


def build_default_chunker() -> TextChunker:
    settings = get_settings()
    return TextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
