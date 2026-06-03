from __future__ import annotations

import random
from dataclasses import dataclass

from models import Chunk


@dataclass
class SyntheticSample:
    question: str
    reference_answer: str
    supporting_chunk_ids: list[str]


def build_synthetic_testset(chunks: list[Chunk], max_samples: int = 50, seed: int = 7) -> list[SyntheticSample]:
    """Generate lightweight synthetic QA samples from ingested chunks."""

    random.seed(seed)
    if not chunks:
        return []

    sampled = random.sample(chunks, k=min(max_samples, len(chunks)))
    out: list[SyntheticSample] = []

    for chunk in sampled:
        question = f"What does the document say about: {chunk.text.split('.')[0][:80]}?"
        reference = chunk.text[:240]
        out.append(
            SyntheticSample(
                question=question,
                reference_answer=reference,
                supporting_chunk_ids=[chunk.chunk_id],
            )
        )

    return out
