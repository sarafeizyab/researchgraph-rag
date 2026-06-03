from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalResult:
    metric_name: str
    value: float
    details: dict[str, Any]


def citation_coverage(answer: str, cited_chunk_ids: list[str]) -> EvalResult:
    claims = [c for c in re.split(r"(?<=[.!?])\s+", answer.strip()) if c]
    cited_claims = [c for c in claims if "[" in c and "]" in c]
    value = len(cited_claims) / max(len(claims), 1)
    return EvalResult("citation_coverage", value, {"claims": len(claims), "cited_claims": len(cited_claims), "unique_citations": len(set(cited_chunk_ids))})


def retrieved_context_overlap(retrieved_texts: list[str], reference_answer: str) -> EvalResult:
    ref_tokens = set(reference_answer.lower().split())
    if not ref_tokens:
        return EvalResult("retrieved_context_overlap", 0.0, {})

    retrieved_tokens = set()
    for text in retrieved_texts:
        retrieved_tokens.update(text.lower().split())

    overlap = len(ref_tokens & retrieved_tokens) / len(ref_tokens)
    return EvalResult("retrieved_context_overlap", overlap, {"reference_tokens": len(ref_tokens)})


def answer_length(answer: str) -> EvalResult:
    tokens = answer.split()
    return EvalResult("answer_length", float(len(tokens)), {"token_count": len(tokens)})


def cited_claim_count(answer: str) -> EvalResult:
    claims = [c for c in re.split(r"(?<=[.!?])\s+", answer.strip()) if c]
    cited_claims = [c for c in claims if re.search(r"\[[^\]]+\]", c)]
    return EvalResult("cited_claim_count", float(len(cited_claims)), {"claims": len(claims)})


def unsupported_claim_detector_placeholder(answer: str) -> EvalResult:
    # Placeholder for a future NLI verifier.
    flagged = 1.0 if "hallucination" in answer.lower() else 0.0
    return EvalResult("unsupported_claim_detector_placeholder", flagged, {"note": "Replace with NLI-based verifier."})
