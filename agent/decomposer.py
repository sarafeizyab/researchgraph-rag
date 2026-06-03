from __future__ import annotations

import logging

from agent.llm import LLMClient

LOGGER = logging.getLogger(__name__)


class QueryDecomposer:
    """Breaks complex scientific questions into focused sub-queries."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def decompose(self, question: str) -> tuple[list[str], float, dict[str, int]]:
        prompt = (
            "Decompose the scientific question into 2-5 concise retrieval sub-queries. "
            "Return JSON only with key sub_queries.\n"
            f"Question: {question}\n"
            'Output format: {"sub_queries": ["...", "..."]}'
        )

        parsed, resp = self.llm.complete_json(prompt=prompt, temperature=0.0)
        raw = parsed.get("sub_queries", []) if isinstance(parsed, dict) else []

        sub_queries = [str(x).strip() for x in raw if str(x).strip()]
        if not sub_queries:
            sub_queries = [question]

        return sub_queries[:5], resp.latency_ms, resp.token_usage
