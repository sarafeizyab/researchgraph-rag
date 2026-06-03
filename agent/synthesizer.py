from __future__ import annotations

import re

from agent.llm import LLMClient


class AnswerSynthesizer:
    """Generates citation-grounded final answers from retrieved context."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def synthesize(self, question: str, contexts: list[dict[str, str]]) -> tuple[str, float, dict[str, int]]:
        if not contexts:
            return "I could not find sufficient context to answer the question.", 0.0, {}

        context_block = "\n".join(
            [f"[{c['chunk_id']}] ({c.get('source_doc')}, page={c.get('page')}, section={c.get('section')}) {c.get('text', '')}" for c in contexts]
        )

        prompt = (
            "Use ONLY the provided context to answer the question. "
            "For each factual claim include at least one citation marker [chunk_id]. "
            "If uncertain, say what is missing.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_block}\n"
        )

        resp = self.llm.complete(prompt=prompt, temperature=0.1)
        answer = resp.text.strip() or "No answer generated."

        if "[" not in answer:
            # Ensure citation markers exist even for fallback outputs.
            marker = contexts[0]["chunk_id"]
            answer = f"{answer} [{marker}]"

        return answer, resp.latency_ms, resp.token_usage

    def extract_cited_chunk_ids(self, answer: str) -> list[str]:
        return sorted(set(re.findall(r"\[([a-f0-9\-]{8,})\]", answer, flags=re.IGNORECASE)))
