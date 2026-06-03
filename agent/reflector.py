from __future__ import annotations

from agent.llm import LLMClient


class SelfReflector:
    """Determines if current evidence is sufficient; proposes next-hop query when needed."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def reflect(
        self,
        original_question: str,
        current_query: str,
        contexts: list[dict[str, str]],
        hop_count: int,
        max_hops: int,
    ) -> tuple[bool, str | None, str, float, dict[str, int]]:
        if hop_count >= max_hops:
            return True, None, "max_hops_reached", 0.0, {}

        context_preview = "\n".join(
            [f"- [{c.get('chunk_id')}] {c.get('text', '')[:180]}" for c in contexts[-5:]]
        )
        prompt = (
            "You are a retrieval quality controller. Decide if evidence is sufficient to answer the original question. "
            "If not sufficient, generate exactly one focused follow-up retrieval query.\n"
            "Return JSON only:\n"
            '{"sufficient": bool, "follow_up_query": str|null, "rationale": str}\n\n'
            f"Original question: {original_question}\n"
            f"Current query: {current_query}\n"
            f"Hop: {hop_count}/{max_hops}\n"
            f"Context snippets:\n{context_preview}\n"
        )

        parsed, resp = self.llm.complete_json(prompt=prompt, temperature=0.0)
        sufficient = bool(parsed.get("sufficient", False)) if isinstance(parsed, dict) else False
        follow_up = parsed.get("follow_up_query") if isinstance(parsed, dict) else None
        rationale = str(parsed.get("rationale", "")) if isinstance(parsed, dict) else ""

        if not sufficient and (not isinstance(follow_up, str) or not follow_up.strip()):
            follow_up = f"additional evidence for: {original_question}"

        return sufficient, (follow_up.strip() if isinstance(follow_up, str) else None), rationale or "no_rationale", resp.latency_ms, resp.token_usage
