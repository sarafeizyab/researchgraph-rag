from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """LangGraph state for multi-hop retrieval-augmented reasoning."""

    original_question: str
    sub_queries: list[str]
    current_query: str
    accumulated_context: list[dict[str, Any]]
    hop_count: int
    max_hops: int
    sufficient: bool
    follow_up_query: str | None
    answer: str
    citations: list[dict[str, Any]]
    reasoning_trace: list[dict[str, Any]]
    latency: dict[str, float]
    token_usage: dict[str, int]
