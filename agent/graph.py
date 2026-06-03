from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional import path
    END = "__end__"
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False

from agent.decomposer import QueryDecomposer
from agent.reflector import SelfReflector
from agent.state import AgentState
from agent.synthesizer import AnswerSynthesizer
from config import get_settings
from models import Citation, QueryMetrics, QueryResult
from retrieval.hybrid import HybridRetriever

LOGGER = logging.getLogger(__name__)


@dataclass
class MultiHopGraphAgent:
    """LangGraph-based multi-hop scientific RAG agent."""

    decomposer: QueryDecomposer
    retriever: HybridRetriever
    reflector: SelfReflector
    synthesizer: AnswerSynthesizer
    max_hops: int

    def __post_init__(self) -> None:
        self._graph = self._build_graph()

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            LOGGER.warning("LangGraph not available; using deterministic fallback executor.")
            return None

        graph = StateGraph(AgentState)
        graph.add_node("decompose", self._decompose_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("synthesize", self._synthesize_node)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "reflect")
        graph.add_conditional_edges(
            "reflect",
            self._route_after_reflection,
            {
                "retrieve": "retrieve",
                "synthesize": "synthesize",
            },
        )
        graph.add_edge("synthesize", END)
        return graph.compile()

    def run(self, question: str) -> QueryResult:
        start_total = time.perf_counter()

        initial: AgentState = {
            "original_question": question,
            "sub_queries": [],
            "current_query": question,
            "accumulated_context": [],
            "hop_count": 0,
            "max_hops": self.max_hops,
            "sufficient": False,
            "follow_up_query": None,
            "answer": "",
            "citations": [],
            "reasoning_trace": [],
            "latency": {"retrieval_ms": 0.0, "reranking_ms": 0.0, "llm_ms": 0.0},
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        if self._graph is None:
            final_state = self._run_fallback(initial)
        else:
            final_state = self._graph.invoke(initial)
        total_latency_ms = (time.perf_counter() - start_total) * 1000.0

        citations = [
            Citation(
                chunk_id=str(c["chunk_id"]),
                source_doc=str(c["source_doc"]),
                page=c.get("page"),
                section=c.get("section"),
                excerpt=str(c.get("excerpt", "")),
            )
            for c in final_state.get("citations", [])
        ]

        metrics = QueryMetrics(
            total_latency_ms=total_latency_ms,
            retrieval_latency_ms=float(final_state["latency"].get("retrieval_ms", 0.0)),
            reranking_latency_ms=float(final_state["latency"].get("reranking_ms", 0.0)),
            llm_latency_ms=float(final_state["latency"].get("llm_ms", 0.0)),
            hops=int(final_state.get("hop_count", 0)),
            retrieved_chunks=len(final_state.get("accumulated_context", [])),
            token_usage={k: int(v) for k, v in final_state.get("token_usage", {}).items()},
        )

        used_chunk_ids = [c.chunk_id for c in citations]
        return QueryResult(
            answer=final_state.get("answer", ""),
            citations=citations,
            reasoning_trace=final_state.get("reasoning_trace", []),
            metrics=metrics,
            sub_queries=final_state.get("sub_queries", []),
            used_chunk_ids=used_chunk_ids,
            generated_at=datetime.utcnow(),
        )

    def _run_fallback(self, state: AgentState) -> AgentState:
        updates = self._decompose_node(state)
        state = {**state, **updates}

        while True:
            updates = self._retrieve_node(state)
            state = {**state, **updates}

            updates = self._reflect_node(state)
            state = {**state, **updates}

            if self._route_after_reflection(state) == "synthesize":
                break

        updates = self._synthesize_node(state)
        return {**state, **updates}

    def _decompose_node(self, state: AgentState) -> dict[str, Any]:
        sub_queries, latency_ms, usage = self.decomposer.decompose(state["original_question"])
        trace = list(state["reasoning_trace"])
        trace.append({"step": "decompose", "sub_queries": sub_queries})

        current = sub_queries[0] if sub_queries else state["original_question"]
        return {
            "sub_queries": sub_queries,
            "current_query": current,
            "reasoning_trace": trace,
            "latency": self._update_latency(state["latency"], llm_ms=latency_ms),
            "token_usage": self._merge_usage(state["token_usage"], usage),
        }

    def _retrieve_node(self, state: AgentState) -> dict[str, Any]:
        if hasattr(self.retriever, "retrieve_with_diagnostics"):
            retrieved, diagnostics = self.retriever.retrieve_with_diagnostics(query=state["current_query"])
            retrieval_ms = float(diagnostics.get("retrieval_ms", 0.0))
            reranking_ms = float(diagnostics.get("reranking_ms", 0.0))
        else:
            start = time.perf_counter()
            retrieved = self.retriever.retrieve(query=state["current_query"])
            retrieval_ms = (time.perf_counter() - start) * 1000.0
            reranking_ms = 0.0

        existing = {item["chunk_id"] for item in state["accumulated_context"]}
        context = list(state["accumulated_context"])
        added = 0
        for item in retrieved:
            cid = item.chunk.chunk_id
            if cid in existing:
                continue
            existing.add(cid)
            context.append(
                {
                    "chunk_id": cid,
                    "source_doc": item.chunk.source_doc,
                    "title": item.chunk.title,
                    "page": item.chunk.page,
                    "section": item.chunk.section,
                    "text": item.chunk.text,
                    "score": item.score,
                }
            )
            added += 1

        trace = list(state["reasoning_trace"])
        trace.append(
            {
                "step": "retrieve",
                "query": state["current_query"],
                "retrieved": len(retrieved),
                "new_chunks_added": added,
            }
        )

        return {
            "accumulated_context": context,
            "reasoning_trace": trace,
            "latency": self._update_latency(state["latency"], retrieval_ms=retrieval_ms, reranking_ms=reranking_ms),
        }

    def _reflect_node(self, state: AgentState) -> dict[str, Any]:
        next_hop = state["hop_count"] + 1
        sufficient, follow_up, rationale, latency_ms, usage = self.reflector.reflect(
            original_question=state["original_question"],
            current_query=state["current_query"],
            contexts=state["accumulated_context"],
            hop_count=next_hop,
            max_hops=state["max_hops"],
        )

        if next_hop >= state["max_hops"]:
            sufficient = True
            follow_up = None

        next_query = self._select_next_query(
            sub_queries=state["sub_queries"],
            completed_hops=next_hop,
            follow_up=follow_up,
            current_query=state["current_query"],
            sufficient=sufficient,
        )
        if next_query != follow_up:
            follow_up = next_query

        trace = list(state["reasoning_trace"])
        trace.append(
            {
                "step": "reflect",
                "hop": next_hop,
                "sufficient": sufficient,
                "follow_up_query": follow_up,
                "rationale": rationale,
            }
        )

        current_query = follow_up if follow_up else state["current_query"]
        return {
            "hop_count": next_hop,
            "sufficient": sufficient,
            "follow_up_query": follow_up,
            "current_query": current_query,
            "reasoning_trace": trace,
            "latency": self._update_latency(state["latency"], llm_ms=latency_ms),
            "token_usage": self._merge_usage(state["token_usage"], usage),
        }

    def _select_next_query(
        self,
        sub_queries: list[str],
        completed_hops: int,
        follow_up: str | None,
        current_query: str,
        sufficient: bool,
    ) -> str | None:
        if sufficient:
            return None

        if completed_hops < len(sub_queries):
            candidate = sub_queries[completed_hops]
            if candidate and candidate != current_query:
                return candidate

        return follow_up

    def _route_after_reflection(self, state: AgentState) -> str:
        if state["sufficient"] or state["hop_count"] >= state["max_hops"]:
            return "synthesize"
        return "retrieve"

    def _synthesize_node(self, state: AgentState) -> dict[str, Any]:
        answer, latency_ms, usage = self.synthesizer.synthesize(
            question=state["original_question"],
            contexts=state["accumulated_context"],
        )

        cited_ids = set(self.synthesizer.extract_cited_chunk_ids(answer))
        citations = []
        for chunk in state["accumulated_context"]:
            if cited_ids and chunk["chunk_id"] not in cited_ids:
                continue
            citations.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source_doc": chunk["source_doc"],
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "excerpt": str(chunk["text"])[:280],
                }
            )

        trace = list(state["reasoning_trace"])
        trace.append({"step": "synthesize", "cited_chunks": [c["chunk_id"] for c in citations]})

        return {
            "answer": answer,
            "citations": citations,
            "reasoning_trace": trace,
            "latency": self._update_latency(state["latency"], llm_ms=latency_ms),
            "token_usage": self._merge_usage(state["token_usage"], usage),
        }

    def _update_latency(
        self,
        previous: dict[str, float],
        retrieval_ms: float = 0.0,
        reranking_ms: float = 0.0,
        llm_ms: float = 0.0,
    ) -> dict[str, float]:
        return {
            "retrieval_ms": float(previous.get("retrieval_ms", 0.0)) + float(retrieval_ms),
            "reranking_ms": float(previous.get("reranking_ms", 0.0)) + float(reranking_ms),
            "llm_ms": float(previous.get("llm_ms", 0.0)) + float(llm_ms),
        }

    def _merge_usage(self, prev: dict[str, int], curr: dict[str, int]) -> dict[str, int]:
        merged = dict(prev)
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            merged[key] = int(merged.get(key, 0)) + int(curr.get(key, 0))
        return merged


def build_default_agent(retriever: HybridRetriever) -> MultiHopGraphAgent:
    settings = get_settings()
    from agent.llm import LLMClient

    llm = LLMClient(model=settings.openai_model, api_key=settings.openai_api_key)
    return MultiHopGraphAgent(
        decomposer=QueryDecomposer(llm),
        retriever=retriever,
        reflector=SelfReflector(llm),
        synthesizer=AnswerSynthesizer(llm),
        max_hops=settings.max_hops,
    )
