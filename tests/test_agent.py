import pytest

pytest.importorskip("pydantic")

from agent.graph import MultiHopGraphAgent


class _FakeDecomposer:
    def decompose(self, question: str):
        return ["subquery-1", "subquery-2"], 1.0, {"total_tokens": 10, "prompt_tokens": 7, "completion_tokens": 3}


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0
        self.queries = []

    def retrieve_with_diagnostics(self, query: str):
        self.calls += 1
        self.queries.append(query)
        chunk_id = f"chunk-{self.calls}"
        retrieved = [
            type(
                "R",
                (),
                {
                    "chunk": type(
                        "C",
                        (),
                        {
                            "chunk_id": chunk_id,
                            "source_doc": "paper.pdf",
                            "title": "Paper",
                            "page": 1,
                            "section": "Results",
                            "text": f"evidence for {query}",
                        },
                    )(),
                    "score": 0.9,
                },
            )()
        ]
        return retrieved, {"retrieval_ms": 2.0, "reranking_ms": 1.0}


class _FakeReflector:
    def __init__(self) -> None:
        self.calls = 0

    def reflect(self, original_question, current_query, contexts, hop_count, max_hops):
        self.calls += 1
        if self.calls == 1:
            return False, "follow-up-query", "need more", 1.5, {"total_tokens": 4, "prompt_tokens": 2, "completion_tokens": 2}
        return True, None, "enough", 1.0, {"total_tokens": 3, "prompt_tokens": 2, "completion_tokens": 1}


class _FakeSynthesizer:
    def synthesize(self, question: str, contexts):
        return "Final answer [chunk-1] [chunk-2]", 2.5, {"total_tokens": 5, "prompt_tokens": 3, "completion_tokens": 2}

    def extract_cited_chunk_ids(self, answer: str):
        return ["chunk-1", "chunk-2"]


def test_agent_multi_hop_state_transitions() -> None:
    retriever = _FakeRetriever()
    agent = MultiHopGraphAgent(
        decomposer=_FakeDecomposer(),
        retriever=retriever,
        reflector=_FakeReflector(),
        synthesizer=_FakeSynthesizer(),
        max_hops=4,
    )

    result = agent.run("compare transformers and cnns")

    assert "Final answer" in result.answer
    assert result.metrics.hops == 2
    assert result.metrics.retrieved_chunks >= 2
    assert len(result.citations) == 2
    assert retriever.queries == ["subquery-1", "subquery-2"]
    assert any(step["step"] == "decompose" for step in result.reasoning_trace)
    assert any(step["step"] == "reflect" for step in result.reasoning_trace)
