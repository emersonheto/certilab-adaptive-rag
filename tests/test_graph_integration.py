"""Integration tests for the Adaptive RAG graph topology.

Tests the full graph end-to-end with a FakeLLM that returns queued responses.
Three scenarios verify:
  1. Happy path: vectorstore route, relevant docs, grounded+useful answer.
  2. Rewrite loop: all docs irrelevant for 3 rounds, then relevant on the 4th.
  3. Regenerate loop: hallucination detected once, then grounded+useful.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.adaptive_rag.graph import build_graph
from app.adaptive_rag.grader import (
    GradeAnswer,
    GradeDocuments,
    GradeHallucinations,
    RouteQuery,
)


class FakeLLM:
    """Fake LLM that returns queued responses for structured and plain invoke.

    Both ``with_structured_output(Schema).invoke(prompt)`` and ``invoke(prompt)``
    pop the next response from a shared queue, preserving execution order.
    """

    def __init__(self) -> None:
        self._responses: list[Any] = []

    def add(self, obj: Any) -> None:
        """Queue a response (Pydantic model or AIMessage)."""

        self._responses.append(obj)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Structured:
            def invoke(self, prompt: Any, **kw: Any) -> Any:
                return fake._pop()

        return _Structured()

    def invoke(self, prompt: Any, **kwargs: Any) -> AIMessage:
        result = self._pop()
        if not isinstance(result, AIMessage):
            raise TypeError(f"FakeLLM.invoke expected AIMessage, got {type(result).__name__}")
        return result

    def _pop(self) -> Any:
        if not self._responses:
            raise RuntimeError("FakeLLM response queue exhausted — unexpected graph path")
        return self._responses.pop(0)


def _build_inputs(question: str = "test question") -> dict[str, Any]:
    """Build the initial graph state with all required fields."""

    return {
        "question": question,
        "generation": "",
        "documents": [],
        "web_results": [],
        "route": "",
        "rewrite_count": 0,
        "regenerate_count": 0,
        "hallucination_verdict": "",
        "answer_verdict": "",
        "principal": None,
        "scope": None,
    }


def _mock_index(docs: list[str]) -> MagicMock:
    """Return a mock VectorIndex that always returns *docs* from search."""

    index = MagicMock()
    index.search.return_value = [(f"c{i}", 0.9, {"text": d}) for i, d in enumerate(docs)]
    return index


def _mock_settings() -> MagicMock:
    """Return a mock Settings with defaults matching production config."""

    settings = MagicMock()
    settings.default_top_k = 4
    settings.openai_chat_model = "gpt-4o-mini"
    settings.openai_temperature = 0.0
    return settings


def _build(index: MagicMock, llm: FakeLLM) -> Any:
    """Build the graph with mock dependencies."""

    web_search = MagicMock()
    web_search.search.return_value = []
    return build_graph(
        index=index,
        embeddings=MagicMock(),
        web_search=web_search,
        settings=_mock_settings(),
        llm=llm,
    )


class TestHappyPath:
    """Scenario 1: route to vectorstore, all docs relevant, answer grounded."""

    def test_happy_path_reaches_end(self) -> None:
        llm = FakeLLM()
        llm.add(RouteQuery(route="vectorstore"))
        llm.add(GradeDocuments(score="yes"))
        llm.add(AIMessage(content="The certificate is valid."))
        llm.add(GradeHallucinations(score="yes"))
        llm.add(GradeAnswer(score="yes"))

        graph = _build(_mock_index(["relevant document text"]), llm)
        result = graph.invoke(_build_inputs("What certificates exist?"))

        assert result["generation"] == "The certificate is valid."
        assert result["regenerate_count"] == 1
        assert result["rewrite_count"] == 0
        assert result["documents"] == ["relevant document text"]


class TestRewriteLoop:
    """Scenario 2: all docs graded 'no' for 3 rounds, then 'yes' on the 4th."""

    def test_rewrite_loop_increments_count_and_recovers(self) -> None:
        llm = FakeLLM()
        llm.add(RouteQuery(route="vectorstore"))

        # Round 1: all no → rewrite (rewrite_count 0→1)
        llm.add(GradeDocuments(score="no"))
        llm.add(AIMessage(content="rewritten 1"))
        # Round 2: all no → rewrite (rewrite_count 1→2)
        llm.add(GradeDocuments(score="no"))
        llm.add(AIMessage(content="rewritten 2"))
        # Round 3: all no → rewrite (rewrite_count 2→3)
        llm.add(GradeDocuments(score="no"))
        llm.add(AIMessage(content="rewritten 3"))
        # Round 4: yes → generate
        llm.add(GradeDocuments(score="yes"))
        llm.add(AIMessage(content="Recovered answer."))
        llm.add(GradeHallucinations(score="yes"))
        llm.add(GradeAnswer(score="yes"))

        graph = _build(_mock_index(["document"]), llm)
        result = graph.invoke(_build_inputs("vague question"))

        assert result["rewrite_count"] == 3
        assert result["generation"] == "Recovered answer."
        assert result["documents"] == ["document"]


class TestRegenerateLoop:
    """Scenario 3: hallucination detected once, then grounded+useful."""

    def test_regenerate_loop_increments_count(self) -> None:
        llm = FakeLLM()
        llm.add(RouteQuery(route="vectorstore"))
        llm.add(GradeDocuments(score="yes"))
        # First generate → hallucinated
        llm.add(AIMessage(content="First attempt answer."))
        llm.add(GradeHallucinations(score="no"))
        # Second generate → grounded + useful
        llm.add(AIMessage(content="Second attempt answer."))
        llm.add(GradeHallucinations(score="yes"))
        llm.add(GradeAnswer(score="yes"))

        graph = _build(_mock_index(["document"]), llm)
        result = graph.invoke(_build_inputs("question"))

        assert result["regenerate_count"] == 2
        assert result["generation"] == "Second attempt answer."
        assert result["hallucination_verdict"] == "grounded"
        assert result["answer_verdict"] == "useful"
