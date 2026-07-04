"""Unit tests for Adaptive RAG node factory functions.

Each factory (``make_*``) injects its dependencies via a closure and returns a
callable ``(state) -> dict``. These tests verify state mutations with mocked
LLM, embeddings, index, and web-search dependencies.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from app.adaptive_rag.grader import (
    GradeAnswer,
    GradeDocuments,
    GradeHallucinations,
    RouteQuery,
)
from app.adaptive_rag.nodes import (
    make_generate,
    make_grade_documents,
    make_hallucination_check,
    make_retrieve,
    make_route_question,
    make_transform_query,
    make_web_search,
)
from app.domain.models import Role
from app.security.access_control import AccessScope
from app.tools.web_search import WebSearchResult


def _mock_settings() -> MagicMock:
    """Return a mock Settings with defaults matching production config."""

    settings = MagicMock()
    settings.default_top_k = 4
    settings.openai_chat_model = "gpt-4o-mini"
    settings.openai_temperature = 0.0
    return settings


class TestRouteQuestion:
    """route_question — classifies question to 'vectorstore' or 'web_search'."""

    def test_routes_to_vectorstore(self) -> None:
        llm = _structured_llm(RouteQuery(route="vectorstore"))
        node = make_route_question(llm, _mock_settings())
        result = node({"question": "What calibration certificates exist?"})
        assert result == {"route": "vectorstore"}

    def test_routes_to_web_search(self) -> None:
        llm = _structured_llm(RouteQuery(route="web_search"))
        node = make_route_question(llm, _mock_settings())
        result = node({"question": "What are the latest ISO standard updates?"})
        assert result == {"route": "web_search"}


class TestRetrieve:
    """retrieve — embeds question, searches index, extracts document texts."""

    def test_returns_document_texts_from_index(self) -> None:
        index = _fake_index([("c1", 0.9, {"text": "doc one"}), ("c2", 0.8, {"text": "doc two"})])
        node = make_retrieve(MagicMock(), index, _mock_settings())
        result = node({"question": "query", "scope": None})
        assert result == {"documents": ["doc one", "doc two"]}

    def test_passes_none_allowed_ids_for_global_scope(self) -> None:
        index = _fake_index([("c1", 0.9, {"text": "doc"})])
        scope = AccessScope(role=Role.ADMIN, customer_id=None)
        node = make_retrieve(MagicMock(), index, _mock_settings())
        node({"question": "query", "scope": scope})
        index.search.assert_called_once_with("query", None, 4)

    def test_passes_customer_ids_for_scoped_retrieval(self) -> None:
        index = _fake_index([("c1", 0.9, {"text": "scoped"})])
        scope = AccessScope(role=Role.CLIENT, customer_id=101)
        node = make_retrieve(MagicMock(), index, _mock_settings())
        result = node({"question": "query", "scope": scope})
        assert result == {"documents": ["scoped"]}
        index.search.assert_called_once_with("query", {101}, 4)

    def test_empty_results_returns_empty_documents(self) -> None:
        index = _fake_index([])
        node = make_retrieve(MagicMock(), index, _mock_settings())
        result = node({"question": "query", "scope": None})
        assert result == {"documents": []}


class TestGradeDocuments:
    """grade_documents — keeps only docs graded relevant ('yes')."""

    def test_keeps_all_relevant_docs(self) -> None:
        llm = _structured_llm(GradeDocuments(score="yes"))
        node = make_grade_documents(llm, _mock_settings())
        result = node({"question": "q", "documents": ["doc1", "doc2"]})
        assert result == {"documents": ["doc1", "doc2"]}

    def test_filters_all_irrelevant_docs(self) -> None:
        llm = _structured_llm(GradeDocuments(score="no"))
        node = make_grade_documents(llm, _mock_settings())
        result = node({"question": "q", "documents": ["doc1", "doc2"]})
        assert result == {"documents": []}

    def test_keeps_only_relevant_in_mixed_set(self) -> None:
        llm = _structured_llm(
            GradeDocuments(score="yes"),
            GradeDocuments(score="no"),
        )
        node = make_grade_documents(llm, _mock_settings())
        result = node({"question": "q", "documents": ["relevant", "irrelevant"]})
        assert result == {"documents": ["relevant"]}


class TestTransformQuery:
    """transform_query — rewrites question, increments rewrite_count."""

    def test_rewrites_question_and_increments_count(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="improved specific query")
        node = make_transform_query(llm, _mock_settings())
        result = node({"question": "vague question", "rewrite_count": 0})
        assert result["question"] == "improved specific query"
        assert result["rewrite_count"] == 1

    def test_increments_existing_count(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="better query")
        node = make_transform_query(llm, _mock_settings())
        result = node({"question": "vague", "rewrite_count": 2})
        assert result["rewrite_count"] == 3


class TestGenerate:
    """generate — produces answer from documents or web results."""

    def test_generates_from_documents(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Answer based on retrieved docs.")
        node = make_generate(llm, _mock_settings())
        result = node({"question": "q", "documents": ["doc1"], "web_results": [], "regenerate_count": 0})
        assert result["generation"] == "Answer based on retrieved docs."
        assert result["regenerate_count"] == 1

    def test_falls_back_to_web_results_when_no_documents(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Answer from web snippets.")
        node = make_generate(llm, _mock_settings())
        web = [WebSearchResult(title="t", url="u", snippet="web snippet text")]
        result = node({"question": "q", "documents": [], "web_results": web, "regenerate_count": 0})
        assert result["generation"] == "Answer from web snippets."

    def test_increments_regenerate_count(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="Regenerated answer.")
        node = make_generate(llm, _mock_settings())
        result = node({"question": "q", "documents": ["d"], "web_results": [], "regenerate_count": 1})
        assert result["regenerate_count"] == 2


class TestWebSearch:
    """web_search_node — calls web_search.search and returns results."""

    def test_returns_web_results(self) -> None:
        results = [WebSearchResult(title="t", url="u", snippet="snippet")]
        web_search = MagicMock()
        web_search.search.return_value = results
        node = make_web_search(web_search, _mock_settings())
        result = node({"question": "query"})
        assert result == {"web_results": results}


class TestHallucinationCheck:
    """hallucination_check — grades groundedness then answer usefulness."""

    def test_grounded_and_useful(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.side_effect = [
            _structured_invoke(GradeHallucinations(score="yes")),
            _structured_invoke(GradeAnswer(score="yes")),
        ]
        node = make_hallucination_check(llm, _mock_settings())
        result = node({"question": "q", "documents": ["doc"], "generation": "answer"})
        assert result == {"hallucination_verdict": "grounded", "answer_verdict": "useful"}

    def test_grounded_but_not_useful(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.side_effect = [
            _structured_invoke(GradeHallucinations(score="yes")),
            _structured_invoke(GradeAnswer(score="no")),
        ]
        node = make_hallucination_check(llm, _mock_settings())
        result = node({"question": "q", "documents": ["doc"], "generation": "answer"})
        assert result == {"hallucination_verdict": "grounded", "answer_verdict": "not_useful"}

    def test_hallucinated_skips_answer_check(self) -> None:
        llm = MagicMock()
        llm.with_structured_output.return_value = _structured_invoke(GradeHallucinations(score="no"))
        node = make_hallucination_check(llm, _mock_settings())
        result = node({"question": "q", "documents": ["doc"], "generation": "answer"})
        assert result["hallucination_verdict"] == "hallucinated"
        assert result["answer_verdict"] == "not_useful"


# --- Helpers ---


def _structured_invoke(pydantic_obj: Any) -> MagicMock:
    """Return a mock whose invoke() always returns *pydantic_obj*."""

    mock = MagicMock()
    mock.invoke.return_value = pydantic_obj
    return mock


def _structured_llm(*pydantic_objs: Any) -> MagicMock:
    """Return a mock LLM whose with_structured_output().invoke() cycles through *pydantic_objs*.

    If a single object is given, every invoke returns it. If multiple are given,
    they are consumed in order (side_effect on the inner invoke).
    """

    llm = MagicMock()
    if len(pydantic_objs) == 1:
        llm.with_structured_output.return_value = _structured_invoke(pydantic_objs[0])
    else:
        llm.with_structured_output.return_value.invoke.side_effect = pydantic_objs
    return llm


def _fake_index(results: list[tuple[str, float, dict[str, Any]]]) -> MagicMock:
    """Return a mock VectorIndex whose search() returns *results*."""

    index = MagicMock()
    index.search.return_value = results
    return index
