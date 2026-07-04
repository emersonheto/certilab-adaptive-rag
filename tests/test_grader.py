"""Unit tests for the Adaptive RAG Pydantic grader schemas.

Each schema is used as a ``with_structured_output`` target for an LLM call.
These tests verify that valid data instantiates correctly and that invalid
data (violating the Literal constraints) raises a ValidationError.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.adaptive_rag.grader import (
    GradeAnswer,
    GradeDocuments,
    GradeHallucinations,
    QuestionRewriter,
    RouteQuery,
)


class TestRouteQuery:
    """RouteQuery schema — binary routing decision for question classification."""

    def test_valid_vectorstore_route(self) -> None:
        result = RouteQuery(route="vectorstore")
        assert result.route == "vectorstore"

    def test_valid_web_search_route(self) -> None:
        result = RouteQuery(route="web_search")
        assert result.route == "web_search"

    def test_invalid_route_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RouteQuery(route="database")

    def test_missing_route_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RouteQuery()  # type: ignore[call-arg]


class TestGradeDocuments:
    """GradeDocuments schema — relevance grading for retrieved documents."""

    def test_valid_yes_score(self) -> None:
        result = GradeDocuments(score="yes")
        assert result.score == "yes"

    def test_valid_no_score(self) -> None:
        result = GradeDocuments(score="no")
        assert result.score == "no"

    def test_invalid_score_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            GradeDocuments(score="maybe")

    def test_missing_score_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            GradeDocuments()  # type: ignore[call-arg]


class TestGradeHallucinations:
    """GradeHallucinations schema — groundedness check for LLM generations."""

    def test_valid_yes_score(self) -> None:
        result = GradeHallucinations(score="yes")
        assert result.score == "yes"

    def test_valid_no_score(self) -> None:
        result = GradeHallucinations(score="no")
        assert result.score == "no"

    def test_invalid_score_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            GradeHallucinations(score="partial")

    def test_missing_score_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            GradeHallucinations()  # type: ignore[call-arg]


class TestGradeAnswer:
    """GradeAnswer schema — usefulness check for LLM answers."""

    def test_valid_yes_score(self) -> None:
        result = GradeAnswer(score="yes")
        assert result.score == "yes"

    def test_valid_no_score(self) -> None:
        result = GradeAnswer(score="no")
        assert result.score == "no"

    def test_invalid_score_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            GradeAnswer(score="unknown")

    def test_missing_score_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            GradeAnswer()  # type: ignore[call-arg]


class TestQuestionRewriter:
    """QuestionRewriter schema — rewritten query output for transform_query node."""

    def test_valid_question(self) -> None:
        result = QuestionRewriter(question="calibration certificates for client 101")
        assert result.question == "calibration certificates for client 101"

    def test_empty_question_is_valid(self) -> None:
        """An empty string is a valid str; the LLM is responsible for non-empty output."""
        result = QuestionRewriter(question="")
        assert result.question == ""

    def test_missing_question_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            QuestionRewriter()  # type: ignore[call-arg]

    def test_non_string_question_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            QuestionRewriter(question=42)  # type: ignore[arg-type]
