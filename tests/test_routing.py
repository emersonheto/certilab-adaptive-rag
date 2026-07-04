"""Unit tests for Adaptive RAG conditional-edge routing functions.

These are pure functions that read the graph state and return the next node
name (or END). They encode the bounded-loop logic for query rewrite (max 3)
and hallucination regeneration (max 2).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from app.adaptive_rag.graph import check_hallucinations, decide_to_generate


def _state(**overrides: Any) -> dict[str, Any]:
    """Build a minimal graph-state dict with sensible defaults."""

    base: dict[str, Any] = {
        "question": "",
        "generation": "",
        "documents": [],
        "web_results": [],
        "route": "vectorstore",
        "rewrite_count": 0,
        "regenerate_count": 0,
        "hallucination_verdict": "grounded",
        "answer_verdict": "useful",
        "principal": None,
        "scope": None,
    }
    base.update(overrides)
    return base


class TestDecideToGenerate:
    """decide_to_generate — routes after grade_documents."""

    def test_empty_docs_within_rewrite_bounds_routes_to_transform(self) -> None:
        state = _state(documents=[], rewrite_count=0)
        assert decide_to_generate(state) == "transform_query"

    def test_empty_docs_just_below_max_still_rewrites(self) -> None:
        state = _state(documents=[], rewrite_count=2)
        assert decide_to_generate(state) == "transform_query"

    def test_empty_docs_at_max_rewrite_routes_to_generate(self) -> None:
        state = _state(documents=[], rewrite_count=3)
        assert decide_to_generate(state) == "generate"

    def test_non_empty_docs_routes_to_generate(self) -> None:
        state = _state(documents=["relevant doc"], rewrite_count=0)
        assert decide_to_generate(state) == "generate"

    def test_non_empty_docs_with_high_rewrite_count_still_generates(self) -> None:
        state = _state(documents=["doc"], rewrite_count=5)
        assert decide_to_generate(state) == "generate"


class TestCheckHallucinations:
    """check_hallucinations — routes after hallucination_check."""

    def test_hallucinated_within_regen_bounds_routes_to_generate(self) -> None:
        state = _state(hallucination_verdict="hallucinated", regenerate_count=0)
        assert check_hallucinations(state) == "generate"

    def test_hallucinated_just_below_max_still_regenerates(self) -> None:
        state = _state(hallucination_verdict="hallucinated", regenerate_count=1)
        assert check_hallucinations(state) == "generate"

    def test_hallucinated_at_max_regen_with_useful_answer_routes_to_end(self) -> None:
        state = _state(hallucination_verdict="hallucinated", regenerate_count=2, answer_verdict="useful")
        result = check_hallucinations(state)
        assert result == END
        assert result != "generate"

    def test_hallucinated_at_max_regen_with_not_useful_routes_to_transform(self) -> None:
        state = _state(
            hallucination_verdict="hallucinated", regenerate_count=2, answer_verdict="not_useful"
        )
        assert check_hallucinations(state) == "transform_query"

    def test_grounded_but_not_useful_routes_to_transform(self) -> None:
        state = _state(hallucination_verdict="grounded", answer_verdict="not_useful", regenerate_count=0)
        assert check_hallucinations(state) == "transform_query"

    def test_grounded_and_useful_routes_to_end(self) -> None:
        state = _state(hallucination_verdict="grounded", answer_verdict="useful", regenerate_count=0)
        assert check_hallucinations(state) == END
