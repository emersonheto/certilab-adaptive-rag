"""Pydantic schemas for Adaptive RAG LLM structured output.

Each model is used as a ``with_structured_output`` target so that an LLM call
returns a validated Pydantic instance instead of free-form text. Literal
constraints enforce that the LLM can only produce the expected enum values.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class RouteQuery(BaseModel):
    """Routing decision for the ``route_question`` node.

    Attributes:
        route: ``"vectorstore"`` for calibration/technical questions,
            ``"web_search"`` for current events or general web knowledge.
    """

    route: Literal["vectorstore", "web_search"]


class GradeDocuments(BaseModel):
    """Relevance grade for a single retrieved document.

    Attributes:
        score: ``"yes"`` if the document is relevant to the question,
            ``"no"`` otherwise.
    """

    score: Literal["yes", "no"]


class GradeHallucinations(BaseModel):
    """Groundedness check for an LLM generation against retrieved facts.

    Attributes:
        score: ``"yes"`` if the generation is grounded in the documents,
            ``"no"`` if it is a hallucination.
    """

    score: Literal["yes", "no"]


class GradeAnswer(BaseModel):
    """Usefulness check for whether an answer resolves the question.

    Attributes:
        score: ``"yes"`` if the answer addresses the question,
            ``"no"`` if it does not.
    """

    score: Literal["yes", "no"]


class QuestionRewriter(BaseModel):
    """Rewritten query output from the ``transform_query`` node.

    Attributes:
        question: The reformulated question, more specific and precise
            for vector search.
    """

    question: str
