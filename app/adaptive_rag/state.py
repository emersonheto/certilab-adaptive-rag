"""Graph state schema for the Adaptive RAG topology.

Defines the ``AdaptiveRAGState`` TypedDict that flows through every node in the
StateGraph. Each node reads specific fields and writes back its output into
the same dict, which LangGraph merges automatically.
"""

from __future__ import annotations

from typing import TypedDict

from app.security.access_control import AccessScope, Principal
from app.tools.web_search import WebSearchResult


class AdaptiveRAGState(TypedDict):
    """Mutable state dict shared across all Adaptive RAG nodes.

    Fields:
        question: The user question (may be rewritten by ``transform_query``).
        generation: Final LLM answer text produced by ``generate``.
        documents: Retrieved and graded document texts (filtered to relevant only).
        web_results: Results from Tavily web search (``web_search`` node).
        route: Routing decision — ``"vectorstore"`` or ``"web_search"``.
        rewrite_count: Number of times ``transform_query`` has rewritten the question.
        regenerate_count: Number of times ``generate`` has been re-invoked after hallucination.
        hallucination_verdict: ``"grounded"`` or ``"hallucinated"`` from ``hallucination_check``.
        answer_verdict: ``"useful"`` or ``"not_useful"`` from ``hallucination_check``.
        principal: Optional authenticated caller; if set, its scope filters retrieval.
        scope: Derived access scope; ``None`` means no tenant filtering.
    """

    question: str
    generation: str
    documents: list[str]
    web_results: list[WebSearchResult]
    route: str
    rewrite_count: int
    regenerate_count: int
    hallucination_verdict: str
    answer_verdict: str
    principal: Principal | None
    scope: AccessScope | None
