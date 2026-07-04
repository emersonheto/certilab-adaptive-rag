"""Conditional-edge routing functions and graph topology for Adaptive RAG.

This module defines the pure routing functions used by LangGraph conditional
edges (``decide_to_generate`` and ``check_hallucinations``) and the
``build_graph`` factory that assembles the full 7-node StateGraph with two
bounded self-correction loops.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI

from app.adaptive_rag.nodes import (
    make_generate,
    make_grade_documents,
    make_hallucination_check,
    make_retrieve,
    make_route_question,
    make_transform_query,
    make_web_search,
)
from app.adaptive_rag.state import AdaptiveRAGState
from app.config import Settings
from app.retrieval.protocols import VectorIndex
from app.tools.web_search import TavilyWebSearch


_MAX_REWRITE = 3
_MAX_REGENERATE = 2


def decide_to_generate(state: AdaptiveRAGState) -> str:
    """After grade_documents: rewrite the query if no docs survived grading.

    Routes to ``transform_query`` when every document was graded irrelevant
    AND the rewrite loop has not yet hit its bound of MAX_REWRITE attempts.
    Otherwise routes to ``generate`` so the caller produces an answer (even
    from an empty document set, which falls back to web results).
    """

    if len(state["documents"]) == 0 and state["rewrite_count"] < _MAX_REWRITE:
        return "transform_query"
    return "generate"


def check_hallucinations(state: AdaptiveRAGState) -> str:
    """After hallucination_check: route based on groundedness and usefulness.

    - Hallucinated AND regenerate_count < MAX_REGENERATE -> regenerate via ``generate``.
    - Grounded but answer not useful AND rewrite_count < MAX_REWRITE -> rewrite via ``transform_query``.
    - Otherwise (grounded + useful, or ALL bounds exhausted) -> END.
    """

    if state.get("hallucination_verdict") == "hallucinated" and state["regenerate_count"] < _MAX_REGENERATE:
        return "generate"
    if state.get("answer_verdict") == "not_useful" and state["rewrite_count"] < _MAX_REWRITE:
        return "transform_query"
    return END


def route_after_question(state: AdaptiveRAGState) -> str:
    """Conditional edge after route_question: dispatch to retrieve or web_search."""

    return state["route"]


def build_graph(
    index: VectorIndex,
    embeddings: object | None,
    web_search: TavilyWebSearch,
    settings: Settings,
    llm: ChatOpenAI | None = None,
) -> CompiledStateGraph[AdaptiveRAGState, Any, Any, Any]:
    """Assemble the 7-node Adaptive RAG StateGraph with self-correction loops.

    Args:
        index: Vector index supporting the ``search(query, allowed_ids, top_k)`` protocol.
        embeddings: Embeddings provider (``None`` for mock/InMemory mode; an
            ``EmbeddingsProvider`` instance for real mode where it is already
            injected into the ``QdrantVectorIndex`` at construction).
        web_search: Tavily web-search client.
        settings: Application settings — used to build the ``ChatOpenAI``
            instance and forwarded to every node factory for ``trace_span``
            attributes and ``default_top_k``.
        llm: Optional ``ChatOpenAI`` override (primarily for tests injecting a
            fake LLM).  When ``None``, built from ``settings.openai_chat_model``
            and ``settings.openai_temperature``.

    Returns:
        A compiled LangGraph ready to ``invoke`` or ``stream``.
    """

    if llm is None:
        llm = ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=settings.openai_temperature,
        )

    route_question = make_route_question(llm, settings)
    retrieve = make_retrieve(embeddings, index, settings)
    grade_documents = make_grade_documents(llm, settings)
    transform_query = make_transform_query(llm, settings)
    web_search_node = make_web_search(web_search, settings)
    generate = make_generate(llm, settings)
    hallucination_check = make_hallucination_check(llm, settings)

    workflow = StateGraph(AdaptiveRAGState)
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination_check", hallucination_check)

    workflow.add_edge(START, "route_question")
    workflow.add_conditional_edges(
        "route_question",
        route_after_question,
        {"vectorstore": "retrieve", "web_search": "web_search"},
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges(
        "hallucination_check",
        check_hallucinations,
        {"generate": "generate", "transform_query": "transform_query", END: END},
    )

    return workflow.compile()
