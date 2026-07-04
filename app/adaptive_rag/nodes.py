"""Plain node functions for the Adaptive RAG StateGraph.

Each factory (``make_*``) injects its dependencies via a closure and returns a
callable ``(state) -> dict`` that LangGraph invokes during graph execution.
Nodes read specific state fields and write back their outputs as a partial
dict that LangGraph merges automatically.

Every LLM call is wrapped in a ``trace_span`` so Phoenix captures the full
adaptive RAG execution path with per-node duration and result metadata.
"""

from __future__ import annotations

import time
from typing import Any, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.adaptive_rag.grader import (
    GradeAnswer,
    GradeDocuments,
    GradeHallucinations,
    RouteQuery,
)
from app.config import Settings
from app.observability import trace_span
from app.retrieval.protocols import VectorIndex
from app.security.access_control import AccessScope
from app.tools.web_search import TavilyWebSearch

State = dict[str, Any]
NodeFn = Any  # Callable[[State], State], kept loose for LangGraph compatibility

# --- Inline prompts (NOT hub.pull — avoids langchain-community dependency) ---

_ROUTE_PROMPT = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to calibration certificates, laboratory data, and technical measurements.
Use the vectorstore for questions about these topics. Otherwise, use web search.

Question: {question}"""

_GRADE_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.

Retrieved document: {document}

User question: {question}"""

_REWRITE_PROMPT = """You are a question re-writer that converts an input question to a better version
that is optimized for vector retrieval. Look at the input and try to reason about the
underlying semantic intent / meaning.

Here is the initial question:
{question}

Formulate an improved question."""

_RAG_PROMPT = """You are an assistant for question-answering tasks. Use the following pieces of \
retrieved context to answer the question. If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""

_GROUNDEDNESS_PROMPT = """You are a grader assessing whether an LLM generation is grounded in / \
supported by a set of retrieved facts. Give a binary score 'yes' or 'no' to indicate whether the answer \
is grounded in the documents.

Set of facts:
{documents}

LLM generation:
{generation}"""

_ANSWER_PROMPT = """You are a grader assessing whether an answer resolves / addresses a question. \
Give a binary score 'yes' or 'no' to indicate whether the answer resolves the question.

Here is the question:
{question}

Here is the answer:
{generation}"""


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed milliseconds since *started_at* (perf_counter epoch)."""

    return int((time.perf_counter() - started_at) * 1000)


def make_route_question(llm: ChatOpenAI, settings: Settings) -> NodeFn:
    """Create the ``route_question`` node: classifies to vectorstore or web_search."""

    def route_question(state: State) -> State:
        question = state["question"]
        prompt = _ROUTE_PROMPT.format(question=question)
        started = time.perf_counter()
        with trace_span(
            "rag.adaptive.route_question",
            {"rag.question_length": len(question)},
        ) as span:
            result = cast(RouteQuery, llm.with_structured_output(RouteQuery).invoke(prompt))
            span.set_attribute("rag.route", result.route)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started))
            return {"route": result.route}

    return route_question


def make_retrieve(embeddings: object | None, index: VectorIndex, settings: Settings) -> NodeFn:
    """Create the ``retrieve`` node: searches the index for relevant documents.

    The ``VectorIndex.search()`` protocol accepts a query **string** and embeds
    it internally — the ``QdrantVectorIndex`` uses the ``EmbeddingsProvider``
    injected at construction time, so Phoenix traces the embed call via the
    provider's own ``trace_span("rag.embed")`` wrapper.  The ``embeddings``
    parameter signals real mode (an ``EmbeddingsProvider`` instance) vs mock
    mode (``None``, where ``InMemoryVectorIndex`` uses token-based similarity).

    Tenant isolation: when ``state["scope"]`` is a non-global ``AccessScope``,
    the caller's ``customer_id`` is passed as ``allowed_customer_ids`` so the
    index filters server-side (Qdrant payload filter), never post-filter.
    """

    def retrieve(state: State) -> State:
        scope = state.get("scope")
        match scope:
            case AccessScope(customer_id=customer_id) if customer_id is not None:
                allowed_customer_ids: set[int] | None = {customer_id}
            case _:
                allowed_customer_ids = None

        question = state["question"]
        started = time.perf_counter()
        with trace_span(
            "rag.adaptive.retrieve",
            {"rag.question_length": len(question)},
        ) as span:
            results = index.search(question, allowed_customer_ids, settings.default_top_k)
            documents = [payload.get("text", "") for _id, _score, payload in results]
            span.set_attribute("rag.source_count", len(results))
            span.set_attribute("rag.scope_is_global", allowed_customer_ids is None)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started))
            return {"documents": documents}

    return retrieve


def make_grade_documents(llm: ChatOpenAI, settings: Settings) -> NodeFn:
    """Create the ``grade_documents`` node: filters docs to relevant ones only."""

    def grade_documents(state: State) -> State:
        question = state["question"]
        docs = state["documents"]
        filtered: list[str] = []
        started = time.perf_counter()
        with trace_span(
            "rag.adaptive.grade_documents",
            {"rag.question_length": len(question), "rag.document_count": len(docs)},
        ) as span:
            for doc in docs:
                prompt = _GRADE_PROMPT.format(question=question, document=doc)
                result = cast(GradeDocuments, llm.with_structured_output(GradeDocuments).invoke(prompt))
                if result.score == "yes":
                    filtered.append(doc)
            span.set_attribute("rag.source_count", len(filtered))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started))
            return {"documents": filtered}

    return grade_documents


def make_transform_query(llm: ChatOpenAI, settings: Settings) -> NodeFn:
    """Create the ``transform_query`` node: rewrites question for better retrieval."""

    def transform_query(state: State) -> State:
        started = time.perf_counter()
        with trace_span(
            "rag.adaptive.transform_query",
            {"rag.question_length": len(state["question"])},
        ) as span:
            prompt = _REWRITE_PROMPT.format(question=state["question"])
            response = llm.invoke(prompt)
            rewritten = StrOutputParser().invoke(response)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started))
            return {"question": rewritten, "rewrite_count": state["rewrite_count"] + 1}

    return transform_query


def make_web_search(web_search: TavilyWebSearch, settings: Settings) -> NodeFn:
    """Create the ``web_search`` node: queries Tavily for public web results."""

    def web_search_node(state: State) -> State:
        started = time.perf_counter()
        with trace_span(
            "rag.adaptive.web_search",
            {"rag.question_length": len(state["question"])},
        ) as span:
            results = web_search.search(state["question"])
            span.set_attribute("rag.web_result_count", len(results))
            span.set_attribute("rag.duration_ms", _elapsed_ms(started))
            return {"web_results": results}

    return web_search_node


def make_generate(llm: ChatOpenAI, settings: Settings) -> NodeFn:
    """Create the ``generate`` node: produces an answer from context."""

    def generate(state: State) -> State:
        documents = state["documents"]
        match documents:
            case [*_]:
                context = "\n".join(documents)
            case []:
                context = "\n".join(r.snippet for r in state.get("web_results", []))
        prompt = _RAG_PROMPT.format(question=state["question"], context=context)
        started = time.perf_counter()
        with trace_span(
            "rag.adaptive.generate",
            {"rag.question_length": len(state["question"]), "rag.source_count": len(documents)},
        ) as span:
            response = llm.invoke(prompt)
            answer = StrOutputParser().invoke(response)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started))
            return {
                "generation": answer,
                "regenerate_count": state.get("regenerate_count", 0) + 1,
            }

    return generate


def make_hallucination_check(llm: ChatOpenAI, settings: Settings) -> NodeFn:
    """Create the ``hallucination_check`` node: grades groundedness and usefulness."""

    def hallucination_check(state: State) -> State:
        started = time.perf_counter()
        with trace_span(
            "rag.adaptive.hallucination_check",
            {"rag.question_length": len(state["question"])},
        ) as span:
            documents_text = "\n".join(state["documents"])
            groundedness_prompt = _GROUNDEDNESS_PROMPT.format(
                documents=documents_text,
                generation=state["generation"],
            )
            grounded_result = cast(
                GradeHallucinations,
                llm.with_structured_output(GradeHallucinations).invoke(groundedness_prompt),
            )

            match grounded_result.score:
                case "yes":
                    hallucination_verdict = "grounded"
                    answer_prompt = _ANSWER_PROMPT.format(
                        question=state["question"],
                        generation=state["generation"],
                    )
                    answer_result = cast(
                        GradeAnswer,
                        llm.with_structured_output(GradeAnswer).invoke(answer_prompt),
                    )
                    answer_verdict = "useful" if answer_result.score == "yes" else "not_useful"
                case _:
                    hallucination_verdict = "hallucinated"
                    answer_verdict = "not_useful"

            span.set_attribute("rag.hallucination_verdict", hallucination_verdict)
            span.set_attribute("rag.answer_verdict", answer_verdict)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started))
            return {
                "hallucination_verdict": hallucination_verdict,
                "answer_verdict": answer_verdict,
            }

    return hallucination_check
