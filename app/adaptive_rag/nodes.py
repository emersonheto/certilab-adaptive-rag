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
from functools import cache
from typing import Any, cast

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


@cache
def _get_llm(settings: Settings) -> Any:
    """Lazy-import and cache a ``ChatOpenAI`` instance (certilab-rag-patterns rule 1)."""
    from langchain_openai import ChatOpenAI  # noqa: PLC0415
    return ChatOpenAI(model=settings.openai_chat_model, temperature=settings.openai_temperature)


def _get_parser() -> Any:
    """Lazy-import ``StrOutputParser`` (used by transform_query node)."""
    from langchain_core.output_parsers import StrOutputParser  # noqa: PLC0415
    return StrOutputParser()


# --- Inline prompts (NOT hub.pull — avoids langchain-community dependency) ---

_ROUTE_PROMPT = """<role>
You are a routing classifier for a RAG system serving Certilab, a calibration laboratory.
</role>

<task>
Classify the user question. Return "vectorstore" if it is about any of these topics:
- Calibration certificates, procedures, or standards (e.g. INDECOPI, ISO)
- Customers or companies (e.g. "ALS PERU", "cliente 101")
- Technical measurements (temperature °C, humidity %, uncertainty, pressure)
- Equipment, instruments, or laboratory data
- Certificate status (Pendiente/Firmado) or type (Acreditado/No acreditado)
- Dates or time periods related to certificates

Return "web_search" ONLY for general knowledge completely unrelated to Certilab
(e.g. news, weather, general science definitions).
</task>

<few_shot_examples>
Q: "¿105°C es una temperatura segura?"
A: vectorstore

Q: "¿Quién ganó las elecciones?"
A: web_search

Q: "¿Cuántos certificados pendientes hay?"
A: vectorstore
</few_shot_examples>

Question: {question}"""

_GRADE_PROMPT = """<role>
You are a document relevance grader for a calibration certificate retrieval system.
</role>

<task>
Determine whether the retrieved document is relevant to the user question.
Return "yes" if the document discusses the same topic, customer, certificate,
measurement, or equipment mentioned in the question — even if the exact words
differ. Return "no" only if the document is clearly unrelated.
</task>

<constraints>
- A document fragment from a Certilab certificate is ALWAYS relevant when the
  question asks about certificates, customers, or calibration data.
- Technical data (numbers, tables) IS relevant to measurement questions even
  if it doesn't repeat the question text.
- "No tengo información" is NOT a valid reason to grade as irrelevant.
</constraints>

<few_shot_examples>
Q: "¿Cuántos certificados tiene ALERTA TECNICA?"
Doc: "Certificado T-032.26-1 | Cliente: ALERTA TECNICA IMPORT EIRL | Fecha: 2026-04-21"
Grade: yes

Q: "¿Temperatura máxima a 105°C?"
Doc: "Parámetro 105°C ±10°C | Máxima temperatura medida: 107.1°C"
Grade: yes

Q: "¿Certificados de mayo 2026?"
Doc: "The weather in Lima was sunny"
Grade: no
</few_shot_examples>

Retrieved document: {document}

User question: {question}"""

_REWRITE_PROMPT = """<role>
You are a query rewriter for a calibration laboratory RAG system.
</role>

<task>
Rewrite the user question to improve vector search retrieval.
Make the question more specific and precise while preserving the original intent.
</task>

<constraints>
- Keep domain context: if the question is about Certilab certificates,
  customers, or measurements, preserve that framing.
- Add specificity: include relevant details (customer name, date range,
  measurement type) extracted from the original question.
- Do NOT change the core question — only make it clearer for retrieval.
- Output ONLY the rewritten question, no explanation.
</constraints>

<few_shot_examples>
Original: "Dame info de calibración"
Rewritten: "¿Qué información hay sobre procedimientos de calibración y certificados emitidos?"

Original: "temperatura"
Rewritten: "¿Qué mediciones de temperatura se registraron en los certificados de calibración?"
</few_shot_examples>

Initial question: {question}"""

_RAG_PROMPT = """<role>
You are a helpful assistant that answers questions about calibration certificates
using ONLY the provided context. You serve Certilab, a calibration laboratory.
</role>

<task>
Answer the user question based on the retrieved context below.
If the context does not contain the answer, say "No tengo suficiente información para responder."
Use 1-3 sentences. Be factual and concise.
</task>

<constraints>
- Only use information present in the context.
- Do NOT invent certificate numbers, dates, or measurements.
- If the context contains certificate codes (e.g. T-043.26-1), include them.
</constraints>

Context:
{context}

Question: {question}
Answer:"""

_GROUNDEDNESS_PROMPT = """<role>
You are a factual accuracy grader for a calibration certificate QA system.
</role>

<task>
Determine whether the LLM answer is grounded in the provided facts.
Return "yes" if every claim in the answer is supported by the facts.
Return "no" if the answer makes claims NOT present in the facts.
</task>

<constraints>
- A claim is grounded if the facts contain the same information,
  even if phrased differently.
- "No tengo información" or "No sé" IS grounded when the facts
  indeed lack the requested data.
</constraints>

Facts:
{documents}

LLM answer:
{generation}"""

_ANSWER_PROMPT = """<role>
You are an answer quality grader for a calibration certificate QA system.
</role>

<task>
Determine whether the answer actually addresses the user question.
Return "yes" if the answer provides useful information or honestly
states it cannot answer. Return "no" only if the answer is off-topic
or provides irrelevant information.
</task>

<constraints>
- "No tengo información" IS a valid answer when the data is absent.
- An answer listing certificate codes IS useful for certificate queries.
</constraints>

Question: {question}

Answer: {generation}"""


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed milliseconds since *started_at* (perf_counter epoch)."""

    return int((time.perf_counter() - started_at) * 1000)


def make_route_question(llm: Any, settings: Settings) -> NodeFn:
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


def make_grade_documents(llm: Any, settings: Settings) -> NodeFn:
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


def make_transform_query(llm: Any, settings: Settings) -> NodeFn:
    """Create the ``transform_query`` node: rewrites question for better retrieval."""

    def transform_query(state: State) -> State:
        started = time.perf_counter()
        with trace_span(
            "rag.adaptive.transform_query",
            {"rag.question_length": len(state["question"])},
        ) as span:
            prompt = _REWRITE_PROMPT.format(question=state["question"])
            response = llm.invoke(prompt)
            rewritten = _get_parser().invoke(response)
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


def make_generate(llm: Any, settings: Settings) -> NodeFn:
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
            answer = _get_parser().invoke(response)
            span.set_attribute("rag.duration_ms", _elapsed_ms(started))
            return {
                "generation": answer,
                "regenerate_count": state.get("regenerate_count", 0) + 1,
            }

    return generate


def make_hallucination_check(llm: Any, settings: Settings) -> NodeFn:
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
