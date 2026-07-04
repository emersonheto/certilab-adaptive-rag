"""CLI entry point for the Adaptive RAG demo.

Runnable as::

    python -m app.adaptive_rag.demo "Tu pregunta aqui"

Validates the ``OPENAI_API_KEY``, then branches on ``APP_MODE``:

- ``mock`` (default): loads JSON fixtures into an ``InMemoryVectorIndex``.
- ``real``: connects to MySQL + Qdrant using the real ``EmbeddingsProvider``
  and ``QdrantVectorIndex`` (the same components used by the production
  pipeline factory).

In both modes the 7-node Adaptive RAG graph is built and streamed
node-by-node to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.adaptive_rag.graph import build_graph
from app.adaptive_rag.state import AdaptiveRAGState
from app.config import Settings
from app.domain.models import Role
from app.ingestion.indexer import InMemoryVectorIndex
from app.ingestion.loader import (
    load_certificates,
    load_customers,
    load_histories,
    load_pdf_texts,
)
from app.ingestion.splitter import build_pdf_chunks
from app.security.access_control import Principal, scope_from_principal
from app.tools.web_search import TavilyWebSearch, WebSearchConfig

DEFAULT_QUESTION = "¿Cuantos certificados vigentes tiene el cliente 101?"

_SEP = "=" * 60


def _build_mock_components(settings: Settings) -> tuple[InMemoryVectorIndex, None, TavilyWebSearch]:
    """Load mock data fixtures and build the graph dependencies.

    Returns ``(index, embeddings, web_search)``.  ``embeddings`` is ``None``
    because the ``InMemoryVectorIndex`` embeds the query internally via token
    cosine similarity; it is passed to ``build_graph`` only for API symmetry.
    """

    data_dir = Path("data")
    certificates = load_certificates(data_dir)
    load_customers(data_dir)
    load_histories(data_dir)
    pdf_texts = load_pdf_texts(data_dir)
    chunks = build_pdf_chunks(certificates, pdf_texts)
    index = InMemoryVectorIndex(chunks=chunks)

    web_search = TavilyWebSearch(WebSearchConfig(tavily_api_key=settings.tavily_api_key))

    return index, None, web_search


def _build_real_components(settings: Settings) -> tuple[Any, Any, TavilyWebSearch]:
    """Build real-stack dependencies: EmbeddingsProvider, Qdrant, web search.

    Uses ``importlib`` for lazy module loading so mock mode never requires
    qdrant-client, and so the import chain does not pull optional real-mode
    dependencies into the type-checker graph at analysis time.

    The Qdrant index is expected to already contain data from a prior
    ``reindex()`` — this function does NOT re-ingest.

    Returns ``(index, embedding_provider, web_search)``.
    """

    import importlib

    qdrant_module = importlib.import_module("app.retrieval.qdrant_index")
    embeddings_module = importlib.import_module("app.tools.embeddings")

    embedding_provider = embeddings_module.EmbeddingsProvider(
        embeddings_module.EmbeddingProviderConfig.from_settings(settings)
    )
    index = qdrant_module.QdrantVectorIndex.from_settings(settings, embedding_provider)

    web_search = TavilyWebSearch(WebSearchConfig(tavily_api_key=settings.tavily_api_key))

    return index, embedding_provider, web_search


def _run_stream(graph: Any, initial_state: AdaptiveRAGState) -> str:
    """Stream the graph, printing each node and its state delta.

    Returns the final ``generation`` text (or a placeholder if absent).
    """

    generation = "(no output)"
    for step in graph.stream(initial_state):
        for node_name, node_output in step.items():
            print(f"--- Node: {node_name} ---")
            if isinstance(node_output, dict):
                for key, value in node_output.items():
                    preview = repr(value)
                    if len(preview) > 120:
                        preview = preview[:117] + "..."
                    print(f"  {key}: {preview}")
                gen = node_output.get("generation")
                if gen:
                    generation = gen
            print()
    return generation


def main(argv: list[str] | None = None) -> None:
    """Run the Adaptive RAG demo CLI.

    Loads ``.env`` from the project root (if present) before reading settings,
    so API keys configured there are picked up automatically.

    Args:
        argv: Optional CLI arguments. ``argv[0]`` is the question string.
            If ``None``, reads from ``sys.argv``. Defaults to
            :data:`DEFAULT_QUESTION` when no argument is provided.
    """

    # Load .env from the project root so keys defined there reach Settings()
    # even when they haven't been exported to the shell environment.
    _project_root_env = Path(__file__).parents[2] / ".env"
    if _project_root_env.exists():
        load_dotenv(_project_root_env, override=False)

    settings = Settings()

    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY no está configurada.")
        print("Definila en .env o exportala antes de correr el demo:")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if argv is None:
        argv = sys.argv[1:]
    question = argv[0] if argv else DEFAULT_QUESTION

    match settings.app_mode:
        case "real":
            index, embeddings, web_search = _build_real_components(settings)
        case _:
            index, embeddings, web_search = _build_mock_components(settings)

    graph = build_graph(index=index, embeddings=embeddings, web_search=web_search, settings=settings)

    principal = Principal(role=Role.ADMIN, customer_id=None, user_id=1)
    scope = scope_from_principal(principal)

    initial_state: AdaptiveRAGState = {
        "question": question,
        "generation": "",
        "documents": [],
        "web_results": [],
        "route": "",
        "rewrite_count": 0,
        "regenerate_count": 0,
        "hallucination_verdict": "",
        "answer_verdict": "",
        "principal": principal,
        "scope": scope,
    }

    print(f"\n{_SEP}")
    print("Certilab Adaptive RAG - Demo CLI")
    print(f"Mode: {settings.app_mode}")
    print(f"Question: {question}")
    print(f"Web search: {'enabled' if settings.tavily_api_key else 'disabled (no TAVILY_API_KEY)'}")
    print(f"{_SEP}\n")

    try:
        generation = _run_stream(graph, initial_state)
        print(f"{_SEP}")
        print(f"Final Generation:\n{generation}")
        print(f"{_SEP}\n")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
