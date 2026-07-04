"""Adaptive RAG module implementing the canonical 7-node self-correcting RAG topology.

Exposes the graph state schema, graph builder, and demo entry point.
"""

from __future__ import annotations

from app.adaptive_rag.graph import build_graph
from app.adaptive_rag.state import AdaptiveRAGState

__all__ = ["AdaptiveRAGState", "build_graph", "run_demo"]


def run_demo(*args: object, **kwargs: object) -> None:
    """Lazy wrapper — avoids circular import when running ``-m app.adaptive_rag.demo``."""
    from app.adaptive_rag.demo import main  # noqa: PLC0415
    main(*args, **kwargs)  # type: ignore[arg-type]
