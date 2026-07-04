"""Ingest certificate PDFs from S3 into the Qdrant vector index.

Pipeline
--------
1. Stream each PDF from S3 into memory (no disk writes except for Camelot
   and Unstructured, which need temporary files).
2. Extract clean text blocks with PyMuPDF, structured tables with Camelot,
   and large images (>50 KB) with PyMuPDF.
3. Run semantic chunking with Unstructured (title-based, table-aware).
4. Enrich every chunk with certificate/customer metadata for tenant isolation.
5. Embed with OpenAI and upsert into Qdrant using deterministic content hashes.

The script is real-mode only: it requires ``APP_MODE=real`` plus working
MySQL, S3, Qdrant, and OpenAI credentials.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any

import fitz

from app.config import Settings
from app.logging import get_logger

logger = get_logger("adaptive_rag.ingest")

# Graphs smaller than this threshold are treated as logos/artifacts and ignored.
_GRAPH_SIZE_THRESHOLD = 50_000

# PyMuPDF text block type for actual text (type 1 is image blocks).
_FITZ_TEXT_BLOCK_TYPE = 0


@dataclass(frozen=True)
class CertMeta:
    """Read-only certificate metadata required for chunk enrichment."""

    id: int
    code: str
    customer_id: int
    customer_name: str
    issue_date: date
    pdf_path: str


@dataclass(frozen=True)
class IngestResult:
    """Per-certificate ingestion statistics."""

    chunks_created: int
    pages: int
    tables: int
    graphs: int


def main() -> None:
    """CLI entry point: load certs from MySQL, extract, embed, and index."""

    settings = Settings()
    if settings.app_mode != "real":
        print("ERROR: ingest requires APP_MODE=real", file=sys.stderr)
        sys.exit(1)

    from app.retrieval.qdrant_index import QdrantVectorIndex
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider
    from app.tools.mysql_connector import MySQLCertificateConnector, MySQLConnectorConfig
    from app.tools.s3_loader import S3LoaderConfig, S3PdfTextLoader

    configure_logging(settings.log_level, json=settings.log_json)

    mysql_config = MySQLConnectorConfig.from_settings(settings)
    connector = MySQLCertificateConnector(mysql_config)
    customers = _load_customers(connector)
    certificates = _load_certificates(connector)

    if not certificates:
        print("No certificates found to ingest.")
        return

    s3_loader = S3PdfTextLoader(S3LoaderConfig.from_settings(settings))
    embedding_provider = EmbeddingsProvider(EmbeddingProviderConfig.from_settings(settings))
    index = QdrantVectorIndex.from_settings(settings, embedding_provider)

    total = len(certificates)
    total_chunks = 0
    stats_list: list[IngestResult] = []

    print(f"Progress: {total} certificates to ingest")
    for current, cert in enumerate(certificates, start=1):
        customer_name = customers.get(cert.customer_id, "Unknown")
        cert_meta = CertMeta(
            id=cert.id,
            code=cert.code,
            customer_id=cert.customer_id,
            customer_name=customer_name,
            issue_date=cert.emitted_at,
            pdf_path=cert.pdf_path,
        )

        try:
            pdf_bytes = s3_loader.fetch_pdf_bytes(cert.pdf_path)
        except Exception as exc:
            logger.warning("ingest.s3_download_failed", error=type(exc).__name__)
            print(f"[{current}/{total}] {cert.code} ({customer_name}) — S3 download failed: {exc}")
            continue

        try:
            result = _process_certificate(
                pdf_bytes=pdf_bytes,
                cert_meta=cert_meta,
                embedding_provider=embedding_provider,
                index=index,
                current=current,
                total=total,
            )
        except Exception as exc:
            logger.exception("ingest.certificate_failed", error=type(exc).__name__)
            print(f"[{current}/{total}] {cert.code} ({customer_name}) — processing failed: {exc}")
            continue

        stats_list.append(result)
        total_chunks += result.chunks_created

    total_pages = sum(s.pages for s in stats_list)
    total_tables = sum(s.tables for s in stats_list)
    total_graphs = sum(s.graphs for s in stats_list)

    print(
        f"Done: {len(stats_list)}/{total} certificates → "
        f"{total_chunks} chunks indexed "
        f"({total_pages} pages, {total_tables} tables, {total_graphs} graphs)"
    )


def _process_certificate(
    pdf_bytes: bytes,
    cert_meta: CertMeta,
    embedding_provider: Any,
    index: Any,
    current: int,
    total: int,
) -> IngestResult:
    """Extract, chunk, embed, and index a single certificate PDF.

    Returns an :class:`IngestResult` with the number of chunks actually indexed
    (after idempotent deduplication) plus per-certificate counts for progress
    reporting.
    """

    doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    pages = len(doc)

    text_blocks = _extract_text_fitz(doc)
    tables = _extract_tables_camelot(pdf_bytes, text_blocks)
    graphs = _extract_graphs_fitz(doc)
    semantic_chunks = _chunk_with_unstructured(pdf_bytes)

    chunks = _build_chunks(
        text_blocks=text_blocks,
        tables=tables,
        graphs=graphs,
        cert_meta=cert_meta,
        semantic_chunks=semantic_chunks,
    )
    chunks_created = _embed_and_index(chunks, embedding_provider, index)

    print(
        f"[{current}/{total}] {cert_meta.code} ({cert_meta.customer_name}) — "
        f"{pages} pages, {len(tables)} tables, {len(graphs)} graphs → "
        f"{chunks_created} chunks"
    )

    return IngestResult(
        chunks_created=chunks_created,
        pages=pages,
        tables=len(tables),
        graphs=len(graphs),
    )


def _extract_text_fitz(doc: fitz.Document) -> list[dict[str, Any]]:
    """Extract clean text blocks from a PyMuPDF document, preserving positions.

    Only text blocks (``type == 0``) are kept; image blocks are skipped here
    because they are handled separately by :func:`_extract_graphs_fitz`.
    """

    text_blocks: list[dict[str, Any]] = []
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _block_no, block_type = block
            if block_type != _FITZ_TEXT_BLOCK_TYPE:
                continue
            cleaned = " ".join(text.split())
            if not cleaned:
                continue
            text_blocks.append(
                {
                    "page": page_num,
                    "text": cleaned,
                    "bbox": (x0, y0, x1, y1),
                }
            )
    return text_blocks


def _extract_tables_camelot(
    pdf_bytes: bytes,
    text_blocks: list[dict[str, Any]],
) -> list[str]:
    """Extract tables from *pdf_bytes* as formatted Markdown text.

    A temporary file is created because Camelot requires a filesystem path.
    The file is deleted immediately after parsing. Each table is prefixed with
    an inferred title taken from the text block immediately above the table on
    the same page.
    """

    try:
        import camelot
    except ImportError as exc:
        logger.warning("ingest.camelot_unavailable", error=str(exc))
        return []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        tables = camelot.read_pdf(tmp_path, pages="all")
    except Exception as exc:
        logger.warning("ingest.camelot_read_failed", error=type(exc).__name__)
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    formatted: list[str] = []
    for table in tables:
        title = _infer_table_title(table, text_blocks)
        try:
            markdown = table.df.to_markdown(index=False)
        except Exception:
            markdown = table.df.to_string(index=False)
        formatted.append(f"TABLA: {title}\n{markdown}")

    return formatted


def _infer_table_title(table: Any, text_blocks: list[dict[str, Any]]) -> str:
    """Best-effort caption lookup for a Camelot table.

    Searches *text_blocks* on the same page for the nearest text block above
    the table's bounding box. Falls back to a generic label.
    """

    table_page = getattr(table, "page", None)
    bbox = getattr(table, "_bbox", None)
    if bbox is None or table_page is None:
        return "Datos del certificado"

    tx0, ty0, _tx1, _ty1 = bbox
    candidates = [
        block
        for block in text_blocks
        if block["page"] == table_page and block["bbox"][3] <= ty0
    ]
    if not candidates:
        return "Datos del certificado"

    candidates.sort(key=lambda b: b["bbox"][3], reverse=True)
    return candidates[0]["text"]


def _extract_graphs_fitz(doc: fitz.Document) -> list[dict[str, Any]]:
    """Extract large images from *doc* that are likely graphs/diagrams.

    Images smaller than :const:`_GRAPH_SIZE_THRESHOLD` are discarded as logos
    or decorative artifacts. Each returned graph includes page number, image
    index, byte size, and a base64 reference that can be stored in metadata.
    """

    graphs: list[dict[str, Any]] = []
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            extracted = doc.extract_image(xref)
            image_bytes = extracted.get("image", b"")
            if len(image_bytes) <= _GRAPH_SIZE_THRESHOLD:
                continue
            graphs.append(
                {
                    "page": page_num,
                    "index": img_index,
                    "xref": xref,
                    "size": len(image_bytes),
                    "ext": extracted.get("ext", "png"),
                    "image_b64": base64.b64encode(image_bytes).decode("ascii"),
                }
            )
    return graphs


def _chunk_with_unstructured(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Run semantic, title-based chunking on *pdf_bytes* using Unstructured.

    Tables are kept intact (``isolate_table=True``) and split tables repeat
    their headers (``repeat_table_headers=True``). The function returns a list
    of chunk dictionaries with text, page, and inferred chunk type.
    """

    try:
        from unstructured.chunking.title import chunk_by_title
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        logger.warning("ingest.unstructured_unavailable", error=str(exc))
        return []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        elements = partition_pdf(
            filename=tmp_path,
            strategy="auto",
            include_page_breaks=True,
        )
        chunks = chunk_by_title(
            elements,
            max_characters=800,
            new_after_n_chars=600,
            overlap=100,
            isolate_table=True,
            repeat_table_headers=True,
        )
    except Exception as exc:
        logger.warning("ingest.unstructured_partition_failed", error=type(exc).__name__)
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    semantic_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        text = getattr(chunk, "text", "")
        if not text:
            continue
        metadata = getattr(chunk, "metadata", None) or {}
        page = getattr(metadata, "page_number", None)
        category = getattr(chunk, "category", "")
        chunk_type = _classify_unstructured_chunk(str(category))
        semantic_chunks.append(
            {
                "page": page,
                "text": " ".join(text.split()),
                "chunk_type": chunk_type,
            }
        )
    return semantic_chunks


def _classify_unstructured_chunk(category: str) -> str:
    """Map an Unstructured element category to a canonical chunk type."""

    match category.lower():
        case "table":
            return "table"
        case "compositeelement" | "composite":
            return "composite"
        case _:
            return "text"


def _build_chunks(
    text_blocks: list[dict[str, Any]],
    tables: list[str],
    graphs: list[dict[str, Any]],
    cert_meta: CertMeta,
    semantic_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble final, metadata-enriched chunk objects ready for embedding.

    The output combines semantic text chunks (Unstructured), structured tables
    (Camelot), and graph references (PyMuPDF). Every chunk carries the
    certificate and customer metadata required for tenant-isolated retrieval.
    """

    chunks: list[dict[str, Any]] = []

    # Prefer semantic chunks when Unstructured succeeded; otherwise fall back
    # to flat PyMuPDF text blocks grouped by page.
    source_chunks = semantic_chunks if semantic_chunks else _blocks_to_chunks(text_blocks)
    for source_chunk in source_chunks:
        page = source_chunk.get("page")
        if page is None:
            page = _estimate_page_for_text(source_chunk["text"], text_blocks)
        chunks.append(
            _enrich_chunk(
                text=source_chunk["text"],
                chunk_type=source_chunk.get("chunk_type", "text"),
                cert_meta=cert_meta,
                page=page,
            )
        )

    # Each Camelot table becomes a self-contained chunk.
    for table_text in tables:
        title, parameter = _parse_table_header(table_text)
        chunks.append(
            _enrich_chunk(
                text=table_text,
                chunk_type="table",
                cert_meta=cert_meta,
                page=_estimate_page_for_text(table_text, text_blocks),
                parameter=parameter,
            )
        )

    # Large images are represented as graph chunks with a descriptive note.
    for graph in graphs:
        page = graph.get("page", 1)
        note = (
            f"[GRÁFICO: imagen de página {page} — "
            f"tamaño {graph['size']} bytes, referencia {graph['index']}]"
        )
        chunks.append(
            _enrich_chunk(
                text=note,
                chunk_type="graph",
                cert_meta=cert_meta,
                page=page,
                image_reference=graph,
            )
        )

    return chunks


def _blocks_to_chunks(text_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert PyMuPDF text blocks into simple text chunks grouped by page."""

    chunks: list[dict[str, Any]] = []
    for block in text_blocks:
        chunks.append(
            {
                "page": block["page"],
                "text": block["text"],
                "chunk_type": "text",
            }
        )
    return chunks


def _estimate_page_for_text(text: str, text_blocks: list[dict[str, Any]]) -> int:
    """Return the most likely page for *text* based on substring overlap."""

    if not text_blocks:
        return 1
    for block in text_blocks:
        snippet = text[:120]
        if snippet and snippet in block["text"] or block["text"][:120] in text:
            return block["page"]
    return text_blocks[0]["page"]


def _parse_table_header(table_text: str) -> tuple[str, str | None]:
    """Parse the ``TABLA: <title>`` header and try to extract a parameter."""

    lines = table_text.splitlines()
    if not lines or not lines[0].startswith("TABLA:"):
        return "Datos del certificado", None

    title = lines[0].replace("TABLA:", "").strip()
    # Heuristic: if the title contains "parámetro" or a temperature/equipment
    # pattern, treat the title itself as the parameter.
    lower = title.lower()
    if "parámetro" in lower or "°c" in lower or "°f" in lower:
        return title, title
    return title, None


def _enrich_chunk(
    text: str,
    chunk_type: str,
    cert_meta: CertMeta,
    page: int | None,
    parameter: str | None = None,
    image_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metadata-rich chunk payload for Qdrant."""

    payload: dict[str, Any] = {
        "certificate_id": cert_meta.id,
        "certificate_code": cert_meta.code,
        "customer_id": cert_meta.customer_id,
        "customer_name": cert_meta.customer_name,
        "issue_date": cert_meta.issue_date.isoformat(),
        "chunk_type": chunk_type,
        "source_type": "pdf_certificate",
        "page": page if page is not None else 1,
        "text": text,
    }
    if parameter:
        payload["parameter"] = parameter
    if image_reference:
        payload["image_reference"] = {
            "page": image_reference.get("page"),
            "index": image_reference.get("index"),
            "size": image_reference.get("size"),
            "ext": image_reference.get("ext"),
        }
    return payload


def _embed_and_index(
    chunks: list[dict[str, Any]],
    embedding_provider: Any,
    index: Any,
) -> int:
    """Embed *chunks* and upsert them into *index*.

    Chunk IDs are deterministic MD5 hashes of the chunk text, making the
    operation idempotent across runs. Already-existing IDs are skipped by the
    index's ``upsert_points`` method.
    """

    if not chunks:
        return 0

    from qdrant_client.models import PointStruct

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_provider.embed_batch(texts)

    points: list[Any] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        doc_id = str(uuid.UUID(hashlib.md5(chunk["text"].encode()).hexdigest()))
        chunk["chunk_id"] = doc_id
        points.append(PointStruct(id=doc_id, vector=vector, payload=chunk))

    new_count, _skipped = index.upsert_points(points)
    return new_count


def _load_customers(connector: Any) -> dict[int, str]:
    """Return a mapping ``customer_id -> company_name`` from MySQL."""

    rows = connector.fetch_customers()
    return {int(row["id"]): str(row.get("company_name", f"Customer-{row['id']}")) for row in rows}


def _load_certificates(connector: Any) -> list[Any]:
    """Load certificates from MySQL, excluding test codes and missing PDFs."""

    from app.domain.models import Certificate

    rows = connector.fetch_certificates()
    certificates: list[Certificate] = []
    for row in rows:
        code = str(row.get("code", ""))
        pdf_path = str(row.get("pdf_document_path", ""))
        if code.lower() == "test" or not pdf_path:
            continue
        certificates.append(
            Certificate(
                id=int(row["id"]),
                code=code,
                customer_id=int(row["customer_id"]),
                status=str(row.get("status", "")),
                emitted_at=_parse_issue_date(row.get("issue_date")),
                technician_id=int(row.get("user_id", 0) or 0),
                equipment="",
                pdf_path=pdf_path,
                document_type=str(row.get("document_type")) if row.get("document_type") else None,
                case_number=None,
                user_id=int(row.get("user_id")) if row.get("user_id") else None,
                qr_code=str(row.get("qr_code")) if row.get("qr_code") else None,
                request_number=str(row.get("request_number")) if row.get("request_number") else None,
                service_date=_parse_optional_date(row.get("service_date")),
            )
        )
    return certificates


def _parse_issue_date(value: object) -> date:
    """Parse a date value from MySQL into a :class:`date`."""

    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError(f"Cannot parse issue_date from {type(value).__name__}: {value!r}")


def _parse_optional_date(value: object) -> date | None:
    """Parse an optional date value from MySQL."""

    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    return None


def configure_logging(level: str, *, json: bool) -> None:
    """Configure application logging for the ingest CLI."""

    from app.logging import configure_logging as _configure_logging

    _configure_logging(level, json=json)


if __name__ == "__main__":
    main()
