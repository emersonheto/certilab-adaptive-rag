"""Unit tests for ingest.py helper functions."""

from __future__ import annotations

from datetime import date

from app.adaptive_rag.ingest import (
    CertMeta,
    _build_metadata_chunk,
    _split_oversized_chunk,
)


def test_build_metadata_chunk_includes_all_fields() -> None:
    """Metadata chunk text contains certificate code, customer, status, type, and dates."""
    meta = CertMeta(
        id=1,
        code="T-043.26-1",
        customer_id=3,
        customer_name="ALERTA TECNICA IMPORT EIRL",
        issue_date=date(2026, 5, 23),
        pdf_path="certificates/x.pdf",
        status="Firmado",
        document_type="Acreditado",
        service_date=date(2026, 5, 16),
        request_number="SST-025.26-1",
    )
    chunk = _build_metadata_chunk(meta)
    text = chunk["text"]
    assert "T-043.26-1" in text
    assert "ALERTA TECNICA" in text
    assert "mayo 2026" in text
    assert "Firmado" in text
    assert "Acreditado" in text
    assert "SST-025.26-1" in text
    assert chunk["chunk_type"] == "metadata"
    assert chunk["customer_id"] == 3


def test_build_metadata_chunk_minimal() -> None:
    """Metadata chunk works with only required fields (no optional)."""
    meta = CertMeta(
        id=2,
        code="P-001.26-1",
        customer_id=5,
        customer_name="TEST CO",
        issue_date=date(2026, 1, 15),
        pdf_path="certificates/y.pdf",
    )
    chunk = _build_metadata_chunk(meta)
    text = chunk["text"]
    assert "P-001.26-1" in text
    assert "TEST CO" in text
    assert "enero 2026" in text
    assert chunk["chunk_type"] == "metadata"


def test_build_metadata_chunk_spanish_months() -> None:
    """All 12 months are correctly translated to Spanish."""
    expected = {
        1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
        5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
        9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
    }
    for month, name in expected.items():
        meta = CertMeta(
            id=month, code="T-X", customer_id=1, customer_name="C",
            issue_date=date(2026, month, 1), pdf_path="x.pdf",
        )
        chunk = _build_metadata_chunk(meta)
        assert name in chunk["text"], f"Month {month}: expected '{name}' in '{chunk['text']}'"


def test_split_oversized_chunk_small_text_unchanged() -> None:
    """Text under the threshold is returned as-is."""
    text = "Short text\nwith two lines"
    result = _split_oversized_chunk(text)
    assert len(result) == 1
    assert result[0] == text


def test_split_oversized_chunk_table_rows() -> None:
    """Table rows are split into sub-chunks with header preserved."""
    header = "TABLA: Datos de calibración\n"
    rows = "\n".join(f"| {i:03d} | data {i} |" for i in range(200))
    text = header + rows

    result = _split_oversized_chunk(text)
    # 200 rows should produce multiple sub-chunks (~1500 chars each)
    assert len(result) > 1
    # Each sub-chunk should contain the header
    for sub in result:
        assert "TABLA: Datos" in sub


def test_split_oversized_chunk_empty() -> None:
    """Single-line or empty text returns truncated result."""
    result = _split_oversized_chunk("A")
    assert len(result) == 1
