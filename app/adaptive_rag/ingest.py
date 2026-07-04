"""Ingest PDFs from S3, embed and index in Qdrant. Real-mode only."""
from __future__ import annotations
import sys
from app.config import Settings

def main() -> None:
    settings = Settings()
    if settings.app_mode != "real":
        print("ERROR: ingest requires APP_MODE=real")
        sys.exit(1)
    from app.ingestion.mysql_loader import MySQLLoader
    from app.ingestion.splitter import build_metadata_chunks, build_s3_pdf_chunks
    from app.ingestion.pdf_extractor import S3PdfTextExtractor
    from app.retrieval.qdrant_index import QdrantVectorIndex
    from app.tools.embeddings import EmbeddingProviderConfig, EmbeddingsProvider
    from app.tools.mysql_connector import MySQLCertificateConnector, MySQLConnectorConfig

    connector = MySQLCertificateConnector(MySQLConnectorConfig.from_settings(settings))
    loader = MySQLLoader(connector)
    _, certificates, _ = loader.load()
    print(f"Loaded {len(certificates)} certificates from MySQL")

    metadata_chunks = build_metadata_chunks(certificates)
    extractor = S3PdfTextExtractor(settings)
    pdf_texts: dict[int, str] = {}
    for cert in certificates:
        if cert.pdf_path:
            try:
                pdf_texts[cert.id] = extractor.extract_text(cert.pdf_path)
            except Exception as e:
                print(f"  Failed to extract {cert.pdf_path}: {e}")

    pdf_chunks = build_s3_pdf_chunks(certificates, pdf_texts)
    chunks = [*metadata_chunks, *pdf_chunks]
    print(f"Built {len(chunks)} chunks ({len(pdf_chunks)} from PDFs)")

    embedding_provider = EmbeddingsProvider(EmbeddingProviderConfig.from_settings(settings))
    index = QdrantVectorIndex.from_settings(settings, embedding_provider)
    index.upsert(chunks)
    print(f"Indexed {len(chunks)} chunks into Qdrant")

if __name__ == "__main__":
    main()
