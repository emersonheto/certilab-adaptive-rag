#!/usr/bin/env python3
"""Restore a Qdrant collection from a gzipped JSONL backup.

Usage:
    python scripts/restore_qdrant.py <backup.jsonl.gz>

Each line in the backup is a JSON object with:
    {"id": "...", "vector": [0.1, 0.2, ...], "payload": {...}}
"""

from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Callable


def main(backup_path: str, qdrant_url: str = "http://localhost:6333", collection: str = "certilab-rag") -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(url=qdrant_url)

    # Recreate collection (fresh start)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    batch: list[PointStruct] = []
    total = 0

    with gzip.open(backup_path, "rt") as f:
        for line in f:
            p = json.loads(line)
            batch.append(PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]))

            if len(batch) >= 100:
                client.upsert(collection_name=collection, points=batch)
                total += len(batch)
                print(f"  {total} puntos...", flush=True)
                batch = []

    if batch:
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)

    print(f"✅ {total} puntos restaurados en '{collection}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/restore_qdrant.py <backup.jsonl.gz>")
        sys.exit(1)
    main(sys.argv[1])
