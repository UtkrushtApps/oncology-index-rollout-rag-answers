from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any
import psycopg
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from app.config import get_settings
from app.embeddings import embed_texts

DATA_PATH = Path('/root/task/data/corpus.jsonl')


def load_corpus() -> list[dict[str, Any]]:
    with DATA_PATH.open('r', encoding='utf-8') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    settings = get_settings()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def ingest() -> int:
    settings = get_settings()
    rows = load_corpus()
    vectors = embed_texts([r['text'] for r in rows])
    client = QdrantClient(url=settings.qdrant_url)
    ensure_collection(client, len(vectors[0]))
    points = []
    with psycopg.connect(settings.database_url) as conn:
        for row, vector in zip(rows, vectors):
            payload = dict(row)
            payload['embedding_version'] = settings.embedding_version
            payload['manifest_id'] = row.get('manifest_id', settings.active_manifest_id)
            points.append(qm.PointStruct(id=_point_id(row['chunk_id']), vector=vector, payload=payload))
            digest = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
            conn.execute(
                """
                INSERT INTO chunk_lineage(chunk_id, document_id, manifest_id, section, char_start, char_end, content_sha256)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                  manifest_id=EXCLUDED.manifest_id,
                  char_start=EXCLUDED.char_start,
                  char_end=EXCLUDED.char_end,
                  content_sha256=EXCLUDED.content_sha256
                """,
                (row['chunk_id'], row['document_id'], payload['manifest_id'], row['section'], row['char_start'], row['char_end'], digest),
            )
        conn.commit()
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


if __name__ == '__main__':
    print(ingest())
