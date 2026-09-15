from __future__ import annotations

import psycopg
from qdrant_client import QdrantClient
from app.cache import redis_client
from app.config import get_settings
from app.ingestion import ingest


def bootstrap() -> None:
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        conn.execute('SELECT 1')
    QdrantClient(url=settings.qdrant_url).get_collections()
    redis_client().set('fixture:bootstrap', 'ready')
    count = ingest()
    print(f'loaded {count} protocol chunks')


if __name__ == '__main__':
    bootstrap()
