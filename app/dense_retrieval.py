from __future__ import annotations

import uuid
from datetime import date
from functools import lru_cache
from typing import Any

import psycopg
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings
from app.embeddings import embed_query
from app.models import Evidence, QueryScope, RetrievalResult


@lru_cache(maxsize=512)
def _site_active(trial_id: str, site_id: str, as_of: date) -> bool:
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute(
            """
            SELECT activated_on, closed_on
            FROM site_activation_windows
            WHERE trial_id=%s AND site_id=%s
            """,
            (trial_id, site_id),
        ).fetchone()
    if not row:
        return False
    activated_on, closed_on = row
    if as_of < activated_on:
        return False
    if closed_on is not None and as_of > closed_on:
        return False
    return True


@lru_cache(maxsize=512)
def _allowed_amendments(trial_id: str, as_of: date) -> frozenset[str]:
    """Return amendment_ids whose document provenance is effective for the requested as_of and not archived."""
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(
            """
            SELECT amendment_id
            FROM document_provenance
            WHERE trial_id=%s
              AND archived=false
              AND effective_from <= %s
              AND (effective_to IS NULL OR effective_to >= %s)
            """,
            (trial_id, as_of, as_of),
        ).fetchall()
    return frozenset(r[0] for r in rows)


def _enrich_evidence_from_postgres(settings: Any, chunk_id: str, current: Evidence) -> Evidence:
    """Fill missing metadata (document_id, amendment_id, section, char offsets, source_uri) for auditability."""
    # Only query when we are missing key provenance fields.
    needs = not current.document_id or not current.amendment_id or not current.section or current.char_end == 0
    if not needs:
        return current

    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute(
            """
            SELECT
              dp.trial_id,
              dp.amendment_id,
              cl.document_id,
              cl.section,
              cl.char_start,
              cl.char_end,
              dp.source_uri
            FROM chunk_lineage cl
            JOIN document_provenance dp ON dp.document_id = cl.document_id
            WHERE cl.chunk_id=%s AND cl.manifest_id=%s
            """,
            (chunk_id, settings.active_manifest_id),
        ).fetchone()

    if not row:
        return current

    trial_id, amendment_id, document_id, section, char_start, char_end, source_uri = row
    return Evidence(
        chunk_id=current.chunk_id,
        document_id=document_id,
        trial_id=trial_id,
        amendment_id=amendment_id,
        section=section,
        text=current.text,
        score=current.score,
        source_uri=source_uri,
        char_start=int(char_start),
        char_end=int(char_end),
        manifest_id=current.manifest_id or settings.active_manifest_id,
        embedding_version=current.embedding_version,
    )


def retrieve_dense(question: str, scope: QueryScope, limit: int = 8) -> RetrievalResult:
    settings = get_settings()

    if not _site_active(scope.trial_id, scope.site_id, scope.as_of):
        return RetrievalResult(
            query_id=str(uuid.uuid4()),
            evidence=[],
            diagnostics={'retriever': 'dense', 'site_active': False, 'scope_seen': scope.model_dump(mode='json')},
        )

    allowed = _allowed_amendments(scope.trial_id, scope.as_of)
    if not allowed:
        return RetrievalResult(
            query_id=str(uuid.uuid4()),
            evidence=[],
            diagnostics={'retriever': 'dense', 'site_active': True, 'allowed_amendments': [], 'scope_seen': scope.model_dump(mode='json')},
        )

    vector = embed_query(question)
    client = QdrantClient(url=settings.qdrant_url)

    # Prefer scoping inside Qdrant using stable metadata fields.
    qdrant_filter = qm.Filter(
        must=[
            qm.FieldCondition(key='trial_id', match=qm.MatchValue(value=scope.trial_id)),
            qm.FieldCondition(key='manifest_id', match=qm.MatchValue(value=settings.active_manifest_id)),
            qm.FieldCondition(key='amendment_id', match=qm.MatchAny(any=list(allowed))),
        ]
    )

    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=vector,
        limit=limit * 3,
        with_payload=True,
        query_filter=qdrant_filter,
    )

    # Ensure deterministic ordering.
    hits = sorted(hits, key=lambda h: (-float(h.score or 0.0), str((h.payload or {}).get('chunk_id', str(h.id)))))

    evidence: list[Evidence] = []
    for hit in hits:
        payload = hit.payload or {}
        chunk_id = payload.get('chunk_id', str(hit.id))
        ev = Evidence(
            chunk_id=str(chunk_id),
            document_id=str(payload.get('document_id', '')),
            trial_id=str(payload.get('trial_id', '')),
            amendment_id=str(payload.get('amendment_id', '')),
            section=str(payload.get('section', '')),
            text=str(payload.get('text', '')),
            score=float(hit.score or 0.0),
            source_uri=payload.get('source_uri'),
            char_start=int(payload.get('char_start', 0) or 0),
            char_end=int(payload.get('char_end', 0) or 0),
            manifest_id=payload.get('manifest_id'),
            embedding_version=payload.get('embedding_version'),
        )
        ev = _enrich_evidence_from_postgres(settings, ev.chunk_id, ev)

        # Fail-closed: drop any evidence not eligible for this exact scope.
        if ev.trial_id != scope.trial_id:
            continue
        if ev.amendment_id not in allowed:
            continue
        evidence.append(ev)
        if len(evidence) >= limit:
            break

    return RetrievalResult(
        query_id=str(uuid.uuid4()),
        evidence=evidence,
        diagnostics={'retriever': 'dense', 'scope_seen': scope.model_dump(mode='json'), 'allowed_amendments': sorted(list(allowed))},
    )
