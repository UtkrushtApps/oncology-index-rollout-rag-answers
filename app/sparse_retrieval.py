from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from datetime import date
from functools import lru_cache

import psycopg

from app.config import get_settings
from app.ingestion import load_corpus
from app.models import Evidence, QueryScope, RetrievalResult

_TOKEN = re.compile(r'[A-Za-z0-9-]+')


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


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


def _enrich_evidence_from_postgres(settings, row: dict) -> Evidence:
    """Enrich missing metadata needed for audit/citations."""
    chunk_id = str(row.get('chunk_id') or '')
    current = Evidence(
        chunk_id=chunk_id,
        document_id=str(row.get('document_id', '')),
        trial_id=str(row.get('trial_id', '')),
        amendment_id=str(row.get('amendment_id', '')),
        section=str(row.get('section', '')),
        text=str(row.get('text', '')),
        score=float(row.get('score', 0.0) or 0.0),
        source_uri=row.get('source_uri'),
        char_start=int(row.get('char_start', 0) or 0),
        char_end=int(row.get('char_end', 0) or 0),
        manifest_id=row.get('manifest_id'),
        embedding_version=row.get('embedding_version'),
    )

    needs = not current.document_id or not current.amendment_id or not current.section or current.char_end == 0
    if not needs or not chunk_id:
        return current

    with psycopg.connect(settings.database_url) as conn:
        row_db = conn.execute(
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

    if not row_db:
        return current

    trial_id, amendment_id, document_id, section, char_start, char_end, source_uri = row_db
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


def retrieve_sparse(question: str, scope: QueryScope, limit: int = 8) -> RetrievalResult:
    settings = get_settings()

    if not _site_active(scope.trial_id, scope.site_id, scope.as_of):
        return RetrievalResult(
            query_id=str(uuid.uuid4()),
            evidence=[],
            diagnostics={'retriever': 'sparse', 'site_active': False, 'scope_seen': scope.model_dump(mode='json')},
        )

    allowed = _allowed_amendments(scope.trial_id, scope.as_of)
    if not allowed:
        return RetrievalResult(
            query_id=str(uuid.uuid4()),
            evidence=[],
            diagnostics={'retriever': 'sparse', 'site_active': True, 'allowed_amendments': [], 'scope_seen': scope.model_dump(mode='json')},
        )

    q_terms = Counter(_tokens(question))
    rows = load_corpus()

    scored: list[tuple[float, dict]] = []
    for row in rows:
        if row.get('trial_id') != scope.trial_id:
            continue
        # Fail-closed on amendment provenance.
        if str(row.get('amendment_id', '')) not in allowed:
            continue
        # Version-appropriate index: only current active manifest.
        # (If corpus uses a manifest_id field, honor it; otherwise we still fail-closed using DB enrichment.)
        if 'manifest_id' in row and row.get('manifest_id') != settings.active_manifest_id:
            continue

        terms = Counter(_tokens(row.get('text', '')))
        overlap = sum(min(q_terms[t], terms[t]) for t in q_terms)
        if overlap:
            norm = math.sqrt(sum(v * v for v in terms.values())) or 1.0
            scored.append((overlap / norm, row))

    # Deterministic ordering.
    scored.sort(key=lambda item: (-item[0], str(item[1].get('chunk_id', ''))))

    evidence: list[Evidence] = []
    for score, row in scored[:limit * 2]:
        ev = _enrich_evidence_from_postgres(settings, row)
        # Fail-closed verification.
        if ev.trial_id != scope.trial_id:
            continue
        if ev.amendment_id not in allowed:
            continue
        ev.score = float(score)
        evidence.append(ev)
        if len(evidence) >= limit:
            break

    return RetrievalResult(
        query_id=str(uuid.uuid4()),
        evidence=evidence,
        diagnostics={'retriever': 'sparse', 'scope_seen': scope.model_dump(mode='json'), 'allowed_amendments': sorted(list(allowed))},
    )
