from __future__ import annotations

import json
import uuid
import psycopg
from app.config import get_settings


def record_event(event_type: str, payload: dict, query_id: str | None = None, manifest_id: str | None = None) -> None:
    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            'INSERT INTO audit_events(event_type, query_id, manifest_id, payload) VALUES (%s,%s,%s,%s)',
            (event_type, query_id, manifest_id, json.dumps(payload)),
        )
        conn.commit()


def active_manifest() -> dict:
    settings = get_settings()
    with psycopg.connect(settings.database_url, row_factory=psycopg.rows.dict_row) as conn:
        row = conn.execute(
            "SELECT * FROM index_manifests WHERE manifest_id=%s",
            (settings.active_manifest_id,),
        ).fetchone()
    return dict(row or {})


def store_evaluation(dataset_id: str, metrics: dict, verdict: str, manifest_id: str | None = None) -> str:
    run_id = str(uuid.uuid4())
    with psycopg.connect(get_settings().database_url) as conn:
        conn.execute(
            'INSERT INTO evaluation_runs(run_id, dataset_id, manifest_id, metrics, verdict) VALUES (%s,%s,%s,%s,%s)',
            (run_id, dataset_id, manifest_id, json.dumps(metrics), verdict),
        )
        conn.commit()
    return run_id


def store_rollout_decision(manifest_id: str, dataset_id: str, decision: str, rationale: str) -> str:
    settings = get_settings()
    decision_id = None
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO rollout_decisions(manifest_id, dataset_id, decision, rationale)
            VALUES (%s,%s,%s,%s)
            RETURNING decision_id
            """,
            (manifest_id, dataset_id, decision, rationale),
        ).fetchone()
        conn.commit()
        decision_id = row[0] if row else None
    return str(decision_id)
