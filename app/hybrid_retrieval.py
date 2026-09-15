from __future__ import annotations

import uuid
from app.dense_retrieval import retrieve_dense
from app.models import Evidence, QueryScope, RetrievalResult
from app.sparse_retrieval import retrieve_sparse


def retrieve_hybrid(question: str, scope: QueryScope, limit: int = 6) -> RetrievalResult:
    dense = retrieve_dense(question, scope, limit=limit)
    sparse = retrieve_sparse(question, scope, limit=limit)
    merged: dict[str, Evidence] = {}
    for rank, ev in enumerate(dense.evidence):
        ev.score = ev.score + 1.0 / (rank + 1)
        merged[ev.chunk_id] = ev
    for rank, ev in enumerate(sparse.evidence):
        if ev.chunk_id in merged:
            merged[ev.chunk_id].score += 0.5 / (rank + 1)
        else:
            ev.score = ev.score + 0.5 / (rank + 1)
            merged[ev.chunk_id] = ev
    evidence = sorted(merged.values(), key=lambda e: (-e.score, e.chunk_id))[:limit]
    return RetrievalResult(
        query_id=str(uuid.uuid4()),
        evidence=evidence,
        diagnostics={'dense': dense.diagnostics, 'sparse': sparse.diagnostics},
    )
