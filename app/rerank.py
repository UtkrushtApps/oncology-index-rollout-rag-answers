from __future__ import annotations

import re
from app.models import Evidence, QueryScope


def rerank(question: str, scope: QueryScope, candidates: list[Evidence], limit: int = 4) -> list[Evidence]:
    """Deterministic rerank primarily by lexical grounding to reduce float/score jitter."""
    terms = {t.lower() for t in re.findall(r'[A-Za-z0-9-]+', question) if len(t) > 2}

    def key(ev: Evidence) -> tuple[int, float, str]:
        text = ev.text.lower()
        overlap = sum(1 for t in terms if t in text)
        # Sort: overlap desc, then original score desc, then chunk_id.
        return (-overlap, -float(ev.score or 0.0), ev.chunk_id)

    ranked = sorted(candidates, key=key)

    # Update scores in-place for transparency.
    out: list[Evidence] = []
    for ev in ranked[:limit]:
        text = ev.text.lower()
        overlap = sum(1 for t in terms if t in text)
        ev.score = float(ev.score or 0.0) + overlap * 0.05
        out.append(ev)
    return out
