from __future__ import annotations

import json
from redis import Redis

from app.config import get_settings
from app.models import AnswerResponse, QueryScope


def redis_client() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


def answer_cache_key(question: str, scope: QueryScope) -> str:
    """Cache key must include clinical scope and rollout state to prevent stale evidence reuse."""
    settings = get_settings()
    scope_part = scope.model_dump(mode='json')
    q = question.strip().lower()
    # include active manifest id and embedding model/version to avoid cross-rollout poisoning
    return (
        'answer:'
        f'manifest={settings.active_manifest_id};'
        f'emb={settings.embedding_version};'
        f'corpus={settings.corpus_id};'
        f'trial={scope_part["trial_id"]};'
        f'site={scope_part["site_id"]};'
        f'as_of={scope_part["as_of"]};'
        f'role={scope_part.get("requester_role", "")}:'
        f'q={q}'
    )


def get_cached_answer(question: str, scope: QueryScope) -> AnswerResponse | None:
    raw = redis_client().get(answer_cache_key(question, scope))
    if not raw:
        return None
    return AnswerResponse.model_validate(json.loads(raw))


def set_cached_answer(question: str, scope: QueryScope, answer: AnswerResponse) -> None:
    redis_client().setex(answer_cache_key(question, scope), 3600, answer.model_dump_json())
