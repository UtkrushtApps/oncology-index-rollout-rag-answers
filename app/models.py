from __future__ import annotations

from datetime import date
from typing import Any
from pydantic import BaseModel, Field


class QueryScope(BaseModel):
    trial_id: str
    site_id: str
    as_of: date
    requester_role: str = 'medical_information'


class AnswerRequest(BaseModel):
    question: str
    scope: QueryScope
    query_id: str | None = None


class Evidence(BaseModel):
    chunk_id: str
    document_id: str
    trial_id: str
    amendment_id: str
    section: str
    text: str
    score: float = 0.0
    source_uri: str | None = None
    char_start: int = 0
    char_end: int = 0
    manifest_id: str | None = None
    embedding_version: str | None = None


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    section: str
    quote: str
    char_start: int
    char_end: int


class AnswerResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[Citation]
    evidence: list[Evidence]
    model: str | None = None
    cache_hit: bool = False


class RetrievalResult(BaseModel):
    query_id: str
    evidence: list[Evidence]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    case_id: str
    question: str
    scope: QueryScope
    expected_trial_id: str
    expected_amendment_id: str
    relevant_terms: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    dataset_id: str
    metrics: dict[str, float]
    dimensions: dict[str, dict[str, Any]]
    verdict: str
