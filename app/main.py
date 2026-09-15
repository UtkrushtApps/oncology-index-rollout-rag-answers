from __future__ import annotations

import uuid
from fastapi import FastAPI, HTTPException

from app.audit import active_manifest, record_event
from app.cache import get_cached_answer, set_cached_answer
from app.context_assembly import build_context
from app.evaluation import run_evaluation
from app.generation import ProviderNotConfigured, generate_answer
from app.hybrid_retrieval import retrieve_hybrid
from app.models import AnswerRequest, AnswerResponse
from app.rerank import rerank

app = FastAPI(title='Oncology Protocol RAG')


@app.get('/health')
def health() -> dict:
    return {'ok': True, 'manifest': active_manifest().get('manifest_id')}


@app.post('/answer', response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    query_id = req.query_id or str(uuid.uuid4())
    cached = get_cached_answer(req.question, req.scope)
    if cached:
        cached.cache_hit = True
        record_event(
            'answer_cache_hit',
            {'scope': req.scope.model_dump(mode='json')},
            query_id=query_id,
            manifest_id=active_manifest().get('manifest_id'),
        )
        return cached

    result = retrieve_hybrid(req.question, req.scope)
    evidence = rerank(req.question, req.scope, result.evidence)
    context, citations = build_context(evidence)

    try:
        text, model = generate_answer(req.question, req.scope, context, citations)
    except ProviderNotConfigured as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc

    response = AnswerResponse(query_id=query_id, answer=text, citations=citations, evidence=evidence, model=model)
    set_cached_answer(req.question, req.scope, response)

    record_event(
        'answer_generated',
        {
            'scope': req.scope.model_dump(mode='json'),
            'chunks': [e.chunk_id for e in evidence],
            'citations': [c.model_dump(mode='json') for c in citations],
            'evidence': [e.model_dump(mode='json') for e in evidence],
            'retrieval_diagnostics': result.diagnostics,
        },
        query_id=query_id,
        manifest_id=active_manifest().get('manifest_id'),
    )
    return response


@app.get('/evaluate')
def evaluate() -> dict:
    return run_evaluation(store=True).model_dump(mode='json')
