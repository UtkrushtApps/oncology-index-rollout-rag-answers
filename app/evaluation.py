from __future__ import annotations

from app.audit import store_evaluation, store_rollout_decision
from app.config import get_settings
from app.hybrid_retrieval import retrieve_hybrid
from app.models import EvalCase, EvalReport
from app.rerank import rerank

import json
from pathlib import Path
from statistics import mean

EVAL_PATH = Path('/root/task/data/eval_queries.jsonl')


def load_eval_cases(path: Path = EVAL_PATH) -> list[EvalCase]:
    with path.open('r', encoding='utf-8') as fh:
        return [EvalCase.model_validate(json.loads(line)) for line in fh if line.strip()]


def evaluate_case(case: EvalCase) -> dict:
    result = retrieve_hybrid(case.question, case.scope, limit=8)
    evidence = rerank(case.question, case.scope, result.evidence, limit=4)
    top = evidence[0] if evidence else None

    trial_ok = bool(top and top.trial_id == case.expected_trial_id)
    version_ok = bool(top and top.amendment_id == case.expected_amendment_id)

    any_relevant = any(term.lower() in ev.text.lower() for ev in evidence for term in case.relevant_terms)
    citation_grounding = bool(evidence) and all(ev.char_end >= ev.char_start and ev.chunk_id for ev in evidence)

    return {
        'case_id': case.case_id,
        'trial_isolation': 1.0 if trial_ok else 0.0,
        'version_attribution': 1.0 if version_ok else 0.0,
        'retrieval_relevance': 1.0 if any_relevant else 0.0,
        'citation_grounding': 1.0 if citation_grounding else 0.0,
        'top_chunks': [ev.chunk_id for ev in evidence],
    }


def _gate_verdict(metrics: dict[str, float]) -> tuple[str, str]:
    """Return (verdict, rationale). Promotion requires PASS with no dimension regressions."""
    # Any dimension < 0.999 counts as regression for promotion.
    promote_threshold = 0.999
    regressions = {k: v for k, v in metrics.items() if v < promote_threshold}

    if not regressions:
        return 'pass', 'all quality dimensions meet promotion threshold'

    # Critical dimensions: fail closed.
    critical = {'trial_isolation', 'version_attribution', 'citation_grounding'}
    critical_regress = any(k in critical for k in regressions)

    if critical_regress:
        return 'hold', 'critical quality regression detected: ' + ', '.join(f'{k}={v:.3f}' for k, v in sorted(regressions.items()))

    return 'review', 'non-critical regression detected: ' + ', '.join(f'{k}={v:.3f}' for k, v in sorted(regressions.items()))


def run_evaluation(store: bool = False) -> EvalReport:
    settings = get_settings()
    case_results = [evaluate_case(case) for case in load_eval_cases()]

    dimensions: dict[str, dict] = {}
    for dim in ['trial_isolation', 'version_attribution', 'retrieval_relevance', 'citation_grounding']:
        scores = [r[dim] for r in case_results]
        dimensions[dim] = {'score': mean(scores) if scores else 0.0, 'cases': case_results}

    metrics = {name: data['score'] for name, data in dimensions.items()}
    verdict, rationale = _gate_verdict(metrics)

    if store:
        store_evaluation(settings.eval_dataset_id, metrics, verdict, settings.active_manifest_id)
        # Gate rollout: only promote if verdict == pass.
        decision = 'promote' if verdict == 'pass' else 'hold'
        store_rollout_decision(settings.active_manifest_id, settings.eval_dataset_id, decision, rationale)

    return EvalReport(dataset_id=settings.eval_dataset_id, metrics=metrics, dimensions=dimensions, verdict=verdict)
