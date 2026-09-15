from __future__ import annotations

import argparse
import json
import sys
from app.audit import active_manifest
from app.bootstrap import bootstrap
from app.context_assembly import build_context
from app.evaluation import run_evaluation
from app.generation import generate_answer
from app.hybrid_retrieval import retrieve_hybrid
from app.models import AnswerRequest
from app.rerank import rerank


def smoke() -> None:
    manifest = active_manifest()
    if not manifest:
        raise SystemExit('no active manifest found')
    print(json.dumps({'ok': True, 'manifest': manifest.get('manifest_id')}))


def ask(payload: str) -> None:
    req = AnswerRequest.model_validate_json(payload)
    result = retrieve_hybrid(req.question, req.scope)
    evidence = rerank(req.question, req.scope, result.evidence)
    context, citations = build_context(evidence)
    answer, model = generate_answer(req.question, req.scope, context, citations)
    print(json.dumps({'answer': answer, 'model': model, 'citations': [c.model_dump() for c in citations]}, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('bootstrap')
    sub.add_parser('smoke')
    sub.add_parser('evaluate')
    ask_p = sub.add_parser('ask')
    ask_p.add_argument('payload')
    args = parser.parse_args(argv)
    if args.cmd == 'bootstrap':
        bootstrap()
    elif args.cmd == 'smoke':
        smoke()
    elif args.cmd == 'evaluate':
        print(run_evaluation(store=True).model_dump_json(indent=2))
    elif args.cmd == 'ask':
        ask(args.payload)


if __name__ == '__main__':
    main(sys.argv[1:])
