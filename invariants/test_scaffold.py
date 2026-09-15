from pathlib import Path
import json


def test_fixture_files_are_populated():
    corpus = Path('/root/task/data/corpus.jsonl')
    evals = Path('/root/task/data/eval_queries.jsonl')
    assert corpus.exists() and evals.exists()
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    assert len(rows) >= 8
    assert {r['trial_id'] for r in rows} >= {'AX-17', 'BC-42'}
    assert all('chunk_id' in r and 'text' in r and 'amendment_id' in r for r in rows)


def test_application_imports_without_provider_key():
    import app.main  # noqa: F401
    import app.evaluation  # noqa: F401
    import app.hybrid_retrieval  # noqa: F401
