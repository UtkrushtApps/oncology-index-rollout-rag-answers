from app.evaluation import load_eval_cases, run_evaluation


def test_eval_cases_carry_scope_and_expectations():
    cases = load_eval_cases()
    assert len(cases) >= 4
    assert all(c.scope.trial_id and c.scope.site_id and c.expected_amendment_id for c in cases)


def test_eval_report_has_separate_dimensions_when_services_are_running():
    report = run_evaluation(store=False)
    for name in ['trial_isolation', 'version_attribution', 'retrieval_relevance', 'citation_grounding']:
        assert name in report.metrics
        assert name in report.dimensions
    assert report.dataset_id
