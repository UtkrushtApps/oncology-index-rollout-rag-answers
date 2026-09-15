# Solution Steps

1. Identify the failing flows in the starter implementation: retrieval ignores scope (trial/site/date/amendment eligibility), cache key ignores scope and rollout state, and evaluation always returns verdict='review' without actionable gate logic.

2. Harden retrieval to be fail-closed for scope: in dense and sparse retrieval, (1) check site activation window for scope.as_of, (2) compute allowed amendment_ids from document_provenance where trial_id matches, archived=false, and effective_from/effective_to cover as_of, (3) filter candidates by trial_id + manifest_id + amendment_id (Qdrant metadata) and then post-filter against allowed amendments to guarantee correctness even if payload fields vary.

3. Ensure audit-traceable evidence provenance: when building Evidence from Qdrant rows, enrich any missing provenance fields (document_id, amendment_id, section, char offsets, source_uri) by joining chunk_lineage->document_provenance in Postgres keyed by chunk_id and active manifest_id. This guarantees citations point to exact fixture passages with correct document_id and character bounds.

4. Stabilize reranking for reproducibility: adjust rerank ordering to prefer lexical overlap (grounding) and then use the original score and chunk_id as deterministic tie-breakers to minimize float-order jitter.

5. Fix cache and rollout state handling: update answer_cache_key to incorporate question, trial_id, site_id, as_of, requester_role, active_manifest_id, embedding_version, and corpus_id. This prevents stale answers/evidence from being served across amendments, sites, and rollout promotions.

6. Make generated citations audit-ready: keep build_context citations derived directly from the retrieved Evidence (chunk_id/document_id/section and quote with bounded char_start/char_end). Update audit_events for answer_generated to include both the evidence and citations payloads (so provenance links back to the exact retrieved passages).

7. Implement evaluation/regression gates: in run_evaluation, keep the separate dimensions (trial_isolation, version_attribution, retrieval_relevance, citation_grounding) and compute deterministic metrics. Add a gate verdict function that promotes only when every dimension meets the promotion threshold; otherwise return 'review' or 'hold' with an explicit rationale.

8. Prevent promotion via gate results: when store=True, write evaluation_runs and also write a rollout_decisions entry (promote only on verdict='pass', else hold) using the gate rationale. This provides a concrete, queryable gate artifact for rollout automation.

9. Validate locally using run.sh: bootstrapping ingests the fixture; smoke imports and health endpoints work without provider keys; pytest invariants and hidden tests confirm (a) scope filtering excludes later amendments/drafts, (b) cache keys change with scope, (c) citations resolve to exact fixture passages, and (d) evaluation metrics match expected 1.0 on unchanged fixture data with a gate verdict in {'pass','review','hold'}.

