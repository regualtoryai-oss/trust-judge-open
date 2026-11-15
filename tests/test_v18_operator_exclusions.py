"""Explicit v18 exclusion provenance through initial, resume and publication.

Synthetic doubles test engineering permission flow, never semantic accuracy.
"""
from copy import deepcopy

import pytest

from app.domain.material_eligibility import validate_workflow_material_eligibility
from app.domain.premise_bank import build_quote_faithful_bank, PremiseBankInvalid
from app.domain.strict_verification import build_fact_views
from app.workflows.analysis import run_analysis, AnalysisCancelled
from app.worker import Worker
from test_premise_bank import Q, source, paragraphs
from test_premise_bank_workflow import V18Provider
from test_premise_bank_repository import setup, MeteredProductionDouble


class ObservedProvider(V18Provider):
    def __init__(self, rows, **kwargs):
        super().__init__(rows, **kwargs)
        self.scope_inputs = []

    async def verify_evidence_scope(self, question, evidence):
        self.scope_inputs.append(deepcopy(evidence))
        return await super().verify_evidence_scope(question, evidence)


def blocked_span(row):
    start = row["quote"].index("预警")
    return (row["paragraph_id"], start, start + 2, "预警")


async def execute(tmp_path, provider, exclusions, *, identity="initial", resume=None, cancel=lambda: False, manual=None):
    return await run_analysis(Q, paragraphs(provider.rows), provider,
        checkpoint_path=tmp_path / "checkpoints.sqlite3", thread_id=identity,
        strict_scope_enabled=True, excluded_spans=exclusions,
        manual_evidence=manual or [], emit=lambda *_args: None,
        cancel_check=cancel, resume_from_thread_id=resume)


@pytest.mark.parametrize("include_manual", [False, True])
async def test_initial_exclusion_keeps_full_replay_proof_without_reviewing_or_reviving_excluded_views(tmp_path, include_manual):
    blocked, allowed = source("blocked"), source("allowed", text="场馆已配备应急广播。")
    provider = ObservedProvider([blocked, allowed])
    manual = [{key: deepcopy(blocked[key]) for key in
        ("paragraph_id", "quote", "start_cp", "end_cp", "fact_des", "argument", "entities")}] if include_manual else []
    result = await execute(tmp_path, provider, {blocked_span(blocked)}, manual=manual)
    candidates = result["candidate_evidence"]
    assert len(candidates) == 2 + int(include_manual)
    excluded = result["strict_scope_review"]["operator_excluded_fact_view_ids"]
    expected = {view["id"] for row in candidates if row["paragraph_id"] == blocked["paragraph_id"]
                for view in build_fact_views([row])}
    assert set(excluded) == expected and len(excluded) == 1 + int(include_manual)
    assert len(result["operator_exclusion_audit"]) == 1 + int(include_manual)
    assert all(row["paragraph_id"] == allowed["paragraph_id"] for batch in provider.scope_inputs for row in batch)
    assert all(row["paragraph_id"] == allowed["paragraph_id"] for row in result["quote_faithful_bank"]["rows"])
    assert all(row["paragraph_id"] == allowed["paragraph_id"] for row in result["evidence"])
    assert all(row["paragraph_id"] == allowed["paragraph_id"] for _role, rows, _sid in provider.analyst_inputs for row in rows)
    assert result["purpose_review"]["manual_pending_ids"] == []
    validate_workflow_material_eligibility(result, Q, paragraphs(provider.rows))


async def test_all_initial_candidates_excluded_yields_empty_internal_and_formal_pool_without_purpose_call(tmp_path):
    provider = ObservedProvider([source("first"), source("second")])
    exclusions = {blocked_span(row) for row in provider.rows}
    result = await execute(tmp_path, provider, exclusions)
    assert len(result["candidate_evidence"]) == len(result["operator_exclusion_audit"]) == 2
    assert provider.scope_inputs == provider.purpose_calls == []
    assert result["quote_faithful_bank"]["rows"] == result["evidence"] == []
    assert result["purpose_review"]["complete"] and result["final"]["decision"] == "undetermined"
    assert all(rows == [] for _role, rows, _sid in provider.analyst_inputs)


@pytest.mark.parametrize("attack", ["invent_unknown_id", "remove_actual_excluded_proof"])
async def test_audit_summary_never_authorizes_unknown_or_missing_excluded_view_id(tmp_path, attack):
    provider = ObservedProvider([source("blocked"), source("allowed", text="场馆已配备应急广播。")])
    result = await execute(tmp_path, provider, {blocked_span(provider.rows[0])})
    candidates, scope = deepcopy(result["candidate_evidence"]), deepcopy(result["strict_scope_review"])
    if attack == "invent_unknown_id":
        scope["operator_excluded_fact_view_ids"].append("F" + "0" * 64)
    else:
        candidates = [row for row in candidates if row["paragraph_id"] != provider.rows[0]["paragraph_id"]]
    # Even a genuine summary with a quote hash is not a substitute for the
    # canonical annotation needed to recompute a fact-view identity.
    scope["operator_exclusion_audit"] = deepcopy(result["operator_exclusion_audit"])
    with pytest.raises(PremiseBankInvalid, match="premise_scope_exclusions"):
        build_quote_faithful_bank(Q, candidates, scope, paragraphs(provider.rows))


async def test_exclusion_proof_survives_checkpoint_resume_without_double_audit_or_repeated_purpose_batch(tmp_path, monkeypatch):
    import app.domain.premise_bank as module
    original = module.plan_fact_purpose_batches
    monkeypatch.setattr(module, "plan_fact_purpose_batches", lambda question, bank: original(question, bank, max_rows=1))
    paused = [False]
    provider = ObservedProvider([source("blocked"), source("one"), source("two")], pause=paused)
    exclusions = {blocked_span(provider.rows[0])}
    with pytest.raises(AnalysisCancelled):
        await execute(tmp_path, provider, exclusions, identity="paused", cancel=lambda: paused[0])
    assert len(provider.purpose_calls) == 1
    provider.pause, paused[0] = None, False
    result = await execute(tmp_path, provider, exclusions, identity="resumed", resume="paused")
    assert len(result["candidate_evidence"]) == 3 and len(result["operator_exclusion_audit"]) == 1
    assert len(provider.purpose_calls) == 2 and not set(provider.purpose_calls[0]) & set(provider.purpose_calls[1])
    assert len(result["evidence"]) == 2
    assert all(row["paragraph_id"] != provider.rows[0]["paragraph_id"] for row in result["evidence"])
    validate_workflow_material_eligibility(result, Q, paragraphs(provider.rows))


async def test_initial_worker_with_operator_exclusion_publishes_and_exports_without_reviving_excluded_content(tmp_path):
    repo, topic, rows = setup(tmp_path)
    blocked = next(row for row in rows if "外墙" not in row["quote"])
    provider = MeteredProductionDouble(rows, repo)
    question_ids = [row["id"] for row in repo.questions(topic)]
    queued = repo.create_run(topic, {"question_ids": question_ids, "base_revision_id": None,
        "mode": "initial", "include_baseline": False}, "operator-exclusion",
        model_available=True, options={"overlays": {"excluded": [list(blocked_span(blocked))],
            "manual_evidence": [], "interpretation_overrides": {}}})
    worker = Worker(repo, provider)
    await worker.execute(repo.claim())
    run = repo.run(queued["id"])
    assert run["status"] == "succeeded", run["error"]
    snapshot = repo.snapshot(topic, run["target_revision_id"])
    result = next(iter(snapshot["question_results"].values()))
    assert len(result["candidate_evidence"]) == 2 and len(result["operator_exclusion_audit"]) == 1
    assert all(row["paragraph_id"] != blocked["paragraph_id"] for row in result["quote_faithful_bank"]["rows"])
    assert snapshot["evidence"] == []
    assert repo.formal_submission_files(topic, run["target_revision_id"])["evidence.json"] == []
    job = repo.create_export(topic, run["target_revision_id"], "excluded-export")
    assert worker.export_one() and repo.export(job["id"])["status"] == "succeeded"
