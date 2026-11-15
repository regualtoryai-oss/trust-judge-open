"""Synthetic UI/native files and synthetic gold only. No models or sample IO."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.capture_bundle_workbench_real import (
    bind_browser_download, bound, capture, capture_episode, read_database, save, usage_observation,
)
import scripts.score_bundle_workbench_real as scorer
from test_source_bundle_publication import fixture, result_for, create_and_build


def ui_fixture(tmp_path, status="failed"):
    observation = tmp_path / "synthetic-ui-observation.txt"
    observation.write_text("Explicit synthetic UI click record for unit tests", encoding="utf-8")
    freeze = tmp_path / "freeze.json"; freeze.write_text("{}", encoding="utf-8")
    run = {"id": "run", "topic_id": "topic", "question_ids": ["question"], "status": status,
        "completed_at": "2026-10-20T12:00:00Z", "usage": {"question": [{"operation": "source_bundle_selection", "usage_source": "provider",
            "input_tokens": 5, "output_tokens": 2, "total_tokens": 7}]}}
    start = tmp_path / "ui-start.json"
    save(start, {"format": "actual-workbench-ui-start-v1", "actual_browser_UI_start": True,
        "run_id": run["id"], "topic_id": run["topic_id"], "question_internal_id": "question",
        "input_freeze_sha256": bound(freeze)["sha256"], "browser_observation": bound(observation)})
    return run, freeze, start


def test_capture_refuses_active_task_before_snapshot_or_reference_access(tmp_path):
    run, freeze, start = ui_fixture(tmp_path, "running")
    with pytest.raises(ValueError, match="only_terminal"):
        capture("run", tmp_path / "episode", freeze, start, tmp_path / "closed", "synthetic-topic", "q", getter=lambda *_: run)
    assert not (tmp_path / "closed").exists()


def test_failed_capture_preserves_known_and_unknown_cost_without_old_result(tmp_path):
    run, freeze, start = ui_fixture(tmp_path)
    run["usage"]["question"].append({"operation": "authority_analyst", "usage_source": "unavailable", "input_tokens": None, "output_tokens": None, "total_tokens": None})
    episode = tmp_path / "episode"; episode.mkdir()
    save(episode / "request-01-preflight.json", {"admitted": True, "operation": "authority_analyst"})
    native = episode / "source-bundle/native"; native.mkdir(parents=True)
    save(native / "closed-receipt.json", {"status": "failed_usage_unknown", "episode_closed": True})
    calls = []
    def getter(path, **_):
        calls.append(path)
        return run if path.endswith("/run") else {"run": run, "usage": run["usage"]}
    binding = capture("run", episode, freeze, start, tmp_path / "closed", "synthetic-topic", "q", getter=getter)
    closure = json.loads(Path(binding["file"]).read_text("utf-8"))
    assert closure["status"] == "failed" and closure["final_precision_recall"] is None
    assert closure["usage_observation"]["known_totals"]["total_tokens"] == 7
    assert closure["usage_observation"]["unknown_usage_records"] == 1
    assert "question_result" not in closure["artifacts"]
    assert calls == ["/runs/run", "/runs/run/activity"]
    result = scorer.score_episode(binding["file"], binding["sha256"], tmp_path / "do-not-read-reference.json", tmp_path / "do-not-read-conclusions.json",
        browser_binding=None, browser_sha256=None)
    assert result["after"] is None and result["before"] is None and result["references_read"] is False


def test_episode_capture_never_reads_session_or_reasoning_files(tmp_path):
    (tmp_path / "source-bundle/native/sessions").mkdir(parents=True)
    (tmp_path / "source-bundle/native/sessions/private.json").write_text("not valid JSON and must never be parsed", encoding="utf-8")
    (tmp_path / "source-bundle/native/native-response-01.sse").write_text("private reasoning fixture", encoding="utf-8")
    result = capture_episode(tmp_path)
    assert "sessions" not in json.dumps(result) and "native-response" not in json.dumps(result)


def test_langchain_front_accounting_has_its_own_receipt_without_a_dsh_ledger(tmp_path):
    usage = {"operation": "strict_selection", "usage_source": "provider", "input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    receipt = save(tmp_path / "explicit-synthetic-lc-receipt.json", {"execution_mode": "workbench", "usage_records": [usage]})
    previous = save(tmp_path / "explicit-synthetic-parents.json", {"records": []})
    response = {"proof": {"format": scorer.LC_PROOF_FORMAT, "closed_receipt": receipt, "previous_usage": previous},
        "usage_report": {"real_api_requests": 1, "new_request_count": 1, "simulated_usage_not_bill": False}}
    result = {"source_bundle_proof": {"response": response}}
    checked = scorer._front_usage(result, [usage], [])
    assert checked["new_records"] == [usage] and checked["previous_records"] == []
    with pytest.raises(ValueError, match="must_equal_workbench"):
        scorer._front_usage(result, [{**usage, "input_tokens": 99, "total_tokens": 119}], [])


def test_langchain_capture_binds_its_actual_front_files_without_synthetic_native_calls(tmp_path):
    front = tmp_path / "source-bundle"
    front.mkdir()
    save(front / "provider-response.json", {"proof": {"format": scorer.LC_PROOF_FORMAT,
        "front_requests": [{"operation": "strict_selection", "actual_wire": {"file": "synthetic-only", "sha256": "a" * 64}}]}})
    capture_files = capture_episode(tmp_path)
    assert capture_files["front"]["native_requests"] == []
    assert capture_files["front"]["langchain_requests"][0]["operation"] == "strict_selection"


def test_capture_includes_graded_mode_lossless_projection_and_nonbilling_forecasts(tmp_path):
    save(tmp_path / "request-03-preflight.json", {"operation": "arbitrate"})
    projected = save(tmp_path / "request-03-analysis-input-projection.json", {"new_model_calls": 0})
    budget = save(tmp_path / "budget-before-first-paid.json", {"is_actual_usage": False})
    save(tmp_path / "source-bundle/provider-response.json", {"proof": {"format": scorer.LC_PROOF_FORMAT,
        "front_requests": [], "selection_only": True, "graded_selection": True, "selection_output_tokens": 2048,
        "compact_analyses": True, "graded_selection_policy": {"maximum_selected_anchors": 10}}})
    captured = capture_episode(tmp_path)
    assert captured["tail"]["requests"][0]["analysis-input-projection"] == projected
    assert captured["tail"]["budget_forecasts"] == [budget]
    assert captured["front"]["selection_mode"]["selection_output_tokens"] == 2048
    assert captured["front"]["selection_mode"]["use_type_is_final_relation"] is False


def test_scoring_graded_mode_matches_frozen_policy_not_a_final_conflict_label():
    from app.config import Settings
    cfg = Settings(_env_file=None, model_source_bundle_enabled=True, source_bundle_front_framework="langchain",
        source_bundle_selection_only=True, source_bundle_graded_selection=True, source_bundle_compact_analyses_enabled=True).source_bundle_config
    proof = {key: deepcopy(cfg[key]) for key in ("selection_only", "graded_selection", "graded_selection_policy",
        "selection_output_tokens", "compact_analyses", "whole_chain_planning")}
    run = {key: True for key in ("source_bundle_selection_only", "source_bundle_graded_selection", "source_bundle_compact_analyses")}
    assert scorer._selection_mode(proof, cfg, run)["front_use_type_is_final_conflict"] is False
    for field, bad in [("graded_selection", False), ("selection_output_tokens", 1024),
                       ("graded_selection_policy", {"maximum_selected_anchors": 9}), ("whole_chain_planning", None)]:
        altered = {**proof, field: bad}
        with pytest.raises(ValueError, match="bundle_frozen_"):
            scorer._selection_mode(altered, cfg, run)
    with pytest.raises(ValueError, match="selection_mode_changed"):
        scorer._selection_mode(proof, cfg, {**run, "source_bundle_graded_selection": False})


@pytest.mark.parametrize("change", [None, "cap", "policy", "mixed_context", "public_flag"])
def test_scoring_paragraph_quality_requires_closed_mode_policy_and_output_budget(change):
    # Synthetic policy: verify receipt binding, not a statement of model quality.
    policy = {"version": "synthetic-complete-parent-fields", "maximum_selected_anchors": None}
    cfg = {"paragraph_quality": True, "paragraph_quality_policy": policy,
           "selection_output_tokens": 8192}
    proof = deepcopy(cfg)
    run = {"source_bundle_paragraph_quality": True}
    if change == "cap":
        proof["selection_output_tokens"] = 2048
    elif change == "policy":
        proof["paragraph_quality_policy"] = {**policy, "maximum_selected_anchors": 10}
    elif change == "mixed_context":
        proof["context_passages"] = True
    elif change == "public_flag":
        run["source_bundle_paragraph_quality"] = False
    if change is None:
        result = scorer._selection_mode(proof, cfg, run)
        assert result["paragraph_quality"] is True
        assert result["front_use_type_is_final_conflict"] is False
    else:
        with pytest.raises(ValueError, match="bundle_frozen_"):
            scorer._selection_mode(proof, cfg, run)


@pytest.mark.parametrize("tamper", [None, "statement", "audit", "original", "evidence_alias", "wire_analyses", "payload"])
def test_offline_analysis_projection_bridge_binds_full_prose_roles_dag_and_actual_wire(tamper):
    from app.domain import analysis_wire_projection as codec
    from test_analysis_wire_projection import analyses, uid
    from scripts.score_workbench_real_episode import remap_aliases, canonical_digest
    original = analyses(3)
    compact = codec.project_analyses(original)
    aliases = {uid("a:evidence"): "E1", uid("b:evidence"): "E2",
        uid("a:opposing-evidence"): "E3", uid("b:opposing-evidence"): "E4"}
    payload = {"analyses": remap_aliases(compact["projected"], aliases), "question": {"question": "合成问题"}}
    receipt = {"format": "strict-analysis-input-projection-v1", "operation": "arbitrate", "new_model_calls": 0,
        "trusted_original": deepcopy(original), **deepcopy(compact), "evidence_aliases": deepcopy(aliases),
        "wire_analyses": deepcopy(payload["analyses"]), "projected_payload_sha256": canonical_digest(payload)}
    if tamper == "statement": receipt["projected"][0]["judgments"][0]["statement"] += "被改写"
    elif tamper == "audit": receipt["audit"]["role_bindings"][0]["alias_prefix"] = "B"
    elif tamper == "original": receipt["trusted_original"][0]["conclusion"] += "被改写"
    elif tamper == "evidence_alias": receipt["evidence_aliases"][uid("a:evidence")] = "E9"
    elif tamper == "wire_analyses": receipt["wire_analyses"].pop()
    elif tamper == "payload": payload["analyses"][0]["judgments"].pop()
    if tamper is None:
        audit = scorer._verify_analysis_input_projection(receipt, payload, original, aliases, codec=codec)
        assert audit["all_prose_and_evidence_refs_preserved"] and audit["dag_preserved"]
    else:
        with pytest.raises(ValueError):
            scorer._verify_analysis_input_projection(receipt, payload, original, aliases, codec=codec)


def test_budget_forecast_capture_checks_bound_policy_math_and_native_wire_without_calling_models(tmp_path):
    from app.agents.episode_budget import POLICY, plan_source_episode, plan_tail, planning_sources
    from app.agents.strict_transport import LIMITS
    from app.domain.lc_bundle_case import prepare_lc_case
    from test_source_bundle_tool_bridge import raw_case
    raw = raw_case()
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    kwargs = {"known_tokens": 0, "known_calls": 0, "threshold": 45000, "max_calls": 10,
        "model_name": "deepseek-v4-pro", "output_limits": {**LIMITS, "strict_selection": 2048},
        "compact_analyses": True, "counter": lambda strings: [10] * len(strings)}
    original = [{"role": role, "evidence_snapshot_id": "synthetic-snapshot", "decision": "undetermined", "conclusion": "待核验。",
        "judgments": [], "supporting_evidence_ids": [], "conflicting_evidence_ids": []}
        for role in ("authority_analyst", "cross_validation_analyst")]
    evidence = planning_sources(prepared, counter=kwargs["counter"])
    initial = plan_source_episode(raw["question"], prepared, **kwargs)
    selected = plan_tail(raw["question"], evidence, **kwargs)
    actual = plan_tail(raw["question"], evidence, analyses=original, **kwargs)
    bindings = [save(tmp_path / f"budget-{i}.json", row) for i, row in enumerate((initial, selected, actual))]
    native = {"front": {}, "tail": {"budget_forecasts": bindings}}
    metadata = {"source_bundle_config": {"whole_chain_planning": POLICY}}
    freeze = {"engineering_episode_stop": 45000, "max_total_requests": 10}
    audits = scorer._budget_forecasts(native, metadata, freeze, [])
    assert len(audits) == 3 and all(row["is_actual_usage"] is False for row in audits)
    broken = deepcopy(actual)
    broken["required_remaining_planning_tokens"] += 1
    wrong = save(tmp_path / "bad-math.json", broken)
    with pytest.raises(ValueError, match="forecast_math_changed"):
        scorer._budget_forecasts({"front": {}, "tail": {"budget_forecasts": [*bindings[:2], wrong]}}, metadata, freeze, [])


def test_read_only_database_capture_preserves_repo_aliases_and_original_manifests(tmp_path):
    repo, topic, question = fixture(tmp_path)
    result = result_for(repo, topic, question)
    run, snapshot = create_and_build(repo, topic, question, result)
    repo.publish(run["run_id"], snapshot, {question["id"]: result})
    captured = read_database(repo.settings.database_path, topic, run["target_revision_id"])
    assert captured["source_documents"] == repo.source_documents(topic)
    assert captured["source_paragraphs"] == repo.paragraphs(topic)
    assert all(p["paragraph_id"] == p["id"] for p in captured["source_paragraphs"])
    assert captured["snapshot"] == repo.snapshot(topic, run["target_revision_id"])


def test_download_binding_checks_real_bytes_not_only_true_claim(tmp_path):
    run, _, _ = ui_fixture(tmp_path, "succeeded"); run["target_revision_id"] = "ready"
    run_d = save(tmp_path / "run.json", run)
    api = tmp_path / "api.json"; api.write_bytes(b'{"files":{},"revision_id":"ready","topic_id":"topic"}')
    download = tmp_path / "download.json"; download.write_bytes(api.read_bytes())
    closure_d = save(tmp_path / "closure.json", {"status": "succeeded", "artifacts": {"run": run_d, "api_submission_preview": bound(api)}})
    proof = {"actual_browser_UI_download": True, "run_id": "run", "revision_id": "ready", "download_file": bound(download),
             "exact_API_preview_bytes_equal": True, "exact_API_preview_sha256": bound(api)["sha256"]}
    proof_d = save(tmp_path / "proof.json", proof)
    result = bind_browser_download(closure_d["file"], proof_d["file"], tmp_path / "binding.json")
    assert json.loads(Path(result["file"]).read_text("utf-8"))["exact_API_preview_bytes_equal"]
    download.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError, match="actual_API_bytes"):
        bind_browser_download(closure_d["file"], proof_d["file"], tmp_path / "binding2.json")


def normalized_fixture():
    fact = {"fact_id": "exported-fact", "doc_id": "synthetic-doc", "para_id": "opaque-parent", "quote": "合成原始记录。",
        "fact_des": "最终导出的合成说明。", "argument": {}, "entities": []}
    question = {"question_id": "q-synthetic", "question": "合成记录是否存在？"}
    normalized = {"question_id": question["question_id"], "conclusion": "资料记载合成记录。", "evidence": [fact], "supporting_facts": [fact], "conflicting_facts": []}
    return question, fact, normalized


def test_export_normalization_scores_downloaded_annotation_not_candidate_proposal():
    question, fact, normalized = normalized_fixture()
    result = {"evidence": [{"raw_doc_id": fact["doc_id"], "raw_para_id": fact["para_id"], "quote": fact["quote"], "fact_des": "旧candidate描述"}],
              "candidate_evidence": [{"quote": "未导出的其他candidate"}]}
    preview = {"files": {"evidence.json": [fact], "conclusions.json": [{"question_id": question["question_id"], "question": question["question"],
        "conclusion": normalized["conclusion"], "supporting_facts": [{**fact, "fact_des": "最终关系内描述"}], "conflicting_facts": []}]}}
    actual = scorer._normalized_export(preview, question, result)
    assert actual["evidence"][0]["fact_des"] == fact["fact_des"]
    assert actual["supporting_facts"][0]["fact_des"] == "最终关系内描述"
    assert len(actual["evidence"]) == 1


def test_invalid_closure_is_rejected_before_any_reference_read(tmp_path, monkeypatch):
    closure = save(tmp_path / "bad.json", {"format": "wrong"})
    forbidden = tmp_path / "never-open-synthetic-gold.json"
    real_read, calls = scorer.read_bound, []
    def tracked(path, *args, **kwargs):
        calls.append(Path(path))
        if Path(path) == forbidden: raise AssertionError("reference opened before closure")
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(scorer, "read_bound", tracked)
    with pytest.raises(ValueError, match="real_UI_bundle"):
        scorer.score_episode(closure["file"], closure["sha256"], forbidden, forbidden, browser_binding=None, browser_sha256=None)
    assert calls == [Path(closure["file"])]


def test_frozen_original_document_binding_checks_gaps_and_parent_coordinates():
    content = "首段。|原始分隔|末段。"
    raw = {"documents": [{"doc_id": "raw", "content": content, "paragraphs": [
        {"para_id": "opaque", "char_start": 0, "char_end": 3}]}]}
    documents = [{"raw_doc_id": "raw", "content": content, "content_sha256": sha256(content.encode()).hexdigest()}]
    paragraphs = [{"raw_doc_id": "raw", "raw_para_id": "opaque", "document_start_cp": 0, "document_end_cp": 3}]
    scorer._verify_original_documents(raw, documents, paragraphs)
    changed = deepcopy(documents); changed[0]["content"] = "首段。|伪造分隔|末段。"
    changed[0]["content_sha256"] = sha256(changed[0]["content"].encode()).hexdigest()
    with pytest.raises(ValueError, match="including_gaps_changed"):
        scorer._verify_original_documents(raw, changed, paragraphs)
    with pytest.raises(ValueError, match="coordinates_changed"):
        scorer._verify_original_documents(raw, documents, [{**paragraphs[0], "document_end_cp": 4}])


def test_closed_only_scoring_uses_existing_engine_with_synthetic_gold_and_unknown_B(tmp_path, monkeypatch):
    question, fact, normalized = normalized_fixture()
    prepared = {"failed": False, "guards": [], "question": question, "normalized": normalized, "baseline": deepcopy(normalized),
        "contexts": {(fact["doc_id"], fact["para_id"]): fact["quote"]},
        "candidates": {(fact["doc_id"], fact["para_id"])}, "read_keys": {(fact["doc_id"], fact["para_id"])},
        "source_scope": {"search_preview_paragraphs": 1, "native_full_read_paragraphs": 1, "final_exported_anchor_records": 1},
        "tail": {"actual_tail_requests": 3}, "front": {"report": {"new_tokens": 7}}, "baseline_usage": {"total_tokens": 1}, "browser_download": {}}
    monkeypatch.setattr(scorer, "prepare_closed", lambda *args: prepared)
    evidence_d = save(tmp_path / "synthetic-evidence.json", [{**fact, "fact_id": "reference-fact"}])
    conclusion_d = save(tmp_path / "synthetic-conclusions.json", [{**question, "conclusion": "合成参考", "supporting_facts": [
        {**fact, "fact_id": "reference-fact", "verdict_label": "full_support"}], "conflicting_facts": []}])
    result = scorer.score_episode("already-verified-fixture", "bound", evidence_d["file"], conclusion_d["file"], browser_binding="bound", browser_sha256="bound")
    assert result["after"]["question_selection"]["A_id_only_proxy"]["matches"] == 1
    assert result["after"]["supporting"]["A_id_only_proxy"]["matches"] == 1
    assert result["semantic_B_point_score"] is None and result["new_model_calls"] == 0
    assert result["actual_export_rows_scored"] and not result["proposed_or_candidate_rows_scored_as_final"]


@pytest.mark.asyncio
@pytest.mark.optional_dsh_runtime
async def test_tail_audit_replays_official_SDK_fixture_and_three_strict_HTTP_mock_calls(tmp_path, monkeypatch):
    from app.agents.bundle_transport import BundleWorkbenchProvider
    from app.workflows.analysis import run_analysis
    from test_bundle_transport import settings, fixture_plan, patch_tail
    from test_source_bundle_tool_bridge import raw_case
    case = raw_case()
    for row in case["paragraphs"]:
        row.update(title=case["documents"][0]["title"], source_credibility_json="0.9")
    plan = fixture_plan(tmp_path / "plan.json", case)
    provider = BundleWorkbenchProvider(settings(tmp_path / "runtime"), fixture_plan=plan)
    patch_tail(monkeypatch)
    parent = {"operation": "explicit_synthetic_previous_failure", "usage_source": "provider", "input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
    provider.set_previous_usage([parent])
    question = {"id": "internal-question", "raw_question_id": case["question"]["question_id"], "question": case["question"]["question"]}
    result = await run_analysis(question, case["paragraphs"], provider, source_documents=case["documents"],
        checkpoint_path=tmp_path / "checkpoints.sqlite", thread_id="explicit-sdk-fixture",
        cancel_check=lambda: False, emit=lambda *_: None)
    native_files = capture_episode(provider.directory)
    source_records = [row for row in result["usage"] if row["operation"] not in scorer.TAIL]
    front = {"previous_records": [parent], "new_records": source_records}
    audit = scorer._tail(result, question, native_files, result["usage"], front,
        {"engineering_episode_stop": 30000, "max_total_requests": 10}, [])
    # Program navigation is local: this fixture pays only read + submit in DSH,
    # then three HTTP-mocked tail calls. The previous failed call remains charged.
    assert len(source_records) == 2 and sum(row["total_tokens"] for row in source_records) == 300
    assert audit["actual_tail_requests"] == 3
    assert audit["whole_episode_actual_requests"] == len(result["usage"]) + 1 == 6
    assert audit["whole_episode_known_tokens"] == sum(row["total_tokens"] for row in result["usage"]) + parent["total_tokens"] == 760
    assert audit["independent_A_B_inputs_verified"] and audit["native_relation_audit"]["enabled"]
    # The overall real-episode scorer must still reject the explicit transport
    # fixture. Passing the tail decoder is not evidence of real paid execution.
    with pytest.raises(ValueError, match="forbids_fixture"):
        scorer._front_usage(result, result["usage"], [])
