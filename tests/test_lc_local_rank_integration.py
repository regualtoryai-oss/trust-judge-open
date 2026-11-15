"""Local rank admission is opt-in and cannot silently replace frozen retrieval."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.config import Settings
from test_local_parent_rank import model, infer  # noqa: F401 - synthetic local-model fixtures


def test_rank_requires_lc_and_a_frozen_model_manifest():
    defaults = Settings(_env_file=None)
    assert "local_parent_rank" not in defaults.source_bundle_config
    values = {"_env_file": None, "model_source_bundle_enabled": True,
        "source_bundle_front_framework": "langchain", "source_bundle_local_rank_enabled": True}
    with pytest.raises(ValueError, match="frozen model manifest"):
        Settings(**values)
    frozen = {**values, "source_bundle_local_rank_manifest": Path("explicit-local-model.json"),
        "source_bundle_local_rank_manifest_sha256": "a" * 64}
    settings = Settings(**frozen)
    assert settings.source_bundle_config["local_parent_rank"] == {
        "enabled": True, "model_manifest_sha256": "a" * 64,
        "candidate_limit": 32, "per_document_soft_cap": 3,
        "policy": "full-question-literal-facets-body-rrf60-v1"}
    with pytest.raises(ValueError, match="LangChain"):
        Settings(**{**frozen, "source_bundle_front_framework": "dsh"})


def test_graded_and_compact_modes_freeze_the_exact_opt_in_policy_and_keep_old_defaults():
    from app.domain.lc_bundle_case import GRADED_POLICY
    from app.agents.episode_budget import POLICY
    base = {"_env_file": None, "model_source_bundle_enabled": True, "source_bundle_front_framework": "langchain"}
    original = Settings(**base).source_bundle_config
    assert "graded_selection" not in original and "compact_analyses" not in original and "whole_chain_planning" not in original
    with pytest.raises(ValueError, match="explicit source-ID-only"):
        Settings(**base, source_bundle_graded_selection=True)
    with pytest.raises(ValueError, match="LangChain"):
        Settings(_env_file=None, source_bundle_compact_analyses_enabled=True)
    cfg = Settings(**base, source_bundle_selection_only=True, source_bundle_graded_selection=True,
        source_bundle_compact_analyses_enabled=True).source_bundle_config
    assert cfg["selection_output_tokens"] == 2048 and cfg["graded_selection_policy"] == GRADED_POLICY
    assert cfg["compact_analyses"] is True and cfg["whole_chain_planning"] == POLICY
    assert cfg["whole_chain_planning"]["future_analysis_allowance_is_not_a_proven_global_bound"] is True


def test_publication_binds_rank_policy_to_run_config(tmp_path):
    # Exercise the repository's trusted-origin gate before its source proof
    # replay; no model/reference and no need to fabricate a rank proof.
    from app.repository import Repository, dumps
    from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL
    from unittest.mock import Mock
    repo = object.__new__(Repository)
    config = {"source_bundle_enabled": True, "source_bundle_protocol": SOURCE_BUNDLE_PROTOCOL,
        "prompt_version": "trusted-analysis-langchain-source-bundles-v1",
        "source_strategy": "langchain_source_book", "source_bundle_config": {
            "front_framework": "langchain", "local_parent_rank": {"enabled": True}}}
    repo.one = Mock(return_value={"topic_id": "t", "question_ids_json": dumps(["q"]),
        "model_config_json": dumps(config)})
    repo.source_documents = Mock(return_value=[])
    repo.paragraphs = Mock(return_value=[])
    result = {"source_bundle_protocol": SOURCE_BUNDLE_PROTOCOL, "evidence": [],
        "source_bundle_proof": {"response": {"proof": {"format": "langchain-source-bundle-workbench-proof-v1"}}}}
    with pytest.raises(ValueError, match="local ranking"):
        repo._validate_result_protocol(result, "run", "t", "q")
    config["source_bundle_config"].pop("local_parent_rank")
    repo.one.return_value["model_config_json"] = dumps(config)
    modified = deepcopy(result)
    modified["source_bundle_proof"]["response"]["proof"]["local_parent_rank"] = {"config": {"enabled": True}}
    with pytest.raises(ValueError, match="local ranking"):
        repo._validate_result_protocol(modified, "run", "t", "q")


@pytest.mark.parametrize("kind", ["graded", "graded_policy", "compact", "planning"])
def test_publication_cannot_downgrade_graded_compact_or_budget_configuration(kind):
    from app.repository import Repository, dumps
    from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL
    from unittest.mock import Mock
    from app.domain.lc_bundle_case import GRADED_POLICY
    repo = object.__new__(Repository)
    mode = {"front_framework": "langchain", "selection_only": True, "selection_output_tokens": 2048,
        "graded_selection": True, "graded_selection_policy": deepcopy(GRADED_POLICY)}
    proof = {"format": "langchain-source-bundle-workbench-proof-v1", "selection_only": True,
        "graded_selection": True, "graded_selection_policy": deepcopy(GRADED_POLICY), "selection_output_tokens": 2048}
    if kind == "graded": proof.pop("graded_selection")
    elif kind == "graded_policy": proof["graded_selection_policy"]["maximum_selected_anchors"] = 9
    elif kind == "compact": proof["compact_analyses"] = True
    else: proof["whole_chain_planning"] = {"version": "unfrozen-policy"}
    cfg = {"source_bundle_enabled": True, "source_bundle_protocol": SOURCE_BUNDLE_PROTOCOL,
        "prompt_version": "trusted-analysis-langchain-source-bundles-v1", "source_strategy": "langchain_source_book", "source_bundle_config": mode}
    repo.one = Mock(return_value={"topic_id": "t", "question_ids_json": dumps(["q"]), "model_config_json": dumps(cfg)})
    repo.source_documents = Mock(return_value=[])
    repo.paragraphs = Mock(return_value=[])
    result = {"source_bundle_protocol": SOURCE_BUNDLE_PROTOCOL, "evidence": [], "source_bundle_proof": {"response": {"proof": proof}}}
    with pytest.raises(ValueError, match="frozen run configuration"):
        repo._validate_result_protocol(result, "run", "t", "q")


@pytest.mark.asyncio
@pytest.mark.parametrize("selection_only,graded_selection", [(False, False), (True, False), (True, True)])
async def test_local_rank_actual_http_audit_replays_through_archived_codec(tmp_path, monkeypatch, model, infer, selection_only, graded_selection):
    from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
    from app.agents.provider import capture_usage
    from app.workflows.lc_bundle_workbench import replay
    from test_lc_bundle_transport import config, inputs, patch_http
    settings = config(tmp_path / "runtime", source_bundle_local_rank_enabled=True, source_bundle_selection_only=selection_only,
        source_bundle_graded_selection=graded_selection,
        source_bundle_local_rank_manifest=model[0], source_bundle_local_rank_manifest_sha256=model[1])
    adapter = LCBundleWorkbenchProvider(settings)
    adapter.set_previous_usage([])
    observed = patch_http(monkeypatch, adapter)
    raw, question = inputs()
    with capture_usage() as usage:
        response = await adapter.select_source_bundles(question, raw["documents"], raw["paragraphs"])
    assert len(observed) == len(usage) == 1  # explicit local HTTP MockTransport
    assert observed[0]["max_tokens"] == (2048 if graded_selection else 1024 if selection_only else 4096)
    assert len(infer) == 1  # synthetic inference injected above; not real ONNX
    artifact, scope = replay(question, raw["paragraphs"], raw["documents"], response, [])
    assert artifact == response["artifact"]
    if selection_only:
        assert artifact["model_generated_annotations"] is False
        assert artifact["annotation_policy"] == "program_literal_lead_sentence_not_abstractive_summary"
    if graded_selection:
        assert artifact["graded_selection"] is True and artifact["use_type_is_final_relation"] is False
        assert response["proof"]["selection_output_tokens"] == 2048
        modified = deepcopy(response)
        modified["proof"]["graded_selection_policy"]["maximum_selected_anchors"] = 9
        with pytest.raises(ValueError, match="policy_changed"):
            replay(question, raw["paragraphs"], raw["documents"], modified, [])
    assert scope["corpus_paragraph_count"] == len(raw["paragraphs"])
    assert response["proof"]["local_parent_rank"]["config"] == settings.source_bundle_config["local_parent_rank"]
    altered = deepcopy(response)
    altered["proof"]["local_parent_rank"]["config"]["model_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        replay(question, raw["paragraphs"], raw["documents"], altered, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("graded_selection,compact_analyses", [(False, False), (True, False), (True, True)])
async def test_ids_only_worker_keeps_three_analysts_and_publishes_literal_fields(tmp_path, monkeypatch, graded_selection, compact_analyses, issue_frame=False):
    from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
    from app.data.ingest import import_topic
    from app.repository import Repository
    from app.worker import Worker
    from test_lc_bundle_transport import config, patch_http
    from test_source_bundle_tool_bridge import raw_case
    raw = raw_case()
    settings = config(tmp_path / "runtime", source_bundle_selection_only=True, source_bundle_graded_selection=graded_selection,
        source_bundle_compact_analyses_enabled=compact_analyses, source_bundle_issue_frame_enabled=issue_frame)
    repo = Repository(settings)
    doc = raw["documents"][0]
    topic = import_topic(settings.database_path,
        [{"doc_id": doc["raw_doc_id"], "title": doc["title"], "content": doc["content"],
          "source_credibility": .9, "paragraphs": [{"para_id": p["raw_para_id"],
            "char_start": p["document_start_cp"], "char_end": p["document_end_cp"]} for p in raw["paragraphs"]]}],
        [raw["question"]], "显式本地HTTP编号选证发布测试")["topic_id"]
    question = repo.questions(topic)[0]
    adapter = LCBundleWorkbenchProvider(settings)
    requests = patch_http(monkeypatch, adapter)
    queued = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "local-mock-ids-only", model_available=True)
    await Worker(repo, adapter).execute(repo.claim())
    run = repo.run(queued["run_id"])
    assert run["status"] == "succeeded", run["error"]
    assert len(requests) == 4 and requests[0]["max_tokens"] == (2048 if graded_selection else 1024)
    result = repo.snapshot(topic, run["target_revision_id"])
    assert result["evidence"]
    for row in result["evidence"]:
        assert row["fact_des"] in row["quote"]
        assert row["argument"]["内容"] in row["quote"]
        assert row["argument"]["时间"] == "" and row["entities"] == []
    assert run["source_bundle_graded_selection"] is graded_selection
    assert run["source_bundle_compact_analyses"] is compact_analyses
    if compact_analyses:
        from app.workflows.analysis import _project_question
        from scripts.capture_bundle_workbench_real import capture_episode
        from scripts.score_bundle_workbench_real import _tail, _selection_mode, TAIL
        completed = result["question_results"][question["id"]]
        native = capture_episode(adapter.directory)
        front = {"previous_records": [], "new_records": [row for row in completed["usage"] if row["operation"] not in TAIL]}
        audit = _tail(completed, _project_question(question), native, completed["usage"], front,
            {"engineering_episode_stop": 45000, "max_total_requests": 10}, [])
        assert len(audit["analysis_input_projection_audits"]) == 1
        assert audit["analysis_input_projection_audits"][0]["all_prose_and_evidence_refs_preserved"] is True
        assert len(audit["whole_chain_forecast_receipts"]) == 3
        assert all(row["is_actual_usage"] is False for row in audit["whole_chain_forecast_receipts"])
        mode = _selection_mode(completed["source_bundle_proof"]["response"]["proof"], settings.source_bundle_config, run)
        assert mode["front_use_type_is_final_conflict"] is False
        if issue_frame:
            assert completed["evidence_relation_scope"]["format"] == "conclusion-aligned-evidence-relations-v3"
            assert "issue_frame" not in json.loads(requests[-1]["messages"][-1]["content"]).get("analyses", [{}])[0]
            assert "issue_frame" in requests[-1]["tools"][0]["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_issue_frame_worker_publishes_and_replays_anchored_directional_roles(tmp_path, monkeypatch):
    await test_ids_only_worker_keeps_three_analysts_and_publishes_literal_fields(
        tmp_path, monkeypatch, True, True, issue_frame=True)


def test_budget_stop_disables_only_an_impossible_same_budget_resume(tmp_path):
    from types import SimpleNamespace
    from app.repository import Repository, dumps
    repo = object.__new__(Repository)
    repo.one = lambda sql, *a, **k: ({"head_revision_id": "r", "source_manifest_sha256": "src"}
        if "FROM topics" in sql else {"input_manifest_sha256": "src"})
    repo.settings = SimpleNamespace(checkpoint_path=tmp_path / "missing.sqlite3")
    config = {"strict_evidence_token_stop": 45000}
    repo.config = lambda: config
    repo.checkpoint_sources = lambda *a: {"q": "run"}
    row = {"id": "run", "status": "failed", "topic_id": "t", "target_revision_id": "r",
        "model_config_json": dumps(config), "question_ids_json": dumps(["q"]),
        "error_json": dumps({"code": "experiment_usage_stop", "details": {
            "known_tokens": 38901, "input_estimate": 11858, "last_max_output_tokens": 8192}})}
    reason = repo._resume_blocked_reason(row)
    assert "6,099" in reason and "20,562" in reason and "不能续跑" in reason
    row["error_json"] = dumps({"code": "experiment_usage_stop", "details": {
        "known_tokens": 3000, "input_estimate": 1000, "last_max_output_tokens": 8192}})
    assert repo._resume_blocked_reason(row) is None
