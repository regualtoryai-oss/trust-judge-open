"""Compact protocol integration uses synthetic HTTP only, never paid inference."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
from app.config import Settings
from app.repository import Repository
from app.worker import Worker
from app.workflows.lc_bundle_workbench import replay
from scripts.lc_bundle_case import execute
from test_context_workbench import _import
from test_lc_context_case import raw_case
from test_lc_local_rank_integration import model, infer  # noqa: F401
from test_paragraph_quality_workbench import settings_for, patch_quality_http


def test_compact_is_explicit_and_archived_v1_is_not_reinterpreted():
    with pytest.raises(ValueError, match="explicit paragraph quality"):
        Settings(_env_file=None, source_bundle_paragraph_quality_compact_enabled=True)
    raw = json.dumps(raw_case(["原始记录仍待核验。"]), ensure_ascii=False).encode("utf8")
    checksum = sha256(raw).hexdigest()
    v1 = execute(raw, checksum, {"operation": "prepare", "paragraph_quality": True})["result"]
    v2 = execute(raw, checksum, {"operation": "prepare", "paragraph_quality": True,
        "paragraph_quality_compact": True})["result"]
    assert v1["function_name"] == "LCParagraphQualityV1"
    assert v2["function_name"] == "LCParagraphQualityV4"
    assert v1["source_catalog_id"] != v2["source_catalog_id"]
    assert "paragraph_quality_compact" not in v1
    with pytest.raises(ValueError, match="compact_mode_required"):
        execute(raw, checksum, {"operation": "prepare", "paragraph_quality_compact": True})


@pytest.mark.asyncio
async def test_32_complete_parent_fields_publish_export_and_replay_without_long_wire_copy(tmp_path, monkeypatch, model, infer):
    # Deliberately larger than the former 12-row integration case. All complete
    # input parents remain in the model input; only redundant output is removed.
    raw = raw_case([f"第{i}条原始记录仍未最终确认。" + "这是完整原始段落的程序背景和限定条件。" * 8 for i in range(32)])
    settings = settings_for(tmp_path / "runtime", model, compact=True)
    repo = Repository(settings)
    topic = _import(repo, raw)
    question = repo.questions(topic)[0]
    provider = LCBundleWorkbenchProvider(settings)
    observed = patch_quality_http(monkeypatch, provider)
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "compact-worker-32", model_available=True)
    await Worker(repo, provider).execute(repo.claim())
    run = repo.run(task["run_id"])
    assert run["status"] == "succeeded", run["error"]
    assert len(observed) == 4 and observed[0]["max_tokens"] == 8192
    assert run["source_bundle_paragraph_quality_compact"] is True
    result = repo.snapshot(topic, run["target_revision_id"])["question_results"][question["id"]]
    assert len(result["evidence"]) == 32
    assert all(row["argument"]["内容"] == row["quote"] for row in result["evidence"])
    response = result["source_bundle_proof"]["response"]
    native = json.loads((provider.bundle_directory / "model-output.json").read_text("utf8"))
    assert all(row["annotation"]["argument"]["content"] == "whole_parent" for row in native["annotations"])
    verbose = deepcopy(native)
    quotes = {row["source_window_id"]: row["quote"] for row in response["artifact"]["evidence"]}
    for row in verbose["annotations"]:
        row["annotation"]["argument"]["content"] = [{"text": quotes[row["source_window_id"]], "start_cp": None}]
    assert len(json.dumps(native, ensure_ascii=False)) < len(json.dumps(verbose, ensure_ascii=False))
    compiled, scope = replay(question, repo.paragraphs(topic), repo.source_documents(topic), response, [])
    assert compiled == response["artifact"] and len(scope["returned_complete_paragraph_ids"]) == 32
    assert len(repo.formal_submission_files(topic, run["target_revision_id"])["evidence.json"]) == 32
    altered = deepcopy(response)
    altered["proof"].pop("paragraph_quality_compact")
    altered["proof"].pop("paragraph_quality_compact_policy")
    with pytest.raises(ValueError):
        replay(question, repo.paragraphs(topic), repo.source_documents(topic), altered, [])
    assert len(observed) == 4 and sum(row["total_tokens"] for row in run["usage"][question["id"]]) == 480


@pytest.mark.asyncio
async def test_pause_after_completed_front_reuses_checkpoint_and_keeps_full_cost_lineage(tmp_path, monkeypatch, model, infer):
    settings = settings_for(tmp_path / "runtime", model, compact=True).model_copy(update={"strict_evidence_max_requests": 5})
    repo = Repository(settings)
    raw = raw_case([f"第{i}条原始记录仍待确认。" for i in range(16)])
    topic = _import(repo, raw)
    question = repo.questions(topic)[0]
    first = LCBundleWorkbenchProvider(settings)
    first_http = patch_quality_http(monkeypatch, first)
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "compact-paused-first", model_available=True)
    original_event = repo.event
    def pause_after_source_checkpoint(run_id, kind, data):
        original_event(run_id, kind, data)
        # The next graph node can only start after the source superstep has
        # committed. Pausing inside select() is too early: its completed HTTP
        # response is not yet a graph checkpoint and may require fresh payment.
        if run_id == task["run_id"] and kind == "node_started" and data.get("node") == "freeze":
            repo.pause(run_id)
    monkeypatch.setattr(repo, "event", pause_after_source_checkpoint)
    await Worker(repo, first).execute(repo.claim())
    paused = repo.run(task["run_id"])
    assert paused["view_status"] == "paused", paused
    assert len(first_http) == 1 and len(paused["usage"][question["id"]]) == 1
    assert repo.topic(topic)["published_revision_id"] is None
    resumed = repo.resume(task["run_id"], "compact-paused-resume", model_available=True)
    second = LCBundleWorkbenchProvider(settings)
    second_http = patch_quality_http(monkeypatch, second)
    await Worker(repo, second).execute(repo.claim())
    final = repo.run(resumed["run_id"])
    assert final["status"] == "succeeded", final["error"]
    assert len(second_http) == 3 and second.front_provider is None
    assert second.prior_known_tokens == 120
    assert len(final["usage"][question["id"]]) == 3
    assert sum(row["total_tokens"] for row in final["usage"][question["id"]]) == 360
    lineage = [row for row in repo.events(resumed["run_id"]) if row["type"] == "cost_lineage_seeded"]
    assert lineage[0]["data"]["prior_known_tokens"] == 120
    result = repo.snapshot(topic, final["target_revision_id"])["question_results"][question["id"]]
    assert "source_bundle_selection" in result["execution_provenance"]["reused_nodes"]
    assert len(repo.formal_submission_files(topic, final["target_revision_id"])["evidence.json"]) == 16
    assert len(first_http) + len(second_http) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["optional_metadata", "required_content"])
async def test_optional_metadata_is_quarantined_but_required_content_still_fails(tmp_path, monkeypatch, model, bad):
    settings = settings_for(tmp_path / "runtime", model, compact=True, local_rank=False)
    repo = Repository(settings)
    topic = _import(repo, raw_case(["原始记录尚未最终确认。", "另一条程序记录仍待核验。 "]))
    question = repo.questions(topic)[0]
    provider = LCBundleWorkbenchProvider(settings)
    observed = patch_quality_http(monkeypatch, provider, bad=bad)
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "compact-invalid-" + bad, model_available=True)
    await Worker(repo, provider).execute(repo.claim())
    run = repo.run(task["run_id"])
    if bad == "required_content":
        assert run["status"] == "failed" and len(observed) == 1
        assert repo.topic(topic)["published_revision_id"] is None
        assert run["usage"][question["id"]][0]["total_tokens"] == 120
        return
    assert run["status"] == "succeeded", run["error"]
    assert len(observed) == 4
    result = repo.snapshot(topic, run["target_revision_id"])["question_results"][question["id"]]
    artifact = result["source_bundle_proof"]["response"]["artifact"]
    audit = artifact["field_extraction_audit"]["optional_field_quarantine"]
    assert audit["quarantined_value_count"] == 2 and audit["affected_annotation_count"] == 1
    assert audit["raw_model_output_modified"] is False and audit["required_fields_quarantined"] is False
    assert artifact["raw_model_selection"]["annotations"][0]["annotation"]["entities"] == ["F-16"]
    assert len(result["evidence"]) == 2 and all("F-16" not in row["entities"] for row in result["evidence"])
    files = repo.formal_submission_files(topic, run["target_revision_id"])
    assert len(files["evidence.json"]) == 2 and all("F-16" not in row["entities"] for row in files["evidence.json"])
    summaries = [row["summary"] for row in repo.activity(task["run_id"])["tool_calls"]]
    assert any("2 项可选字段原文不符已隔离" in value for value in summaries)


@pytest.mark.asyncio
async def test_v4_front_incremental_budget_keeps_paid_ancestors_and_replays(tmp_path, monkeypatch, model):
    settings = settings_for(tmp_path / "runtime", model, compact=True, local_rank=False).model_copy(update={
        "source_bundle_front_token_stop": 20000, "source_bundle_tail_reserved_tokens": 60000,
        "strict_evidence_token_stop": 110000, "strict_evidence_max_requests": 6})
    repo = Repository(settings)
    topic = _import(repo, raw_case(["原始记录尚未最终确认。", "另一条程序记录仍待核验。 "]))
    question = repo.questions(topic)[0]
    provider = LCBundleWorkbenchProvider(settings)
    prior = [{"operation": "synthetic_failed_parent", "model": "deepseek-v4-pro", "usage_source": "provider",
        "input_tokens": 10010, "output_tokens": 2323, "total_tokens": 12333},
        {"operation": "synthetic_failed_parent", "model": "deepseek-v4-pro", "usage_source": "provider",
        "input_tokens": 10000, "output_tokens": 1865, "total_tokens": 11865}]
    provider.set_previous_usage(prior)
    observed = patch_quality_http(monkeypatch, provider)
    response = await provider.select_source_bundles(question, repo.source_documents(topic), repo.paragraphs(topic))
    assert len(observed) == 1
    launch = json.loads((provider.bundle_directory / "launch-manifest.json").read_text("utf8"))
    assert launch["parent_tokens"] == 24198 and launch["parent_request_count"] == 2
    assert launch["configured_new_front_token_budget"] == 20000
    assert launch["front_token_stop"] == launch["cumulative_front_guard_threshold"] == 44198
    assert response["usage_report"]["whole_episode_tokens"] == 24318
    rebuilt, scope = replay(question, repo.paragraphs(topic), repo.source_documents(topic), response, [])
    assert rebuilt == response["artifact"] and len(scope["returned_complete_paragraph_ids"]) == 2
