"""Opt-in full-parent/field integration with in-process synthetic HTTP only.

No provider credentials, real models, sample answers or quality claims. The
worker, immutable proof replay and formal export are production components.
"""
from copy import deepcopy
from hashlib import sha256
import json

import httpx
import pytest

from app.agents import strict_transport
from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
from app.config import Settings
from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL
from app.repository import Repository
from app.worker import Worker
from app.workflows.lc_bundle_workbench import replay
from test_context_workbench import _import
from test_lc_context_case import raw_case
from test_lc_bundle_transport import config
from test_lc_local_rank_integration import model, infer  # noqa: F401 - synthetic local inference


def settings_for(path, model, *, local_rank=True, compact=False):
    rank = ({"source_bundle_local_rank_enabled": True,
        "source_bundle_local_rank_manifest": model[0],
        "source_bundle_local_rank_manifest_sha256": model[1]} if local_rank else {})
    return config(path, **rank,
        source_bundle_compact_analyses_enabled=True, source_bundle_issue_frame_enabled=True,
        source_bundle_paragraph_quality_enabled=True,
        source_bundle_paragraph_quality_compact_enabled=compact).model_copy(update={
            "strict_evidence_max_requests": 4, "strict_evidence_token_stop": 80000})


def patch_quality_http(monkeypatch, adapter, *, bad=None):
    from synthetic_planning import patch_planning
    patch_planning(monkeypatch)
    observed = []
    async def handle(request):
        body = json.loads(request.content)
        observed.append(body)
        tool = body["tools"][0]["function"]["name"]
        payload = json.loads(body["messages"][-1]["content"])
        if tool in {"LCParagraphQualityV1", "LCParagraphQualityV2", "LCParagraphQualityV3", "LCParagraphQualityV4", "LCParagraphQualityV5"}:
            prepared = json.loads((adapter.bundle_directory / "prepared-case.json").read_text("utf8"))
            decisions, annotations = {}, []
            for row in prepared["catalog"]:
                quote = row["source_bundle"]["quote"]
                argument = {k: [] for k in ("时间", "主体", "地点", "内容", "影响", "数值口径")}
                argument["内容"] = "whole_parent" if tool != "LCParagraphQualityV1" else [{"text": quote, "start_cp": None}]
                if tool in {"LCParagraphQualityV3", "LCParagraphQualityV4", "LCParagraphQualityV5"}:
                    aliases = dict(zip(("时间", "主体", "地点", "内容", "影响", "数值口径"),
                        ("time", "subject", "location", "content", "impact", "numeric_scope")))
                    argument = {aliases[key]: value for key, value in argument.items()}
                decisions[row["source_window_id"]] = True
                annotations.append({"source_window_id": row["source_window_id"],
                    "annotation": {"fact_des": quote[:30] if tool != "LCParagraphQualityV1" else quote,
                        "argument": argument, "entities": []}})
            result = {"source_catalog_id": prepared["source_catalog_id"], "decisions": decisions, "annotations": annotations}
            if bad == "missing": result["decisions"].pop(next(iter(decisions)))
            if bad == "invented_time": annotations[0]["annotation"]["argument"]["时间"] = [{"text": "2099年", "start_cp": None}]
            if bad == "optional_metadata":
                annotations[0]["annotation"]["entities"] = ["F-16"]
                annotations[0]["annotation"]["argument"]["time"] = ["2099年"]
            if bad == "required_content": annotations[0]["annotation"]["argument"]["content"] = ["原文不存在的内容。"]
            if bad == "disjoint_time": annotations[0]["annotation"]["argument"]["time"] = ["10:58", "下午时段"]
        else:
            ids = [row["id"] for row in payload["evidence"]]
            result = {"decision": "undetermined", "conclusion": "原始记录未能确定最终结果。",
                "judgments": [{"id": "J1", "statement": "原始记录保留未决状态。",
                    "evidence_ids": ids, "prerequisite_judgment_ids": []}],
                "supporting_evidence_ids": ids, "conflicting_evidence_ids": []}
            if tool == "ScopedArbitrationWireOutput":
                result.pop("supporting_evidence_ids"); result.pop("conflicting_evidence_ids")
                result.update(issue_frame={"question_anchor": payload["question"]["question"],
                    "claim_kind": "official_finding", "scope": "仅当前原始记录。",
                    "alternatives": {"X": "", "Y": ""}, "alternatives_relation": "not_binary"},
                    relations={ref: {"assessment": "QS_procedure", "side": "neither"} for ref in ids}, conflicts=[])
        response = {"id": "synthetic-quality-only", "object": "chat.completion", "created": 1,
            "model": "deepseek-v4-pro", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "tool_calls": [{"id": "mock-quality-call", "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(result, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
        return httpx.Response(200, json=response)
    monkeypatch.setattr(strict_transport.httpx, "AsyncHTTPTransport", lambda **_kw: httpx.MockTransport(handle))
    class MockAudit(strict_transport.AuditedTransport, httpx.MockTransport):
        pass
    monkeypatch.setattr(strict_transport, "AuditedTransport", MockAudit)
    monkeypatch.setattr(strict_transport, "count_wire", lambda body, directory, ordinal: (100,
        strict_transport.freeze(directory / f"request-{ordinal:02d}-wire.json", body)))
    return observed


def test_quality_admission_is_distinct_and_old_defaults_remain():
    assert Settings(_env_file=None).source_bundle_paragraph_quality_enabled is False
    with pytest.raises(ValueError, match="compact LangChain"):
        Settings(_env_file=None, source_bundle_paragraph_quality_enabled=True)
    values = {"_env_file": None, "model_source_bundle_enabled": True,
        "source_bundle_front_framework": "langchain", "source_bundle_compact_analyses_enabled": True,
        "source_bundle_paragraph_quality_enabled": True}
    settings = Settings(**values)
    assert settings.source_bundle_config["paragraph_quality"] is True
    assert settings.source_bundle_local_rank_enabled is False  # base search40 is an explicit supported source book
    assert settings.source_bundle_config["selection_output_tokens"] == 8192
    assert "graded_selection" not in settings.source_bundle_config
    with pytest.raises(ValueError, match="mutually exclusive"):
        Settings(**values, source_bundle_selection_only=True)
    for feature in ("source_bundle_graded_selection", "source_bundle_context_passages_enabled",
                    "source_bundle_complete_grading_enabled", "source_bundle_hybrid_retrieval_enabled"):
        with pytest.raises(ValueError):
            Settings(**values, **{feature: True})


def test_quality_archived_cli_allows_explicit_false_legacy_flags_without_changing_protocol():
    from scripts.lc_bundle_case import execute
    raw = json.dumps(raw_case(["原始记录尚待核验。"]), ensure_ascii=False).encode("utf8")
    request = {"operation": "prepare", "paragraph_quality": True}
    expected = execute(raw, sha256(raw).hexdigest(), request)
    actual = execute(raw, sha256(raw).hexdigest(), {**request,
        "selection_only": False, "graded_selection": False, "context_passages": False,
        "complete_grading": False, "hybrid_retrieval": False})
    assert actual == expected and actual["model_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("local_rank", [False, True], ids=["base-search40", "verified-local-rank32"])
async def test_full_parent_fields_survive_worker_publication_export_and_replay(tmp_path, monkeypatch, model, infer, local_rank):
    raw = raw_case([f"第{i}条原始记录仍未最终确认。" for i in range(12)])
    settings = settings_for(tmp_path / "runtime", model, local_rank=local_rank)
    repo = Repository(settings)
    topic = _import(repo, raw)
    question = repo.questions(topic)[0]
    adapter = LCBundleWorkbenchProvider(settings)
    observed = patch_quality_http(monkeypatch, adapter)
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "quality-synthetic-worker", model_available=True)
    await Worker(repo, adapter).execute(repo.claim())
    run = repo.run(task["run_id"])
    assert run["status"] == "succeeded", run["error"]
    assert run["source_bundle_paragraph_quality"] is True and run["source_bundle_context_passages"] is False
    assert len(observed) == len(run["usage"][question["id"]]) == 4
    assert observed[0]["max_tokens"] == 8192
    snapshot = repo.snapshot(topic, run["target_revision_id"])
    result = snapshot["question_results"][question["id"]]
    assert result["source_bundle_protocol"] == SOURCE_BUNDLE_PROTOCOL
    assert len(result["evidence"]) == 12  # no hidden top-ten truncation
    parents = {p["id"]: p for p in repo.paragraphs(topic)}
    for row in result["evidence"]:
        assert row["quote"] == parents[row["paragraph_id"]]["text"]
        assert row["fact_des"] == row["argument"]["内容"] == row["quote"]
        assert row["argument"]["时间"] == "" and row["entities"] == []
    front = json.loads(observed[0]["messages"][-1]["content"])
    assert "source_runs" in front and "source_contexts" not in front
    for body in observed[1:]:
        assert len(json.loads(body["messages"][-1]["content"])["evidence"]) == 12
    response = result["source_bundle_proof"]["response"]
    assert ("local_parent_rank" in response["proof"]) is local_rank
    artifact, scope = replay(question, repo.paragraphs(topic), repo.source_documents(topic), response, [])
    assert artifact == response["artifact"] and len(scope["returned_complete_paragraph_ids"]) == 12
    files = repo.formal_submission_files(topic, run["target_revision_id"])
    assert len(files["evidence.json"]) == 12
    assert sum(row["total_tokens"] for row in run["usage"][question["id"]]) == 480  # mock only
    changed = deepcopy(response)
    changed["proof"]["paragraph_quality_policy"]["changed"] = True
    with pytest.raises(ValueError):
        replay(question, repo.paragraphs(topic), repo.source_documents(topic), changed, [])
    assert len(observed) == 4  # offline replay never invokes a model


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["missing", "invented_time"])
async def test_field_structure_or_grounding_failure_preserves_one_cost_and_no_publication(tmp_path, monkeypatch, model, infer, bad):
    settings = settings_for(tmp_path / "runtime", model)
    repo = Repository(settings)
    raw = raw_case(["原始记录尚未确认。", "另一条记录仍待核验。"])
    topic = _import(repo, raw)
    question = repo.questions(topic)[0]
    adapter = LCBundleWorkbenchProvider(settings)
    observed = patch_quality_http(monkeypatch, adapter, bad=bad)
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "quality-synthetic-" + bad, model_available=True)
    await Worker(repo, adapter).execute(repo.claim())
    run = repo.run(task["run_id"])
    assert run["status"] == "failed"
    assert len(observed) == len(run["usage"][question["id"]]) == 1
    assert sum(r["total_tokens"] for r in run["usage"][question["id"]]) == 120  # mock only
    assert repo.topic(topic)["published_revision_id"] is None
    assert adapter._tail_seeded is False
