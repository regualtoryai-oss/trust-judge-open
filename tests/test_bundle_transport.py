"""Official native SDK fixture plus HTTP-mocked strict tail; no paid traffic."""
from copy import deepcopy
import asyncio
from hashlib import sha256
import json
from pathlib import Path

import httpx
import pytest

from app.agents.bundle_transport import BundleWorkbenchProvider
from app.agents.provider import ModelOutputInvalid, capture_usage
from app.agents import strict_transport
from app.config import Settings
from app.workflows.bundle_workbench import compile_result
from app.workflows.analysis import run_analysis
from test_source_bundle_tool_bridge import raw_case, annotation


def settings(path, **changes):
    return Settings(_env_file=None, data_dir=path, model_provider="deepseek", model_name="deepseek-v4-pro",
        model_api_key="explicit-local-test-not-real", model_source_bundle_enabled=True,
        strict_evidence_max_requests=10, strict_evidence_token_stop=30000, **changes)


def fixture_plan(path, case, *, incomplete=False):
    selected = annotation(case)
    first = case["paragraphs"][0]["id"]
    plan = [
        {"name": "read_source_bundles", "arguments": {"anchor_paragraph_ids": [first], "before": 0, "after": 1}},
    ]
    if not incomplete:
        plan.append({"name": "submit_bundle_evidence", "arguments": {"selected": [selected]}})
    path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return path


def patch_tail(monkeypatch):
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        name = body["tools"][0]["function"]["name"]
        if name == "ScopedArbitrationWireOutput":
            result = {"principal_claim": "公开记录只表明项目申请已受理而尚未批准。", "relations": {"E1": "qualification"},
                "decision": "undetermined", "conclusion": "公开记录只表明项目申请已受理而尚未批准。", "judgments": [{
                    "id": "J1", "statement": "公开记录保留项目尚未批准的限定。", "evidence_ids": ["E1"], "prerequisite_judgment_ids": []}], "conflicts": []}
        else:
            result = {"decision": "undetermined", "conclusion": "公开记录尚未确认项目批准。", "judgments": [{
                "id": "J1", "statement": "公开记录保留项目尚未批准的限定。", "evidence_ids": ["E1"], "prerequisite_judgment_ids": []}],
                "supporting_evidence_ids": ["E1"], "conflicting_evidence_ids": []}
        return httpx.Response(200, json={"id": "fixture-tail", "object": "chat.completion", "created": 1, "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tail-call", "type": "function", "function": {"name": name, "arguments": json.dumps(result, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
    monkeypatch.setattr(strict_transport.httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(respond))
    class MockAudit(strict_transport.AuditedTransport, httpx.MockTransport):
        pass
    monkeypatch.setattr(strict_transport, "AuditedTransport", MockAudit)
    monkeypatch.setattr(strict_transport, "count_wire", lambda body, directory, ordinal: (100,
        strict_transport.freeze(directory / f"request-{ordinal:02d}-wire.json", body)))
    return calls


@pytest.mark.optional_dsh_runtime
async def test_official_source_loop_compiles_real_native_proof_and_reuses_normal_three_call_tail(tmp_path, monkeypatch):
    case = raw_case()
    for row in case["paragraphs"]:
        row.update(title=case["documents"][0]["title"], source_credibility_json="0.9")
    plan = fixture_plan(tmp_path / "plan.json", case)
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime"), fixture_plan=plan)
    calls = patch_tail(monkeypatch)
    events = []
    adapter.workbench_event = lambda event, data: events.append((event, data))
    question = {"id": "internal-question", "raw_question_id": case["question"]["question_id"], "question": case["question"]["question"]}
    parent = {"operation": "previous_failure", "usage_source": "provider", "input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
    adapter.set_previous_usage([parent])
    result = await run_analysis(question, case["paragraphs"], adapter, source_documents=case["documents"],
        checkpoint_path=tmp_path / "checkpoints.sqlite", thread_id="actual-native-fixture-graph",
        cancel_check=lambda: False, emit=lambda event, data: events.append((event, data)))
    response = json.loads((adapter.bundle_directory / "provider-response.json").read_text("utf-8"))
    usage, final, evidence = result["usage"], result["final"], result["evidence"]
    assert response["usage_report"]["new_request_count"] == 2
    assert response["usage_report"]["new_tokens"] == 300
    assert response["usage_report"]["parent_tokens"] == 100
    assert adapter.prior_known_tokens == 400
    assert compile_result(question, case["paragraphs"], case["documents"], response)["evidence"] == evidence
    assert len(calls) == 3
    assert len(usage) == 5
    assert sum(row["total_tokens"] for row in usage) == 660  # Fixture usage only, no real account use.
    assert adapter.guard.report()["actual_requests_started"] == 3
    assert adapter.guard.report()["known_total_tokens"] == 360
    assert adapter.guard.policy.max_requests == 7  # 10 minus one parent and two actual DSH requests.
    assert final["supporting_evidence_ids"] == [evidence[0]["id"]]
    assert [name for name, _ in events].count("source_bundle_native_usage_settled") == 2
    assert any(name == "source_bundle_tool_started" for name, _ in events)
    actual_tools = [data for event, data in events if event == "tool_started" and data.get("engine") == "official-dsh"]
    assert {data["name"] for data in actual_tools} == {"source_bundle_select", "read_source_bundles", "submit_bundle_evidence"}
    assert len(actual_tools) == 4 and len({data["call_id"] for data in actual_tools}) == 4
    assert all("question_id" not in data for event, data in events if event == "source_bundle_scope_prepared")
    prefetched = [data for event, data in events if event == "tool_completed" and data.get("name") == "search_source_bundles"]
    assert len(prefetched) == 1 and prefetched[0]["model_calls"] == 0
    assert prefetched[0]["completed_at"] and prefetched[0]["navigation_archive"]["sha256"]
    assert json.loads((adapter.directory / "episode.json").read_text())["balanced_scoped_enabled"] is True
    metadata = json.loads((adapter.directory / "source-bundle-episode.json").read_text())
    assert metadata["base_episode_scope"] == "inherited_strict_tail_only"
    assert "app/workflows/bundle_workbench.py" in metadata["source_archives"]
    for descriptor in metadata["source_archives"].values():
        assert sha256(Path(descriptor["file"]).read_bytes()).hexdigest() == descriptor["sha256"]
    assert response["usage_report"]["real_api_requests"] == 0
    assert not response["artifact"]["publishable"]
    nav_descriptor = response["proof"]["program_navigation"]
    navigation = json.loads(Path(nav_descriptor["file"]).read_text("utf8"))
    assert sha256(Path(nav_descriptor["file"]).read_bytes()).hexdigest() == nav_descriptor["sha256"]
    assert navigation["model_calls"] == 0 and navigation["candidate_count"] <= 40
    assert navigation["backend_result"]["results"] and navigation["initial_view"]
    ledger = [json.loads(line) for line in Path(response["proof"]["usage_ledger"]["file"]).read_text().splitlines() if line.strip()]
    starts = [row for row in ledger if row["status"] == "request_started"]
    bodies = [json.loads(Path(row["file"]).read_text()) for row in response["proof"]["native_requests"]]
    assert [row["reserved_output_tokens"] for row in starts] == [body["max_tokens"] for body in bodies] == [1024, 4096]
    assert adapter.source_bundle_config["native_max_output_tokens"] == 4096  # Upper bound, never the per-wire reservation.
    assert "program-original-question-rrf40-v1" in bodies[0]["system"]
    assert "program-original-question-rrf40-v1" not in json.dumps(bodies[1])
    assert '"preview_text_withdrawn":true' in bodies[1]["system"]
    assert all("search_source_bundles" not in [row["name"] for row in body["tools"]] for body in bodies)


@pytest.mark.parametrize("repeat_same_arguments", [False, True])
@pytest.mark.optional_dsh_runtime
async def test_program_navigation_executes_two_same_step_reads_and_denies_extra_without_codec_call(tmp_path, repeat_same_arguments):
    case = raw_case()
    first, last = case["paragraphs"][0]["id"], case["paragraphs"][2]["id"]
    read_first = {"name": "read_source_bundles", "arguments": {"anchor_paragraph_ids": [first], "before": 0, "after": 1}}
    read_last = {"name": "read_source_bundles", "arguments": {"anchor_paragraph_ids": [last], "before": 0, "after": 0}}
    if repeat_same_arguments:
        read_last = deepcopy(read_first)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"tool_calls": [read_first, read_last, read_last]},
        {"name": "submit_bundle_evidence", "arguments": {"selected": [annotation(case)]}}], ensure_ascii=False), encoding="utf8")
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime"), fixture_plan=plan)
    events = []
    adapter.workbench_event = lambda event, data: events.append((event, data))
    question = {"id": "internal-question", "raw_question_id": case["question"]["question_id"], "question": case["question"]["question"]}
    with capture_usage() as usage:
        response = await adapter.select_source_bundles(question, case["documents"], case["paragraphs"])
    compile_result(question, case["paragraphs"], case["documents"], response)
    assert len(usage) == 2 and response["usage_report"]["real_api_requests"] == 0
    request = json.loads(Path(response["proof"]["submit_request"]["file"]).read_text("utf8"))
    history = request["history"]
    assert [row["operation"] for row in history] == ["search", "read", "read"]
    assert history[0]["arguments"] == {"query": case["question"]["question"], "limit": 40, "exclude_paragraph_ids": []}
    assert all(row["status"] == "ok" for row in history)
    bodies = [json.loads(Path(row["file"]).read_text("utf8")) for row in response["proof"]["native_requests"]]
    assert [len(json.loads(line)["read_receipts"]) for line in Path(response["proof"]["presence"]["file"]).read_text("utf8").splitlines()] == [0, 2]
    results = [item for message in bodies[-1]["messages"] for item in message.get("content", []) if item.get("type") == "tool_result"]
    denied = [item for item in results if item.get("is_error")]
    assert len(denied) == 1 and all("NOT_READ" in json.dumps(item) for item in denied)
    deferred = [data for event, data in events if event == "tool_deferred"]
    assert len(deferred) == 1 and deferred[0]["call_id"].endswith(":dsh-tool:" + denied[0]["tool_use_id"])
    assert deferred[0]["name"] == "read_source_bundles" and "未读取" in deferred[0]["summary"]
    assert not any(event in {"tool_failed", "tool_completed"} and data.get("call_id") == deferred[0]["call_id"]
        for event, data in events)
    assert all("UNKNOWN_TOOL" not in json.dumps(item) for item in results)
    assert "program-original-question-rrf40-v1" not in json.dumps(bodies[1])
    assert [row["name"] for row in bodies[0]["tools"]] == ["read_source_bundles"]
    assert [row["name"] for row in bodies[-1]["tools"]] == ["submit_bundle_evidence"]


@pytest.mark.optional_dsh_runtime
async def test_program_navigation_unknown_anchor_never_consumes_python_read_allowance(tmp_path):
    case = raw_case()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"name": "read_source_bundles", "arguments": {
        "anchor_paragraph_ids": ["not-in-visible-candidate-set"], "before": 0, "after": 0}},
        {"text_only": True, "text": "显式合成终止，不提交任何证据。"}]), encoding="utf8")
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime"), fixture_plan=plan)
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid):
            await adapter.select_source_bundles(case["question"], case["documents"], case["paragraphs"])
    assert usage and len(usage) == 2 and all(row["usage_source"] == "provider" for row in usage)  # Fixture usage only.
    transcript = [json.loads(line) for line in (adapter.bundle_directory / "native/business-transcript.jsonl").read_text().splitlines()]
    assert [row["operation"] for row in transcript] == ["search"]
    assert not (adapter.bundle_directory / "native/artifact.json").exists()


@pytest.mark.parametrize("invalid_anchor", [False, True])
@pytest.mark.optional_dsh_runtime
async def test_actual_official_sdk_provider_through_worker_publishes_only_complete_verified_fixture(tmp_path, monkeypatch, invalid_anchor):
    from app.repository import Repository
    from app.data.ingest import import_topic
    from app.worker import Worker

    raw = raw_case()
    config = settings(tmp_path / "runtime")
    repo = Repository(config)
    doc = raw["documents"][0]
    imported = import_topic(config.database_path, [{"doc_id": doc["raw_doc_id"], "title": doc["title"],
        "content": doc["content"], "source_credibility": .9, "paragraphs": [{"para_id": p["raw_para_id"],
            "char_start": p["document_start_cp"], "char_end": p["document_end_cp"]} for p in raw["paragraphs"]]}],
        [{"question_id": raw["question"]["question_id"], "question": raw["question"]["question"]}], "显式SDK全链合成数据")
    topic = imported["topic_id"]
    question = repo.questions(topic)[0]
    case = {"question": raw["question"], "documents": repo.source_documents(topic), "paragraphs": repo.paragraphs(topic)}
    plan = tmp_path / "worker-plan.json"
    if invalid_anchor:
        plan.write_text(json.dumps([{"name": "read_source_bundles", "arguments": {"anchor_paragraph_ids": ["not-visible"],
            "before": 0, "after": 0}}, {"text_only": True, "text": "合成终止，未提交。"}]), encoding="utf8")
    else:
        fixture_plan(plan, case)
    provider = BundleWorkbenchProvider(config, fixture_plan=plan)
    tail = patch_tail(monkeypatch)
    queued = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "explicit-actual-sdk-worker-fixture", model_available=True)
    await Worker(repo, provider).execute(repo.claim())
    run = repo.run(queued["run_id"])
    if invalid_anchor:
        assert run["status"] == "failed", run
        assert tail == [] and repo.topic(topic)["published_revision_id"] is None
        assert repo.all("SELECT id FROM export_jobs") == []
    else:
        assert run["status"] == "succeeded", run["error"]
        assert len(tail) == 3 and repo.topic(topic)["published_revision_id"] == run["target_revision_id"]
        assert (provider.bundle_directory / "native/program-navigation.json").exists()
    assert (provider.bundle_directory / "native/closed-receipt.json").exists()
    assert (provider.bundle_directory / "native/native-response-01.sse").exists()


async def test_completed_selection_resume_lazily_seeds_all_parent_calls_before_first_tail(tmp_path, monkeypatch):
    adapter = BundleWorkbenchProvider(settings(tmp_path))
    prior = [{"usage_source": "provider", "input_tokens": 100, "output_tokens": 50, "total_tokens": 150}] * 4
    adapter.set_previous_usage(prior)
    patch_tail(monkeypatch)
    with capture_usage() as usage:
        result = await adapter.analyze("authority_analyst", {"question": "是否批准？"}, [{"id": "source-evidence", "quote": "尚未批准。"}], "snapshot")
    assert adapter.prior_known_tokens == 600
    assert adapter.guard.policy.max_requests == 6
    assert len(usage) == 1 and usage[0]["total_tokens"] == 120
    assert adapter._selection_started is False
    assert result["supporting_evidence_ids"] == ["source-evidence"]
    with pytest.raises(ValueError, match="once_before_calls"):
        adapter.set_previous_usage([])


async def test_no_source_stage_or_parent_cannot_call_tail(tmp_path, monkeypatch):
    adapter = BundleWorkbenchProvider(settings(tmp_path))
    calls = patch_tail(monkeypatch)
    with pytest.raises(ModelOutputInvalid) as error:
        await adapter.analyze("authority_analyst", {"question": "是否批准？"}, [], "snapshot")
    assert error.value.code == "source_bundle_tail_not_admitted"
    assert calls == []


def test_unknown_parent_and_duplicate_set_do_not_reset_budget(tmp_path):
    adapter = BundleWorkbenchProvider(settings(tmp_path))
    with pytest.raises(ModelOutputInvalid) as error:
        adapter.set_previous_usage([{"usage_source": "unavailable", "input_tokens": None, "output_tokens": None, "total_tokens": None}])
    assert error.value.code == "experiment_usage_unknown"
    adapter.set_previous_usage([])
    with pytest.raises(ValueError, match="once_before_calls"):
        adapter.set_previous_usage([])


async def test_native_started_without_known_settlement_is_captured_and_blocks_tail(tmp_path, monkeypatch):
    case = raw_case()
    adapter = BundleWorkbenchProvider(settings(tmp_path))
    async def interrupted(manifest, descriptor):
        native = adapter.bundle_directory / "native"
        native.mkdir()
        (native / "budget.jsonl").write_text(json.dumps({"status": "request_started", "request_sequence": 1}) + "\n")
    monkeypatch.setattr(adapter, "_launch", interrupted)
    question = {"raw_question_id": case["question"]["question_id"], "question": case["question"]["question"]}
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid) as error:
            await adapter.select_source_bundles(question, case["documents"], case["paragraphs"])
    assert error.value.code == "experiment_usage_unknown"
    assert len(usage) == 1 and usage[0]["usage_source"] == "unavailable"
    with pytest.raises(ModelOutputInvalid):
        await adapter.analyze("authority_analyst", question, [], "snapshot")


@pytest.mark.optional_dsh_runtime
async def test_actual_native_preflight_reserves_tail_and_stops_before_provider_when_front_is_too_small(tmp_path):
    case = raw_case()
    plan = fixture_plan(tmp_path / "plan.json", case)
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime", source_bundle_front_token_stop=1000), fixture_plan=plan)
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid) as error:
            await adapter.select_source_bundles(case["question"], case["documents"], case["paragraphs"])
    assert error.value.code == "whole_episode_budget_exceeded"
    assert error.value.details["known_tokens"] == 0
    assert error.value.details["input_estimate"] > 0
    assert error.value.details["last_max_output_tokens"] == 1024
    assert error.value.details["tail_reserved_tokens"] == 18000
    assert usage == []
    native = adapter.bundle_directory / "native"
    receipt = json.loads((native / "closed-receipt.json").read_text())
    assert receipt["new_request_count"] == 0 and receipt["new_tokens"] == 0
    assert receipt["reserved_tail_tokens"] == 18000
    assert not (native / "fixture-native-wire.jsonl").exists()
    assert not adapter._tail_seeded


async def test_known_native_bill_is_retained_even_when_response_failed(tmp_path, monkeypatch):
    case = raw_case()
    adapter = BundleWorkbenchProvider(settings(tmp_path))
    async def failed(manifest, descriptor):
        native = adapter.bundle_directory / "native"
        native.mkdir()
        events = [{"status": "request_started", "request_sequence": 1}, {"status": "usage_settled", "request_sequence": 1,
            "provider_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            "response_completed": False, "halted": True, "unknown_usage_requests": 0}]
        (native / "budget.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
        (native / "closed-receipt.json").write_text(json.dumps({"known_usage_complete": True, "status": "failed_native_response",
            "failure_code": "bundle_workbench_native_response_incomplete"}))
    monkeypatch.setattr(adapter, "_launch", failed)
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid) as error:
            await adapter.select_source_bundles(case["question"], case["documents"], case["paragraphs"])
    assert error.value.code == "bundle_workbench_native_response_incomplete"
    assert usage[0]["operation"] == "source_bundle_select"
    assert usage[0]["total_tokens"] == 120 and usage[0]["usage_source"] == "provider"
    assert not adapter._tail_seeded


@pytest.mark.parametrize("omit_usage", [False, True])
@pytest.mark.optional_dsh_runtime
async def test_actual_native_no_submit_or_unknown_usage_is_failure_and_retains_bill(tmp_path, omit_usage):
    case = raw_case()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"text_only": True, "text": "显式合成未提交响应。", "omit_usage": omit_usage}]), encoding="utf-8")
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime"), fixture_plan=plan)
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid) as error:
            await adapter.select_source_bundles(case["question"], case["documents"], case["paragraphs"])
    assert len(usage) == 1
    assert usage[0]["usage_source"] == ("unavailable" if omit_usage else "provider")
    assert error.value.code == ("experiment_usage_unknown" if omit_usage else "bundle_workbench_submission_missing")
    native = adapter.bundle_directory / "native"
    assert (native / "native-response-01.sse").exists()
    assert not (native / "artifact.json").exists()
    assert not adapter._tail_seeded
    with pytest.raises(ModelOutputInvalid):
        adapter._ensure_tail_seeded()


@pytest.mark.optional_dsh_runtime
async def test_cancel_actual_native_episode_keeps_partial_journal_and_never_admits_tail(tmp_path):
    case = raw_case()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"text_only": True, "text": "不应作为提交。", "delay_ms": 10000}]), encoding="utf-8")
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime"), fixture_plan=plan)
    started = asyncio.Event()
    adapter.workbench_event = lambda event, data: started.set() if event == "source_bundle_native_request_started" else None
    with capture_usage() as usage:
        task = asyncio.create_task(adapter.select_source_bundles(case["question"], case["documents"], case["paragraphs"]))
        await asyncio.wait_for(started.wait(), timeout=30)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    native = adapter.bundle_directory / "native"
    assert (native / "cancel-requested.json").exists()
    assert (native / "budget.jsonl").exists()
    assert not (native / "artifact.json").exists()
    assert usage and usage[0]["usage_source"] == "unavailable"
    assert not adapter._tail_seeded

