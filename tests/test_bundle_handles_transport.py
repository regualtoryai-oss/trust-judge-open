"""Compact handles through the official SDK and real graph, with zero network use."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from app.agents.bundle_transport import BundleWorkbenchProvider
from app.agents.provider import ModelOutputInvalid, capture_usage
from app.workflows.analysis import run_analysis
from app.workflows.bundle_workbench import compile_result
from test_bundle_transport import patch_tail, settings
from test_source_bundle_tool_bridge import annotation, raw_case


def compact_selection(case):
    row = annotation(case)
    return {"selected": {"P1": {"end_ref": "P2", **{
        key: value for key, value in row.items() if key not in {"start_paragraph_id", "end_paragraph_id"}
    }}}}


def write_plan(path, case, *, repeated=False, invalid=False):
    read = {"name": "read_source_bundles", "arguments": {"bundle_refs": {"B1": True}}}
    calls = [{"tool_calls": [deepcopy(read), deepcopy(read), deepcopy(read)]}] if repeated else [read]
    if invalid:
        calls = [{"name": "read_source_bundles", "arguments": {
            "anchor_paragraph_ids": [case["paragraphs"][0]["id"]], "before": 1, "after": 4}},
            {"text_only": True, "text": "显式合成终止；未读取、未提交。"}]
    else:
        calls.append({"name": "submit_bundle_evidence", "arguments": compact_selection(case)})
    path.write_text(json.dumps(calls, ensure_ascii=False), encoding="utf8")
    return path


def load_descriptor(descriptor):
    path = Path(descriptor["file"])
    assert sha256(path.read_bytes()).hexdigest() == descriptor["sha256"]
    return json.loads(path.read_text("utf8"))


@pytest.mark.optional_dsh_runtime
async def test_compact_handles_official_sdk_full_graph_uses_actual_short_schema_and_full_original_bytes(tmp_path, monkeypatch):
    case = raw_case()
    for row in case["paragraphs"]:
        row.update(title=case["documents"][0]["title"], source_credibility_json="0.9")
    plan = write_plan(tmp_path / "plan.json", case)
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime", source_bundle_compact_handles=True), fixture_plan=plan)
    assert adapter.source_bundle_config["compact_handles"] is True
    calls = patch_tail(monkeypatch)
    events = []
    adapter.workbench_event = lambda event, data: events.append((event, data))
    question = {"id": "internal", "raw_question_id": case["question"]["question_id"], "question": case["question"]["question"]}
    result = await run_analysis(question, case["paragraphs"], adapter, source_documents=case["documents"],
        checkpoint_path=tmp_path / "checkpoint.sqlite", thread_id="explicit-compact-official-sdk-graph",
        cancel_check=lambda: False, emit=lambda event, data: events.append((event, data)))
    response = json.loads((adapter.bundle_directory / "provider-response.json").read_text("utf8"))
    assert response["usage_report"]["real_api_requests"] == 0
    assert response["usage_report"]["new_request_count"] == 2
    assert len(calls) == 3 and len(result["usage"]) == 5
    assert compile_result(question, case["paragraphs"], case["documents"], response)["evidence"] == result["evidence"]
    assert not response["artifact"]["publishable"]
    proof = response["proof"]
    handles = load_descriptor(proof["program_projection"])
    projections_path = Path(proof["native_projections"]["file"])
    assert sha256(projections_path.read_bytes()).hexdigest() == proof["native_projections"]["sha256"]
    projections = [json.loads(line) for line in projections_path.read_text("utf8").splitlines()]
    assert [row["operation"] for row in projections] == ["read", "submit"]
    assert handles["format"] == "source-bundle-native-handles-v1"
    assert all(1 <= len(row["paragraph_refs"]) <= 5 for row in handles["bundles"])
    bodies = [load_descriptor(row) for row in proof["native_requests"]]
    read_schema = bodies[0]["tools"][0]["input_schema"]
    assert set(read_schema["properties"]) == {"bundle_refs"}
    assert "before" not in json.dumps(read_schema) and "after" not in json.dumps(read_schema)
    assert set(read_schema["properties"]["bundle_refs"]["properties"]) == {row["bundle_ref"] for row in handles["bundles"]}
    submit_schema = next(row["input_schema"] for row in bodies[1]["tools"] if row["name"] == "submit_bundle_evidence")
    selected = submit_schema["properties"]["selected"]
    assert selected["type"] == "object" and selected["additionalProperties"] is False
    assert selected["properties"]["P1"]["properties"]["end_ref"]["enum"] == ["P1", "P2", "P3"]
    assert set(selected["properties"]["P1"]["properties"]["argument"]["properties"]) == {"时间", "主体", "地点", "内容", "影响", "数值口径"}
    assert selected["properties"]["P1"]["properties"]["entities"]["type"] == "array"
    read = projections[0]
    assert isinstance(read["raw_native_arguments"], str)
    assert json.loads(read["raw_native_arguments"]) == {"bundle_refs": {"B1": True}}
    canonical_quote = read["canonical_result"]["bundles"][0]["quote"]
    assert "😀" in canonical_quote and "\r\n<实际分隔>" in canonical_quote
    assert read["native_result"]["bundles"][0]["quote"] == canonical_quote
    assert read["canonical_result_sha256"] != read["native_read_view_sha256"]
    texts = [block for message in bodies[1]["messages"] for block in message.get("content", []) if block.get("type") == "tool_result"]
    actual = next(block for block in texts if block["tool_use_id"] == read["tool_use_id"])
    assert [json.loads(block["text"]) for block in actual["content"]] == [read["native_result"]]
    presence = [json.loads(line) for line in Path(proof["presence"]["file"]).read_text("utf8").splitlines()]
    assert [row["format"] for row in presence] == ["dsh-source-bundle-native-presence-v2"] * 2
    assert presence[-1]["read_receipts"][0]["native_read_view_sha256"] == read["native_read_view_sha256"]
    assert any(name == "tool_completed" and data.get("name") == "read_source_bundles" for name, data in events)
    manifest = case["documents"][0]["paragraph_manifest"]
    expected = case["documents"][0]["content"][manifest[0]["start_cp"]:manifest[1]["end_cp"]]
    assert result["evidence"][0]["quote"] == expected


@pytest.mark.optional_dsh_runtime
async def test_compact_handles_two_same_step_repeated_reads_deliver_distinct_full_receipts_and_defer_third(tmp_path):
    case = raw_case()
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime", source_bundle_compact_handles=True),
        fixture_plan=write_plan(tmp_path / "plan.json", case, repeated=True))
    events = []
    adapter.workbench_event = lambda event, data: events.append((event, data))
    with capture_usage() as usage:
        response = await adapter.select_source_bundles(case["question"], case["documents"], case["paragraphs"])
    compile_result({"id": "internal", **case["question"]}, case["paragraphs"], case["documents"], response)
    assert len(usage) == 2 and response["usage_report"]["real_api_requests"] == 0
    projections = [json.loads(line) for line in Path(response["proof"]["native_projections"]["file"]).read_text("utf8").splitlines()]
    reads = [row for row in projections if row["operation"] == "read"]
    assert len(reads) == 2 and reads[0]["native_arguments"] == reads[1]["native_arguments"]
    assert reads[0]["canonical_result"]["bundles"] == reads[1]["canonical_result"]["bundles"]
    assert reads[0]["native_read_view_sha256"] != reads[1]["native_read_view_sha256"]
    assert [row["native_result"]["delivery"]["complete_delivery_index"] for row in reads] == [1, 2]
    assert [row["native_result"]["delivery"]["read_call_index"] for row in reads] == [3, 4]  # Search + program-only plan precede delivery.
    assert reads[1]["native_result"]["delivery"]["new_paragraph_refs"] == []
    deferred = [data for name, data in events if name == "tool_deferred"]
    assert len(deferred) == 1 and deferred[0]["name"] == "read_source_bundles"
    assert not any(name in {"tool_failed", "tool_completed"} and data.get("call_id") == deferred[0]["call_id"] for name, data in events)


@pytest.mark.optional_dsh_runtime
async def test_compact_handles_old_six_paragraph_request_is_rejected_without_business_read_or_artifact(tmp_path):
    case = raw_case()
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime", source_bundle_compact_handles=True),
        fixture_plan=write_plan(tmp_path / "plan.json", case, invalid=True))
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid):
            await adapter.select_source_bundles(case["question"], case["documents"], case["paragraphs"])
    assert len(usage) == 2 and all(row["total_tokens"] == 150 for row in usage)  # Fixture counters only.
    transcript = [json.loads(line) for line in (adapter.bundle_directory / "native/business-transcript.jsonl").read_text("utf8").splitlines()]
    assert [row["operation"] for row in transcript] == ["search", "plan"]
    assert not (adapter.bundle_directory / "native/artifact.json").exists()


@pytest.mark.optional_dsh_runtime
async def test_compact_handles_official_sse_duplicate_anchor_key_is_not_silently_overwritten(tmp_path):
    case = raw_case()
    selected = compact_selection(case)["selected"]["P1"]
    literal = json.dumps(selected, ensure_ascii=False)
    duplicate = '{"selected":{"P1":' + literal + ',"P\\u0031":' + literal + '}}'
    plan = tmp_path / "raw-plan.json"
    plan.write_text(json.dumps([
        {"name": "read_source_bundles", "arguments": {"bundle_refs": {"B1": True}}},
        {"tool_calls": [{"name": "submit_bundle_evidence", "raw_arguments": duplicate}]},
        {"text_only": True, "text": "显式合成终止；重复起点未提交。"},
    ], ensure_ascii=False), encoding="utf8")
    adapter = BundleWorkbenchProvider(settings(tmp_path / "runtime", source_bundle_compact_handles=True), fixture_plan=plan)
    events = []
    adapter.workbench_event = lambda event, data: events.append((event, data))
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid):
            await adapter.select_source_bundles(case["question"], case["documents"], case["paragraphs"])
    assert len(usage) == 3 and sum(row["total_tokens"] for row in usage) == 450  # Explicit no-network fixture counters.
    native = adapter.bundle_directory / "native"
    transcript = [json.loads(line) for line in (native / "business-transcript.jsonl").read_text("utf8").splitlines()]
    assert [row["operation"] for row in transcript] == ["search", "plan", "read"]
    projections = [json.loads(line) for line in (native / "native-projections.jsonl").read_text("utf8").splitlines()]
    assert [row["operation"] for row in projections] == ["read"]
    assert not (native / "artifact.json").exists() and not (native / "submit-request.json").exists()
    body3 = json.loads((native / "native-request-03.json").read_text("utf8"))
    failures = [block for message in body3["messages"] for block in message.get("content", [])
        if block.get("type") == "tool_result" and block.get("is_error")]
    assert len(failures) == 1 and "原始工具JSON含重复字段" in json.dumps(failures[0], ensure_ascii=False)
    assert "raw_arguments_sha256=" + sha256(duplicate.encode("utf8")).hexdigest() in json.dumps(failures[0], ensure_ascii=False)
    assert any(name == "tool_deferred" and data["name"] == "submit_bundle_evidence" for name, data in events)


@pytest.mark.optional_dsh_runtime
async def test_compact_handles_actual_sdk_worker_persists_config_and_publishes_verified_fixture_only(tmp_path, monkeypatch):
    from app.data.ingest import import_topic
    from app.repository import Repository
    from app.worker import Worker

    raw = raw_case()
    config = settings(tmp_path / "runtime", source_bundle_compact_handles=True)
    repo = Repository(config)
    doc = raw["documents"][0]
    imported = import_topic(config.database_path, [{"doc_id": doc["raw_doc_id"], "title": doc["title"],
        "content": doc["content"], "source_credibility": .9, "paragraphs": [{"para_id": p["raw_para_id"],
            "char_start": p["document_start_cp"], "char_end": p["document_end_cp"]} for p in raw["paragraphs"]]}],
        [{"question_id": raw["question"]["question_id"], "question": raw["question"]["question"]}], "显式紧凑句柄SDK合成测试")
    topic = imported["topic_id"]
    question = repo.questions(topic)[0]
    case = {"question": raw["question"], "documents": repo.source_documents(topic), "paragraphs": repo.paragraphs(topic)}
    adapter = BundleWorkbenchProvider(config, fixture_plan=write_plan(tmp_path / "worker-plan.json", case))
    calls = patch_tail(monkeypatch)
    queued = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "explicit-compact-official-worker-fixture", model_available=True)
    await Worker(repo, adapter).execute(repo.claim())
    run = repo.run(queued["run_id"])
    assert run["status"] == "succeeded", run["error"]
    assert len(calls) == 3 and repo.topic(topic)["published_revision_id"] == run["target_revision_id"]
    response = json.loads((adapter.bundle_directory / "provider-response.json").read_text("utf8"))
    assert response["usage_report"]["real_api_requests"] == 0
    assert load_descriptor(response["proof"]["launch_manifest"])["compact_handles"] is True
    assert load_descriptor(response["proof"]["program_projection"])["bundles"]
    assert (adapter.bundle_directory / "native/artifact.json").exists()
