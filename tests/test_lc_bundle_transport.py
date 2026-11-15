"""Explicit local HTTP MockTransport only; no credentials, samples or account."""
import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import httpx
import pytest

from app.agents import strict_transport
from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider, PROMPT_VERSION, COMPILE_FAILURE_MESSAGES, _compile_failure
from app.agents.provider import ModelOutputInvalid, capture_usage
from app.config import Settings
from test_source_bundle_tool_bridge import raw_case


def config(path, **changes):
    return Settings(_env_file=None, data_dir=path, model_provider="deepseek", model_name="deepseek-v4-pro",
        model_api_key="explicit-http-mock-not-a-credential", model_source_bundle_enabled=True,
        source_bundle_front_framework="langchain", source_bundle_front_token_stop=30000,
        source_bundle_tail_reserved_tokens=15000, strict_evidence_token_stop=45000,
        strict_evidence_max_requests=10, **changes)


def parents():
    # Deliberately synthetic counters; all failed-parent costs must survive.
    return [{"operation": "synthetic_failed_parent", "model": "deepseek-v4-pro", "usage_source": "provider",
        "input_tokens": 6000, "output_tokens": 187, "total_tokens": 6187},
        {"operation": "synthetic_failed_parent", "model": "deepseek-v4-pro", "usage_source": "provider",
        "input_tokens": 7400, "output_tokens": 243, "total_tokens": 7643}]


def inputs():
    raw = raw_case()
    for row in raw["paragraphs"]:
        row.update(title=raw["documents"][0]["title"], source_credibility_json="0.9")
    question = {"id": "internal-question", "raw_question_id": raw["question"]["question_id"],
        "question": raw["question"]["question"]}
    return raw, question


def _enum(schema, key):
    if isinstance(schema, dict):
        properties = schema.get("properties", {})
        if key in properties and isinstance(properties[key], dict):
            field = properties[key]
            if "const" in field: return [field["const"]]
            if "enum" in field: return field["enum"]
        for value in schema.values():
            found = _enum(value, key)
            if found: return found
    elif isinstance(schema, list):
        for value in schema:
            found = _enum(value, key)
            if found: return found
    return None


def _first_quote(prepared):
    def find(value):
        if isinstance(value, dict):
            if isinstance(value.get("quote"), str) and value["quote"]: return value["quote"]
            for child in value.values():
                result = find(child)
                if result: return result
        elif isinstance(value, list):
            for child in value:
                result = find(child)
                if result: return result
    return find(prepared["catalog"])


def patch_http(monkeypatch, adapter, *, mode="valid", started=None):
    from synthetic_planning import patch_planning
    patch_planning(monkeypatch)
    requests = []
    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        name = body["tools"][0]["function"]["name"]
        prepared_path = adapter.bundle_directory / "prepared-case.json"
        prepared = json.loads(prepared_path.read_text("utf8")) if prepared_path.is_file() else {}
        is_front = name == prepared.get("function_name")
        if is_front:
            if started is not None:
                started.set()
            if mode == "cancel":
                await asyncio.Event().wait()
            properties = body["tools"][0]["function"]["parameters"]
            argument = {key: None for key in ("时间", "主体", "地点", "内容", "影响", "数值口径")}
            argument["内容"] = _first_quote(prepared)
            legal_ranges = ([row["source_window_id"] for row in prepared["catalog"]]
                if prepared.get("selection_only") else _enum(properties, "source_window_id"))
            assert legal_ranges
            selected = {"source_window_id": legal_ranges[0],
                "fact_des": "原文保留项目程序状态。", "argument": argument, "entities": []}
            # The helper may expose optional proposed-use labels in its schema.
            if '"proposed_roles"' in json.dumps(properties): selected["proposed_roles"] = ["L"]
            result = {"source_catalog_id": _enum(properties, "source_catalog_id")[0], "evidence": [selected]}
            if prepared.get("function_name") == "LCSourceBundleIDsV1":
                result = {"source_catalog_id": prepared["source_catalog_id"],
                    "source_window_ids": [prepared["catalog"][0]["source_window_id"]]}
            if prepared.get("function_name") == "LCSourceBundleRankedIDsV1":
                result = {"source_catalog_id": prepared["source_catalog_id"], "ranked_candidates": [
                    {"source_window_id": legal_ranges[0], "relevance_score": 3, "use_type": "direct_answer"}]}
                if mode == "graded_background":
                    result["ranked_candidates"][0]["use_type"] = "background"
            if mode == "unknown_id": selected["source_window_id"] = "W999999"
            elif mode == "literal_failure": argument["时间"] = "2099年"
        else:
            payload = json.loads(body["messages"][-1]["content"])
            ids = [row["id"] for row in payload["evidence"]]
            result = {"decision": "undetermined", "conclusion": "公开记录保留项目尚未批准的程序限定。",
                "judgments": [{"id": "J1", "statement": "项目程序状态尚待核验。", "evidence_ids": ids,
                    "prerequisite_judgment_ids": []}], "supporting_evidence_ids": ids, "conflicting_evidence_ids": []}
            if name == "ScopedArbitrationWireOutput":
                result.pop("supporting_evidence_ids"); result.pop("conflicting_evidence_ids")
                if "issue_frame" in body["tools"][0]["function"]["parameters"].get("properties", {}):
                    result.update(issue_frame={"question_anchor": payload["question"]["question"],
                        "proposition": "所给原文尚未显示项目已经批准。", "claim_kind": "official_finding",
                        "scope": "仅限当前已给原文的程序状态。"}, relations={ref: "QS" for ref in ids}, conflicts=[])
                else:
                    result.update(principal_claim="现有公开材料只能确认程序记录，不能据此认定项目已经批准。",
                        relations={ref: "qualification" for ref in ids}, conflicts=[])
        raw_arguments = json.dumps(result, ensure_ascii=False)
        if mode == "duplicate_raw_key" and is_front:
            marker = '"fact_des": "原文保留项目程序状态。"'
            assert marker in raw_arguments
            raw_arguments = raw_arguments.replace(marker, marker + ', ' + marker)
        response = {"id": "local-mock-only", "object": "chat.completion", "created": 1, "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "finish_reason": "length" if mode == "length" and is_front else "tool_calls",
                "message": {"role": "assistant", "content": "", "tool_calls": [{"id": "mock-call", "type": "function",
                    "function": {"name": name, "arguments": raw_arguments}}]}}]}
        if mode != "unknown_usage" or not is_front:
            response["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return httpx.Response(200, json=response)
    monkeypatch.setattr(strict_transport.httpx, "AsyncHTTPTransport", lambda **_kwargs: httpx.MockTransport(handle))
    class MockAudit(strict_transport.AuditedTransport, httpx.MockTransport):
        pass
    monkeypatch.setattr(strict_transport, "AuditedTransport", MockAudit)
    monkeypatch.setattr(strict_transport, "count_wire", lambda body, directory, ordinal: (100,
        strict_transport.freeze(directory / f"request-{ordinal:02d}-wire.json", body)))
    return requests


def test_explicit_framework_and_parent_accounting_are_required(tmp_path):
    settings = config(tmp_path)
    with pytest.raises(ValueError, match="explicit_single_call"):
        LCBundleWorkbenchProvider(settings.model_copy(update={"source_bundle_front_framework": "dsh"}))
    adapter = LCBundleWorkbenchProvider(settings)
    with pytest.raises(ModelOutputInvalid) as failure:
        adapter.set_previous_usage([{"usage_source": "unavailable", "total_tokens": None}])
    assert failure.value.code == "experiment_usage_unknown"
    adapter.set_previous_usage(parents())
    with pytest.raises(ValueError, match="once_before_calls"):
        adapter.set_previous_usage([])


def test_compile_failure_diagnostics_are_fixed_rule_labels_never_exception_text():
    for check, message in COMPILE_FAILURE_MESSAGES.items():
        failure = _compile_failure(ValueError(check))
        assert failure.code == "lc_bundle_source_compile_failed"
        assert failure.stage == "source_bundle_selection"
        assert failure.details == {"check": check}
        assert str(failure) == message
    class Unprintable(ValueError):
        def __str__(self):
            raise AssertionError("untrusted exception text must never be formatted")
    for error in [ValueError("private original/provider body"), KeyError("private-key"),
                  TypeError({"response": "private"}), ValueError(), Unprintable("private")]:
        failure = _compile_failure(error)
        assert failure.details == {"check": "lc_bundle_source_compile_failed"}
        assert str(failure) == "所选来源编号或注释未通过业务校验，未发布。"
        assert "private" not in json.dumps(failure.details)


@pytest.mark.asyncio
async def test_single_real_transport_shape_full_sources_and_exact_once_parent_to_tail(tmp_path, monkeypatch):
    adapter = LCBundleWorkbenchProvider(config(tmp_path))
    adapter.set_previous_usage(parents())
    requests = patch_http(monkeypatch, adapter)
    events = []
    adapter.workbench_event = lambda event, data: events.append((event, data))
    raw, question = inputs()
    with capture_usage() as usage:
        response = await adapter.select_source_bundles(question, raw["documents"], raw["paragraphs"])
        evidence = [{"id": "actual-evidence-id", "raw_doc_id": "原文档", "raw_para_id": "原段0", "quote": "项目申请已受理😀。",
            "fact_des": "程序状态记录", "grade_score": .9, "source_group_id": "group"}]
        a = await adapter.analyze("authority_analyst", question, evidence, "snapshot")
        b = await adapter.analyze("cross_validation_analyst", question, evidence, "snapshot")
        await adapter.arbitrate(question, evidence, [a, b])
    assert len(requests) == len(usage) == 4
    assert [row["operation"] for row in usage] == ["strict_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert sum(row["total_tokens"] for row in usage) == 480
    assert adapter.prior_known_tokens == 13950  # 13830 prior + 120 new front exactly once.
    assert adapter.guard.report()["known_total_tokens"] == 360
    assert adapter.guard.policy.max_requests == 7
    report = response["usage_report"]
    assert report["new_request_count"] == 1 and report["new_tokens"] == 120
    assert report["parent_tokens"] == 13830 and report["whole_episode_tokens"] == 13950
    assert adapter.front_provider.guard.policy.soft_total_tokens == 16170
    assert adapter.front_provider.single_invocation_entered
    proof = response["proof"]
    assert proof["format"] == "langchain-source-bundle-workbench-proof-v1"
    assert not {"native_requests", "presence", "business_transcript", "submit_request"} & set(proof)
    item = proof["front_requests"][0]
    actual = Path(item["actual_wire"]["file"]).read_bytes()
    assert sha256(actual).hexdigest() == item["actual_wire"]["sha256"]
    assert json.loads(actual) == requests[0]
    prepared = json.loads(Path(proof["prepared_case"]["file"]).read_text("utf8"))
    assert requests[0]["messages"] == prepared["messages"]
    assert requests[0]["tools"][0]["function"]["name"] == prepared["function_name"]
    assert requests[0]["max_tokens"] == 4096 and requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["parallel_tool_calls"] is False
    assert adapter.prompt_version == PROMPT_VERSION
    metadata = json.loads((adapter.directory / "source-bundle-episode.json").read_text("utf8"))
    assert metadata["pipeline"][0] == "langchain_single_source_selection" and metadata["framework"] == "langchain"
    assert len([1 for event, _ in events if event == "tool_started"]) == 1
    assert all(data.get("engine") != "official-dsh" for _, data in events)
    assert "explicit-http-mock-not-a-credential" not in "".join(p.read_text("utf8") for p in adapter.bundle_directory.glob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["unknown_id", "literal_failure", "length", "unknown_usage", "duplicate_raw_key"])
async def test_paid_failure_is_not_retried_or_admitted_to_tail(tmp_path, monkeypatch, mode):
    adapter = LCBundleWorkbenchProvider(config(tmp_path))
    adapter.set_previous_usage(parents())
    requests = patch_http(monkeypatch, adapter, mode=mode)
    raw, question = inputs()
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid) as invalid:
            await adapter.select_source_bundles(question, raw["documents"], raw["paragraphs"])
    if mode == "literal_failure":
        assert invalid.value.details["check"] == "lc_bundle_time_not_literal"
        assert "逐字文本" in str(invalid.value)
    assert len(requests) == len(usage) == 1
    assert usage[0]["usage_source"] == ("unavailable" if mode == "unknown_usage" else "provider")
    receipt = json.loads((adapter.bundle_directory / "closed-receipt.json").read_text("utf8"))
    assert receipt["status"] != "succeeded" and not adapter._tail_seeded
    assert receipt["new_tokens"] == (None if mode == "unknown_usage" else 120)
    assert receipt["parent_tokens"] == 13830
    with pytest.raises(ModelOutputInvalid):
        await adapter.analyze("authority_analyst", question, [], "snapshot")
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_graded_background_inconsistency_preserves_actual_cost_and_blocks_every_tail_call(tmp_path, monkeypatch):
    adapter = LCBundleWorkbenchProvider(config(tmp_path, source_bundle_selection_only=True, source_bundle_graded_selection=True))
    adapter.set_previous_usage(parents())
    requests = patch_http(monkeypatch, adapter, mode="graded_background")
    raw, question = inputs()
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid) as invalid:
            await adapter.select_source_bundles(question, raw["documents"], raw["paragraphs"])
    assert invalid.value.details["check"] == "lc_bundle_graded_selection_inconsistent_background_score"
    assert len(requests) == len(usage) == 1 and requests[0]["max_tokens"] == 2048
    receipt = json.loads((adapter.bundle_directory / "closed-receipt.json").read_text("utf8"))
    assert receipt["status"] == "failed" and receipt["new_tokens"] == 120 and receipt["parent_tokens"] == 13830
    with pytest.raises(ModelOutputInvalid):
        await adapter.analyze("authority_analyst", question, [], "snapshot")
    assert len(requests) == 1 and not (adapter.bundle_directory / "artifact.json").exists()


@pytest.mark.asyncio
async def test_preflight_rejects_before_sending_and_records_zero_new_cost(tmp_path, monkeypatch):
    settings = config(tmp_path).model_copy(update={"source_bundle_front_token_stop": 16000})
    adapter = LCBundleWorkbenchProvider(settings)
    adapter.set_previous_usage(parents())
    requests = patch_http(monkeypatch, adapter)
    raw, question = inputs()
    with capture_usage() as usage:
        with pytest.raises(ModelOutputInvalid) as error:
            await adapter.select_source_bundles(question, raw["documents"], raw["paragraphs"])
    assert error.value.code == "experiment_usage_stop" and requests == usage == []
    receipt = json.loads((adapter.bundle_directory / "closed-receipt.json").read_text("utf8"))
    assert receipt["new_request_count"] == 0 and receipt["new_tokens"] == 0
    assert receipt["whole_episode_tokens"] == receipt["parent_tokens"] == 13830
    finished = json.loads((adapter.front_directory / "request-01-finished.json").read_text("utf8"))
    assert finished["request_started"] is False


@pytest.mark.asyncio
async def test_completed_source_checkpoint_resume_does_not_charge_front_again(tmp_path, monkeypatch):
    adapter = LCBundleWorkbenchProvider(config(tmp_path))
    prior = parents() + [{"operation": "strict_selection", "usage_source": "provider", "input_tokens": 100,
        "output_tokens": 20, "total_tokens": 120}]
    adapter.set_previous_usage(prior)
    requests = patch_http(monkeypatch, adapter)
    raw, question = inputs()
    with capture_usage() as usage:
        await adapter.analyze("authority_analyst", question, [{"id": "E-real", "quote": "尚未批准。"}], "snapshot")
    assert len(requests) == len(usage) == 1 and usage[0]["operation"] == "authority_analyst"
    assert adapter.front_provider is None and adapter.prior_known_tokens == 13950
    with pytest.raises(ValueError, match="once_per_provider"):
        await adapter.select_source_bundles(question, raw["documents"], raw["paragraphs"])


@pytest.mark.asyncio
async def test_cancel_after_actual_start_preserves_unknown_cost_and_blocks_tail(tmp_path, monkeypatch):
    adapter = LCBundleWorkbenchProvider(config(tmp_path))
    adapter.set_previous_usage(parents())
    started = asyncio.Event()
    requests = patch_http(monkeypatch, adapter, mode="cancel", started=started)
    raw, question = inputs()
    with capture_usage() as usage:
        task = asyncio.create_task(adapter.select_source_bundles(question, raw["documents"], raw["paragraphs"]))
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(requests) == len(usage) == 1 and usage[0]["usage_source"] == "unavailable"
    receipt = json.loads((adapter.bundle_directory / "closed-receipt.json").read_text("utf8"))
    assert receipt["new_tokens"] is None and receipt["known_usage_complete"] is False
    assert receipt["failure_code"] == "cancelled" and not adapter._tail_seeded
    with pytest.raises(ModelOutputInvalid):
        await adapter.analyze("authority_analyst", question, [], "snapshot")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "unknown_id"])
async def test_lc_http_mock_worker_publication_and_zip_are_source_and_usage_bound(tmp_path, monkeypatch, mode):
    """Four local HTTP responses exercise real queue/graph/compiler/export code."""
    from zipfile import ZipFile
    from app.data.ingest import import_topic
    from app.repository import Repository
    from app.worker import Worker

    raw = raw_case()
    settings = config(tmp_path / "runtime")
    repo = Repository(settings)
    document = raw["documents"][0]
    imported = import_topic(settings.database_path,
        [{"doc_id": document["raw_doc_id"], "title": document["title"], "content": document["content"],
          "source_credibility": .9, "paragraphs": [{"para_id": p["raw_para_id"],
              "char_start": p["document_start_cp"], "char_end": p["document_end_cp"]} for p in raw["paragraphs"]]}],
        [raw["question"]], "显式本地HTTP合成LC发布测试")
    topic = imported["topic_id"]
    question = repo.questions(topic)[0]
    adapter = LCBundleWorkbenchProvider(settings)
    requests = patch_http(monkeypatch, adapter, mode=mode)
    queued = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "http-mock-lc-worker-" + mode, model_available=True)
    worker = Worker(repo, adapter)
    publication_errors = []
    original_publish = repo.publish
    def observe_publish(*args, **kwargs):
        try:
            return original_publish(*args, **kwargs)
        except ValueError as error:
            publication_errors.append(str(error))
            raise
    monkeypatch.setattr(repo, "publish", observe_publish)
    await worker.execute(repo.claim())
    run = repo.run(queued["run_id"])
    assert run["source_bundle_front_framework"] == "langchain"
    usage = run["usage"][question["id"]]
    if mode != "valid":
        assert run["status"] == "failed", run
        assert run["error"]["code"] in {"model_output_invalid", "lc_bundle_source_compile_failed"}, run["error"]
        assert len(requests) == len(usage) == 1
        assert sum(row["total_tokens"] for row in usage) == 120
        assert repo.topic(topic)["published_revision_id"] is None
        assert repo.all("SELECT id FROM export_jobs") == []
        return
    assert run["status"] == "succeeded", (run["error"], publication_errors)
    assert len(requests) == len(usage) == 4
    assert [row["operation"] for row in usage] == ["strict_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert sum(row["total_tokens"] for row in usage) == 480
    assert adapter.prior_known_tokens == 120 and adapter.guard.report()["known_total_tokens"] == 360
    revision = run["target_revision_id"]
    assert repo.topic(topic)["published_revision_id"] == revision
    readiness = repo.export_readiness(topic, revision)
    assert readiness["can_export"] and readiness["completed_questions"] == readiness["total_questions"] == 1
    files = repo.formal_submission_files(topic, revision)
    assert files["evidence.json"] and len(files["conclusions.json"]) == 1
    for evidence in files["evidence.json"]:
        assert evidence["doc_id"] == document["raw_doc_id"]
        assert evidence["para_id"] in {p["raw_para_id"] for p in raw["paragraphs"]}
        assert evidence["quote"] in document["content"]
    job = repo.create_export(topic, revision, "http-mock-local-zip")
    assert worker.export_one()
    exported = repo.export(job["id"])
    assert exported["status"] == "succeeded", exported["error"]
    path = Path(repo.one("SELECT artifact_path FROM export_jobs WHERE id=?", (job["id"],))["artifact_path"])
    with ZipFile(path) as archive:
        assert {"manifest.json", "field_definitions.json"} <= set(archive.namelist())
        for name, value in files.items():
            assert json.loads(archive.read(name)) == value
    assert len(requests) == 4  # Export never invokes either model transport.
    # A trusted origin configured as DSH cannot publish the valid LC proof.
    result = repo.snapshot(topic, revision)["question_results"][question["id"]]
    original_one = repo.one
    def dsh_origin(query, *args, **kwargs):
        row = original_one(query, *args, **kwargs)
        if query == "SELECT topic_id,question_ids_json,model_config_json FROM runs WHERE id=?":
            from app.workflows.bundle_workbench import PROMPT_VERSION as DSH_PROMPT
            settings_row = json.loads(row["model_config_json"])
            settings_row["source_bundle_config"].pop("front_framework", None)
            settings_row.update(prompt_version=DSH_PROMPT, source_strategy="dsh_source_bundles")
            row = {**row, "model_config_json": json.dumps(settings_row, ensure_ascii=False)}
        return row
    with monkeypatch.context() as isolated:
        isolated.setattr(repo, "one", dsh_origin)
        with pytest.raises(ValueError, match="source bundle run configuration"):
            repo._validate_result_protocol(result, run["id"], topic, question["id"])
    assert len(requests) == 4
