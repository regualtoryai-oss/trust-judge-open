"""Production context workflow with four in-process HTTP mocks, never real API.

All source/model fixtures are synthetic. Counters such as 120/480 below are
mock response fields for accounting tests, not provider billing or P/R.
"""
from copy import deepcopy
import json
import httpx
import pytest

from app.agents import strict_transport
from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
from app.data.ingest import import_topic
from app.domain.snapshot import SOURCE_CONTEXT_PROTOCOL, competition_files
from app.domain.source_passages import SOURCE_GEOMETRY, validate_source_passage
from app.repository import Repository
from app.worker import Worker
from app.workflows.analysis import _project_question
from app.workflows.lc_bundle_workbench import replay as source_replay
from scripts.capture_bundle_workbench_real import capture_episode
from scripts.score_bundle_workbench_real import TAIL, _selection_mode, _tail as _score_tail
from test_lc_bundle_transport import config, patch_http
from test_lc_context_case import raw_case
from test_lc_local_rank_integration import model, infer  # noqa: F401 - explicit local synthetic inference


def _settings(tmp_path, model):
    return config(tmp_path / "runtime", source_bundle_local_rank_enabled=True,
        source_bundle_local_rank_manifest=model[0], source_bundle_local_rank_manifest_sha256=model[1],
        source_bundle_selection_only=True, source_bundle_graded_selection=True,
        source_bundle_compact_analyses_enabled=True, source_bundle_issue_frame_enabled=True,
        source_bundle_context_passages_enabled=True).model_copy(update={"strict_evidence_max_requests": 4})


def _sources():
    # Several native windows map to genuinely partial, complete-sentence
    # contexts. The whole source and its real separators remain immutable.
    text = "".join(f"项目第{i}句仍待核验。" + "原始上下文😀" * 6 + "。\r\n" for i in range(8))
    raw = raw_case([text, "项目申请已经受理，尚未获得批准。", "另一说法仍未得到确认。"])
    raw["question"]["question"] = "项目已经批准还是仍待核验？"
    return raw


def _import(repo, raw):
    documents = [{"doc_id": doc["raw_doc_id"], "title": doc["title"], "content": doc["content"],
        "source_credibility": .9, "paragraphs": [{"para_id": p["raw_para_id"],
            "char_start": p["document_start_cp"], "char_end": p["document_end_cp"]}
            for p in raw["paragraphs"] if p["document_id"] == doc["id"]]} for doc in raw["documents"]]
    return import_topic(repo.settings.database_path, documents, [raw["question"]],
        "显式HTTPMock完整句context工作台测试")["topic_id"]


def _patch_unknown_context(monkeypatch, adapter):
    """Keep the shared strict HTTP audit mock; change one returned enum only."""
    requests = patch_http(monkeypatch, adapter)
    shared_factory = strict_transport.httpx.AsyncHTTPTransport
    shared_transport = shared_factory()
    async def handle(request):
        response = await shared_transport.handle_async_request(request)
        data = response.json()
        body = json.loads(request.content)
        prepared_path = adapter.bundle_directory / "prepared-case.json"
        prepared = json.loads(prepared_path.read_text("utf8")) if prepared_path.is_file() else {}
        if body["tools"][0]["function"]["name"] == prepared.get("function_name"):
            arguments = json.loads(data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
            arguments["ranked_candidates"][0]["source_window_id"] = "C999999"
            data["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
        return httpx.Response(200, json=data)
    monkeypatch.setattr(strict_transport.httpx, "AsyncHTTPTransport", lambda **_kwargs: httpx.MockTransport(handle))
    return requests


@pytest.mark.asyncio
async def test_context_four_http_mock_worker_publishes_exports_and_independently_replays(tmp_path, monkeypatch, model, infer):
    raw, settings = _sources(), _settings(tmp_path, model)
    repo = Repository(settings)
    topic = _import(repo, raw)
    question = repo.questions(topic)[0]
    adapter = LCBundleWorkbenchProvider(settings)
    requests = patch_http(monkeypatch, adapter)
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "context-http-mock-complete", model_available=True)
    publication_errors = []
    publish = repo.publish
    def observed_publish(*args, **kwargs):
        try:
            return publish(*args, **kwargs)
        except ValueError as error:
            publication_errors.append(str(error))
            raise
    monkeypatch.setattr(repo, "publish", observed_publish)
    await Worker(repo, adapter).execute(repo.claim())
    run = repo.run(task["run_id"])
    assert run["status"] == "succeeded", (run["error"], publication_errors)
    assert len(infer) == 1  # Injected synthetic local scores, not ONNX.
    usage = run["usage"][question["id"]]
    assert len(requests) == len(usage) == 4
    assert [r["operation"] for r in usage] == ["strict_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert sum(r["total_tokens"] for r in usage) == 480  # Mock counters only.
    assert run["source_bundle_context_passages"] is True
    snapshot = repo.snapshot(topic, run["target_revision_id"])
    result = snapshot["question_results"][question["id"]]
    assert result["source_bundle_protocol"] == SOURCE_CONTEXT_PROTOCOL
    assert result["evidence_relation_scope"]["format"] == "conclusion-aligned-evidence-relations-v3"
    documents, paragraphs = repo.source_documents(topic), repo.paragraphs(topic)
    parents = {p["id"]: p for p in paragraphs}
    assert result["evidence"]
    for row in result["evidence"]:
        validate_source_passage(row, documents, paragraphs)
        parent = parents[row["paragraph_id"]]
        assert row["source_geometry"] == SOURCE_GEOMETRY
        assert parent["text"][row["local_start_cp"]:row["local_end_cp"]] == row["quote"]
        assert row["start_cp"] == parent["document_start_cp"] + row["local_start_cp"]
        assert row["end_cp"] == parent["document_start_cp"] + row["local_end_cp"]
    assert any(row["local_start_cp"] > 0 or row["local_end_cp"] < len(parents[row["paragraph_id"]]["text"])
        for row in result["evidence"])

    response = result["source_bundle_proof"]["response"]
    artifact, scope = source_replay(question, paragraphs, documents, response, [])
    assert artifact == response["artifact"]
    assert scope["context_passages"] is True
    assert scope["local_window_coverage_is_native_reading"] is False
    for pid in scope["partially_provided_paragraph_ids"]:
        assert pid not in scope["returned_complete_paragraph_ids"]
    front_payload = json.loads(requests[0]["messages"][-1]["content"])
    assert "source_contexts" in front_payload and "source_runs" not in front_payload
    for source in front_payload["source_contexts"]:
        assert source["quote"] in raw["documents"][0]["content"]
    assert "issue_frame" in requests[-1]["tools"][0]["function"]["parameters"]["properties"]

    files = competition_files(snapshot, source_documents=documents, source_paragraphs=paragraphs)
    assert files == repo.formal_submission_files(topic, run["target_revision_id"])
    assert len(files["conclusions.json"]) == 1 and files["evidence.json"]
    for row in files["evidence.json"]:
        assert row["quote"] in raw["documents"][0]["content"]
        assert row["doc_id"] == raw["documents"][0]["raw_doc_id"]
    native = capture_episode(adapter.directory)
    assert len(native["tail"]["requests"]) == 3
    front = {"previous_records": [], "new_records": [r for r in usage if r["operation"] not in TAIL]}
    audit = _score_tail(result, _project_question(question), native, usage, front,
        {"engineering_episode_stop": 45000, "max_total_requests": 4}, [])
    assert audit["actual_tail_requests"] == 3 and audit["whole_episode_actual_requests"] == 4
    assert audit["whole_episode_known_tokens"] == 480  # Mock counters, not a real provider bill.
    assert audit["independent_A_B_inputs_verified"]
    assert len(audit["whole_chain_forecast_receipts"]) == 3
    mode = _selection_mode(response["proof"], settings.source_bundle_config, run)
    assert mode["front_use_type_is_final_conflict"] is False

    # These are independent integrity negatives over the same paid/mock
    # history; none can cause another model request or rewrite the old result.
    changed_source = deepcopy(documents)
    changed_source[0]["content"] += "并非原文"
    with pytest.raises(ValueError):
        source_replay(question, paragraphs, changed_source, response, [])
    changed_proof = deepcopy(response)
    changed_proof["proof"]["context_plan"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        source_replay(question, paragraphs, documents, changed_proof, [])
    changed_geometry = deepcopy(result)
    changed_geometry["evidence"][0]["local_end_cp"] += 1
    with pytest.raises(ValueError):
        repo._validate_result_protocol(changed_geometry, run["id"], topic, question["id"])
    original_one = repo.one
    def old_mode_origin(sql, *args, **kwargs):
        row = original_one(sql, *args, **kwargs)
        if sql == "SELECT topic_id,question_ids_json,model_config_json FROM runs WHERE id=?":
            cfg = json.loads(row["model_config_json"])
            cfg["source_bundle_config"].pop("context_passages", None)
            cfg["source_bundle_config"].pop("context_policy", None)
            row = {**row, "model_config_json": json.dumps(cfg, ensure_ascii=False)}
        return row
    with monkeypatch.context() as isolated:
        isolated.setattr(repo, "one", old_mode_origin)
        with pytest.raises(ValueError, match="protocol|context|geometry"):
            repo._validate_result_protocol(result, run["id"], topic, question["id"])
    assert len(requests) == 4


@pytest.mark.asyncio
async def test_unknown_context_alias_preserves_one_mock_cost_and_cannot_publish(tmp_path, monkeypatch, model, infer):
    raw, settings = _sources(), _settings(tmp_path, model)
    repo = Repository(settings)
    topic = _import(repo, raw)
    question = repo.questions(topic)[0]
    adapter = LCBundleWorkbenchProvider(settings)
    requests = _patch_unknown_context(monkeypatch, adapter)
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "context-http-mock-unknown", model_available=True)
    await Worker(repo, adapter).execute(repo.claim())
    run = repo.run(task["run_id"])
    assert run["status"] == "failed"
    assert run["error"]["code"] in {"model_output_invalid", "lc_bundle_source_compile_failed"}
    assert len(requests) == len(run["usage"][question["id"]]) == 1
    assert sum(r["total_tokens"] for r in run["usage"][question["id"]]) == 120
    assert repo.topic(topic)["published_revision_id"] is None
    assert repo.all("SELECT id FROM export_jobs") == []
    assert adapter._tail_seeded is False
