"""Independent single-call LC proof replay; synthetic sources and billing only."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from app.workflows import lc_bundle_workbench as lc
from app.workflows.bundle_workbench import _json
from test_bundle_workbench import descriptor
from test_source_bundle_tool_bridge import raw_case


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("text", ['{"same":1,"same":2}', '{"same":1,"s\\u0061me":2}', '{"value":NaN}'])
def test_original_visible_json_rejects_duplicate_and_nonfinite_fields(text):
    with pytest.raises(ValueError, match="lc_bundle_invalid_original_json"):
        lc._provider_json(text)


@pytest.mark.parametrize("relative", ["../scripts/lc_bundle_case.py", "app/../secret.py", "C:/secret.py", "app/not-python.js"])
def test_archived_codec_paths_cannot_escape_or_add_an_uncontrolled_app_program(tmp_path, relative):
    archives = {relative: descriptor(tmp_path / "source.bin", b"pass\n", raw=True)}
    with pytest.raises(ValueError, match="lc_bundle_archive_relative_path_required"):
        lc._sources(archives)


def test_missing_frozen_lc_entry_does_not_fall_back_to_live_helper(tmp_path):
    archives = {"app/__init__.py": descriptor(tmp_path / "init.py", b"", raw=True)}
    with pytest.raises(ValueError, match="lc_bundle_codec_archives_required"):
        lc._sources(archives)


def test_isolated_archived_entry_runs_its_old_bytes(tmp_path):
    code = b"import json,sys; request=json.loads(sys.stdin.read()); print(json.dumps({'old_codec':request['operation']}))\n"
    sources = {"app/__init__.py": b"", lc._ENTRY: code}
    raw = b'{}'
    with lc._archived_codec(sources, raw, sha256(raw).hexdigest()) as execute:
        assert execute({"operation": "prepare"}) == {"old_codec": "prepare"}


def test_missing_archived_dependency_fails_instead_of_importing_live_workspace(tmp_path):
    code = b"from app.domain.source_bundles import assemble_source_bundle\n"
    sources = {"app/__init__.py": b"", lc._ENTRY: code}
    raw = b'{}'
    with lc._archived_codec(sources, raw, sha256(raw).hexdigest()) as execute:
        with pytest.raises(ValueError, match="lc_bundle_archived_replay_failed"):
            execute({"operation": "prepare"})


def _frozen_app_bytes(revision):
    available = subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if available.returncode:
        pytest.skip("private historical Git snapshot is intentionally absent from the clean public history")
    archived = subprocess.run(["git", "archive", "--format=tar", revision, "app", lc._ENTRY],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    with tarfile.open(fileobj=BytesIO(archived)) as archive:
        return {row.name: archive.extractfile(row).read() for row in archive.getmembers()
            if row.isfile() and (row.name == lc._ENTRY or row.name.startswith("app/") and row.name.endswith(".py"))}


def proof_fixture(tmp_path, *, parent=100, case=None, frozen_revision=None):
    from app.domain.lc_bundle_case import prepare_lc_case, compile_business
    case = deepcopy(case) if case is not None else raw_case()
    question = {"id": "internal-lc", **case["question"]}
    raw = descriptor(tmp_path / "raw-case.json", case)
    source_bytes = (_frozen_app_bytes(frozen_revision) if frozen_revision else
        {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in [*(ROOT / "app").rglob("*.py"), ROOT / lc._ENTRY]})
    archives = {name: descriptor(tmp_path / "archive" / name, data, raw=True) for name, data in source_bytes.items()}
    def archived(operation, **arguments):
        with lc._archived_codec(lc._sources(archives), Path(raw["file"]).read_bytes(), raw["sha256"]) as execute:
            return lc._codec_result(execute, raw["sha256"], {"operation": operation, "excluded_spans": [], **arguments})
    prepared = archived("prepare") if frozen_revision else prepare_lc_case(case)
    target = next(row for row in prepared["catalog"] if row["source_bundle"]["source_paragraph_ids"] == [row["id"] for row in case["paragraphs"][:2]])
    output = {"source_catalog_id": prepared["source_catalog_id"], "evidence": [{
        "source_window_id": target["source_window_id"], "fact_des": "公开资料保留项目尚未批准的限定。",
        "argument": {"时间": None, "主体": None, "地点": None, "内容": "公开记录尚未确认批准", "影响": None, "数值口径": None},
        "entities": []}]}
    compiled = archived("compile", output=output) if frozen_revision else compile_business(case, prepared, output)
    body = {"model": "deepseek-v4-pro", "temperature": 0, "messages": prepared["messages"],
        "tools": [prepared["wire_tool"]], "tool_choice": {"type": "function", "function": {"name": prepared["function_name"]}},
        "parallel_tool_calls": False, "thinking": {"type": "disabled"}, "max_tokens": 4096, "stream": False}
    docs = descriptor(tmp_path / "documents.json", case["documents"])
    paragraphs = descriptor(tmp_path / "paragraphs.json", case["paragraphs"])
    prep = descriptor(tmp_path / "prepared-case.json", prepared)
    actual = descriptor(tmp_path / "actual-wire.json", json.dumps(body, ensure_ascii=False).encode("utf8"), raw=True)
    wire = descriptor(tmp_path / "wire.json", body)
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    name = prepared["function_name"]
    visible = {"tool_calls_wire": [{"id": "actual-call", "type": "function", "function": {"name": name, "arguments": _json(output)}}],
        "tool_calls": [{"id": "actual-call", "name": name, "args": output, "type": "tool_call"}], "usage": usage, "finish_reason": "tool_calls"}
    guard_messages = [*body["messages"], {"role": "system", "content": _json({"tools": body["tools"], "tool_choice": body["tool_choice"]}, sort=True)}]
    guard = _json(guard_messages, sort=True).encode("utf8")
    settlement = {"operation": "strict_selection", "ticket_id": 1, "actual_request_started": True,
        "usage_known": True, "outcome": "success", "provider_http_status": 200, "structure_repair": False,
        "length_recovery": False, "is_usage": True, "known_tokens_before": 0, "known_total_tokens_after": 150,
        "credited_total_tokens": 150, "stop_reason": None, "requested_max_output_tokens": 4096, "reserved_max_output_tokens": 4096,
        "soft_total_token_threshold": 12000 - parent, "message_content_sha256": sha256(guard).hexdigest(),
        "context_sha256": sha256(guard).hexdigest(), "message_content_utf8_bytes": len(guard), **usage}
    request = {"operation": "strict_selection", "wire": wire, "actual_wire": actual,
        "preflight": descriptor(tmp_path / "preflight.json", {"operation": "strict_selection", "wire": wire, "known_tokens_before": parent,
            "serialized_json_token_estimate": 20, "reserved_output_tokens": 4096, "safety_tokens": 512,
            "experimental_stop_threshold": 12000, "admitted": True, "is_actual_usage": False}),
        "settlement": descriptor(tmp_path / "settlement.json", settlement), "response": descriptor(tmp_path / "response.json", visible),
        "finished": descriptor(tmp_path / "finished.json", {"operation": "strict_selection", "request_started": True,
            "outcome": "success", "cache_hit": False, "safe_error_code": None, "known_total_tokens": 150,
            "known_episode_tokens_with_ancestors": parent + 150, "unknown_usage_requests": 0})}
    front_source = {"app/agents/strict_transport.py": archives["app/agents/strict_transport.py"]}
    front_episode = descriptor(tmp_path / "episode.json", {"method": "function_calling", "strict": True,
        "automatic_retries": 0, "thinking": "disabled", "model": body["model"], "experimental_stop_threshold": 12000,
        "max_requests": 7, "source_archive": front_source, "sources": {key: value["sha256"] for key, value in front_source.items()}})
    launch = {"format": "langchain-source-bundle-launch-v1", "raw_case_sha256": raw["sha256"], "source_binding": prepared["source_binding"],
        "prepared_case": prep, "front_episode": front_episode, "model": body["model"], "front_token_stop": 12000,
        "whole_token_stop": 30000, "max_requests": 10, "front_guard_max_requests": 7, "front_max_requests": 1,
        "parent_request_count": int(parent > 0), "parent_tokens": parent, "required_tail_requests": 3,
        "reserved_output_tokens": 4096, "safety_tokens": 512, "tail_reserved_tokens": 18000}
    artifact = descriptor(tmp_path / "artifact.json", compiled["artifact"])
    report = {"new_request_count": 1, "new_tokens": 150, "parent_tokens": parent, "whole_episode_tokens": parent + 150, "known_usage_complete": True}
    closed = {"format": "langchain-source-bundle-closed-receipt-v1", "status": "succeeded", "episode_closed": True,
        "source_binding": prepared["source_binding"], "raw_case_sha256": raw["sha256"], "artifact_sha256": artifact["sha256"],
        "front_requests": [request], "failure_code": None, "usage_records": [{"operation": "strict_selection", "model": body["model"], "usage_source": "provider", **usage}], **report}
    previous = [] if not parent else [{"operation": "ancestor_failure", "usage_source": "provider", "input_tokens": parent, "output_tokens": 0, "total_tokens": parent}]
    proof = {"format": lc.PROOF_FORMAT, "raw_case": raw, "documents": docs, "paragraphs": paragraphs,
        "prepared_case": prep, "model_output": descriptor(tmp_path / "model-output.json", output), "artifact": artifact,
        "scope": descriptor(tmp_path / "scope.json", compiled["scope"]), "front_episode": front_episode,
        "front_requests": [request], "previous_usage": descriptor(tmp_path / "previous.json", {"records": previous}),
        "launch_manifest": descriptor(tmp_path / "launch.json", launch), "closed_receipt": descriptor(tmp_path / "closed.json", closed),
        "source_archives": archives, "excluded_spans": [], "usage_report": report}
    return question, case, {"artifact": compiled["artifact"], "proof": proof, "usage_report": report}, {"prepared": prepared, "output": output, "body": body}


@pytest.mark.parametrize("parent", [0, 100])
def test_single_lc_request_independently_replays_full_original_source_and_preserves_null_semantics(tmp_path, parent):
    from app.workflows.bundle_workbench import compile_result, validate_result
    question, case, response, values = proof_fixture(tmp_path, parent=parent)
    result = compile_result(question, case["paragraphs"], case["documents"], response)
    assert validate_result(result, question, case["paragraphs"], case["documents"]) == result
    evidence = result["evidence"][0]
    assert "😀" in evidence["quote"] and "\r\n<实际分隔>" in evidence["quote"]
    assert evidence["argument"]["时间"] == "" and "proposed_roles" not in evidence
    assert evidence["native_model_delivery_verified"] is True and evidence["annotation_semantics_verified"] is False
    assert response["proof"]["front_requests"][0]["wire"]["sha256"] != response["proof"]["front_requests"][0]["actual_wire"]["sha256"]
    assert values["output"]["evidence"][0]["argument"]["时间"] is None


@pytest.mark.parametrize("change", ["source_text", "non_strict", "parallel", "extra_history", "wrong_function",
    "duplicate_original_keys", "guard_hash", "unknown_usage", "parent_double_count", "extra_front_request", "missing_dependency"])
def test_source_delivery_structure_or_cost_tampering_fails_before_canonical_evidence(tmp_path, change):
    from app.workflows.bundle_workbench import compile_result
    question, case, response, values = proof_fixture(tmp_path)
    proof, entry = response["proof"], response["proof"]["front_requests"][0]
    if change in {"source_text", "non_strict", "parallel", "extra_history"}:
        body = deepcopy(values["body"])
        if change == "source_text": body["messages"][1]["content"] = body["messages"][1]["content"].replace("😀", "")
        elif change == "non_strict": body["tools"][0]["function"]["strict"] = False
        elif change == "parallel": body["parallel_tool_calls"] = True
        else: body["messages"].append({"role": "assistant", "content": "额外未授权历史"})
        entry["wire"] = descriptor(tmp_path / "changed-wire.json", body)
        entry["actual_wire"] = descriptor(tmp_path / "changed-actual.json", _json(body).encode("utf8"), raw=True)
    elif change in {"wrong_function", "duplicate_original_keys"}:
        visible = json.loads(Path(entry["response"]["file"]).read_text("utf8"))
        if change == "wrong_function": visible["tool_calls_wire"][0]["function"]["name"] = "AnotherTool"
        else: visible["tool_calls_wire"][0]["function"]["arguments"] = '{"evidence":[],"evidence":[]}'
        entry["response"] = descriptor(tmp_path / "changed-response.json", visible)
    elif change in {"guard_hash", "unknown_usage"}:
        settled = json.loads(Path(entry["settlement"]["file"]).read_text("utf8"))
        if change == "guard_hash": settled["message_content_sha256"] = "0" * 64
        else: settled["usage_known"] = False
        entry["settlement"] = descriptor(tmp_path / "changed-settlement.json", settled)
    elif change == "parent_double_count":
        records = [{"usage_source": "provider", "input_tokens": 200, "output_tokens": 0, "total_tokens": 200}]
        proof["previous_usage"] = descriptor(tmp_path / "changed-parent.json", {"records": records})
    elif change == "extra_front_request": proof["front_requests"].append(deepcopy(entry))
    else: proof["source_archives"].pop("app/domain/question_recall.py")
    closed = json.loads(Path(proof["closed_receipt"]["file"]).read_text("utf8"))
    closed["front_requests"] = proof["front_requests"]
    proof["closed_receipt"] = descriptor(tmp_path / "changed-closed.json", closed)
    with pytest.raises(ValueError, match="lc_bundle_"):
        compile_result(question, case["paragraphs"], case["documents"], response)


def test_future_live_source_helper_cannot_replace_the_frozen_codec(tmp_path, monkeypatch):
    import app.domain.lc_bundle_case as live
    question, case, response, _ = proof_fixture(tmp_path)
    def unavailable(*args, **kwargs):
        raise AssertionError("current helper must not be used for archived proof")
    monkeypatch.setattr(live, "prepare_lc_case", unavailable)
    artifact, scope = lc.replay(question, case["paragraphs"], case["documents"], response, [])
    assert artifact == response["artifact"]
    assert scope["native_model_delivery_verified"] is False


def _overlap_case():
    from app.domain.source_bundles import paragraph_manifest_sha256
    case = raw_case()
    case["question"]["question"] = "项目 ABCDEFGHI、ABCD、NOPQR 的原始记录是什么？"
    content = "项目😀ABCDEFGHIJKLMNOPQRSTUVWXYZ末尾"
    intervals = [(0, 12), (3, 7), (3, 7), (16, 21)]
    rows, manifest = [], []
    for pid, (start, end) in zip(["opaque-z", "opaque-a", "opaque-duplicate", "opaque-final"], intervals):
        text = content[start:end]
        rows.append({"id": pid, "document_id": "doc-internal", "raw_doc_id": "原文档", "raw_para_id": pid,
            "document_start_cp": start, "document_end_cp": end, "text": text})
        manifest.append({"id": pid, "raw_para_id": pid, "start_cp": start, "end_cp": end, "text_sha256": sha256(text.encode()).hexdigest()})
    case["paragraphs"] = rows
    case["documents"][0].update(title=case["question"]["question"], content=content, content_sha256=sha256(content.encode()).hexdigest(),
        paragraph_manifest=manifest, paragraph_manifest_sha256=paragraph_manifest_sha256(manifest))
    return case


def _assert_independent_lossless_overlay(case, prepared):
    docs, parents = {r["id"]: r for r in case["documents"]}, {r["id"]: r for r in case["paragraphs"]}
    seen = []
    assert prepared["model_input"]["contiguous_quote_payload_claimed"] is False
    for run, native in zip(prepared["source_runs"], prepared["model_input"]["source_runs"], strict=True):
        original = docs[run["document_id"]]["content"]
        overlay, frontier = {}, run["start_cp"]
        assert native["id"] == run["run_id"] and native["document"] == run["document_ref"]
        assert run["contiguous_quote_payload_claimed"] is False and "quote" not in native
        for segment, delivered in zip(run["segments"], native["segments"], strict=True):
            a, b = segment["start_cp"], segment["end_cp"]
            assert segment["text"] == original[a:b] and delivered["text"] == segment["text"]
            assert len(segment["text"]) == b - a and not {"start_cp", "end_cp", "quote"} & set(delivered)
            if segment["kind"] == "paragraph":
                parent = parents[segment["paragraph_id"]]
                assert (a, b, segment["text"]) == (parent["document_start_cp"], parent["document_end_cp"], parent["text"])
                assert delivered["id"] == segment["ref"]
                assert delivered.get("overlaps_prior_source", False) is (a < frontier)
                seen.append(parent["id"])
            else:
                next_parent = next(row for row in run["paragraphs"] if row["ref"] == segment["before_ref"])
                assert a == frontier and b == next_parent["start_cp"] and b > a
                assert delivered["before"] == segment["before_ref"]
            for cp, char in enumerate(segment["text"], a):
                assert overlay.get(cp, char) == char
                overlay[cp] = char
            frontier = max(frontier, b)
        assert set(overlay) == set(range(run["start_cp"], run["end_cp"]))
        rebuilt = "".join(overlay[cp] for cp in range(run["start_cp"], run["end_cp"]))
        assert rebuilt == original[run["start_cp"]:run["end_cp"]]
        assert sha256(rebuilt.encode()).hexdigest() == run["original_slice_sha256"]
    assert len(seen) == len(set(seen)) and seen == prepared["scope"]["returned_complete_paragraph_ids"]
    assert "source_window_catalog" not in prepared["model_input"]
    assert prepared["model_input"]["source_ranges"] == [row["source_window_id"] for row in prepared["catalog"]]


@pytest.mark.parametrize("overlap", [False, True])
def test_v2_actual_source_wire_replays_lossless_complete_paragraphs_gaps_and_overlap_overlay(tmp_path, overlap):
    case = _overlap_case() if overlap else raw_case()
    question, case, response, values = proof_fixture(tmp_path, case=case)
    prepared = values["prepared"]
    assert prepared["function_name"] == "LCSourceBundleSelectionV2"
    _assert_independent_lossless_overlay(case, prepared)
    artifact, scope = lc.replay(question, case["paragraphs"], case["documents"], response, [])
    assert artifact == response["artifact"] and scope["segmented_original_delivery_lossless_checked"] is True
    if overlap:
        run = prepared["source_runs"][0]
        assert [row["text"] for row in run["segments"] if row["kind"] == "original_gap"] == [case["documents"][0]["content"][12:16]]
        assert len([row for row in run["segments"] if row["kind"] == "paragraph"]) == 4
        assert "".join(row["text"] for row in run["segments"]) != case["documents"][0]["content"][0:21]
        assert artifact["evidence"][0]["quote"] == case["documents"][0]["content"][:12]
    else:
        assert "😀" in prepared["model_input"]["source_runs"][0]["segments"][0]["text"]
        assert any(row["text"] == "\r\n<实际分隔>" for row in prepared["model_input"]["source_runs"][0]["segments"] if row["kind"] == "original_gap")


@pytest.mark.parametrize("change", ["paragraph", "gap"])
def test_v2_partial_actual_segment_delivery_is_not_a_complete_source_proof(tmp_path, change):
    question, case, response, values = proof_fixture(tmp_path)
    body = deepcopy(values["body"])
    model_input = json.loads(body["messages"][1]["content"])
    segment = next(row for row in model_input["source_runs"][0]["segments"]
        if row["kind"] == ("paragraph" if change == "paragraph" else "original_gap"))
    segment["text"] = segment["text"][:-1]
    body["messages"][1]["content"] = _json(model_input, sort=True)
    request = response["proof"]["front_requests"][0]
    request["wire"] = descriptor(tmp_path / "partial-wire.json", body)
    request["actual_wire"] = descriptor(tmp_path / "partial-actual.json", _json(body).encode(), raw=True)
    with pytest.raises(ValueError, match="lc_bundle_actual_full_source_or_strict_tool_missing"):
        lc.replay(question, case["paragraphs"], case["documents"], response, [])


def test_frozen_v1_source_and_old_window_ids_replay_without_the_live_v2_helper(tmp_path, monkeypatch):
    import app.domain.lc_bundle_case as live
    question, case, response, values = proof_fixture(tmp_path, frozen_revision="06bdcf1da4d460eaf2c226156ab1ea50efb911a4")
    assert values["prepared"]["format"] == "lc-source-bundle-prepared-v1"
    assert values["prepared"]["function_name"] == "LCSourceBundleSelection"
    assert values["output"]["evidence"][0]["source_window_id"].startswith("W")
    def unavailable(*args, **kwargs):
        raise AssertionError("current v2 helper cannot interpret frozen v1 source")
    monkeypatch.setattr(live, "prepare_lc_case", unavailable)
    artifact, scope = lc.replay(question, case["paragraphs"], case["documents"], response, [])
    assert artifact == response["artifact"] and scope["binding"]["format"] == "lc-source-bundle-input-v1"


@pytest.mark.asyncio
async def test_actual_langchain_http_wire_replays_through_full_graph_and_independent_tail(tmp_path, monkeypatch):
    from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
    from app.workflows.analysis import run_analysis
    from app.workflows.bundle_workbench import validate_result
    from test_lc_bundle_transport import config, inputs, parents, patch_http
    adapter = LCBundleWorkbenchProvider(config(tmp_path / "runtime"))
    adapter.set_previous_usage(parents())
    requests = patch_http(monkeypatch, adapter)
    case, question = inputs()
    result = await run_analysis(question, case["paragraphs"], adapter, source_documents=case["documents"],
        checkpoint_path=tmp_path / "checkpoint.sqlite", thread_id="single-lc-full-graph-synthetic-only",
        cancel_check=lambda: False, emit=lambda *_args: None)
    assert len(requests) == len(result["usage"]) == 4
    assert sum(row["total_tokens"] for row in result["usage"]) == 480
    assert adapter.prior_known_tokens == 13950 and adapter.guard.report()["known_total_tokens"] == 360
    assert result["source_bundle_proof"]["response"]["proof"]["format"] == lc.PROOF_FORMAT
    assert result["evidence"] and "proposed_roles" not in result["evidence"][0]
    assert validate_result(result, question, case["paragraphs"], case["documents"])["evidence"] == result["evidence"]
    tail = requests[1:]
    assert [row["tools"][0]["function"]["name"] for row in tail] == ["AnalysisOutput", "AnalysisOutput", "ScopedArbitrationWireOutput"]
    a, b = (json.loads(row["messages"][-1]["content"]) for row in tail[:2])
    assert "agent_outputs" not in a and "agent_outputs" not in b


@pytest.mark.asyncio
async def test_completed_lc_source_checkpoint_resumes_with_only_three_new_tail_calls_and_original_cost(tmp_path, monkeypatch):
    from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
    from app.workflows.analysis import AnalysisCancelled, run_analysis
    from test_lc_bundle_transport import config, inputs, parents, patch_http
    case, question = inputs()
    path, pause = tmp_path / "checkpoint.sqlite", {"stop": False}
    first = LCBundleWorkbenchProvider(config(tmp_path / "runtime"))
    first.set_previous_usage(parents())
    first_requests = patch_http(monkeypatch, first)
    select = first.select_source_bundles
    async def select_then_pause(*args, **kwargs):
        value = await select(*args, **kwargs)
        pause["stop"] = True
        return value
    first.select_source_bundles = select_then_pause
    with pytest.raises(AnalysisCancelled):
        await run_analysis(question, case["paragraphs"], first, source_documents=case["documents"],
            checkpoint_path=path, thread_id="lc-source-paid-once", cancel_check=lambda: pause["stop"], emit=lambda *_args: None)
    assert len(first_requests) == 1 and first.guard.report()["actual_requests_started"] == 0
    second = LCBundleWorkbenchProvider(config(tmp_path / "runtime"))
    second.set_previous_usage([*parents(), *deepcopy(first._front_records)])
    second_requests = patch_http(monkeypatch, second)
    result = await run_analysis(question, case["paragraphs"], second, source_documents=case["documents"],
        checkpoint_path=path, thread_id="lc-tail-resume", resume_from_thread_id="lc-source-paid-once",
        cancel_check=lambda: False, emit=lambda *_args: None)
    assert len(second_requests) == len(result["usage"]) == 3
    assert sum(row["total_tokens"] for row in result["usage"]) == 360
    assert second.front_provider is None and second.prior_known_tokens == 13950
    assert second.guard.policy.max_requests == 7
    assert "source_bundle_selection" in result["execution_provenance"]["reused_nodes"]
    assert result["source_bundle_proof"]["response"]["usage_report"]["whole_episode_tokens"] == 13950
    assert all(row["evidence_snapshot_id"] == result["evidence_snapshot_id"] for row in result["analyses"])
