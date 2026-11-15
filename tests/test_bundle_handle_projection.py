"""Synthetic short-handle native wires retain full, replayable source bytes."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

import app.workflows.bundle_workbench as wb
from scripts.source_bundle_tool import execute
from test_bundle_workbench import Q, descriptor, packed, sources


ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, repeat=False):
    docs, rows = sources()
    question = {"question_id": Q["raw_question_id"], "question": Q["question"]}
    case = {"format": "frozen-source-bundle-case-v1", "question": question, "documents": docs, "paragraphs": rows, "limits": {}}
    case_d = descriptor(tmp_path / "case.json", case)
    case_bytes, history = Path(case_d["file"]).read_bytes(), []
    def call(op, args):
        result = execute(case_bytes, case_d["sha256"], {"operation": op, "arguments": args, "history": history})
        assert result["status"] == "ok"
        if op != "submit": history.append(result["history_entry"])
        return result
    navigation = call("search", {"query": question["question"], "limit": 40, "exclude_paragraph_ids": []})["result"]
    plan = call("plan", {"anchor_paragraph_ids": [r["paragraph_id"] for r in navigation["results"]], "before": 1, "after": 3})["result"]
    handles = wb._handle_manifest(case, navigation, plan, case_d["sha256"])
    handles_d = descriptor(tmp_path / "handles.json", handles)
    native_args = {"bundle_refs": {handles["bundles"][0]["bundle_ref"]: True}}
    business_args = wb._handle_read_arguments(handles, native_args)
    reads, projections, uses, results, receipts = [], [], [], [], []
    for index in range(2 if repeat else 1):
        value = call("read", business_args)["result"]
        use_id = f"compact-read-{index+1}"
        reads.append({"arguments": business_args, "result": value})
        view = wb._handle_read_view(handles, value, native_args)
        rendered = wb._json(view, sort=True)
        projections.append({"operation": "read", "tool_use_id": use_id,
            "raw_native_arguments": wb._json(native_args), "native_arguments": deepcopy(native_args),
            "business_arguments": deepcopy(business_args), "canonical_result": value,
            "canonical_result_sha256": wb.digest(value), "native_result": view,
            "native_read_view_sha256": sha256(rendered.encode()).hexdigest(), "handles_sha256": handles_d["sha256"]})
        uses.append({"type": "tool_use", "name": "read_source_bundles", "id": use_id, "input": deepcopy(native_args)})
        results.append({"type": "tool_result", "tool_use_id": use_id, "content": [{"type": "text", "text": rendered}]})
        receipts.append({"tool_use_id": use_id, "canonical_read_result_sha256": wb.digest(value),
            "native_read_view_sha256": sha256(rendered.encode()).hexdigest(), "handles_sha256": handles_d["sha256"],
            "bundle_ids": [b["bundle_id"] for b in value["bundles"]],
            "source_bundle_set_sha256": value["source_bundle_set_sha256"], "complete_original_slices_present": True})
    annotation = {"end_ref": handles["paragraphs"][1]["ref"], "fact_des": "经营方与主管部门对甲设备批准状态的说法不同。",
        "argument": {"时间": "", "主体": "", "地点": "", "内容": "公开说法尚有分歧", "影响": "", "数值口径": ""},
        "entities": ["甲设备"], "proposed_roles": ["L"]}
    native_submit = {"selected": {handles["paragraphs"][0]["ref"]: annotation}}
    business_submit = wb._handle_submit_arguments(handles, reads, native_submit)
    submitted = call("submit", business_submit)
    projections.append({"operation": "submit", "tool_use_id": "compact-submit",
        "raw_native_arguments": wb._json(native_submit), "native_arguments": native_submit,
        "business_arguments": business_submit, "canonical_result": submitted["result"],
        "canonical_result_sha256": wb.digest(submitted["result"]), "handles_sha256": handles_d["sha256"]})
    question_data = wb._json({"original_question": json.loads(packed(question)), "corpus_document_count": len(docs),
        "corpus_paragraph_count": len(rows), "source_binding": json.loads(packed(submitted["scope"]["binding"])),
        "unseen_context": "unknown_not_irrelevant", "publishable": False})
    body = json.loads(packed({"system": question_data, "messages": [{"role": "assistant", "content": uses},
        {"role": "user", "content": results}], "tools": [{"name": "submit_bundle_evidence"}],
        "max_tokens": 100, "thinking": {"type": "disabled"}}))
    native_d = descriptor(tmp_path / "native.json", body)
    presence = {"format": "dsh-source-bundle-native-presence-v2", "handles_sha256": handles_d["sha256"],
        "read_receipts": receipts, "native_messages_sha256": sha256(wb._json(body["messages"]).encode()).hexdigest(),
        "source_bytes_present_in_current_request": True, "semantic_verified": False, "publishable": False}
    artifact = {**submitted["result"], "native_model_delivery_proof": presence,
        "source_bytes_present_in_submission_request": True, "semantic_verified": False, "publishable": False}
    artifact_d = descriptor(tmp_path / "artifact.json", artifact)
    usage = {"new_request_count": 1, "new_tokens": 7, "parent_tokens": 0, "whole_episode_tokens": 7, "known_usage_complete": True}
    closed = {"format": "dsh-source-bundle-closed-receipt-v1", "status": "succeeded", "episode_closed": True,
        "source_binding": submitted["scope"]["binding"], "raw_case_sha256": case_d["sha256"], "artifact_sha256": artifact_d["sha256"],
        "native_requests": [native_d], **{k: usage[k] for k in ("new_request_count", "new_tokens", "parent_tokens", "whole_episode_tokens")}}
    started = {"status": "request_started", "request_sequence": 1, "wire_sha256": native_d["sha256"], "known_episode_tokens_before": 0, "reserved_output_tokens": 100}
    settled = {**started, "status": "usage_settled", "provider_usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        "known_episode_tokens_after": 7, "response_completed": True, "halted": False, "unknown_usage_requests": 0}
    proof = {"format": wb.PROOF_FORMAT, "raw_case": case_d,
        "submit_request": descriptor(tmp_path / "submit-request.json", {"operation": "submit", "arguments": business_submit, "history": history}),
        "tool_submission": descriptor(tmp_path / "tool-submission.json", submitted), "artifact": artifact_d,
        "closed_receipt": descriptor(tmp_path / "closed.json", closed), "native_requests": [native_d],
        "presence": descriptor(tmp_path / "presence.jsonl", [presence], lines=True),
        "usage_ledger": descriptor(tmp_path / "budget.jsonl", [started, settled], lines=True),
        "program_projection": handles_d, "native_projections": descriptor(tmp_path / "projections.jsonl", projections, lines=True),
        "source_archives": {name: descriptor(tmp_path / "archive" / name, (ROOT / name).read_bytes(), raw=True) for name in wb.REPLAY_SOURCES},
        "excluded_spans": [], "usage_report": usage}
    return docs, rows, {"artifact": artifact, "proof": proof, "usage_report": usage}, {
        "body": body, "presence": presence, "reads": reads, "question_data": question_data,
        "handles": handles, "handles_sha256": handles_d["sha256"], "projections": projections,
        "case": case, "navigation": navigation, "plan": plan}


@pytest.mark.parametrize("repeat", [False, True])
def test_compact_view_preserves_exact_full_read_and_replays_full_inline_submission(tmp_path, repeat):
    docs, rows, response, native = fixture(tmp_path, repeat)
    result = wb.compile_result(Q, rows, docs, response)
    assert result["evidence"][0]["quote"] == response["artifact"]["evidence"][0]["quote"]
    assert result["source_bundle_proof"]["native_source_delivery_replayed"]
    assert wb.validate_result(result, Q, rows, docs) == result
    assert native["presence"]["read_receipts"][0]["canonical_read_result_sha256"] != native["presence"]["read_receipts"][0]["native_read_view_sha256"]


def test_JS_manifest_view_and_presence_match_independent_Python_verifier(tmp_path):
    _, _, _, native = fixture(tmp_path, repeat=True)
    state = {"question_data": native["question_data"], "tool_names": ["submit_bundle_evidence"],
        "handles": {"manifest": native["handles"], "sha256": native["handles_sha256"]},
        "reads": [{**row, "tool_use_id": p["tool_use_id"]} for row, p in zip(native["reads"], native["projections"])],
        "native_reads": native["projections"][:-1]}
    script = """
import {buildHandleManifest,handleReadView,stableHandleJson} from './integrations/dsh/bundle_handles.mjs';
import {verifyBundleNativePresence} from './integrations/dsh/bundle_presence.mjs';
let input='';for await(const chunk of process.stdin)input+=chunk;const v=JSON.parse(input);
const handles=buildHandleManifest(v.case,v.navigation,v.plan,{case_sha256:v.handles.case_sha256,backend_navigation_sha256:v.handles.backend_navigation_sha256});
console.log(JSON.stringify({handles,view:handleReadView(handles,v.reads[0].result,v.projections[0].native_arguments),presence:verifyBundleNativePresence(v.body,v.state)}));
"""
    completed = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT,
        input=packed({**native, "state": state}), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=True)
    actual = json.loads(completed.stdout)
    assert actual["handles"] == native["handles"]
    assert actual["view"] == native["projections"][0]["native_result"]
    assert actual["presence"] == native["presence"]


@pytest.mark.parametrize("mutation", ["handle_coordinate", "handle_quote", "view_quote", "business_result",
    "duplicate_raw_key", "unknown_handle", "submit_annotation", "submit_end", "missing_projection", "repeat_tool_id"])
def test_changed_handle_projection_or_source_mapping_fails_before_adoption(tmp_path, mutation):
    docs, rows, response, native = fixture(tmp_path, repeat=True)
    proof = response["proof"]
    if mutation.startswith("handle_"):
        value = deepcopy(native["handles"])
        if mutation == "handle_coordinate": value["paragraphs"][0]["end_cp"] += 1
        else: value["bundles"][0]["canonical_bundle"]["quote"] += "伪造"
        proof["program_projection"] = descriptor(tmp_path / "changed-handles.json", value)
    else:
        value = deepcopy(native["projections"])
        if mutation == "view_quote": value[0]["native_result"]["bundles"][0]["quote"] += "伪造"
        elif mutation == "business_result": value[0]["canonical_result"]["bundles"][0]["quote"] += "伪造"
        elif mutation == "duplicate_raw_key": value[0]["raw_native_arguments"] = '{"bundle_refs":{"B1":true,"B1":true}}'
        elif mutation == "unknown_handle": value[0]["native_arguments"] = {"bundle_refs": {"B999": True}}
        elif mutation == "submit_annotation": value[-1]["native_arguments"]["selected"]["P1"]["fact_des"] = "被程序换掉的说明"
        elif mutation == "submit_end": value[-1]["native_arguments"]["selected"]["P1"]["end_ref"] = "P999"
        elif mutation == "missing_projection": value.pop(0)
        else: value[-1]["tool_use_id"] = value[0]["tool_use_id"]
        proof["native_projections"] = descriptor(tmp_path / "changed-projections.jsonl", value, lines=True)
    with pytest.raises(ValueError, match="source_bundle_(?:projection|native_projection)"):
        wb.compile_result(Q, rows, docs, response)
