from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.domain.frozen_selection_inputs import canonical
from app.domain.source_bundles import paragraph_manifest_sha256
from scripts.source_bundle_tool import decode, execute


def raw_case():
    pieces = ["项目申请已受理😀。", "项目尚未批准，须进一步审核。", "项目后来出现新记录。"]
    content, paragraphs, manifest = "源前缀|", [], []
    for index, text in enumerate(pieces):
        start = len(content)
        content += text
        pid = f"source-{chr(937-index)}"
        paragraphs.append({"id": pid, "document_id": "doc-internal", "raw_doc_id": "原文档",
            "raw_para_id": f"原段{index}", "document_start_cp": start, "document_end_cp": len(content), "text": text})
        manifest.append({"id": pid, "raw_para_id": f"原段{index}", "start_cp": start, "end_cp": len(content),
                         "text_sha256": sha256(text.encode()).hexdigest()})
        content += "\r\n<实际分隔>"
    doc = {"id": "doc-internal", "raw_doc_id": "原文档", "title": "项目程序记录", "content": content,
           "content_sha256": sha256(content.encode()).hexdigest(), "paragraph_manifest": manifest,
           "paragraph_manifest_sha256": paragraph_manifest_sha256(manifest)}
    return {"format": "frozen-source-bundle-case-v1", "question": {"question_id": "q-original", "question": "项目是否已经批准？"},
            "documents": [doc], "paragraphs": paragraphs, "limits": {"max_read_calls": 2}}


def request(case, operation, arguments, history=()):
    data = canonical(case)
    return execute(data, sha256(data).hexdigest(), {"operation": operation, "arguments": arguments, "history": list(history)})


def read_history(case):
    first = request(case, "search", {"query": case["question"]["question"], "limit": 60})
    second = request(case, "read", {"anchor_paragraph_ids": [case["paragraphs"][0]["id"]], "before": 0, "after": 1}, [first["history_entry"]])
    return [first["history_entry"], second["history_entry"]], second


def annotation(case):
    argument = {key: "" for key in ["时间", "主体", "地点", "内容", "影响", "数值口径"]}
    argument["内容"] = case["paragraphs"][0]["text"]
    return {"start_paragraph_id": case["paragraphs"][0]["id"], "end_paragraph_id": case["paragraphs"][1]["id"],
            "fact_des": "仅供后续独立核验的程序状态概括。", "argument": argument,
            "entities": [], "proposed_roles": ["L"]}


def test_replay_reconstructs_read_source_and_program_fills_continuous_quote():
    case = raw_case()
    history, read = read_history(case)
    result = request(case, "submit", {"selected": [annotation(case)]}, history)
    assert result["status"] == "ok"
    proposed = result["result"]
    assert proposed["status"] == "proposed_unverified"
    assert proposed["publishable"] is False
    assert proposed["native_model_delivery_verified"] is False
    assert proposed["basic_literal_grounding_checked"] is True
    assert proposed["annotation_semantics_verified"] is False
    source = proposed["evidence"][0]
    assert source["quote"] == read["result"]["bundles"][0]["quote"]
    assert "\r\n<实际分隔>" in source["quote"]
    assert source["source_paragraph_ids"] == [row["id"] for row in case["paragraphs"][:2]]
    assert result["model_calls"] == 0
    assert "quote" not in json.dumps(history, ensure_ascii=False)


@pytest.mark.parametrize("mutation", ["hash", "arguments", "result_status", "extra"])
def test_history_replay_rejects_mutation(mutation):
    case = raw_case()
    history, _ = read_history(case)
    if mutation == "hash":
        history[1]["result_sha256"] = "0" * 64
    elif mutation == "arguments":
        history[1]["arguments"]["after"] = 0
    elif mutation == "result_status":
        history[1]["status"] = "rejected"
    else:
        history[1]["extra"] = "unknown"
    with pytest.raises(ValueError, match="history_replay_mismatch"):
        request(case, "inspect", {}, history)


def test_rejected_reads_are_replayed_and_do_not_reset_attempt_caps():
    case = raw_case()
    history = []
    for _ in range(2):
        result = request(case, "read", {"anchor_paragraph_ids": ["unknown"], "before": 0, "after": 0}, history)
        assert result["status"] == "rejected"
        history.append(result["history_entry"])
    result = request(case, "read", {"anchor_paragraph_ids": ["unknown"], "before": 0, "after": 0}, history)
    assert result["error_code"] == "source_bundle_tools_read_call_limit_reached"
    assert result["scope"]["returned_complete_paragraph_ids"] == []


@pytest.mark.parametrize("field,value", [("quote", "模型写的引用"), ("start_cp", 0), ("document_content_sha256", "0" * 64), ("path", "C:/unbound.txt")])
def test_model_cannot_supply_quote_offsets_hash_or_path(field, value):
    case = raw_case()
    history, _ = read_history(case)
    row = annotation(case)
    row[field] = value
    response = request(case, "submit", {"selected": [row]}, history)
    assert response["status"] == "rejected"
    assert response["error_code"] == "bundle_bridge_unknown_or_missing_fields"


def test_unread_context_and_duplicate_final_range_rejected():
    case = raw_case()
    row = annotation(case)
    assert request(case, "submit", {"selected": [row]})["status"] == "rejected"
    history, _ = read_history(case)
    duplicate = request(case, "submit", {"selected": [row, deepcopy(row)]}, history)
    assert duplicate["error_code"] == "bundle_bridge_duplicate_selected_range"
    row["end_paragraph_id"] = case["paragraphs"][2]["id"]
    unread = request(case, "submit", {"selected": [row]}, history)
    assert unread["error_code"] == "source_bundle_tools_citation_range_not_completely_returned"


def test_same_start_anchor_with_different_end_is_rejected_without_merging_source_ranges():
    case = raw_case()
    history, _ = read_history(case)
    first = annotation(case)
    second = deepcopy(first)
    first["end_paragraph_id"] = first["start_paragraph_id"]
    response = request(case, "submit", {"selected": [first, second]}, history)
    assert response["status"] == "rejected"
    assert response["error_code"] == "bundle_bridge_duplicate_selected_start_anchor"
    assert "result" not in response


@pytest.mark.parametrize("mutation,code", [
    ("empty_content", "nonempty_argument_content_required"),
    ("invented_time", "time_must_be_literal_or_empty"),
    ("new_fact_number", "annotation_number_not_in_source"),
    ("new_argument_number", "annotation_number_not_in_source"),
    ("new_entity_number", "annotation_number_not_in_source"),
])
def test_literal_annotation_grounding_rejects_without_repair_or_semantic_truth_claim(mutation, code):
    case = raw_case()
    history, _ = read_history(case)
    row = annotation(case)
    if mutation == "empty_content":
        row["argument"]["内容"] = " "
    elif mutation == "invented_time":
        row["argument"]["时间"] = "2026年10月20日"
    elif mutation == "new_fact_number":
        row["fact_des"] = "新增99人。"
    elif mutation == "new_argument_number":
        row["argument"]["内容"] = "新增99人。"
    else:
        row["entities"] = ["99人"]
    original = deepcopy(row)
    response = request(case, "submit", {"selected": [row]}, history)
    assert response["status"] == "rejected"
    assert response["error_code"] == "bundle_bridge_" + code
    assert row == original
    assert "result" not in response


def test_raw_case_changes_hash_or_reference_extra_are_not_admitted():
    case = raw_case()
    data = canonical(case)
    with pytest.raises(ValueError, match="case_sha256_mismatch"):
        execute(data + b" ", sha256(data).hexdigest(), {"operation": "inspect", "arguments": {}, "history": []})
    case["reference_answers"] = []
    with pytest.raises(ValueError, match="unknown_or_missing_fields"):
        request(case, "inspect", {})


@pytest.mark.parametrize("body", [b'{"a":1,"a":2}', b'{"a":NaN}', b'\xef\xbb\xbf{}'])
def test_input_json_is_lossless_and_rejects_duplicate_or_nonfinite_fields(body):
    with pytest.raises((ValueError, UnicodeError)):
        decode(body)


def test_actual_isolated_cli_replays_bound_case_and_never_needs_credentials(tmp_path):
    case = raw_case()
    path = tmp_path / "raw-case.json"
    data = canonical(case)
    path.write_bytes(data)
    script = Path(__file__).resolve().parents[1] / "scripts/source_bundle_tool.py"
    payload = canonical({"operation": "inspect", "arguments": {}, "history": []})
    result = subprocess.run([sys.executable, "-I", str(script), str(path), sha256(data).hexdigest()], input=payload,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    assert result.returncode == 0
    response = json.loads(result.stdout)
    assert response["status"] == "ok"
    assert response["scope"]["corpus_paragraph_count"] == 3
    assert response["model_calls"] == 0
    assert response["publishable"] is False
