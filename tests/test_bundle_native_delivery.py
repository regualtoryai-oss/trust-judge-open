"""Same-argument full reads remain separate exact native deliveries."""
from copy import deepcopy
from hashlib import sha256

import pytest

from app.domain.source_bundle_tools import SourceBundleTools
from app.workflows.bundle_workbench import _json, _native_presence
from test_bundle_workbench import Q, sources


def repeated_delivery():
    docs, rows = sources()
    question = {"question_id": Q["raw_question_id"], "question": Q["question"]}
    tools = SourceBundleTools(docs, rows, question)
    tools.search(question["question"], limit=10)
    args = {"anchor_paragraph_ids": [rows[0]["id"]], "before": 0, "after": 1}
    reads = [{"arguments": deepcopy(args), "result": tools.read(**args)} for _ in range(2)]
    question_data = _json({"original_question": question, "source_binding": tools.scope()["binding"]})
    calls, results, receipts = [], [], []
    for i, read in enumerate(reads, 1):
        use_id = f"read-{i}"
        rendered = _json(read["result"], sort=True)
        calls.append({"type": "tool_use", "id": use_id, "name": "read_source_bundles", "input": deepcopy(args)})
        results.append({"type": "tool_result", "tool_use_id": use_id,
            "content": [{"type": "text", "text": rendered}], "is_error": False})
        receipts.append({"tool_use_id": use_id, "native_read_result_sha256": sha256(rendered.encode()).hexdigest(),
            "bundle_ids": [b["bundle_id"] for b in read["result"]["bundles"]],
            "source_bundle_set_sha256": read["result"]["source_bundle_set_sha256"],
            "complete_original_slices_present": True})
    body = {"system": question_data, "tools": [{"name": "submit_bundle_evidence"}], "messages": [
        {"role": "assistant", "content": calls}, {"role": "user", "content": results}]}
    presence = {"format": "dsh-source-bundle-native-presence-v1", "read_receipts": receipts,
        "source_bytes_present_in_current_request": True, "semantic_verified": False, "publishable": False}
    bind_body(body, presence)
    return body, presence, reads, question_data


def bind_body(body, presence):
    presence["native_messages_sha256"] = sha256(_json(body["messages"]).encode()).hexdigest()


def test_identical_arguments_and_bundles_match_unique_real_complete_deliveries():
    body, presence, reads, question_data = repeated_delivery()
    assert reads[0]["arguments"] == reads[1]["arguments"]
    assert reads[0]["result"]["bundles"] == reads[1]["result"]["bundles"]
    assert reads[0]["result"]["source_bundle_set_sha256"] == reads[1]["result"]["source_bundle_set_sha256"]
    assert [r["result"]["delivery"]["complete_delivery_index"] for r in reads] == [1, 2]
    assert presence["read_receipts"][0]["native_read_result_sha256"] != presence["read_receipts"][1]["native_read_result_sha256"]
    _native_presence(body, presence, reads, question_data, final=True)


@pytest.mark.parametrize("mutation", ["reuse_first_delivery", "swap_receipt", "truncate", "split_text",
    "splice_quote", "duplicate_body_result", "ambiguous_replay", "changed_arguments", "reused_receipt",
    "missing_receipt", "changed_delivery_metadata", "duplicate_outside_result"])
def test_full_canonical_result_must_bind_each_read_once_without_repair(mutation):
    body, presence, reads, question_data = repeated_delivery()
    actual = body["messages"][1]["content"][1]
    if mutation == "reuse_first_delivery":
        actual["content"] = deepcopy(body["messages"][1]["content"][0]["content"])
    elif mutation == "swap_receipt":
        presence["read_receipts"][1]["native_read_result_sha256"] = presence["read_receipts"][0]["native_read_result_sha256"]
    elif mutation in {"truncate", "splice_quote", "changed_delivery_metadata"}:
        forged = deepcopy(reads[1]["result"])
        if mutation == "truncate": forged["bundles"][0]["quote"] = forged["bundles"][0]["quote"][:-1]
        elif mutation == "splice_quote": forged["bundles"][0]["quote"] += reads[0]["result"]["bundles"][0]["quote"]
        else: forged["delivery"]["complete_delivery_index"] = 99
        actual["content"][0]["text"] = _json(forged, sort=True)
    elif mutation == "split_text":
        text = actual["content"][0]["text"]
        actual["content"] = [{"type": "text", "text": text[:50]}, {"type": "text", "text": text[50:]}]
    elif mutation == "duplicate_body_result":
        body["messages"][1]["content"].append(deepcopy(actual))
    elif mutation == "ambiguous_replay":
        reads.append(deepcopy(reads[1]))
    elif mutation == "changed_arguments":
        body["messages"][0]["content"][1]["input"]["before"] = 1
    elif mutation == "reused_receipt":
        presence["read_receipts"][1] = deepcopy(presence["read_receipts"][0])
    elif mutation == "missing_receipt":
        presence["read_receipts"].pop()
    else:
        body["messages"].append({"role": "assistant", "content": [{"type": "text", "text": actual["content"][0]["text"]}]})
    bind_body(body, presence)
    with pytest.raises(ValueError, match="source_bundle_native_"):
        _native_presence(body, presence, reads, question_data, final=True)
