"""Only synthetic sources/receipts/reference objects. No competition gold reads."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from app.domain.frozen_selection_inputs import build_frozen_selection_input, canonical, POOL_FORMAT
from app.domain.obligation_selection import plan_obligation_selection
from app.domain.paragraph_contributions import source_blocks
from scripts import score_obligation_selection_probe as scorer


def fixture(tmp_path):
    def write(name, value):
        path = tmp_path/name
        path.write_bytes(canonical(value))
        return {"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    text = "合成程序材料😀。\r\n"
    raw = {"title": "全新合成主题", "questions": [{"question_id": "q", "question": "程序是否完成？"}],
        "documents": [{"doc_id": "d", "title": "完整原标题", "content": text, "paragraphs": [
            {"para_id": "p", "char_start": 0, "char_end": len(text), "text": text}]}]}
    raw_desc = write("raw.json", raw)
    candidates = [{"id": "t/d/p", "raw_doc_id": "d", "raw_para_id": "p"}]
    pool = {"format": POOL_FORMAT, "reference_reads": [], "inputs": [{"topic_id": "t", "title": raw["title"], "sha256": raw_desc["sha256"]}],
        "cases": [{"topic_id": "t", "question_id": "q", "title": raw["title"], "question": raw["questions"][0]["question"],
            "outputs": [{"variant": "four_channel_quota400", "limit": 400, "candidates": candidates, "candidate_sha256": scorer.digest(candidates)}]}]}
    pool_desc = write("pool.json", pool)
    case = build_frozen_selection_input(raw_input_bytes=Path(raw_desc["file"]).read_bytes(), pool_input_bytes=Path(pool_desc["file"]).read_bytes(),
        topic_id="t", question_id="q", expected_raw_sha256=raw_desc["sha256"], expected_pool_sha256=pool_desc["sha256"])
    case_desc = write("case.json", case)
    index = write("index.json", {"cases": [case_desc], "reference_reads": []})
    freeze = write("freeze.json", {"status": "CLOSED_RAW_INPUT_MAPPING_ONLY", "index": index, "reference_reads": []})
    tool = write("synthetic-tool.json", {"fixture": "not production or reference"})
    files = {d["file"]: d["sha256"] for d in (raw_desc, pool_desc, case_desc, index, freeze)}
    config = {"case_input": case_desc["file"], "case_sha256": case_desc["sha256"], "input_index": index["file"],
        "input_freeze": freeze["file"], "raw_input": raw_desc["file"], "pool_input": pool_desc["file"], "prior_receipts": []}
    protocol = {"format": "one-uniform-tool-reading-probe-v1", "config": config, "bound_input_files": files,
        "source_bundle": {"files": {tool["file"]: tool["sha256"]}, "sha256": scorer.digest({tool["file"]: tool["sha256"]})},
        "case_identity": {"topic_id": "t", "question_id": "q"}, "prior": {"tokens": 0}, "reference_reads": []}
    protocol_desc = write("probe-protocol.json", protocol)
    scope = {"exposed_paragraph_refs": ["P1"], "read_paragraph_refs": ["P1"]}
    events = [{"call_index": 1, "operation": "search", "arguments": {"query": raw["questions"][0]["question"], "limit": 5},
        "status": "completed", "returned_paragraph_refs": ["P1"], "local_only": True, "model_calls": 0},
        {"call_index": 2, "operation": "read", "arguments": {"paragraph_refs": ["P1"]}, "status": "completed",
         "returned_paragraph_refs": ["P1"], "local_only": True, "model_calls": 0}]
    parent = {k: case["catalog"][0][k] for k in ("paragraph_ref", "doc_id", "para_id", "title")}
    parent.update(blocks=source_blocks(text), paragraph_complete=True, parent_codepoints=len(text), semantic_verified=False)
    messages = [{"role": "system", "content": "合成系统契约"}, {"role": "human", "content": json.dumps({
        "question": {"question": raw["questions"][0]["question"]}, "full_read_paragraphs": [parent]}, ensure_ascii=False)}]
    messages_desc = write("messages.json", messages)
    admission = {"admitted": True, "messages_sha256": scorer.digest(messages)}
    pre = write("preflight.json", {"protocol_sha256": protocol_desc["sha256"], "admission": admission,
        "message_file": messages_desc, "counter_receipts": []})
    ledger = {"actual_request_started": True, "usage_known": True, "input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
        "message_content_sha256": scorer.digest(messages), "outcome": "success", "stop_reason": None}
    step = {"messages": messages_desc, "preflight": pre, "stage": "final", "messages_sha256": scorer.digest(messages),
        "full_context_refs": ["P1"], "search_excerpt_refs": [], "complete_search_excerpt_refs": [],
        "actual_request_started": True, "response_completed": True, "usage_settled": True, "unknown_usage_requests": 0,
        "settlement": {k: ledger[k] for k in ("usage_known", "input_tokens", "output_tokens", "total_tokens", "outcome", "stop_reason")}}
    step["receipt"] = write("request-receipt.json", deepcopy(step))
    graph = {"obligations": [{"obligation_id": "O1", "aspect_id": "A1", "role": "P", "ask": "程序状态？"}],
        "judgments": [{"judgment_id": "J1", "proposition_to_test": "程序说法待核验", "obligation_ids": ["O1"], "depends_on": [],
            "proposed_state": "unresolved", "links": [{"obligation_id": "O1", "paragraph_ref": "P1", "span_ref": "S1", "relation": "limit",
                "usage_kind": "limitation", "usage": "限定状态"}]}], "answer_depends_on": ["J1"]}
    selection = plan_obligation_selection(case["question"], case["catalog"], graph, input_token_cost=lambda _: 10,
        token_limit=20000, reserved_tokens=537, read_refs=["P1"])
    result = {"status": "finished_unverified", "usage_after_boundary_unknown": False, "known_episode_tokens": 25,
        "tool_trace": events, "scope": scope, "fetched_full_paragraph_refs": ["P1"], "presented_full_paragraph_refs": ["P1"],
        "bound_full_context_paragraph_refs": ["P1"], "final_proposed_plan": graph, "selection_plan": selection}
    artifact = write("result.json", result)
    final = {"closed": True, "source_unchanged": True, "reference_reads": [], "publishable": False, "formal_evidence_generated": False,
        "status": "completed_unverified", "protocol_sha256": protocol_desc["sha256"], "case_sha256": case_desc["sha256"],
        "guard": {"unknown_usage_requests": 0, "pending_request": None, "actual_requests_started": 1, "known_total_tokens": 25, "records": [ledger]},
        "prior_known_tokens": 0, "whole_episode_total_tokens": 25, "steps": [step], "preflights": [{"messages": messages_desc, "preflight": pre, "admission": admission}],
        "tool_trace": events, "tool_scope": scope, "output_artifact": artifact}
    final_desc = write("probe-final-receipt.json", final)
    return {"write": write, "final": final, "descriptor": final_desc, "case": case, "protocol": protocol, "result": result, "messages": messages}


def reference(case):
    return [{"fact_id": "f1", "doc_id": "d", "para_id": "p"}, {"fact_id": "f2", "doc_id": "d", "para_id": "p"}], {
        "question_id": "q", "question": case["question"]["question"], "supporting_facts": [{"fact_id": "f1"}, {"fact_id": "f2"}],
        "conflicting_facts": [{"fact_id": "f1"}, {"doc_id": "d", "para_id": "missing"}]}, []


def run(f, loader=reference):
    d = f["descriptor"]
    return scorer.score_closed_probe(d["file"], d["sha256"], reference_loader=loader)


def test_unique_source_denominator_and_stage_only_coverage(tmp_path):
    f = fixture(tmp_path); result = run(f)
    assert result["reference_raw_union_records"] == 3
    assert result["selected_source_A_proxy"]["reference_records"] == 2
    assert result["selected_source_A_proxy"]["matches"] == 1
    assert result["stage_source_coverage"]["fixed_pool"]["recall"] == .5
    assert "precision" not in result["stage_source_coverage"]["fixed_pool"]
    assert result["misses"] == [{"doc_id": "d", "para_id": "missing", "reason": "pool_missing"}]
    assert not result["semantic_score"] and not result["final_evidence_or_relation_score"]


@pytest.mark.parametrize("change", ["notclosed", "pending", "unknown", "usage", "total", "source", "artifact", "step", "messages", "protocol", "expectedsha"])
def test_any_incomplete_binding_blocks_before_reference(tmp_path, change):
    f = fixture(tmp_path); final = f["final"]; calls = []
    if change == "notclosed": final["closed"] = False
    elif change == "pending": final["guard"]["pending_request"] = {}
    elif change == "unknown": final["guard"]["unknown_usage_requests"] = 1
    elif change == "usage": final["guard"]["records"][0]["usage_known"] = False
    elif change == "total": final["whole_episode_total_tokens"] = 26
    elif change == "source": Path(f["protocol"]["config"]["raw_input"]).write_text("{}", encoding="utf-8")
    elif change == "artifact": Path(final["output_artifact"]["file"]).write_text("{}", encoding="utf-8")
    elif change == "step": Path(final["steps"][0]["receipt"]["file"]).write_text("{}", encoding="utf-8")
    elif change == "messages": Path(final["steps"][0]["messages"]["file"]).write_text("[]", encoding="utf-8")
    elif change == "protocol": Path(tmp_path/"probe-protocol.json").write_text("{}", encoding="utf-8")
    f["descriptor"] = f["write"]("probe-final-receipt.json", final)
    if change == "expectedsha": f["descriptor"]["sha256"] = "0"*64
    with pytest.raises((ValueError, KeyError)):
        run(f, lambda case: calls.append(case) or reference(case))
    assert calls == []


def test_receipt_with_rehashed_but_nonliteral_context_still_blocks_before_gold(tmp_path):
    f = fixture(tmp_path); final = f["final"]; messages = f["messages"]
    payload = json.loads(messages[1]["content"]); payload["full_read_paragraphs"][0]["blocks"][0]["text"] += "编造"
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
    m = f["write"]("messages.json", messages); h = scorer.digest(messages)
    a = {"admitted": True, "messages_sha256": h}
    pre = f["write"]("preflight.json", {"protocol_sha256": final["protocol_sha256"], "admission": a, "message_file": m, "counter_receipts": []})
    final["preflights"] = [{"messages": m, "preflight": pre, "admission": a}]
    step = final["steps"][0]; step.update(messages=m, preflight=pre, messages_sha256=h)
    step["receipt"] = f["write"]("request-receipt.json", {k: v for k,v in step.items() if k != "receipt"})
    final["guard"]["records"][0]["message_content_sha256"] = h
    f["descriptor"] = f["write"]("probe-final-receipt.json", final)
    calls = []
    with pytest.raises(ValueError, match="blocks_changed"): run(f, lambda case: calls.append(case))
    assert not calls


def test_unexposed_fetch_is_rejected_before_reference(tmp_path):
    f = fixture(tmp_path); f["final"]["tool_trace"][0]["returned_paragraph_refs"] = []
    f["descriptor"] = f["write"]("probe-final-receipt.json", f["final"])
    calls = []
    with pytest.raises(ValueError, match="previous_exposure"): run(f, lambda case: calls.append(case))
    assert not calls


def test_known_closed_stop_with_no_output_has_empty_selection_not_success(tmp_path):
    f = fixture(tmp_path); final = f["final"]
    final.update(status="stopped", output_artifact=None)
    f["descriptor"] = f["write"]("probe-final-receipt.json", final)
    result = run(f)
    assert result["probe_status"] == "stopped"
    assert result["selected_source_A_proxy"]["submitted_records"] == 0
    assert result["stage_source_coverage"]["completed_response_full_context"]["matches"] == 1
    assert {r["reason"] for r in result["misses"]} == {"not_linked", "pool_missing"}


def test_reference_relation_question_mismatch_never_scored(tmp_path):
    f = fixture(tmp_path)
    def wrong(case):
        evidence, conclusion, guards = reference(case); conclusion["question"] = "另一道题"
        return evidence, conclusion, guards
    with pytest.raises(ValueError, match="question_changed"): run(f, wrong)


def test_cli_reference_loader_uses_adapter_topic_title_not_invented_title(tmp_path):
    f = fixture(tmp_path); case = f["case"]
    assert "topic_title" in case and "title" not in case
    sample_root = tmp_path/"synthetic-reference-only"; folder = sample_root/case["topic_title"]
    folder.mkdir(parents=True)
    evidence, conclusion, _ = reference(case)
    (folder/"evidence_samples.json").write_bytes(canonical(evidence))
    (folder/"conclusions.json").write_bytes(canonical([conclusion]))
    result = run(f, lambda current: scorer.load_reference(current, sample_root))
    assert result["selected_source_A_proxy"]["matches"] == 1
    assert len([r for r in result["input_guards"] if Path(r["file"]).parent == folder]) == 2
