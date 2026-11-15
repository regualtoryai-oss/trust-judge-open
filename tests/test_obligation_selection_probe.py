"""Synthetic injected boundaries only. Fake units are NOT tokenizer measurements."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from app.domain.paragraph_contributions import source_blocks
from app.workflows.obligation_selection_probe import (
    BoundedObligationPlanOutput, ProbeActionOutput, run_obligation_selection_probe,
)


QUESTION = {"question_id": "synthetic", "question": "排期还是程序状态影响进度，是否有关键限定？"}
CATALOG = [{"paragraph_ref": f"P{i}", "doc_id": f"doc{i}", "para_id": "p1", "title": f"原标题{i}",
            "text": f"原始材料{i}：程序状态仍待核实。\r\n😀", "paragraph_complete": True,
            "source_credibility": .9, "private_metadata": "DO_NOT_FORWARD"} for i in (1, 2)]


def graph(ref="P1", span="S1"):
    return {"obligations": [{"obligation_id": "O1", "aspect_id": "A1", "role": "P", "ask": "相关程序状态是什么？"}],
        "judgments": [{"judgment_id": "J1", "proposition_to_test": "某材料提供待核验的程序状态说法",
            "obligation_ids": ["O1"], "depends_on": [], "proposed_state": "unresolved", "links": [{
                "obligation_id": "O1", "paragraph_ref": ref, "span_ref": span, "relation": "limit",
                "usage_kind": "limitation", "usage": "限定程序状态尚未确定"}]}], "answer_depends_on": ["J1"]}


class FakeTools:
    def __init__(self, catalog=CATALOG):
        self.catalog = deepcopy(catalog); self.exposed = []; self.read_refs = []; self.events = []

    def search(self, query, limit=5):
        self.events.append({"operation": "search", "query": query, "status": "completed", "local_only": True})
        rows = self.catalog[:limit]
        for row in rows:
            if row["paragraph_ref"] not in self.exposed: self.exposed.append(row["paragraph_ref"])
        return {"original_question": QUESTION["question"], "results": [{"paragraph_ref": row["paragraph_ref"], "title": row["title"],
            "excerpt_text": row["text"][:80], "paragraph_complete": len(row["text"]) <= 80,
            "unseen_context_state": "unknown_not_irrelevant"} for row in rows]}

    def read(self, refs):
        event = {"operation": "read", "paragraph_refs": refs, "status": "started", "local_only": True}
        self.events.append(event)
        if len(set(refs)) != len(refs) or any(ref not in self.exposed or ref in self.read_refs for ref in refs):
            event["status"] = "rejected"; raise ValueError("PRIVATE_TOOL_ERROR")
        rows = []
        for row in self.catalog:
            if row["paragraph_ref"] in refs:
                self.read_refs.append(row["paragraph_ref"])
                rows.append({**{k: row[k] for k in ("paragraph_ref", "doc_id", "para_id", "title")},
                    "blocks": source_blocks(row["text"]), "paragraph_complete": True,
                    "private_metadata": "DO_NOT_FORWARD"})
        event["status"] = "completed"
        return {"paragraphs": rows}

    def scope(self):
        return {"read_paragraph_refs": list(self.read_refs), "exposed_paragraph_refs": list(self.exposed),
                "model_comprehension_verified": False}

    def trace(self):
        return deepcopy(self.events)


class Boundaries:
    def __init__(self, outputs):
        self.outputs = list(outputs); self.prepared = []; self.requested = []; self.known = 0

    async def prepare(self, spec):
        self.prepared.append(deepcopy(spec))
        return {"opaque_request": spec, "admission": {"admitted": True, "messages_sha256": "a" * 64,
            "input_tokens": 100, "known_episode_tokens": self.known,
            "reserved_output_tokens": spec["max_output_tokens"], "reserved_final_tokens": spec["reserved_final_tokens"],
            "reserved_final_input_tokens": 100 if spec["stage"] == "action" else 0,
            "reserved_tokens": spec["reserved_tokens"], "safety_tokens": spec["safety_tokens"],
            "token_limit": spec["token_limit"], "cost_basis": "actual_final_messages_tokenizer",
            "reasoning": "PRIVATE_REASONING"}}

    async def request(self, opaque):
        self.requested.append(opaque)
        value = self.outputs.pop(0)
        if isinstance(value, Exception): raise value
        self.known += 5
        return {"output": deepcopy(value), "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                "known_episode_tokens_after": self.known, "unknown_usage_requests": 0, "reasoning": "PRIVATE_REASONING",
                "response_completed": True,
                "request_receipt": {"actual_request_started": True, "messages_sha256": "a" * 64,
                    "full_context_refs": [row["paragraph_ref"] for row in opaque["payload"]["full_read_paragraphs"]]}}


async def run(outputs, *, tools=None, boundaries=None, **kwargs):
    tools = tools or FakeTools(); boundary = boundaries or Boundaries(outputs)
    out = await run_obligation_selection_probe(QUESTION, CATALOG, tools=tools, prepare_request_fn=boundary.prepare,
        request_fn=boundary.request, selection_input_token_cost=lambda payload: 50, **kwargs)
    return out, boundary, tools


@pytest.mark.asyncio
async def test_early_finish_is_separate_final_call_full_question_and_actual_tool_trace_preserved():
    out, boundary, tools = await run([{"action": "read", "paragraph_refs": ["P1"]}, {"action": "finish"}, graph()])
    assert out["status"] == "finished_unverified" and out["request_boundary_attempts"] == 3
    assert [spec["stage"] for spec in boundary.requested] == ["action", "action", "final"]
    assert [row["operation"] for row in tools.trace()] == ["search", "read"]
    assert tools.trace()[0]["query"] == QUESTION["question"]
    assert all(spec["payload"]["question"] == {"question": QUESTION["question"]} for spec in boundary.prepared)
    assert boundary.prepared[-1]["payload"]["aspects"][0]["text"] == QUESTION["question"]
    assert out["known_episode_tokens"] == 15
    assert out["selection_plan"]["proposed_selection_refs"] == ["P1"]
    assert out["unread_paragraph_refs"] == ["P2"]
    assert out["bound_full_context_paragraph_refs"] == out["successful_model_context_refs"] == ["P1"]
    assert out["fetched_full_paragraph_refs"] == out["presented_full_paragraph_refs"] == ["P1"]
    assert not out["publishable"] and not out["selection_plan"]["publishable"]


@pytest.mark.asyncio
async def test_real_raw_adapter_and_local_tools_integrate_with_injected_model_boundaries():
    from app.domain.frozen_selection_inputs import POOL_FORMAT, build_frozen_selection_input, canonical
    from app.domain.selection_read_tools import SelectionReadTools
    documents = [{"doc_id": row["doc_id"], "title": row["title"], "content": row["text"], "paragraphs": [{
        "para_id": row["para_id"], "text": row["text"], "char_start": 0, "char_end": len(row["text"])}]}
        for row in CATALOG]
    raw = {"title": "通用合成测试", "documents": documents, "questions": [QUESTION]}; raw_bytes = canonical(raw)
    candidates = [{"id": f"t/{row['doc_id']}/{row['para_id']}", "raw_doc_id": row["doc_id"], "raw_para_id": row["para_id"]}
                  for row in CATALOG]
    pool = {"format": POOL_FORMAT, "reference_reads": [], "inputs": [{"topic_id": "t", "title": raw["title"],
        "sha256": sha256(raw_bytes).hexdigest()}], "cases": [{"topic_id": "t", "question_id": QUESTION["question_id"],
        "title": raw["title"], "question": QUESTION["question"], "outputs": [{"variant": "four_channel_quota400",
            "candidates": candidates, "candidate_sha256": sha256(canonical(candidates)).hexdigest()}]}]}
    pool_bytes = canonical(pool)
    case = build_frozen_selection_input(raw_input_bytes=raw_bytes, pool_input_bytes=pool_bytes, topic_id="t",
        question_id=QUESTION["question_id"], expected_raw_sha256=sha256(raw_bytes).hexdigest(),
        expected_pool_sha256=sha256(pool_bytes).hexdigest())
    tools = SelectionReadTools(case, max_search_calls=6, max_read_calls=5)
    boundary = Boundaries([{"action": "read", "paragraph_refs": ["P1"]}, graph()])
    out = await run_obligation_selection_probe(case["question"], case["catalog"], tools=tools,
        prepare_request_fn=boundary.prepare, request_fn=boundary.request, selection_input_token_cost=lambda payload: 50,
        max_steps=2)
    assert out["status"] == "finished_unverified" and out["fetched_full_paragraph_refs"] == ["P1"]
    assert out["successful_model_context_refs"] == ["P1"]
    assert [row["operation"] for row in out["tool_trace"]] == ["search", "read"]


@pytest.mark.asyncio
async def test_last_available_call_is_final_not_seventh_and_no_implicit_retry():
    actions = [{"action": "read", "paragraph_refs": ["P1"]}] + [{"action": "search", "query": "待查程序"}] * 4
    out, boundary, _ = await run(actions + [graph()])
    assert out["status"] == "finished_unverified" and len(boundary.requested) == 6
    assert boundary.requested[-1]["response_model"] is BoundedObligationPlanOutput
    assert all(spec["response_model"] is ProbeActionOutput for spec in boundary.requested[:-1])
    assert [spec["reserved_final_tokens"] for spec in boundary.prepared] == [4096] * 5 + [0]


@pytest.mark.asyncio
async def test_two_call_limit_read_then_direct_final():
    out, boundary, _ = await run([{"action": "read", "paragraph_refs": ["P1"]}, graph()], max_steps=2)
    assert out["status"] == "finished_unverified" and len(boundary.requested) == 2


@pytest.mark.asyncio
async def test_read_order_does_not_break_original_catalog_order_context_receipts():
    out, boundary, _ = await run([{"action": "read", "paragraph_refs": ["P2"]},
        {"action": "read", "paragraph_refs": ["P1"]}, {"action": "finish"}, graph()])
    assert out["status"] == "finished_unverified"
    assert [row["paragraph_ref"] for row in boundary.prepared[-1]["payload"]["full_read_paragraphs"]] == ["P1", "P2"]
    assert out["presented_full_paragraph_refs"] == out["successful_model_context_refs"] == ["P1", "P2"]


@pytest.mark.asyncio
async def test_no_full_read_means_no_paid_final_and_no_fabricated_success():
    out, boundary, _ = await run([{"action": "finish"}])
    assert out["status"] == "stopped" and out["code"] == "probe_no_full_sources_read"
    assert len(boundary.requested) == 1 and out["final_proposed_plan"] is None
    one, b, _ = await run([], max_steps=1)
    assert one["status"] == "stopped" and not b.prepared and not b.requested


@pytest.mark.asyncio
async def test_action_cannot_spend_closing_input_reservation_but_can_finish_once_without_paying_action():
    b = Boundaries([{"action": "read", "paragraph_refs": ["P1"]}, graph()]); real = b.prepare
    async def prepare(spec):
        result = await real(spec)
        if spec["stage"] == "action" and spec["payload"]["full_read_paragraphs"]:
            result["admission"].update(admitted=False, code="action_budget_reserved_for_final",
                reserved_final_input_tokens=19000)
        return result
    b.prepare = prepare
    out, _, _ = await run([], boundaries=b)
    assert out["status"] == "finished_unverified"
    assert [spec["stage"] for spec in b.prepared] == ["action", "action", "final"]
    assert [spec["stage"] for spec in b.requested] == ["action", "final"]
    assert out["request_boundary_attempts"] == 2 and out["known_episode_tokens"] == 10
    assert out["model_boundary_trace"][1]["status"] == "not_invoked_budget_reserved_for_final"
    assert b.prepared[1]["final_request_spec"]["payload"] == b.prepared[2]["payload"]


@pytest.mark.asyncio
async def test_closing_fallback_never_retries_if_final_itself_over_budget():
    b = Boundaries([{"action": "read", "paragraph_refs": ["P1"]}]); real = b.prepare
    async def prepare(spec):
        result = await real(spec)
        if spec["payload"]["full_read_paragraphs"]:
            if spec["stage"] == "action":
                result["admission"].update(admitted=False, code="action_budget_reserved_for_final",
                    reserved_final_input_tokens=19000)
            else:
                result["admission"].update(admitted=False, code="budget_exhausted", input_tokens=30000)
        return result
    b.prepare = prepare
    out, _, _ = await run([], boundaries=b)
    assert out["status"] == "stopped" and len(b.requested) == 1 and len(b.prepared) == 3
    assert out["fetched_full_paragraph_refs"] == ["P1"] and out["presented_full_paragraph_refs"] == []


@pytest.mark.asyncio
async def test_current_final_shadow_and_real_tool_history_are_payload_bound_not_hidden_reasoning():
    out, b, _ = await run([{"action": "read", "paragraph_refs": ["P1"]}, {"action": "finish"}, graph()])
    action = b.prepared[1]
    assert action["payload"]["program_budget"]["known_episode_tokens"] == 5
    assert action["payload"]["tool_history"][-1] == {"operation": "read", "status": "completed", "paragraph_refs": ["P1"]}
    shadow = action["final_request_spec"]
    assert shadow["response_model"] is BoundedObligationPlanOutput and shadow["max_output_tokens"] == 4096
    assert "latest_search_results" not in shadow["payload"] and "aspects" in shadow["payload"]
    assert "final_request_spec" not in shadow
    assert out["status"] == "finished_unverified"


@pytest.mark.asyncio
async def test_full_source_sent_once_as_blocks_and_private_metadata_or_reasoning_never_persisted():
    out, boundary, _ = await run([{"action": "read", "paragraph_refs": ["P1"]}, {"action": "finish"}, graph()])
    second = boundary.prepared[1]["payload"]
    wire = json.dumps(second, ensure_ascii=False)
    assert wire.count(CATALOG[0]["text"].replace("\r", "\\r").replace("\n", "\\n")) == 1
    assert all(row["paragraph_ref"] != "P1" for row in second["latest_search_results"])
    assert "text" not in second["full_read_paragraphs"][0]
    assert "DO_NOT_FORWARD" not in str(boundary.prepared)
    assert "PRIVATE_REASONING" not in str(out)
    assert "source_credibility" not in str(boundary.prepared)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["over_budget", "unknown", "bool", "wrong_before", "wrong_sha", "wrong_final_reservation"])
async def test_preflight_refusal_prevents_request_boundary(damage):
    b = Boundaries([]); real = b.prepare
    async def prepare(spec):
        result = await real(spec); receipt = result["admission"]
        key, value = {"over_budget": ("input_tokens", 20000), "unknown": ("input_tokens", None),
            "bool": ("input_tokens", True), "wrong_before": ("known_episode_tokens", 1),
            "wrong_sha": ("messages_sha256", "bad"), "wrong_final_reservation": ("reserved_final_tokens", 0)}[damage]
        receipt[key] = value; return result
    b.prepare = prepare
    out, _, tools = await run([], boundaries=b)
    assert out["status"] in {"failed", "stopped"} and not b.requested
    assert out["stage"] == "prepare_action" and tools.trace()[0]["operation"] == "search"


@pytest.mark.asyncio
async def test_unknown_usage_stops_before_next_action_and_does_not_claim_zero_cost():
    b = Boundaries([{"action": "read", "paragraph_refs": ["P1"]}]); real = b.request
    async def request(opaque):
        value = await real(opaque); value["usage"]["total_tokens"] = None; return value
    b.request = request
    out, _, tools = await run([], boundaries=b)
    assert out["status"] == "failed" and out["usage_after_boundary_unknown"]
    assert len(b.requested) == 1 and len(b.prepared) == 1 and tools.scope()["read_paragraph_refs"] == []


@pytest.mark.asyncio
async def test_tools_bound_to_other_question_rejected_before_first_paid_boundary():
    tools = FakeTools(); real = tools.search
    def search(query, limit):
        out = real(query, limit); out["original_question"] = "另一个原问题"; return out
    tools.search = search
    out, boundary, _ = await run([], tools=tools)
    assert out["status"] == "failed" and out["stage"] == "bootstrap" and not boundary.prepared


@pytest.mark.asyncio
async def test_provider_failure_has_stage_prior_known_usage_and_read_scope_without_private_error():
    out, b, _ = await run([{"action": "read", "paragraph_refs": ["P1"]}, RuntimeError("PRIVATE_TOKEN_OR_RESPONSE")])
    assert out["status"] == "failed" and out["stage"] == "request_action"
    assert out["known_episode_tokens"] == 5 and out["usage_after_boundary_unknown"]
    assert out["scope"]["read_paragraph_refs"] == ["P1"] and len(b.requested) == 2
    assert out["fetched_full_paragraph_refs"] == ["P1"]
    assert out["presented_full_paragraph_refs"] == out["successful_model_context_refs"] == []
    assert out["context_submission_unknown"] is True
    assert "PRIVATE_TOKEN_OR_RESPONSE" not in str(out)


@pytest.mark.asyncio
async def test_local_full_read_then_preflight_blocked_is_not_model_reading():
    b = Boundaries([{"action": "read", "paragraph_refs": ["P1"]}]); real = b.prepare
    async def prepare(spec):
        value = await real(spec)
        if spec["payload"]["full_read_paragraphs"]:
            value["admission"]["input_tokens"] = 30000
        return value
    b.prepare = prepare
    out, _, _ = await run([], boundaries=b)
    assert len(b.requested) == 1 and len(b.prepared) == 2 and out["status"] == "stopped"
    assert out["fetched_full_paragraph_refs"] == out["bound_full_context_paragraph_refs"] == ["P1"]
    assert out["presented_full_paragraph_refs"] == out["successful_model_context_refs"] == []
    assert out["context_submission_unknown"] is False  # only first empty-context request was started


@pytest.mark.asyncio
async def test_mismatched_source_receipt_never_claims_model_presented_context():
    b = Boundaries([{"action": "read", "paragraph_refs": ["P1"]}, {"action": "finish"}]); real = b.request
    async def request(opaque):
        value = await real(opaque)
        if opaque["payload"]["full_read_paragraphs"]: value["request_receipt"]["full_context_refs"] = ["P2"]
        return value
    b.request = request
    out, _, _ = await run([], boundaries=b)
    assert out["status"] == "stopped" and out["code"] == "probe_request_source_receipt_invalid"
    assert out["presented_full_paragraph_refs"] == [] and out["usage_after_boundary_unknown"]


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [graph(ref="P2"), graph(span="S2")])
async def test_final_graph_cannot_borrow_unread_parent_or_unprovided_span(output):
    out, boundary, _ = await run([{"action": "read", "paragraph_refs": ["P1"]}, output], max_steps=2)
    assert out["status"] == "failed" and out["stage"] == "validate_final"
    assert out["final_proposed_plan"] is None and out["selection_plan"] is None
    assert out["presented_full_paragraph_refs"] == ["P1"]
    assert out["successful_model_context_refs"] == []  # usage alone does not validate the final graph
    assert len(boundary.requested) == 2 and out["known_episode_tokens"] == 10


@pytest.mark.asyncio
async def test_guessed_unknown_read_rejected_without_hidden_fallback():
    out, boundary, tools = await run([{"action": "read", "paragraph_refs": ["P999"]}])
    assert out["status"] == "failed" and out["stage"] == "tool_read"
    assert tools.trace()[-1]["status"] == "rejected" and len(boundary.requested) == 1


@pytest.mark.asyncio
async def test_local_read_binding_mismatch_prevents_next_model_call():
    tools = FakeTools(); real = tools.read
    def read(refs):
        output = real(refs); output["paragraphs"][0]["blocks"][0]["text"] = "改写内容"; return output
    tools.read = read
    out, boundary, _ = await run([{"action": "read", "paragraph_refs": ["P1"]}], tools=tools)
    assert out["status"] == "failed" and out["stage"] == "tool_read" and len(boundary.requested) == 1
    assert out["bound_full_context_paragraph_refs"] == []


@pytest.mark.asyncio
async def test_invalid_action_not_repaired_or_retried():
    out, boundary, _ = await run([{"action": "search", "query": "待查", "quote": "模型写原文"}])
    assert out["status"] == "failed" and out["stage"] == "validate_action" and len(boundary.requested) == 1


def test_graph_capacity_is_rejected_without_silently_trimming_items():
    value = graph(); value["judgments"][0]["links"] *= 25
    with pytest.raises(ValueError, match="at most 24"):
        BoundedObligationPlanOutput.model_validate(value)


def test_whole_graph_capacity24_accepted25_rejected_and_public_instructions_state_limit():
    from app.workflows.obligation_selection_probe import FINAL_INSTRUCTIONS
    value = graph(); second = deepcopy(value["judgments"][0]); second["judgment_id"] = "J2"
    value["judgments"][0]["links"] *= 12; second["links"] *= 12; value["judgments"].append(second)
    assert sum(len(row.links) for row in BoundedObligationPlanOutput.model_validate(value).judgments) == 24
    value["judgments"][1]["links"].append(deepcopy(second["links"][0]))
    with pytest.raises(ValueError, match="capacity_exceeded_no_truncation"):
        BoundedObligationPlanOutput.model_validate(value)
    assert "links合计至多24条" in FINAL_INSTRUCTIONS


def test_question_conditions_are_not_assumed_true_and_earlier_premises_not_auto_excluded():
    from app.workflows.obligation_selection_probe import ACTION_INSTRUCTIONS, FINAL_INSTRUCTIONS
    for instructions in (ACTION_INSTRUCTIONS, FINAL_INSTRUCTIONS):
        assert "均为待检验命题" in instructions and "不能先当事实排除反证" in instructions
        assert "较早记录仍可为前提" in instructions and "原文缺时间不从题面补全" in instructions


@pytest.mark.asyncio
async def test_usage_settled_without_completed_response_not_successful_context():
    b = Boundaries([{"action": "read", "paragraph_refs": ["P1"]}, graph()]); real = b.request
    async def request(opaque):
        value = await real(opaque)
        if opaque["stage"] == "final": value["response_completed"] = False
        return value
    b.request = request
    out, _, _ = await run([], boundaries=b, max_steps=2)
    assert out["status"] == "stopped" and out["known_episode_tokens"] == 10
    assert out["presented_full_paragraph_refs"] == ["P1"] and out["successful_model_context_refs"] == []
    assert out["usage_after_boundary_unknown"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, 0, 7, 1.5])
async def test_total_step_count_cannot_reset_or_exceed_six(value):
    with pytest.raises(ValueError, match="bounded_total_model_calls"):
        await run([], max_steps=value)
