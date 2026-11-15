"""Explicit v18 capability doubles exercise checkpoints and authorization."""
from copy import deepcopy
import json
import socket

import pytest
import pytest_asyncio

from app.agents.provider import capture_usage, request_output_budget, ModelOutputInvalid
from app.domain.premise_bank import build_fact_purpose_request, resolve_fact_purpose
from app.domain.strict_verification import build_scope_request, resolve_scope_verification, build_narrowing_request, resolve_annotation_narrowing
from app.workflows.analysis import run_analysis, AnalysisCancelled
from app.domain.material_eligibility import validate_workflow_material_eligibility
from test_deepseek_provider import intercepted_adapter
from test_premise_bank import Q, source, paragraphs, decision, bank_for


class V18Provider:
    premise_bank_enabled = True

    def __init__(self, rows, *, pause=None, corrupt=False):
        self.rows, self.pause, self.corrupt = deepcopy(rows), pause, corrupt
        self.purpose_calls = []
        self.analyst_inputs = []

    async def extract(self, question, batch):
        ids = {row["id"] for row in batch}
        return [{key: deepcopy(row[key]) for key in ("paragraph_id", "quote", "start_cp", "end_cp", "fact_des", "argument", "entities")}
                for row in self.rows if row["paragraph_id"] in ids]

    async def verify_evidence_scope(self, question, evidence):
        payload, _, views = build_scope_request(question, evidence, detailed=True)
        return resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": {
            f"C{i + 1}": decision(view) for i, view in enumerate(views)}}, question, evidence)

    async def narrow_evidence_annotations(self, question, evidence, ids):
        payload, _, views = build_narrowing_request(question, evidence, ids)
        return resolve_annotation_narrowing({"candidate_catalog_id": payload["candidate_catalog_id"], "corrections": {
            f"C{i + 1}": {"status": "no_supported_annotation", "annotations": []} for i, _view in enumerate(views)}}, question, evidence, ids)

    async def review_fact_purpose(self, question, bank, ids):
        self.purpose_calls.append(list(ids))
        payload, _, _ = build_fact_purpose_request(question, bank, ids)
        ledger = resolve_fact_purpose({"bank_id": bank["bank_id"], "purpose_catalog_id": payload["purpose_catalog_id"], "decisions": {
            row["id"]: {"purpose_binding_id": "background" if "外墙" in row["quote"] else "T0.necessary_premise",
                "introduces_unstated_source_fact": False, "equivalent_to": None,
                "reason": "原文具体功能可作分类前提；外墙颜色不决定该分类。"}
            for row in payload["candidate_catalog"]}}, question, bank, ids)
        if self.corrupt: ledger["selected_evidence_ids"] = []
        if self.pause is not None: self.pause[0] = True
        return ledger

    async def analyze(self, role, question, evidence, snapshot_id):
        self.analyst_inputs.append((role, deepcopy(evidence), snapshot_id))
        return self.answer()

    async def arbitrate(self, question, evidence, analyses):
        return {**self.answer(), "conflicts": []}

    @staticmethod
    def answer():
        return {"decision": "undetermined", "conclusion": "材料体现部分用途，完整设施分类仍待核实。", "judgments": [],
                "supporting_evidence_ids": [], "conflicting_evidence_ids": []}


async def run(tmp_path, provider, *, identity="run", cancel=lambda: False, resume=None):
    return await run_analysis(provider=provider, question=Q, paragraphs=paragraphs(provider.rows),
        checkpoint_path=tmp_path / "checkpoints.sqlite3", thread_id=identity,
        emit=lambda *_: None, cancel_check=cancel, strict_scope_enabled=True,
        resume_from_thread_id=resume)


async def test_v18_full_graph_keeps_old_empty_selection_but_permits_independent_premises(tmp_path):
    provider = V18Provider([source(), source("b", text="场馆外墙为蓝色。")])
    result = await run(tmp_path, provider)
    assert result["strict_scope_review"]["selected_fact_view_ids"] == []
    assert len(result["quote_faithful_bank"]["rows"]) == 2
    assert len(result["evidence"]) == 1
    assert len(provider.purpose_calls) == 1
    assert provider.analyst_inputs[0][1:] == provider.analyst_inputs[1][1:]
    assert provider.analyst_inputs[0][1] == result["evidence"]
    assert "外墙" not in provider.analyst_inputs[0][1][0]["quote"]
    assert result["purpose_review"]["complete"]
    validate_workflow_material_eligibility(result, Q, paragraphs(provider.rows))


async def test_purpose_batches_are_durable_and_resume_does_not_repeat_completed_paid_work(tmp_path, monkeypatch):
    import app.domain.premise_bank as module
    original = module.plan_fact_purpose_batches
    monkeypatch.setattr(module, "plan_fact_purpose_batches", lambda question, bank: original(question, bank, max_rows=1))
    paused = [False]
    provider = V18Provider([source(), source("b")], pause=paused)
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, identity="before-pause", cancel=lambda: paused[0])
    assert len(provider.purpose_calls) == 1 and provider.analyst_inputs == []
    provider.pause = None
    paused[0] = False
    result = await run(tmp_path, provider, identity="after-pause", resume="before-pause")
    assert len(provider.purpose_calls) == 2
    assert not set(provider.purpose_calls[0]) & set(provider.purpose_calls[1])
    assert len(result["purpose_review"]["ledgers"]) == 2
    assert len(result["evidence"]) == 2


async def test_v18_rejects_forged_purpose_selection_before_analysts_or_freeze(tmp_path):
    provider = V18Provider([source()], corrupt=True)
    with pytest.raises(ModelOutputInvalid) as failure:
        await run(tmp_path, provider)
    assert failure.value.stage == "purpose_review"
    assert len(provider.purpose_calls) == 1 and provider.analyst_inputs == []


async def test_true_v18_capability_cannot_silently_fall_back_when_purpose_api_missing(tmp_path):
    from app.agents.provider import ModelNotConfigured
    provider = V18Provider([source()])
    provider.review_fact_purpose = None
    with pytest.raises(ModelNotConfigured):
        await run(tmp_path, provider)


@pytest_asyncio.fixture
async def block_socket(monkeypatch):
    def blocked(*_args, **_kwargs): pytest.fail("HTTP interception must never use a network socket")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def sdk_reply(*, duplicate=False, missing=False):
    def reply(body):
        payload = json.loads(body["messages"][1]["content"])
        decisions = {row["id"]: {"purpose_binding_id": "T0.necessary_premise", "introduces_unstated_source_fact": False,
            "equivalent_to": None, "reason": "该对象的原文功能作为分类前提，不将未述分类伪称为原文事实。"}
            for row in payload["candidate_catalog"]}
        if missing: decisions = {}
        result = json.dumps({"bank_id": payload["bank_id"], "purpose_catalog_id": payload["purpose_catalog_id"],
                             "decisions": decisions}, ensure_ascii=False)
        return result.replace('"C1":', '"C1": {}, "C1":', 1) if duplicate else result
    return reply


async def test_real_sdk_purpose_is_independent_of_old_target_labels_and_records_budget_usage(block_socket):
    _rows, _audit, bank = bank_for(target="direct_answer", target_bridge=True)
    ids = [row["id"] for row in bank["rows"]]
    async with intercepted_adapter(content=sdk_reply(), reasoning_effort="low", premise_bank_enabled=True) as (provider, requests):
        with capture_usage() as usage, request_output_budget(65536):
            result = await provider.review_fact_purpose({**Q, "conclusion": "PRIVATE_GOLD"}, bank, ids)
    assert result["selected_evidence_ids"] == ids and len(requests) == len(usage) == 1
    wire = json.loads(requests[0].content)
    payload = json.loads(wire["messages"][1]["content"])
    assert wire["max_tokens"] == 65536 and wire["thinking"] == {"type": "enabled"}
    assert usage[0]["operation"] == "review_fact_purpose"
    assert "PRIVATE" not in requests[0].content.decode()
    for forbidden in ("original_scope", "target_requires_unstated_bridge", "quote_requires_unstated_bridge", "selected", "field_checks", "analyses"):
        assert forbidden not in json.dumps(payload)


@pytest.mark.parametrize("kind", ["duplicate", "missing"])
async def test_purpose_http_rejects_incomplete_or_duplicate_decisions_with_safe_errors(block_socket, kind):
    _rows, _audit, bank = bank_for()
    ids = [row["id"] for row in bank["rows"]]
    async with intercepted_adapter(content=sdk_reply(duplicate=kind == "duplicate", missing=kind == "missing"), premise_bank_enabled=True) as (provider, requests):
        with pytest.raises(ModelOutputInvalid) as failure:
            await provider.review_fact_purpose(Q, bank, ids)
    assert len(requests) == 1 and failure.value.stage == "purpose_review"
    assert failure.value.code == "model_output_invalid"
