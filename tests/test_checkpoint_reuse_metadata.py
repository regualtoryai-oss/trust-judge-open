"""Synthetic checkpoint receipts never represent new model work or inference."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.workflows.analysis import run_analysis
from app.workflows.checkpoint_resume import fork_checkpoint
from test_premise_bank import Q, source, paragraphs, bank_for, purpose, result_for
from test_premise_bank_workflow import V18Provider, run


class MemorySaver:
    def __init__(self, channels):
        self.channels = deepcopy(channels)

    async def aget_tuple(self, config):
        if config["configurable"]["thread_id"] != "source":
            return None
        return SimpleNamespace(checkpoint={"channel_values": self.channels, "channel_versions": {}},
            pending_writes=[], config={"configurable": {"checkpoint_id": "stored-checkpoint"}}, metadata={})

    async def aput(self, *args):
        return {"configurable": {"thread_id": "target"}}

    async def aput_writes(self, *args):
        raise AssertionError("This fixture has no pending writes")


async def receipt(projection):
    initial = {"question": Q, "paragraphs": paragraphs(projection["candidate_evidence"]),
        "manual_candidates": [], "interpretation_overrides": {}, "candidate_artifact": None}
    channels = {**deepcopy(initial), **deepcopy(projection), "workflow_input_sha256": "fixture-bound-input"}
    return await fork_checkpoint(MemorySaver(channels), source_thread_id="source", target_thread_id="target",
        initial_state=initial, input_sha256="fixture-bound-input", snapshot_id=lambda _: "new-snapshot")


@pytest.mark.parametrize("role", ["necessary_premise", "background"])
async def test_completed_purpose_counts_replay_stored_projection_including_zero(role):
    rows, audit, bank = bank_for()
    projection = result_for(rows, audit, bank, [purpose(bank, role=role)])
    before = deepcopy(projection)
    result = await receipt(projection)
    assert result["reused_purpose_coverage"] == {"bank_id": bank["bank_id"], "bank_fact_count": 1,
        "formal_fact_count": int(role != "background"), "complete": True, "reused": True}
    assert result["kind"] == "checkpoint_continuation"
    assert result["checkpoint_id"] == "stored-checkpoint"
    assert projection == before


@pytest.mark.parametrize("damage", ["incomplete", "missing_bank", "changed_bank", "changed_formal", "missing_material", "incomplete_ledger"])
async def test_unprovable_purpose_counts_remain_absent(damage):
    rows, audit, bank = bank_for()
    projection = result_for(rows, audit, bank, [purpose(bank)])
    if damage == "incomplete": projection["purpose_review"]["complete"] = False
    elif damage == "missing_bank": projection.pop("quote_faithful_bank")
    elif damage == "changed_bank": projection["quote_faithful_bank"]["rows"][0]["fact_des"] += "伪造"
    elif damage == "changed_formal": projection["evidence"] = []
    elif damage == "missing_material": projection.pop("material_eligibility")
    elif damage == "incomplete_ledger": projection["purpose_review"]["ledgers"] = []
    assert "reused_purpose_coverage" not in await receipt(projection)


async def test_completed_real_v18_graph_resume_reports_reused_counts_without_new_calls(tmp_path):
    rows = [source(), source("b", text="场馆外墙为蓝色。")]
    first = await run(tmp_path, V18Provider(rows), identity="original")
    resumed = V18Provider(rows)

    async def unexpected(*args, **kwargs):
        pytest.fail("A complete checkpoint must not repeat a provider call")

    for name in ("extract", "verify_evidence_scope", "narrow_evidence_annotations", "review_fact_purpose", "analyze", "arbitrate"):
        setattr(resumed, name, unexpected)
    events = []
    result = await run_analysis(provider=resumed, question=Q, paragraphs=paragraphs(rows),
        checkpoint_path=tmp_path / "checkpoints.sqlite3", thread_id="continued",
        emit=lambda kind, data: events.append((kind, data)), cancel_check=lambda: False,
        strict_scope_enabled=True, resume_from_thread_id="original")
    reused = next(data for kind, data in events if kind == "checkpoint_reused")
    assert reused["reused_purpose_coverage"] == {"bank_id": first["quote_faithful_bank"]["bank_id"],
        "bank_fact_count": 2, "formal_fact_count": 1, "complete": True, "reused": True}
    assert result["usage"] == []
    assert result["evidence"] == first["evidence"]
    assert not any(kind.startswith("tool_") or kind in {"node_started", "node_completed", "purpose_review_progress"}
        for kind, _ in events)
