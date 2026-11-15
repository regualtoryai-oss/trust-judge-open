"""Synthetic program eligibility tests; no provider/network/answer inputs."""
from copy import deepcopy

import pytest

from app.domain.material_eligibility import (
    Invalid, apply_workflow_material_eligibility, digest,
    validate_workflow_material_eligibility,
)
from app.domain.scope_batches import deduplicate_verified_facts
from app.workflows.analysis import run_analysis
from test_scope_refine_workflow import ScopeProvider, annotation


QUESTION = {"id": "question-test", "question": "仅依据截至2038年5月9日18:00（北京时间）已发布的材料，设备是否通过安全检测？"}


def paragraph(identity, published, *, doc=None, text="设备已经通过安全检测。"):
    return {"id": identity, "document_id": doc or "doc-" + identity,
        "raw_doc_id": "raw-" + (doc or identity), "raw_para_id": "para-" + identity,
        "text": text, "content_sha256": digest(text), "raw_publish_time": published}


def fact(source, *, identity=None):
    return {"id": identity or "evidence-" + source["id"], "paragraph_id": source["id"],
        "raw_doc_id": source["raw_doc_id"], "raw_para_id": source["raw_para_id"],
        "quote": source["text"], "start_cp": 0, "end_cp": len(source["text"]),
        **annotation(source["text"]), "fact_view_id": "view-" + source["id"],
        "scope_verified": True, "interpretation_origin": "model"}


def projected_result(question, paragraphs, facts):
    qualified, audit = apply_workflow_material_eligibility(question, paragraphs, facts)
    selected, duplicates = deduplicate_verified_facts(qualified)
    return {"evidence": selected, "material_eligibility": audit,
        "strict_scope_review": {"exact_annotation_duplicate_map": duplicates}}


def test_gate_preserves_all_original_annotations_and_excludes_late_and_unknown_before_selection():
    paragraphs = [paragraph("early", "2038-05-09T17:00:00+08:00"),
        paragraph("late", "2038-05-10T09:00:00+08:00"), paragraph("unknown", "2038-05-09")]
    facts = [fact(p) for p in paragraphs]
    original = deepcopy((paragraphs, facts))
    selected, audit = apply_workflow_material_eligibility(QUESTION, paragraphs, facts)
    assert [row["raw_doc_id"] for row in selected] == [paragraphs[0]["raw_doc_id"]]
    assert audit["counts"] == {"reviewed": 3, "qualified": 1, "excluded": 1, "unknown": 1}
    assert audit["input_evidence"] == facts
    assert (paragraphs, facts) == original
    statuses = [row["decision"]["status"] for row in audit["decisions"]]
    assert statuses == ["within_cutoff", "after_cutoff", "unknown"]
    assert selected[0]["program_material_eligibility"]["source_authority_assessment"] == "not_performed"


@pytest.mark.parametrize("question", [
    "设备是否通过安全检测？", "截至2038年5月9日，设备是否通过安全检测？",
    "仅判断截至2038年5月9日18:00（北京时间）已发生的事件，设备是否通过安全检测？",
    "仅依据截至2038年5月9日18:00（北京时间）已发布的材料，设备是否通过安全检测？但允许使用截止后发布的材料。",
])
def test_unspecified_event_only_or_overridden_policy_does_not_impose_publication_limit(question):
    paragraphs = [paragraph("late", None)]
    selected, audit = apply_workflow_material_eligibility(question, paragraphs, [fact(paragraphs[0])])
    assert len(selected) == 1 and not audit["applied"]
    assert audit["decisions"] == [] and "program_material_eligibility" not in selected[0]


def test_explicit_policy_with_unknown_cutoff_keeps_unknown_audit_without_throwing():
    paragraphs = [paragraph("first", "2038-05-09T17:00:00+08:00")]
    question = "仅依据截至2038年5月9日已发布的材料，设备情况如何？"
    selected, audit = apply_workflow_material_eligibility(question, paragraphs, [fact(paragraphs[0])])
    assert selected == [] and audit["applied"]
    assert audit["decisions"][0]["decision"]["reason"] == "cutoff_date_only_uncertain"
    assert audit["unknown_evidence_ids"] == ["evidence-first"]


def test_source_dates_come_only_from_canonical_repository_metadata_never_annotation_or_old_decision():
    paragraphs = [paragraph("late", "2038-05-10T09:00:00+08:00")]
    row = fact(paragraphs[0])
    row.update(raw_publish_time="2038-05-01T00:00:00Z", eligible=True,
        program_material_eligibility={"status": "within_cutoff"})
    row["argument"]["时间"] = "2038年5月8日"
    selected, audit = apply_workflow_material_eligibility(QUESTION, paragraphs, [row])
    assert selected == [] and audit["decisions"][0]["decision"]["status"] == "after_cutoff"
    assert "program_material_eligibility" not in audit["input_evidence"][0]


@pytest.mark.parametrize("change", ["source_date", "question", "formal_annotation", "formal_source", "audit", "duplicate_map"])
def test_freeze_publication_or_export_replay_rejects_changed_metadata_or_projection(change):
    paragraphs = [paragraph("early", "2038-05-09T17:00:00+08:00")]
    q, result = deepcopy(QUESTION), projected_result(QUESTION, paragraphs, [fact(paragraphs[0])])
    validate_workflow_material_eligibility(result, q, paragraphs)
    if change == "source_date":
        paragraphs[0]["raw_publish_time"] = "2038-05-10T09:00:00+08:00"
    elif change == "question":
        q["question"] = q["question"].replace("18:00", "16:00")
    elif change == "formal_annotation":
        result["evidence"][0]["fact_des"] = "凭空增加正式结论"
    elif change == "formal_source":
        result["evidence"][0]["raw_doc_id"] = "other-source"
    elif change == "audit":
        result["material_eligibility"]["counts"]["qualified"] = 999
    else:
        result["strict_scope_review"]["exact_annotation_duplicate_map"] = [{"fact_evidence_id": "fake"}]
    with pytest.raises(Invalid):
        validate_workflow_material_eligibility(result, q, paragraphs)


def test_eligible_fact_cannot_be_replaced_with_a_late_source_at_final_boundary():
    paragraphs = [paragraph("early", "2038-05-09T17:00:00+08:00"), paragraph("late", "2038-05-10T09:00:00+08:00")]
    result = projected_result(QUESTION, paragraphs, [fact(p) for p in paragraphs])
    result["evidence"] = [fact(paragraphs[1])]
    with pytest.raises(Invalid, match="eligibility_formal_projection_replay"):
        validate_workflow_material_eligibility(result, QUESTION, paragraphs)


def test_historical_explicit_publication_result_requires_new_gate_but_no_policy_result_stays_compatible():
    paragraphs = [paragraph("late", "2038-05-10T09:00:00+08:00")]
    result = {"evidence": [fact(paragraphs[0])]}
    with pytest.raises(Invalid, match="eligibility_missing_workflow_audit"):
        validate_workflow_material_eligibility(result, QUESTION, paragraphs)
    validate_workflow_material_eligibility(result, "设备是否通过安全检测？", paragraphs)


def test_canonical_document_metadata_conflicts_do_not_let_earliest_alternative_win():
    paragraphs = [paragraph("one", "2038-05-09T17:00:00+08:00", doc="same"),
        paragraph("two", "2038-05-10T09:00:00+08:00", doc="same")]
    with pytest.raises(Invalid, match="eligibility_source_metadata_conflict"):
        apply_workflow_material_eligibility(QUESTION, paragraphs, [fact(p) for p in paragraphs])


@pytest.mark.parametrize("change", ["quote", "raw_doc_id", "raw_para_id", "start_cp"])
def test_gate_rejects_foreign_or_mutated_source_geometry(change):
    paragraphs = [paragraph("early", "2038-05-09T17:00:00+08:00")]
    row = fact(paragraphs[0])
    row[change] = 1 if change == "start_cp" else "changed"
    with pytest.raises(Invalid):
        apply_workflow_material_eligibility(QUESTION, paragraphs, [row])


class AllSourcesProvider(ScopeProvider):
    def __init__(self):
        super().__init__()
        self.analysis_inputs = []

    async def extract(self, question, paragraphs):
        self.calls.append("extract")
        return [{"paragraph_id": row["id"], "quote": row["text"], **annotation(row["text"])} for row in paragraphs]

    async def analyze(self, role, question, evidence, snapshot_id):
        self.analysis_inputs.append(deepcopy(evidence))
        return await super().analyze(role, question, evidence, snapshot_id)


async def test_real_langgraph_applies_gate_before_dedup_freeze_and_analysts(tmp_path, monkeypatch):
    import app.domain.scope_batches as batches
    original = batches.deduplicate_verified_facts
    dedup_inputs = []
    def record(rows):
        dedup_inputs.append(deepcopy(rows))
        return original(rows)
    monkeypatch.setattr(batches, "deduplicate_verified_facts", record)
    paragraphs = [paragraph("early", "2038-05-09T17:00:00+08:00"),
        paragraph("late", "2038-05-10T09:00:00+08:00"), paragraph("unknown", None)]
    provider, events = AllSourcesProvider(), []
    result = await run_analysis(QUESTION, paragraphs, provider, checkpoint_path=tmp_path / "checkpoint.sqlite3",
        thread_id="eligibility-integration", cancel_check=lambda: False, emit=lambda kind, data: events.append((kind, data)))
    assert len(result["candidate_evidence"]) == 3
    assert result["strict_scope_review"]["verified_fact_view_count"] == 3
    assert result["material_eligibility"]["counts"] == {"reviewed": 3, "qualified": 1, "excluded": 1, "unknown": 1}
    assert len(result["evidence"]) == 1
    assert all([row["raw_doc_id"] for row in rows] == ["raw-early"] for rows in dedup_inputs)
    assert len(provider.analysis_inputs) == 2
    assert all(len(rows) == 1 and rows[0]["raw_doc_id"] == "raw-early" for rows in provider.analysis_inputs)
    assert all(not {"source_lineage", "source_counts", "source_inventory_basis"} & set(row)
               for rows in provider.analysis_inputs for row in rows)
    validate_workflow_material_eligibility(result, QUESTION, paragraphs)
    assert any(kind == "material_eligibility_checked" and data["excluded"] == 1 for kind, data in events)


async def test_unknown_publication_metadata_returns_unresolved_result_with_complete_audit(tmp_path):
    paragraphs = [paragraph("unknown", "2038-05-09")]
    result = await run_analysis(QUESTION, paragraphs, AllSourcesProvider(), checkpoint_path=tmp_path / "unknown.sqlite3",
        thread_id="unknown-publication", cancel_check=lambda: False, emit=lambda *_: None)
    assert result["evidence"] == [] and result["final"]["decision"] == "undetermined"
    assert result["material_eligibility"]["counts"]["unknown"] == 1
    assert len(result["candidate_evidence"]) == 1
