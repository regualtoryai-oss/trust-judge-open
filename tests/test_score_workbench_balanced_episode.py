"""Offline v2 regression coverage with synthetic sources and zero model calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import score_workbench_real_episode as scorer
from scripts.workbench_source_archive import SourceArchive


SOURCE_FILES = (
    "app/agents/strict_evidence.py", "app/agents/compact_selection.py", "app/agents/schemas.py",
    "app/agents/scoped_arbitration.py", "app/domain/numeric_literals.py", "app/domain/evidence_balance.py",
    "app/domain/material_eligibility.py", "app/domain/paragraph_ranked.py", "app/domain/retrieval.py",
    "app/domain/source_metadata.py", "app/domain/question_recall.py", "app/workflows/strict_workbench.py",
)


def archive(tmp_path):
    result = {}
    for name in SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((scorer.ROOT / name).read_bytes())
        result[name] = scorer.bound(path)[1]
    return result


def balanced_fixture():
    from app.agents.compact_selection import canonical_compact_selection
    from app.agents.strict_evidence import canonical_annotation_wire
    from app.workflows.strict_workbench import annotation_inputs, compile_result, prepare_case
    question = {"id": "internal-q", "raw_question_id": "q", "question": "甲设备验收已经确认还是尚未确认？"}
    paragraphs = [{"id": f"p{i}", "document_id": f"d{i // 3}", "raw_doc_id": f"doc{i // 3}",
        "raw_para_id": f"para{i}", "title": "甲设备验收", "text": "主管部门表示甲设备验收尚未确认。",
        "source_credibility_json": "0.95", "raw_publish_time": "未知"} for i in range(12)]
    raw = prepare_case(question, paragraphs, recall_strategy="balanced")
    wire = {"hypothesis": "甲设备验收已经确认", "decisions": {
        row["paragraph_ref"]: (5, 6, 7)[i % 3] for i, row in enumerate(raw["catalog"])}}
    full = canonical_compact_selection(raw, wire).model_dump()
    policy = {"target": 3, "per_doc_soft_cap": 1}
    subset, chosen, _ledger = annotation_inputs(raw, full, policy)
    annotations = {"annotations": {row["paragraph_ref"]: {"fact_des": "主管部门表示验收尚未确认。",
        "argument": {"时间": "", "主体": "主管部门", "地点": "", "内容": "验收尚未确认", "影响": "", "数值口径": ""},
        "entities": ["甲设备"]} for row in subset["catalog"]}}
    annotation = canonical_annotation_wire(subset, chosen, annotations).model_dump()
    result = compile_result(question, paragraphs, raw, full, annotation,
        balance_policy=policy, recall_strategy="balanced")
    responses = [{"tool_calls": [{"args": wire}]}, {"tool_calls": [{"args": annotations}]}]
    hashes = {name: scorer.bound(scorer.ROOT / name)[1]["sha256"] for name in SOURCE_FILES}
    return question, paragraphs, raw, result, responses, hashes, policy


def verify(raw, result, responses, hashes, policy):
    return scorer.verify_annotations(result, raw, responses,
        decoder_sha256=hashes["app/agents/strict_evidence.py"], source_hashes=hashes,
        expected_balance_policy=policy, expected_recall_strategy="balanced")


def test_v2_replays_archived_integer_decoder_balance_and_annotations_even_when_live_code_changes(tmp_path, monkeypatch):
    question, paragraphs, raw, result, responses, hashes, policy = balanced_fixture()
    descriptors = archive(tmp_path)
    import app.workflows.strict_workbench as live_workflow
    import app.agents.compact_selection as live_codec
    original = deepcopy((result, responses))
    monkeypatch.setattr(live_workflow, "annotation_inputs", lambda *_args: pytest.fail("mutable_live_workflow"))
    monkeypatch.setattr(live_codec, "canonical_compact_selection", lambda *_args: pytest.fail("mutable_live_codec"))
    with SourceArchive(descriptors):
        audit = verify(raw, result, responses, hashes, policy)
        workflow = scorer.bound_code_module("app/workflows/strict_workbench.py", hashes["app/workflows/strict_workbench.py"])
        assert workflow.validate_result(result, question, paragraphs) == result
    assert audit["complete_candidate_decisions"] == 12
    assert audit["actual_model_eligible_parents"] == 12
    assert audit["actual_selected_parents"] == 3
    assert (result, responses) == original


@pytest.mark.parametrize("change", ["policy", "ledger", "subset_hash", "proof_hash", "selection_field", "missing_wire_key"])
def test_v2_rejects_forged_projection_or_incomplete_full_model_selection(change):
    _q, _p, raw, result, responses, hashes, policy = balanced_fixture()
    proof = result["strict_evidence_proof"]
    if change == "policy": policy = {"target": 4, "per_doc_soft_cap": 1}
    elif change == "ledger": proof["balance_ledger"]["retained_refs"].pop()
    elif change == "subset_hash": proof["annotation_raw_case_sha256"] = "0" * 64
    elif change == "proof_hash": proof["proof_sha256"] = "0" * 64
    elif change == "selection_field": result["evidence"][0]["selection_relevance"] = 0
    else: responses[0]["tool_calls"][0]["args"]["decisions"].pop("P12")
    with pytest.raises(ValueError): verify(raw, result, responses, hashes, policy)


def scoped_fixture():
    from app.agents.scoped_arbitration import canonical_scoped_arbitration
    facts = [{"id": f"uuid-{i}", "raw_doc_id": f"doc{i}", "raw_para_id": f"p{i}",
        "quote": f"完整原段{i}", "fact_des": f"原注释{i}"} for i in range(1, 4)]
    aliases = {f"E{i}": row["id"] for i, row in enumerate(facts, 1)}
    evidence = [{**row, "id": f"E{i}"} for i, row in enumerate(facts, 1)]
    raw = {"decision": "undetermined", "conclusion": "验收尚未获程序记录证实，仍需进一步核验。",
        "principal_claim": "现有材料尚不能证实正式验收已完成，试开放不能替代程序记录。",
        "judgments": [], "conflicts": [], "relations": {
            "E1": "support_principal_claim", "E2": "oppose_principal_claim", "E3": "qualification"}}
    canonical, scope = canonical_scoped_arbitration(raw, evidence)
    canonical, scope = scorer.remap_aliases(canonical, aliases), scorer.remap_aliases(scope, aliases)
    wire = {"tools": [{"function": {"name": "ScopedArbitrationWireOutput"}}],
        "messages": [{"role": "user", "content": json.dumps({"evidence": evidence})}]}
    response = {"tool_calls": [{"name": "ScopedArbitrationWireOutput", "args": raw}],
        "tool_calls_wire": [{"function": {"name": "ScopedArbitrationWireOutput", "arguments": json.dumps(raw)}}]}
    native = {"logical_operations": ["arbitrate"], "relation_projections": [{"operation": "arbitrate",
        "source_response_sha256": "response-hash", "canonical_output_sha256": scorer.canonical_digest(canonical),
        "scope": scope, "role_assignment_origin": "model", "new_model_calls": 0}],
        "response_file_sha256": ["response-hash"], "runtime_sources_sha256": {
            name: scorer.bound(scorer.ROOT / name)[1]["sha256"] for name in SOURCE_FILES}}
    result = {"verification_protocol": "strict-workbench-evidence-balanced-v2", "evidence": facts,
        "final": canonical, "evidence_relation_scope": scope}
    return result, [wire], [response], native


def test_scoped_replay_maps_role_dictionary_keys_and_all_evidence_references_to_final_UUIDs(tmp_path):
    result, wires, responses, native = scoped_fixture()
    with SourceArchive(archive(tmp_path)):
        audit = scorer.verify_scoped_relations(result, wires, responses, native, question_internal_id="internal-q")
    assert set(audit["evidence_roles"]) == {"uuid-1", "uuid-2", "uuid-3"}
    assert audit["semantic_target_verified"] is False
    assert result["final"]["conflicting_evidence_ids"] == ["uuid-2"]


@pytest.mark.parametrize("change", ["scope", "response_hash", "output_hash", "source", "role", "nested_judgment"])
def test_scoped_replay_rejects_modified_model_roles_or_source_identity(change):
    result, wires, responses, native = scoped_fixture()
    if change == "scope": native["relation_projections"][0]["scope"] = {}
    elif change == "response_hash": native["relation_projections"][0]["source_response_sha256"] = "different"
    elif change == "output_hash": native["relation_projections"][0]["canonical_output_sha256"] = "different"
    elif change == "source": result["evidence"][0]["raw_para_id"] = "wrong-source"
    elif change == "role": result["final"]["conflicting_evidence_ids"] = []
    else: result["final"]["judgments"] = [{"id": "invented", "statement": "invented"}]
    with pytest.raises(ValueError):
        scorer.verify_scoped_relations(result, wires, responses, native, question_internal_id="internal-q")


def test_stage_losses_distinguish_recall_model_allocation_and_conclusion_without_inventing_errors():
    catalog = [{"paragraph_ref": f"P{i}", "doc_id": "d", "para_id": f"p{i}"} for i in range(1, 5)]
    rows = [{"doc_id": "d", "para_id": f"p{i}"} for i in range(1, 6)]
    proof = {"selection": {"decisions": [{"p": f"P{i}", "relevance": 1 if i == 2 else 3}
        for i in range(1, 5)]}, "balance_ledger": {"retained_refs": ["P1", "P4"]}}
    submitted = {"evidence": [rows[0], rows[3]], "supporting_facts": [rows[0]], "conflicting_facts": []}
    audit = scorer.staged_source_errors(submitted, {"supporting_facts": rows, "conflicting_facts": []},
        {"catalog": catalog}, proof, {})
    assert audit["evidence"]["false_negative_count"] == 3
    assert audit["evidence"]["false_positive_count"] == 0
    assert audit["supporting_facts"]["false_negative_stage_counts"] == {
        "model_selection_below_2": 1, "balance_allocation_loss": 1,
        "conclusion_relation_loss": 1, "candidate_recall_loss": 1}


@pytest.mark.parametrize("change", [None, "annotation", "relation", "revision", "unknown_usage"])
def test_actual_UI_download_must_preserve_published_fields_relations_revision_and_usage(tmp_path, change):
    def store(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return scorer.bound(path)[1]
    row = {"fact_id": "internal-fact", "doc_id": "d", "para_id": "p", "quote": "完整原文。",
        "fact_des": "原注释。", "argument": {}, "entities": []}
    normalized = {"evidence": [row], "conclusion": "原结论。", "supporting_facts": [row], "conflicting_facts": []}
    download = {"revision_id": "rev", "topic_id": "topic", "formal_submission": False, "partial": True,
        "files": {"evidence.json": [{**row, "fact_id": "export-fact"}], "conclusions.json": [{
            "question_id": "q", "question": "原题？", "conclusion": "原结论。",
            "supporting_facts": [{**row, "fact_id": "export-fact"}], "conflicting_facts": []}]}}
    if change == "annotation": download["files"]["evidence.json"][0]["fact_des"] = "被改写。"
    elif change == "relation": download["files"]["conclusions.json"][0]["supporting_facts"] = []
    elif change == "revision": download["revision_id"] = "other"
    binding = store("download.json", download)
    proof = {"actual_browser_UI_download": True, "run_id": "run", "revision_id": "rev",
        "whole_question_tokens": None if change == "unknown_usage" else 12,
        "exact_API_preview_bytes_equal": True, "exact_API_preview_sha256": binding["sha256"], "download_file": binding}
    proof_binding = store("proof.json", proof)
    prepared = {"guards": [], "run": {"id": "run", "target_revision_id": "rev", "topic_id": "topic"},
        "native": {"whole_episode_known_usage": {"total_tokens": 12}}, "normalized": normalized,
        "question": {"question_id": "q", "question": "原题？"}}
    if change is None:
        audit = scorer.bind_ui_export(prepared, proof_binding["file"], proof_binding["sha256"])
        assert audit["all_source_and_annotation_fields_equal_published_snapshot"] is True
        assert audit["new_model_calls"] == 0
    else:
        with pytest.raises(ValueError):
            scorer.bind_ui_export(prepared, proof_binding["file"], proof_binding["sha256"])
