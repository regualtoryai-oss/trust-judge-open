"""Synthetic compatibility/allocation tests; no API or reference answers."""
from copy import deepcopy

import pytest

from app.workflows.strict_workbench import (
    BALANCED_VERSION, VERSION, annotation_inputs, compile_result, digest,
    prepare_case, validate_result,
)


Q = {"id": "question-internal", "raw_question_id": "q-test", "question": "甲设备批准已经确认还是尚未确认？"}
TEXT = "甲设备批准状态说明：" + "完整原文😀。" * 90 + "主管部门表示批准尚未确认。"


def sources(count=2, *, per_document=1):
    return [{"id": f"p{i}", "document_id": f"d{i // per_document}", "raw_doc_id": f"doc{i // per_document}",
             "raw_para_id": f"doc{i // per_document}#p{i + 1}", "title": "甲设备批准说明", "text": TEXT,
             "source_credibility_json": "0.95", "raw_publish_time": "刚刚"} for i in range(count)]


def selection(raw, *, score=2):
    return {"hypotheses": [{"id": "H1", "claim": "甲设备批准已确认"}],
            "decisions": [{"p": row["paragraph_ref"], "relevance": score, "stance": "qualification", "hypothesis_id": "H1"}
                          for row in raw["catalog"]]}


def annotations(raw):
    return {"annotations": [{"p": row["paragraph_ref"], "fact_des": "主管部门表示批准尚未确认。",
             "argument": {"时间": "", "主体": "主管部门", "地点": "", "内容": "批准尚未确认", "影响": "", "数值口径": ""},
             "entities": ["甲设备"]} for row in raw["catalog"]]}


def built(*, count=24, per_document=4, policy=None, recall="window_balanced"):
    paragraphs = sources(count, per_document=per_document)
    raw = prepare_case(Q, paragraphs, recall_strategy=recall)
    full = selection(raw)
    for index, row in enumerate(full["decisions"]):
        row["stance"] = ("supports_hypothesis", "opposes_hypothesis", "qualification")[index % 3]
        row["relevance"] = 3 if index % 2 else 2
    policy = {"target": 10, "per_doc_soft_cap": 2} if policy is None else policy
    subset, chosen, ledger = annotation_inputs(raw, full, policy)
    out = compile_result(Q, paragraphs, raw, full, annotations(subset), balance_policy=policy, recall_strategy=recall)
    return paragraphs, raw, full, subset, chosen, ledger, out


def test_exact_legacy_default_raw_and_output_hashes_remain_unchanged():
    # These digests were captured from the original v1 implementation, before
    # introducing the optional policy. Use the exact original synthetic input.
    paragraphs = sources()
    paragraphs[1]["raw_para_id"] = "doc1#p1"
    raw = prepare_case(Q, paragraphs)
    out = compile_result(Q, paragraphs, raw, selection(raw), annotations(raw))
    assert digest(raw) == "52f34d61ab5c2c4da69ed61daa2a8fc43f4148ac6f99f1d440fa53b83e198901"
    assert digest(out) == "d316371fdbaabe2e5d9f4fe650a70014177bb8953a553527f1372798599c6067"
    assert out["verification_protocol"] == VERSION
    assert validate_result(out, Q, paragraphs) == out
    assert prepare_case(Q, paragraphs, recall_strategy=None) == raw
    assert compile_result(Q, paragraphs, raw, selection(raw), annotations(raw),
                          balance_policy=None, recall_strategy=None) == out
    assert "hypotheses" not in out["evidence_selection"]
    assert all("selection_stance" not in row for row in out["evidence"])


def test_balanced_recall_titles_cannot_admit_body_zero_hits():
    question = {**Q, "question": "transport safety cause"}
    paragraphs = sources(24, per_document=20)
    for index, row in enumerate(paragraphs):
        row.update(title="transport safety cause" if index < 20 else "inspection record",
                   text="unrelated weather garden" if index < 20 else "transport safety cause inspection")
    before = deepcopy(paragraphs)
    old = prepare_case(question, paragraphs)
    new = prepare_case(question, paragraphs, recall_strategy="window_balanced")
    assert len(old["catalog"]) == 24
    assert len(new["catalog"]) == 4
    assert all(row["doc_id"] == "doc1" for row in new["catalog"])
    assert all(row["text"] == "transport safety cause inspection" for row in new["catalog"])
    assert paragraphs == before
    assert new["retrieval"]["recall"]["model_calls"] == 0
    assert new["retrieval"]["recall"]["reference_inputs"] is False


def test_annotation_subset_keeps_nonsequential_aliases_and_full_decisions_immutable():
    raw = prepare_case(Q, sources(16))
    full = selection(raw, score=1)
    for ref, stance in (("P10", "supports_hypothesis"), ("P12", "opposes_hypothesis"), ("P14", "qualification")):
        next(row for row in full["decisions"] if row["p"] == ref).update(relevance=3, stance=stance)
    before = deepcopy((raw, full))
    subset, chosen, ledger = annotation_inputs(raw, full, {"target": 3, "per_doc_soft_cap": 1})
    assert [row["paragraph_ref"] for row in subset["catalog"]] == ["P10", "P12", "P14"]
    assert [row["p"] for row in chosen["decisions"]] == ["P10", "P12", "P14"]
    assert set(subset["source_metadata"]) == {"P10", "P12", "P14"}
    assert ledger["audit"]["candidate_count"] == 16
    assert ledger["audit"]["canonical_selection_sha256"] == digest(full)
    assert (raw, full) == before
    subset["catalog"][0]["text"] = "detached"
    chosen["decisions"][0]["relevance"] = 0
    assert (raw, full) == before


def test_no_policy_preserves_legacy_annotation_scope_and_detaches_inputs():
    raw = prepare_case(Q, sources(3)); full = selection(raw)
    subset, chosen, ledger = annotation_inputs(raw, full)
    assert subset == raw and chosen == full and ledger is None
    subset["catalog"].clear(); chosen["decisions"].clear()
    assert len(raw["catalog"]) == len(full["decisions"]) == 3
    assert annotation_inputs(raw, full, {}) == annotation_inputs(raw, full, {"target": 10, "per_doc_soft_cap": 2})


def test_v2_annotations_only_allocated_full_parents_and_complete_audits_replay():
    paragraphs, raw, full, subset, chosen, ledger, out = built()
    proof = out["strict_evidence_proof"]
    assert out["verification_protocol"] == BALANCED_VERSION
    assert len(full["decisions"]) == len(raw["catalog"]) == 24
    assert len(out["evidence"]) == len(subset["catalog"]) == 10
    assert proof["selection"] == full
    assert proof["balance_ledger"] == ledger
    assert proof["annotation_raw_case_sha256"] == digest(subset)
    assert proof["annotation_selection_sha256"] == digest(chosen)
    assert proof["source_metadata_sha256"] == digest(raw["source_metadata"])
    assert proof["retrieval_sha256"] == digest(raw["retrieval"])
    assert validate_result(out, Q, paragraphs) == out
    decisions = {raw["source_metadata"][row["p"]]["paragraph_id"]: row for row in full["decisions"]}
    for row in out["evidence"]:
        assert row["quote"] == TEXT and row["start_cp"] == 0 and row["end_cp"] == len(TEXT)
        assert row["selection_relevance"] == decisions[row["paragraph_id"]]["relevance"]
        assert row["selection_stance"] == decisions[row["paragraph_id"]]["stance"]
        assert row["selection_hypothesis_id"] == "H1"
        assert row["selection_semantics_verified"] is False
        assert row["selection_interpretation_origin"] == "model"
    assert out["evidence_selection"]["hypotheses"] == full["hypotheses"]
    assert out["evidence_selection"]["semantic_verified"] is False


def test_cross_document_exact_mirrors_keep_distinct_rows_but_shared_vote_group():
    paragraphs, _, _, _, _, _, out = built(count=4, per_document=1)
    assert len(out["evidence"]) == 4
    assert len({row["raw_doc_id"] for row in out["evidence"]}) == 4
    assert len({row["source_group_id"] for row in out["evidence"]}) == 1
    assert out["strict_evidence_proof"]["balance_ledger"]["policy"]["mirrors_deduplicated"] is False
    assert validate_result(out, Q, paragraphs) == out


@pytest.mark.parametrize("change", ["full_selection", "balance_ledger", "annotation", "evidence_stance", "hypothesis", "version"])
def test_v2_replay_rejects_tampering_even_after_claimed_proof_rehash(change):
    paragraphs, _, _, _, _, _, out = built()
    broken = deepcopy(out); proof = broken["strict_evidence_proof"]
    if change == "full_selection": proof["selection"]["decisions"][0]["relevance"] = 0
    elif change == "balance_ledger": proof["balance_ledger"]["audit"]["truncated_refs"] = []
    elif change == "annotation": proof["annotation"]["annotations"][0]["fact_des"] = "错误改写"
    elif change == "evidence_stance": broken["evidence"][0]["selection_stance"] = "qualification"
    elif change == "hypothesis": broken["evidence_selection"]["hypotheses"][0]["claim"] = "改动主张"
    else: proof["version"] = VERSION
    proof.pop("proof_sha256", None); proof["proof_sha256"] = digest(proof)
    with pytest.raises(ValueError): validate_result(broken, Q, paragraphs)


def test_v2_replay_rejects_original_text_metadata_retrieval_and_coordinates_change():
    paragraphs, raw, full, subset, _, _, out = built()
    for field, value in (("text", TEXT + "修改"), ("source_credibility_json", "0.7")):
        changed = deepcopy(paragraphs); changed[0][field] = value
        with pytest.raises(ValueError): validate_result(out, Q, changed)
    for mutate in (lambda value: value["retrieval"].update(candidate_limit=1),
                   lambda value: value["source_metadata"]["P1"].update(parent_end_cp=1)):
        changed = deepcopy(raw); mutate(changed)
        with pytest.raises(ValueError):
            compile_result(Q, paragraphs, changed, full, annotations(subset),
                           balance_policy={"target": 10}, recall_strategy="window_balanced")


def test_unallocated_or_missing_annotation_cannot_enter_balanced_evidence():
    paragraphs, raw, full, subset, _, _, _ = built()
    good = annotations(subset)
    for value in (annotations(raw), {"annotations": good["annotations"][:-1]}):
        with pytest.raises(ValueError):
            compile_result(Q, paragraphs, raw, full, value, balance_policy={"target": 10}, recall_strategy="window_balanced")


def test_all_background_has_empty_annotation_scope_without_fake_evidence():
    paragraphs = sources(5)
    raw = prepare_case(Q, paragraphs, recall_strategy="window_balanced"); full = selection(raw, score=1)
    subset, chosen, ledger = annotation_inputs(raw, full, {})
    assert subset["catalog"] == [] and subset["source_metadata"] == {} and chosen["decisions"] == []
    assert ledger["audit"]["decision"] == "undetermined"
    out = compile_result(Q, paragraphs, raw, full, {"annotations": []}, balance_policy={}, recall_strategy="window_balanced")
    assert out["evidence"] == [] and out["source_coverage"]["selected_paragraphs"] == 0
    assert validate_result(out, Q, paragraphs) == out


@pytest.mark.parametrize("policy", [True, [], {"target": True}, {"target": 0}, {"per_doc_soft_cap": 0}, {"unexpected": 1}])
def test_unknown_or_unsafe_balance_configuration_is_rejected(policy):
    raw = prepare_case(Q, sources())
    with pytest.raises(ValueError, match="strict_balance_policy_invalid"):
        annotation_inputs(raw, selection(raw), policy)


def test_unknown_recall_strategy_and_version_are_rejected():
    with pytest.raises(ValueError, match="strict_recall_strategy_invalid"):
        prepare_case(Q, sources(), recall_strategy="custom")
    _, _, _, _, _, _, out = built()
    out["verification_protocol"] = "unknown"
    with pytest.raises(ValueError): validate_result(out, Q, sources(24, per_document=4))


def test_optional_recall_only_or_balance_only_versions_replay_without_implicit_policy():
    for policy, recall in ((None, "window_balanced"), ({"target": 1}, None)):
        paragraphs = sources(3); raw = prepare_case(Q, paragraphs, recall_strategy=recall)
        full = selection(raw); subset, _, _ = annotation_inputs(raw, full, policy)
        out = compile_result(Q, paragraphs, raw, full, annotations(subset), balance_policy=policy, recall_strategy=recall)
        assert out["verification_protocol"] == BALANCED_VERSION
        assert validate_result(out, Q, paragraphs) == out


def test_nonwindow_balanced_full_parents_replay_and_strategy_tampering_rejected():
    paragraphs, raw, full, subset, _, _, out = built(recall="balanced")
    assert raw["retrieval"]["strategy"] == "question_balanced_full_parents"
    assert raw["retrieval"]["recall_strategy"] == "balanced"
    assert raw["retrieval"]["recall"]["variant"] == "balanced"
    assert out["strict_evidence_proof"]["recall_strategy"] == "balanced"
    assert all(row["text"] == TEXT and row["parent_complete"] for row in subset["catalog"])
    assert validate_result(out, Q, paragraphs) == out
    for strategy in (None, "window_balanced"):
        changed = deepcopy(out)
        proof = changed["strict_evidence_proof"]
        proof["recall_strategy"] = strategy
        proof.pop("proof_sha256"); proof["proof_sha256"] = digest(proof)
        with pytest.raises(ValueError): validate_result(changed, Q, paragraphs)
    changed = deepcopy(raw)
    changed["retrieval"]["recall_strategy"] = "window_balanced"
    with pytest.raises(ValueError, match="strict_raw_case_replay_changed"):
        compile_result(Q, paragraphs, changed, full, annotations(subset), balance_policy={"target": 10}, recall_strategy="balanced")
