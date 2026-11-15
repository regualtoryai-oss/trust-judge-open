from copy import deepcopy

import pytest

from scripts.final_evidence_scoring import normalize_dsh, normalize_legacy, score_final, score_reference_conclusion


Q = {"question_id": "q_demo", "question": "方案甲是否通过程序确认，还是仅为建议？"}


def fact(identifier="f1", doc="d1", para="p1", text="登记仅为建议，尚未确认。"):
    return {"fact_id": identifier, "doc_id": doc, "para_id": para, "quote": text,
            "fact_des": text, "argument": {}, "entities": []}


def dsh(rows=None, supporting=None, conflicting=None):
    rows = [fact()] if rows is None else rows
    ref = lambda f: {k: f[k] for k in ("fact_id", "doc_id", "para_id", "quote")}
    return {"format": "dsh-proposed-evidence-v1", "question_id": Q["question_id"], "evidence": rows,
            "conclusion": {"question_id": Q["question_id"], "conclusion": "仍未确认。",
                           "supporting_facts": [ref(rows[0])] if supporting is None and rows else supporting or [],
                           "conflicting_facts": conflicting or []}}


def test_legacy_and_dsh_same_final_rows_and_relations():
    row = fact()
    legacy_row = {**row, "id": row["fact_id"], "raw_doc_id": row["doc_id"], "raw_para_id": row["para_id"]}
    legacy = {"evidence": [legacy_row], "final": {"conclusion": "仍未确认。", "supporting_evidence_ids": ["f1"], "conflicting_evidence_ids": []}}
    a, b = normalize_legacy(legacy, Q), normalize_dsh(dsh(), Q)
    for key in ("evidence", "supporting_facts", "conflicting_facts"):
        assert a[key] == b[key]


def test_dsh_proposed_reading_roles_never_infer_support_or_conflict():
    a = dsh(supporting=[])
    a["evidence"][0]["proposed_roles"] = ["D", "C"]
    result = normalize_dsh(a, Q)
    assert result["supporting_facts"] == result["conflicting_facts"] == []


def test_legacy_null_argument_matches_historical_public_export_without_claim_edit():
    row = {**fact(), "id": "f1", "raw_doc_id": "d1", "raw_para_id": "p1", "argument": {"时间": None}}
    original = {"evidence": [row], "final": {"conclusion": "未知。", "supporting_evidence_ids": ["f1"], "conflicting_evidence_ids": []}}
    normalized = normalize_legacy(original, Q)
    assert normalized["evidence"][0]["argument"] == {"时间": ""}
    assert normalized["supporting_facts"][0]["argument"] == {"时间": ""}
    assert original["evidence"][0]["argument"]["时间"] is None


@pytest.mark.parametrize("mutation", ["wrong_question", "wrong_source", "missing_fact", "duplicate_fact", "duplicate_relation"])
def test_unknown_or_changed_identity_fails(mutation):
    a = dsh()
    if mutation == "wrong_question": a["question_id"] = "elsewhere"
    elif mutation == "wrong_source": a["conclusion"]["supporting_facts"][0]["para_id"] = "different"
    elif mutation == "missing_fact": a["conclusion"]["supporting_facts"][0]["fact_id"] = "absent"
    elif mutation == "duplicate_fact": a["evidence"].append(deepcopy(a["evidence"][0]))
    else: a["conclusion"]["supporting_facts"] *= 2
    with pytest.raises(ValueError): normalize_dsh(a, Q)


def test_raw_duplicates_unique_sources_and_unknown_denominators_with_real_evaluator():
    rows = [fact(), fact("f2"), fact("f3", "mirror", "p9")]
    normalized = normalize_dsh(dsh(rows), Q)
    refs = [fact("g1"), fact("g2"), {"fact_id": "unlocated", "quote": "未定位", "fact_des": "未定位"}]
    contexts = {("d1", "p1"): rows[0]["quote"], ("mirror", "p9"): rows[2]["quote"]}
    result = score_final(Q, normalized, refs, [refs[0]], [], contexts)
    category = result["categories"]["evidence"]
    raw = category["A_source_id_proxy"]["raw_records"]["A_id_only"]["metrics"]
    unique = category["A_source_id_proxy"]["unique_paragraphs"]["A_id_only"]["metrics"]
    assert (raw["matches"], raw["submitted_records"], raw["reference_records"]) == (2, 3, 3)
    assert (unique["matches"], unique["submitted_records"], unique["reference_records"]) == (1, 2, 2)
    semantic = category["two_interpretation_bounds"]["unique_paragraphs"]
    assert semantic["counts"]["unlocated_reference_units"] == 1
    assert semantic["counts"]["semantic_unknown_pairs"] == 6  # raw pair variants; units still 2x2
    assert semantic["interpretations"]["II"]["lower"]["metrics"]["matches"] == 0
    assert semantic["interpretations"]["II"]["upper"]["metrics"]["matches"] == 1


def test_nonliteral_quote_is_reported_without_dropping_output_or_semantic_success():
    normalized = normalize_dsh(dsh(), Q)
    result = score_final(Q, normalized, [fact()], [], [], {("d1", "p1"): "别的原文"})
    assert len(result["literal_quote_issues"]) == 1
    raw = result["categories"]["evidence"]["A_source_id_proxy"]["raw_records"]["A_id_only"]["metrics"]
    assert raw["submitted_records"] == 1  # no silent denominator repair
    semantic = result["categories"]["evidence"]["two_interpretation_bounds"]["raw_records"]
    assert semantic["interpretations"]["I"]["lower"]["metrics"]["matches"] == 1
    assert semantic["interpretations"]["II"]["lower"]["metrics"]["matches"] == 0
    assert semantic["counts"]["pending_judge_pairs"] == 0


def test_empty_final_does_not_get_fabricated_support_or_precision_one():
    normalized = normalize_dsh(dsh([], []), Q)
    result = score_final(Q, normalized, [fact()], [], [], {})
    m = result["categories"]["evidence"]["A_source_id_proxy"]["raw_records"]["A_id_only"]["metrics"]
    assert m["precision"] is None and m["recall"] == 0


def test_pinned_reference_relation_mapper_keeps_duplicate_and_unknown_ref():
    normalized = normalize_dsh(dsh(), Q)
    ref = {**Q, "supporting_facts": [{"fact_id": "g1"}, {"fact_id": "g1"}, {"fact_id": "missing"}], "conflicting_facts": []}
    result = score_reference_conclusion(Q, normalized, [fact("g1")], ref, {("d1", "p1"): fact()["quote"]})
    score = result["categories"]["supporting_facts"]["A_source_id_proxy"]
    assert score["raw_records"]["A_id_only"]["metrics"]["reference_records"] == 3
    assert score["unique_paragraphs"]["A_id_only"]["metrics"]["reference_records"] == 2
    assert result["reference_issues"]


def test_reference_wrong_question_fails_before_mapping():
    with pytest.raises(ValueError, match="same_original_reference"):
        score_reference_conclusion(Q, normalize_dsh(dsh(), Q), [fact()], {**Q, "question": "另一题"}, {})
