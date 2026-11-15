import copy

import pytest

from app.domain.paragraph_outputs import consolidate_evidence, consolidate_submission, whole_source_quote


def row(doc="d1", para="p1", quote="先停运", fact="早期停运", value=None):
    return {"doc_id": doc, "para_id": para, "quote": quote, "fact_des": fact,
            "argument": {"时间": value}, "entities": ["管理处"]}


def source(doc="d1", para="p1", text="先停运，后恢复。"):
    return {"doc_id": doc, "para_id": para, "text": text}


def test_merges_paragraph_without_changing_source_or_inventing_claims():
    sources = [source()]
    rows = [row(), row(quote="后恢复", fact="后续恢复", value="2026年6月12日")]
    before = copy.deepcopy((rows, sources))
    merged, mapping, audit = consolidate_evidence(rows, sources)
    assert mapping == {0: 0, 1: 0}
    assert merged[0]["quote"] == sources[0]["text"]
    assert merged[0]["fact_des"] == "早期停运；后续恢复"
    assert merged[0]["argument"]["时间"] == "2026年6月12日"
    assert audit["new_model_calls"] == 0 and audit["semantic_revalidation_performed"] is False
    assert (rows, sources) == before


def test_mirror_document_identity_is_never_removed():
    merged, mapping, _ = consolidate_evidence([row(), row(doc="d2")], [source(), source(doc="d2")])
    assert len(merged) == 2 and mapping == {0: 0, 1: 1}


def test_relation_dedup_preserves_both_membership_sets_without_promoting_candidates():
    submission = {"evidence": [row(), row(quote="后恢复", fact="恢复")],
                  "conclusions": [{"question_id": "q1", "supporting_facts": [{"evidence_index": 0}, {"evidence_index": 1}],
                                   "conflicting_facts": [{"evidence_index": 1}]}]}
    result, audit = consolidate_submission(submission, [source()])
    assert result["conclusions"][0]["supporting_facts"] == [{"evidence_index": 0}]
    assert result["conclusions"][0]["conflicting_facts"] == [{"evidence_index": 0}]
    assert audit["relation_counts"][0] == {"question_id": "q1", "side": "supporting_facts", "before": 2, "after": 1}


@pytest.mark.parametrize("rows,sources,check", [
    ([row(quote="不存在")], [source()], "cross_span_requires_original_document"),
    ([row()], [], "unknown_source"),
    ([row()], [source(), source()], "duplicate_source_key"),
    ([row(fact="")], [source()], "missing_fact_description"),
])
def test_invalid_sources_fail_explicitly(rows, sources, check):
    with pytest.raises(ValueError, match=check):
        consolidate_evidence(rows, sources)


def test_cross_paragraph_uses_original_document_geometry_and_starting_id():
    original = "😀首段。\r\n次段末。"
    p = source(text="😀首段。")
    p.update(document_text=original, document_start_cp=0,
             document_paragraphs=[{"para_id": "p1", "text": "😀首段。", "start_cp": 0, "end_cp": 4},
                                  {"para_id": "p2", "text": "次段末。", "start_cp": 6, "end_cp": 10}])
    quote, geometry = whole_source_quote(p, "首段。\r\n次段")
    assert quote == original and geometry["covered_para_ids"] == ["p1", "p2"]
    merged, _, _ = consolidate_evidence([row(quote="首段。\r\n次段")], [p])
    assert merged[0]["para_id"] == "p1" and merged[0]["quote"] == original


def test_cross_paragraph_never_guesses_separator_or_moves_starting_paragraph():
    original = "首段。\n第二段。"
    p = source(text="首段。")
    p.update(document_text=original, document_start_cp=0,
             document_paragraphs=[{"para_id": "p1", "text": "首段。", "start_cp": 0, "end_cp": 3},
                                  {"para_id": "p2", "text": "第二段。", "start_cp": 4, "end_cp": 8}])
    with pytest.raises(ValueError, match="not_in_declared"):
        whole_source_quote(p, "首段。第二段")
    with pytest.raises(ValueError, match="not_in_declared"):
        whole_source_quote(p, "第二段")


def test_unknown_relation_index_is_not_silently_dropped():
    with pytest.raises(ValueError, match="unknown_relation_index"):
        consolidate_submission({"evidence": [row()], "conclusions": [{"supporting_facts": [{"evidence_index": 9}]}]}, [source()])
