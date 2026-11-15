"""Pure source-only mirror expansion fixtures, never retrieval quality estimates."""
from copy import deepcopy

import pytest

from app.domain.paragraph_mirror_recall import expand_base_mirrors


def rows(*texts):
    return [{"id": f"p{i}", "raw_doc_id": f"d{i}", "raw_para_id": f"d{i}#p1", "text": text}
            for i, text in enumerate(texts)]


def test_exact_whole_source_bucket_expands_beyond_base_with_own_identity_and_no_votes():
    source = rows("甲设备尚未确认。", "另一个无关段落。", "甲设备尚未确认。")
    before = deepcopy(source)
    audit = expand_base_mirrors(source, ["p0"])
    assert source == before and audit["added_count"] == 1 and audit["bitset_lcs_calls"] == 0
    assert audit["additions"][0]["paragraph_id"] == "p2"
    assert audit["additions"][0]["origins"] == [{"paragraph_id": "p0", "raw_doc_id": "d0", "raw_para_id": "d0#p1", "match": "exact", "similarity": 1.0}]
    assert audit["independent_source_votes_added"] == 0 and audit["evidence_created_or_deleted"] is False


def test_direct_near_edges_do_not_expand_transitively_or_treat_shared_fragment_as_whole_mirror():
    anchor = "甲" * 100
    direct = "甲" * 90 + "乙" * 10
    transitive_only = "甲" * 80 + "乙" * 20
    source = rows(anchor, direct, transitive_only, "共同原句至少十二个文字字数够用。" + "丙" * 200)
    audit = expand_base_mirrors(source, ["p0"])
    assert [r["paragraph_id"] for r in audit["additions"]] == ["p1"]
    assert audit["additions"][0]["origins"][0]["match"] == "similar"
    assert audit["additions"][0]["origins"][0]["similarity"] == .9


def test_cross_document_only_stable_budget_never_replaces_base():
    source = rows(*(["相同原段。"] * 5)); source[1]["raw_doc_id"] = "d0"
    audit = expand_base_mirrors(source, ["p0"], candidate_limit=1)
    assert audit["matched_before_budget"] == 3 and audit["budget_truncated"] is True
    assert [r["paragraph_id"] for r in audit["additions"]] == ["p2"]
    assert expand_base_mirrors(source, ["p0"], candidate_limit=0)["additions"] == []
    assert expand_base_mirrors(source, [])["additions"] == []


@pytest.mark.parametrize("damage", ["duplicate_source", "duplicate_base", "unknown_base", "boolean_budget", "too_large_budget", "empty_text"])
def test_invalid_source_or_budget_never_produces_partial_plan(damage):
    source = rows("原段。", "原段。"); base = ["p0"]; budget = 60
    if damage == "duplicate_source": source.append(deepcopy(source[0]))
    elif damage == "duplicate_base": base += base
    elif damage == "unknown_base": base = ["unknown"]
    elif damage == "boolean_budget": budget = True
    elif damage == "too_large_budget": budget = 121
    else: source[1]["text"] = ""
    with pytest.raises(ValueError): expand_base_mirrors(source, base, candidate_limit=budget)
