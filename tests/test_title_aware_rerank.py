from copy import deepcopy

import pytest

from app.domain.title_aware_rerank import prepare_title_pairs, rank_title_parents, fuse_title_hybrid


def row(ref="P1", text="完整原文。", title="原标题", doc="d1", para="p1"):
    return {"paragraph_ref": ref, "doc_id": doc, "para_id": para, "text": text, "title": title,
            "paragraph_complete": True, "parent_complete": True, "source_credibility": 0.99}


def prepare(rows=None):
    return prepare_title_pairs("原题完整条件：甲还是乙？尚未确认是否成立。", [row()] if rows is None else rows, model_fingerprint="test-model")


def test_original_title_full_query_and_unicode_body_unchanged():
    a = row(text="甲😀\r\n乙" * 170, title="原标题${untrusted}\r\n")
    plan = prepare([a])
    assert plan["question"].endswith("尚未确认是否成立。")
    assert plan["records"][0]["start_cp"] == 0
    assert plan["records"][-1]["end_cp"] == len(a["text"])
    for r in plan["records"]:
        assert r["window_text"] == a["text"][r["start_cp"]:r["end_cp"]]
        assert r["model_document"] == a["title"] + "\n" + r["window_text"]
        assert "source_credibility" not in r
    assert plan["scores_are_truth_probabilities"] is False


def test_title_model_question_and_formatter_bound_pair_keys():
    plan = prepare(); original = plan["records"][0]["pair_sha256"]
    assert prepare([row(title="另一个原标题")])["records"][0]["pair_sha256"] != original
    assert prepare_title_pairs("不同原题", [row()], model_fingerprint="test-model")["records"][0]["pair_sha256"] != original
    assert prepare_title_pairs(plan["question"], [row()], model_fingerprint="different-model")["records"][0]["pair_sha256"] != original


def test_full_tail_window_can_win_parent_max():
    plan = prepare([row(text="原" * 641)])
    values = {r["pair_sha256"]: float(i) for i, r in enumerate(plan["records"])}
    ranked = rank_title_parents(plan, values)
    assert ranked[0]["best_window_end_cp"] == 641


@pytest.mark.parametrize("bad", ["missing", "extra", "nan", "boolean"])
def test_missing_or_invalid_window_score_never_zero_filled(bad):
    p = prepare(); key = p["records"][0]["pair_sha256"]; values = {key: 1.0}
    if bad == "missing": values = {}
    elif bad == "extra": values["not_frozen"] = 3.0
    elif bad == "nan": values[key] = float("nan")
    else: values[key] = True
    with pytest.raises(ValueError): rank_title_parents(p, values)


def test_same_body_same_title_different_sources_keep_both_parent_aliases():
    p = prepare([row(), row("P2", doc="d2", para="p2")])
    assert len({r["pair_sha256"] for r in p["records"]}) == 1
    ranked = rank_title_parents(p, {p["records"][0]["pair_sha256"]: 0.5})
    assert [r["paragraph_ref"] for r in ranked] == ["P1", "P2"]


def test_rrf60_same_forty_stable_source_ties_and_no_title_body_merge():
    p = prepare([row("P1"), row("P2", doc="d2", para="p2", text="另段原文")])
    scores = {r["pair_sha256"]: float(i) for i, r in enumerate(p["records"])}
    title = rank_title_parents(p, scores)
    fused = fuse_title_hybrid(p, title, ["P1", "P2"], limit=2)
    assert [r["paragraph_ref"] for r in fused] == ["P1", "P2"]
    assert len(fused) == 2


@pytest.mark.parametrize("refs", [["P999"], ["P1", "P1"]])
def test_hybrid_outside_same_pool_or_duplicate_rejected(refs):
    p = prepare(); ranked = rank_title_parents(p, {p["records"][0]["pair_sha256"]: 0.0})
    with pytest.raises(ValueError): fuse_title_hybrid(p, ranked, refs)


def test_partial_or_duplicate_source_cannot_be_declared_full():
    for rows in ([{**row(), "parent_complete": False}], [row(), row("P2")]):
        with pytest.raises(ValueError): prepare(rows)


def test_empty_parent_stays_unscored_and_prevents_fake_complete_ranking():
    p = prepare([row(text="  \r\n")])
    assert p["parents_without_windows"] == ["P1"]
    with pytest.raises(ValueError): rank_title_parents(p, {})
