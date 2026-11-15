"""Synthetic source/score fixtures; no ONNX, paid models or answer data."""
from copy import deepcopy
import re

import pytest

from app.domain import local_parent_rank as rank
from app.domain.context_candidates import build_context_plan, validate_context_plan, _context_range
from app.domain.local_passage_rank import load_bound_window_scores
from app.domain.source_passages import build_sentence_manifest
from test_local_parent_rank import model
from test_source_passages import corpus


def raw_case(texts, *, separate_documents=False):
    docs, parents = [], []
    if separate_documents:
        for i, text in enumerate(texts):
            ds, ps = corpus([text], doc_id=f"internal-{100-i}", raw_doc_id=f"raw-{100-i}")
            docs.extend(ds)
            parents.extend(ps)
    else:
        docs, parents = corpus(texts)
    return {"format": "frozen-source-bundle-case-v1", "question": {
        "question_id": "q-synthetic", "question": "甲主张还是乙主张？"},
        "documents": docs, "paragraphs": parents, "limits": {"max_read_calls": 2}}


def bound_windows(tmp_path, model, monkeypatch, raw, scorer=lambda payload: float(len(payload["document"]))):
    def infer(job, output_dir):
        bound = rank._save(output_dir, "inference-job.json", job)
        scores = {key: scorer(payload) for key, payload in job["pairs"].items()}
        result = {"format": "local-parent-inference-result-v1", "status": "CLOSED",
            "job_sha256": bound["sha256"], "scores": scores,
            "pair_token_counts": {key: 10 for key in scores}, "runtime_packages": rank.RUNTIME_PACKAGES,
            "truncation": False, "provider_calls": 0, "external_network_calls": 0}
        return result, rank._save(output_dir, "inference-result.json", result)
    monkeypatch.setattr(rank, "_run_inference", infer)
    result = rank.build_local_rank(raw, raw_case_sha256=rank.fingerprint(raw), model_manifest=model[0],
        expected_manifest_sha256=model[1], cache_dir=tmp_path / "cache", output_dir=tmp_path / "rank")
    return load_bound_window_scores(raw, result["rank_plan"], result["proof"], model[1], rank.fingerprint(raw))


def make_plan(raw, windows):
    return build_context_plan(raw, windows, full_window_scores_digest=rank.fingerprint(windows))


def test_complete_long_sentence_tail_and_duplicate_contexts_are_not_clipped(tmp_path, model, monkeypatch):
    long_sentence = "完整长句😀" * 220 + "原文最后限定尚未确认。"
    raw = raw_case([long_sentence, ""])
    windows = bound_windows(tmp_path, model, monkeypatch, raw)
    plan = make_plan(raw, windows)
    assert len(plan["candidates"]) == 1
    candidate = plan["candidates"][0]
    assert candidate["source_passage"]["quote"] == long_sentence
    assert len(candidate["source_passage"]["quote"]) > 1000
    assert candidate["source_passage"]["local_end_cp"] == len(long_sentence)
    assert len(candidate["window_origins"]) > 1
    assert plan["audit"]["full_query_window_count"] == windows["query_window_count"]
    assert candidate["rrf_score"] == max(row["rrf_score"] for row in candidate["window_origins"])
    assert any(row["reason"] == "empty-or-blank-original-window" for row in plan["audit"]["unquotable_original_windows"])
    assert plan["semantic_verified"] is plan["model_delivery_verified"] is False


def test_facet_rank_fusion_preserves_opposing_query_navigation_despite_logit_scale(tmp_path, model, monkeypatch):
    raw = raw_case([f"项目编号{i}的完整原句。" for i in range(40)], separate_documents=True)
    query_ids = {q["text"]: q["query_id"] for q in rank.prepare_pairs(raw, model[1])[0]["queries"]}
    orders = {"F0": [0, 39, *range(1, 39)], "F1": list(range(40)), "F2": [39, *range(39)]}
    def score(payload):
        i = int(re.search(r"编号(\d+)", payload["document"]).group(1))
        qid = query_ids[payload["query"]]
        position = orders[qid].index(i)
        return {"F0": 100.0, "F1": 1_000_000_000.0, "F2": -100.0}[qid] - position
    windows = bound_windows(tmp_path, model, monkeypatch, raw, score)
    plan = make_plan(raw, windows)
    right_pid = raw["paragraphs"][39]["id"]
    right = next(c for c in plan["candidates"] if c["source_passage"]["paragraph_id"] == right_pid)
    facets = {f["query_facet"]: f for f in right["window_origins"][0]["facet_scores"]}
    assert facets["F1"]["rank"] == 40 and facets["F2"]["rank"] == 1
    assert right["rrf_score"] == sum(1 / (60 + f["rank"]) for f in facets.values())
    assert len(plan["candidates"]) == 32
    # A cross-facet raw-logit maximum would omit this row; ranks retain the
    # opposing query's navigation signal without treating its stance as true.
    assert plan["audit"]["corpus_parent_count"] == 40


def test_raw_order_ties_and_source_geometry_are_replay_stable(tmp_path, model, monkeypatch):
    raw = raw_case([f"完整来源{i}。" for i in range(36)], separate_documents=True)
    windows = bound_windows(tmp_path, model, monkeypatch, raw, lambda _: 1.0)
    plan = make_plan(raw, windows)
    assert [c["source_passage"]["paragraph_id"] for c in plan["candidates"]] == [p["id"] for p in raw["paragraphs"][:32]]
    assert validate_context_plan(raw, windows, plan, full_window_scores_digest=rank.fingerprint(windows)) == plan
    assert plan == make_plan(raw, windows)


def test_parent_and_document_soft_caps_have_explicit_deterministic_fill(tmp_path, model, monkeypatch):
    text = "".join((f"原始完整句{i}：" + "文字" * 75 + "。") for i in range(24))
    raw = raw_case([text])
    windows = bound_windows(tmp_path, model, monkeypatch, raw)
    plan = make_plan(raw, windows)
    assert len(plan["candidates"]) > 2
    assert [c["selection_pass"] for c in plan["candidates"][:2]] == ["document-and-parent-diversity"] * 2
    assert all(c["selection_pass"] == "fill-remaining-ranked-contexts" for c in plan["candidates"][2:])
    assert plan["policy"]["per_document_soft_cap"] == 3
    assert plan["policy"]["per_parent_soft_cap"] == 2


def test_document_cap_is_soft_and_fill_does_not_silently_discard_sources(tmp_path, model, monkeypatch):
    raw = raw_case([f"完整原句{i}。" for i in range(40)])
    windows = bound_windows(tmp_path, model, monkeypatch, raw, lambda _: 1.0)
    plan = make_plan(raw, windows)
    assert len(plan["candidates"]) == 32
    assert [c["selection_pass"] for c in plan["candidates"][:3]] == ["document-and-parent-diversity"] * 3
    assert all(c["selection_pass"] == "relax-document-retain-parent" for c in plan["candidates"][3:])
    audit = plan["audit"]["selection_pass_audit"]
    assert len(audit[0]["deferred_contexts"]) == 37
    assert all(row["reasons"] == ["document-soft-cap"] for row in audit[0]["deferred_contexts"])
    assert len(audit[1]["selected_context_ids"]) == 29


def test_neighbor_groups_skip_blank_count_but_keep_original_separators():
    raw = raw_case(["前句。\n\n命中句。\r\n\r\n后句。"])
    sentences = build_sentence_manifest(raw["documents"], raw["paragraphs"])["paragraphs"][0]["sentences"]
    hit = next(s for s in sentences if "命中句" in s["text"])
    span = _context_range(sentences, hit["local_start_cp"], hit["local_end_cp"])
    text = "".join(s["text"] for s in sentences[span[0]:span[1]])
    assert text == raw["paragraphs"][0]["text"]


@pytest.mark.parametrize("mutation", ["drop_facet", "span", "text", "query", "score_nonfinite"])
def test_incomplete_or_changed_window_source_is_rejected_even_if_rehashed(tmp_path, model, monkeypatch, mutation):
    raw = raw_case(["完整正文。" * 80])
    windows = bound_windows(tmp_path, model, monkeypatch, raw)
    changed = deepcopy(windows)
    if mutation == "drop_facet":
        changed["records"] = [r for r in changed["records"] if r["query_facet"] != "F2"]
        changed["query_window_count"] = len(changed["records"])
    elif mutation == "span":
        changed["records"][0]["end_cp"] += 1
    elif mutation == "text":
        changed["records"][0]["text"] += "不是原文"
    elif mutation == "query":
        changed["records"][0]["query"] += "非原题线索"
    else:
        changed["records"][0]["maxlogit"] = float("nan")
    with pytest.raises(ValueError):
        make_plan(raw, changed)


@pytest.mark.parametrize("mutation", ["geometry", "quote", "score", "policy", "audit", "delivery"])
def test_context_receipt_tampering_is_rejected(tmp_path, model, monkeypatch, mutation):
    raw = raw_case(["原句。" * 120])
    windows = bound_windows(tmp_path, model, monkeypatch, raw)
    plan = make_plan(raw, windows)
    changed = deepcopy(plan)
    if mutation == "geometry":
        changed["candidates"][0]["source_passage"]["local_start_cp"] += 1
    elif mutation == "quote":
        changed["candidates"][0]["source_passage"]["quote"] += "伪造"
    elif mutation == "score":
        changed["candidates"][0]["rrf_score"] += 1
    elif mutation == "policy":
        changed["policy"]["candidate_limit"] = 40
    elif mutation == "audit":
        changed["audit"]["full_query_window_count"] -= 1
    else:
        changed["model_delivery_verified"] = True
    with pytest.raises(ValueError, match="plan_changed"):
        validate_context_plan(raw, windows, changed, full_window_scores_digest=rank.fingerprint(windows))


def test_digest_and_original_source_bindings_cannot_be_replaced(tmp_path, model, monkeypatch):
    raw = raw_case(["原文。"])
    windows = bound_windows(tmp_path, model, monkeypatch, raw)
    with pytest.raises(ValueError, match="window_scores_digest_changed"):
        build_context_plan(raw, windows, full_window_scores_digest="0" * 64)
    other = deepcopy(raw)
    other["paragraphs"][0]["text"] += "篡改"
    with pytest.raises(ValueError, match="window_source_binding_changed"):
        make_plan(other, windows)
