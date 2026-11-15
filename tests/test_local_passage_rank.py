"""Synthetic CLOSED score loading only; no ONNX, provider or answer data."""
from copy import deepcopy
from pathlib import Path

import pytest

from app.domain import local_parent_rank as rank
from app.domain.local_passage_rank import load_bound_window_scores
from test_local_parent_rank import build, case, infer, model


def load(raw, result, model):
    return load_bound_window_scores(raw, result["rank_plan"], result["proof"],
        model[1], rank.fingerprint(raw))


def test_preserves_every_facet_final_tail_empty_parent_and_duplicate_payload(tmp_path, model, infer):
    raw = case(texts=["完整😀" * 190 + "不得漏掉最终尾部", "重复原文乙", "重复原文乙", ""])
    result = build(tmp_path, model, raw)
    closed = rank._read(result["proof"])
    plan, scores = rank._read(closed["window_plan"]), rank._read(closed["pair_scores"])
    loaded = load(raw, result, model)
    assert len(infer) == 1
    assert loaded["query_window_count"] == len(plan["records"])
    assert loaded["unique_pair_count"] < loaded["query_window_count"]
    assert loaded["provider_calls"] == loaded["new_local_inference_calls"] == 0
    assert loaded["selected_top32_filter_applied"] is False
    for actual, expected in zip(loaded["records"], plan["records"]):
        assert actual["paragraph_id"] == expected["paragraph_id"]
        assert actual["query_facet"] == expected["query_id"]
        assert actual["start_cp"] == expected["start_cp"]
        assert actual["end_cp"] == expected["end_cp"]
        assert actual["text"] == expected["text"]
        assert actual["maxlogit"] == scores[expected["pair_sha256"]]
    for parent in raw["paragraphs"]:
        for facet in loaded["queries"]:
            windows = [row for row in loaded["records"]
                if row["paragraph_id"] == parent["id"] and row["query_facet"] == facet["query_id"]]
            assert windows[0]["start_cp"] == 0
            assert windows[-1]["end_cp"] == len(parent["text"])
    assert any(row["text"] == "" for row in loaded["records"])


def test_does_not_filter_to_selected_parent_top32(tmp_path, model, infer):
    raw = case(texts=[f"完整原段{i}内容" for i in range(40)])
    result = build(tmp_path, model, raw)
    assert len(result["rank_plan"]["selected_paragraph_ids"]) == 32
    loaded = load(raw, result, model)
    assert {row["paragraph_id"] for row in loaded["records"]} == {p["id"] for p in raw["paragraphs"]}
    assert loaded["query_window_count"] == 40 * 3
    assert len(infer) == 1


@pytest.mark.parametrize("artifact,mutation", [
    ("window_plan", "changed"), ("window_plan", "missing"),
    ("pair_scores", "changed"), ("pair_scores", "missing"),
    ("pair_payloads", "changed"), ("pair_payloads", "missing"),
])
def test_changed_or_missing_bound_artifact_rejects_without_inference(tmp_path, model, infer, artifact, mutation):
    raw = case()
    result = build(tmp_path, model, raw)
    closed = rank._read(result["proof"])
    path = Path(closed[artifact]["file"])
    if mutation == "changed":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="artifact_changed"):
        load(raw, result, model)
    assert len(infer) == 1


def test_rejects_caller_reported_scores_foreign_raw_or_model(tmp_path, model, infer):
    raw = case()
    result = build(tmp_path, model, raw)
    claimed = deepcopy(result)
    claimed["rank_plan"]["parent_scores"][0]["facet_scores"][0]["max_window_logit"] = 123456.0
    with pytest.raises(ValueError, match="rank_plan_changed"):
        load(raw, claimed, model)
    with pytest.raises(ValueError, match="raw_source_changed"):
        load_bound_window_scores(raw, result["rank_plan"], result["proof"], model[1], "0" * 64)
    with pytest.raises(ValueError, match="expected_model_changed"):
        load_bound_window_scores(raw, result["rank_plan"], result["proof"], "0" * 64, rank.fingerprint(raw))
    assert len(infer) == 1


def test_current_source_proof_guard_is_not_bypassed(tmp_path, model, infer, monkeypatch):
    raw = case()
    result = build(tmp_path, model, raw)
    sources = rank._sources()
    sources["scripts/local_parent_rank.py"]["sha256"] = "0" * 64
    monkeypatch.setattr(rank, "_sources", lambda: sources)
    with pytest.raises(ValueError, match="runner_source_changed"):
        load(raw, result, model)
    assert len(infer) == 1
