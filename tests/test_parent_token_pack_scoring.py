"""Offline delivery accounting: local recall is never promoted to a model read."""
from copy import deepcopy

import pytest

from scripts.capture_bundle_workbench_real import capture_episode, save
from scripts.score_bundle_workbench_real import _delivery_sets, _selection_mode
import scripts.score_bundle_workbench_real as scorer


def policy_fixture():
    proof = {"paragraph_quality": True, "paragraph_quality_policy": {"version": "synthetic-v5"},
        "paragraph_quality_compact": True, "paragraph_quality_compact_policy": {"version": "synthetic-v5"},
        "parent_token_pack": True, "parent_token_pack_policy": {"version": "synthetic-pack"}, "selection_output_tokens": 8192}
    run = {"source_bundle_paragraph_quality": True, "source_bundle_paragraph_quality_compact": True,
        "source_bundle_parent_token_pack": True}
    return proof, deepcopy(proof), run


@pytest.mark.parametrize("change", [None, "policy", "worker_flag", "missing_public_flag", "missing_compact"])
def test_v5_selection_policy_is_closed_to_worker_and_public_run(change):
    proof, config, run = policy_fixture()
    if change == "policy": proof["parent_token_pack_policy"] = {"version": "another-pack"}
    elif change == "worker_flag": config["parent_token_pack"] = False
    elif change == "missing_public_flag": run.pop("source_bundle_parent_token_pack")
    elif change == "missing_compact": proof["paragraph_quality_compact"] = config["paragraph_quality_compact"] = False
    if change is None:
        assert _selection_mode(proof, config, run)["parent_token_pack"] is True
    else:
        with pytest.raises(ValueError): _selection_mode(proof, config, run)


@pytest.mark.parametrize("change", [None, "duplicate_pool", "pool_count", "read_outside_pool", "exposed_unprovided", "policy"])
def test_recall_pool_and_complete_offered_paragraphs_have_separate_denominators(change):
    proof, _, _ = policy_fixture()
    paragraphs = [{"id": f"p{i}", "raw_doc_id": "synthetic-doc", "raw_para_id": f"raw{i}"} for i in range(6)]
    scope = {"parent_token_pack": True, "parent_token_pack_policy": proof["parent_token_pack_policy"],
        "recall_pool_paragraph_ids": [f"p{i}" for i in range(5)], "recall_pool_count": 5,
        "exposed_paragraph_ids": ["p0", "p1"], "returned_complete_paragraph_ids": ["p0", "p1"]}
    if change == "duplicate_pool": scope["recall_pool_paragraph_ids"].append("p0")
    elif change == "pool_count": scope["recall_pool_count"] = 6
    elif change == "read_outside_pool":
        scope["returned_complete_paragraph_ids"].append("p5"); scope["exposed_paragraph_ids"].append("p5")
    elif change == "exposed_unprovided": scope["exposed_paragraph_ids"].append("p2")
    elif change == "policy": scope["parent_token_pack_policy"] = {"version": "other"}
    if change:
        with pytest.raises(ValueError): _delivery_sets(scope, proof, paragraphs)
    else:
        candidates, read, _, _, counts = _delivery_sets(scope, proof, paragraphs)
        assert len(candidates) == 5 and len(read) == 2
        assert counts["recalled_but_not_delivered_parents"] == 3
        assert counts["program_recall_is_model_reading"] is False
        assert counts["parent_text_truncated"] is False


def test_v4_neighbor_complete_reads_keep_old_accounting():
    paragraphs = [{"id": f"p{i}", "raw_doc_id": "raw", "raw_para_id": f"paragraph{i}"} for i in range(3)]
    candidates, read, _, _, counts = _delivery_sets({"exposed_paragraph_ids": ["p0"], "returned_complete_paragraph_ids": ["p0", "p1"]}, {}, paragraphs)
    assert len(candidates) == len(read) == 2 and counts == {}


def test_capture_binds_v5_pack_policy_without_inventing_extra_native_requests(tmp_path):
    proof, _, _ = policy_fixture()
    proof.update(format="langchain-source-bundle-workbench-proof-v1", front_requests=[])
    save(tmp_path / "source-bundle/provider-response.json", {"proof": proof})
    captured = capture_episode(tmp_path)
    assert captured["front"]["selection_mode"]["parent_token_pack"] is True
    assert captured["front"]["selection_mode"]["parent_token_pack_policy"] == proof["parent_token_pack_policy"]
    assert captured["front"]["native_requests"] == []


def test_explicit_missing_historical_result_has_no_invented_before_or_cost(tmp_path):
    missing = {"exists": False, "file": str(tmp_path / "do-not-read-question_result.json"),
        "body_parsed": False, "hash_deferred_until_output_closed": True, "reference_answer_read": False}
    before, cost, availability = scorer._historical_baseline({"result": missing}, {}, {}, {}, [])
    assert before is None and cost is None and availability["available"] is False
    assert availability["baseline_generated_for_comparison"] is False
    assert not (tmp_path / "do-not-read-question_result.json").exists()


def test_valid_current_export_is_scored_when_historical_baseline_is_unavailable(tmp_path, monkeypatch):
    from test_bundle_real_capture_score import normalized_fixture
    question, fact, normalized = normalized_fixture()
    prepared = {"failed": False, "guards": [], "question": question, "normalized": normalized, "baseline": None,
        "contexts": {(fact["doc_id"], fact["para_id"]): fact["quote"]},
        "candidates": {(fact["doc_id"], fact["para_id"])}, "read_keys": {(fact["doc_id"], fact["para_id"])},
        "source_scope": {"program_recall_pool_parents": 1, "actually_delivered_complete_parents": 1},
        "tail": {"actual_tail_requests": 3}, "front": {"report": {"new_tokens": 7}}, "baseline_usage": None,
        "baseline_availability": {"available": False, "reason": "historical_result_missing_in_prelaunch_metadata"}, "browser_download": {}}
    monkeypatch.setattr(scorer, "prepare_closed", lambda *args: prepared)
    reference = save(tmp_path / "synthetic-evidence.json", [{**fact, "fact_id": "reference"}])
    conclusion = save(tmp_path / "synthetic-conclusions.json", [{**question, "conclusion": "合成参考", "supporting_facts": [
        {**fact, "fact_id": "reference", "verdict_label": "full_support"}], "conflicting_facts": []}])
    result = scorer.score_episode("synthetic-verified", "hash", reference["file"], conclusion["file"], browser_binding=None, browser_sha256=None)
    assert result["status"] == "succeeded" and result["after"]["question_selection"]["A_id_only_proxy"]["recall"] == 1
    assert result["before"] is None and result["before_usage"] is None
    assert result["historical_before_actual_after_comparison"] is False
