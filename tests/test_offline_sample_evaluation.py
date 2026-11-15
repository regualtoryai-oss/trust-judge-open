"""Synthetic checks of offline diagnostics, never gold-fed runtime/model tests."""

import copy
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_samples import (compare_reports, comparison_markdown, evidence_metrics, quote_audit,
                                      ro_connect, set_metrics, side_diagnostic, source_scan_audit,
                                      stored_span_audit, selection_audit)


def fact(quote="原文", para="p1", identity="f1"):
    return {"doc_id": "d1", "para_id": para, "quote": quote, "fact_id": identity}


def test_paragraph_hit_is_not_strict_quote_hit_or_semantic_match():
    result = evidence_metrics([fact("第一种事实")], [fact("另一种事实")])
    assert result["paragraph_set"]["f1"] == 1
    assert result["strict_quote_set"]["f1"] == 0
    assert result["one_to_one_paragraph_filter_upper_bound"]["potential_matches"] == 1


def test_deduplication_and_one_to_one_upper_bound_do_not_count_cross_product():
    result = evidence_metrics([fact("甲"), fact("乙"), fact("甲")], [fact("丙")])
    assert result["raw_counts"]["predicted"] == 3
    assert result["paragraph_set"]["predicted"] == 1
    assert result["strict_quote_set"]["predicted"] == 2
    assert result["one_to_one_paragraph_filter_upper_bound"]["potential_matches"] == 1
    assert result["one_to_one_paragraph_filter_upper_bound"]["precision_upper_bound"] == .5


def test_empty_predictions_are_not_perfect_precision():
    result = set_metrics([], ["one"])
    assert result == {"matched": 0, "predicted": 0, "reference": 1, "precision": None, "recall": 0, "f1": 0}
    assert set_metrics([], [])["f1"] is None


def test_literal_quote_audit_keeps_wrong_paragraph_and_smart_quotes_invalid():
    raw = [{"doc_id": "d1", "content": "‘原文’\n别段", "paragraphs": [
        {"para_id": "p1", "char_start": 0, "char_end": 4},
        {"para_id": "p2", "char_start": 5, "char_end": 7}]}]
    first = fact("‘原文’"); first.update(char_start=0, char_end=4)
    result = quote_audit([first, fact("‘原文’", "p2", "f2"), fact("“原文”", identity="f3")], raw)
    assert result["document_literal_hits"] == 2
    assert result["declared_paragraph_literal_hits"] == 1
    assert result["declared_document_spans_exact"] == 1
    assert [r["fact_id"] for r in result["unlocatable_declared_paragraph_quotes"]] == ["f2", "f3"]


def test_retrieval_extraction_and_final_loss_are_separate_sets():
    facts = {f"f{i}": fact(str(i), f"p{i}", f"f{i}") for i in range(1, 4)}
    retrieved = [{"raw_doc_id": "d1", "raw_para_id": f"p{i}"} for i in (1, 2)]
    extracted = [{"raw_doc_id": "d1", "raw_para_id": "p1"}]
    result = side_diagnostic([], list(facts.values()), facts, retrieved, extracted)
    assert result["not_retrieved"] == [{"doc_id": "d1", "para_id": "p3"}]
    assert result["retrieved_not_extracted"] == [{"doc_id": "d1", "para_id": "p2"}]
    assert result["extracted_not_final_linked"] == [{"doc_id": "d1", "para_id": "p1"}]
    assert result["paragraph_set"]["recall"] == 0


def test_readonly_connection_rejects_mutation(tmp_path):
    path = tmp_path / "readonly.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE source(value TEXT)")
        connection.execute("INSERT INTO source VALUES ('immutable')")
    with ro_connect(path) as connection:
        assert connection.execute("SELECT value FROM source").fetchone()[0] == "immutable"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM source")


def scan_connection(events=()):
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE run_events(run_id TEXT,seq INTEGER,type TEXT,data_json TEXT)")
    for seq, kind, data in events:
        connection.execute("INSERT INTO run_events VALUES (?,?,?,?)", ("r", seq, kind, json.dumps(data)))
    return connection


def test_full_source_plan_is_not_completed_model_scanning():
    retrieved = [{"raw_doc_id": "d1", "raw_para_id": f"p{i}"} for i in range(1, 4)]
    planned = {"question_id": "q", "strategy": "full_source_scan", "batch_count": 2, "total_paragraphs": 3, "complete": True}
    with scan_connection([(1, "source_coverage_planned", planned)]) as connection:
        report, actual = source_scan_audit(connection, "r", "q", retrieved)
    assert report["planned_source_paragraphs"] == 3
    assert report["recorded_scanned_paragraphs"] is None
    assert report["recorded_scan_complete"] is None
    assert actual is None


def test_scan_audit_scopes_question_and_uses_actual_cumulative_events():
    retrieved = [{"raw_doc_id": "d1", "raw_para_id": f"p{i}"} for i in range(1, 4)]
    base = {"question_id": "q", "strategy": "full_source_scan", "batch_count": 2, "total_paragraphs": 3}
    partial = {**base, "completed_batches": 1, "scanned_paragraphs": 2, "complete": False}
    full = {**base, "completed_batches": 2, "scanned_paragraphs": 3, "complete": True}
    events = [(1, "source_coverage_planned", base), (2, "source_coverage_progress", partial),
              (3, "source_coverage_progress", {**full, "question_id": "other"})]
    with scan_connection(events) as connection:
        report, actual = source_scan_audit(connection, "r", "q", retrieved)
        assert report["recorded_scanned_paragraphs"] == 2
        assert report["source_paragraph_scan_coverage"] == pytest.approx(2 / 3)
        assert report["recorded_scan_complete"] is False
        assert actual is None  # A partial cumulative count does not identify gold paragraph membership.
        connection.execute("INSERT INTO run_events VALUES (?,?,?,?)", ("r", 4, "source_coverage_progress", json.dumps(full)))
        report, actual = source_scan_audit(connection, "r", "q", retrieved)
        assert report["recorded_scan_complete"] is True
        assert actual == {("d1", "p1"), ("d1", "p2"), ("d1", "p3")}
        assert report["progress_event_sequences"] == [1, 2, 4]


def test_checkpoint_source_scan_supports_resume_but_legacy_plan_complete_alone_cannot():
    retrieved = [{"raw_doc_id": "d1", "raw_para_id": "p1"}]
    with scan_connection() as connection:
        unknown, actual = source_scan_audit(connection, "r", "q", retrieved, {"complete": True, "paragraph_count": 1})
        assert unknown["recorded_scanned_paragraphs"] is None
        assert actual is None
        complete, actual = source_scan_audit(connection, "r", "q", retrieved, {
            "strategy": "full_source_scan", "batch_count": 1, "total_paragraphs": 1,
            "completed_batches": 1, "scanned_paragraphs": 1, "complete": True})
        assert complete["actual_progress_source"] == "persisted_checkpoint_source_coverage"
        assert actual == {("d1", "p1")}


def test_actual_scanned_gold_count_is_not_inferred_from_retrieved_plan():
    facts = {"f1": fact()}
    retrieved = [{"raw_doc_id": "d1", "raw_para_id": "p1"}]
    unknown = side_diagnostic([], list(facts.values()), facts, retrieved, [])
    assert unknown["gold_paragraphs_planned"] == 1
    assert unknown["gold_paragraphs_actual_scanned"] is None
    known = side_diagnostic([], list(facts.values()), facts, retrieved, [], {("d1", "p1")})
    assert known["gold_paragraphs_actual_scanned"] == 1
    assert known["paragraph_set"]["recall"] == 0  # Scanning all input still need not extract or cite it.


def test_internal_span_audit_keeps_emoji_codepoints_distinct_from_quote_matching():
    raw = [{"doc_id": "d1", "content": "🙂原文", "paragraphs": [{"para_id": "p1", "char_start": 0, "char_end": 3}]}]
    rows = [{**fact(), "start_cp": 1, "end_cp": 3}, {**fact(), "start_cp": 2, "end_cp": 4}]
    audit = stored_span_audit(rows, raw)
    assert audit["stored_codepoint_spans_exact"] == 1
    assert audit["original_raw_literal_audit"]["declared_paragraph_literal_hits"] == 2


def comparison_report(*, predicted=("p1",), referenced=("p1", "p2"), origin="old", question="同一个问题"):
    predicted_facts = [fact(para=para) for para in predicted]
    reference_facts = [fact(para=para) for para in referenced]
    question_row = {"question_id": "q1", "question": question, "origin_run_id": origin, "prompt_version": origin,
                    "question_referenced_gold_evidence": evidence_metrics(predicted_facts, reference_facts),
                    "support": {"paragraph_set": set_metrics(predicted, referenced)},
                    "conflict": {"paragraph_set": set_metrics([], [])}}
    return {"sample_input_sha256": {"fixture": "same"}, "source_fingerprint_before": "source",
            "topics": [{"topic_id": "t1", "topic": "测试专题", "revision_id": origin, "questions": [question_row]}]}


def test_matched_question_comparison_preserves_baseline_and_excludes_new_questions_from_delta():
    baseline = comparison_report()
    frozen = copy.deepcopy(baseline)
    current = comparison_report(predicted=("p1", "p2"), origin="new")
    extra = copy.deepcopy(current["topics"][0]["questions"][0]); extra["question_id"] = "q2"
    current["topics"][0]["questions"].append(extra)
    comparison = compare_reports(baseline, current)
    assert baseline == frozen
    assert comparison["matched_question_count"] == 1
    assert comparison["changed_origin_question_count"] == 1
    assert comparison["after_only_questions"] == [{"topic_id": "t1", "question_id": "q2"}]
    assert comparison["matched_question_micro"]["paragraph_set"]["before"]["recall"] == .5
    assert comparison["matched_question_micro"]["paragraph_set"]["after"]["recall"] == 1
    assert comparison["matched_question_micro"]["paragraph_set"]["delta"]["recall"] == .5
    assert comparison["questions"][0]["source_scan"]["before"] is None
    assert "50.0% → 100.0%" in comparison_markdown(comparison)
    assert "仅新版 1 题" in comparison_markdown(comparison)


def test_same_origin_is_explicit_and_changed_question_or_reference_is_excluded():
    baseline = comparison_report()
    assert compare_reports(baseline, baseline)["changed_origin_question_count"] == 0
    assert "复用原结果" in comparison_markdown(compare_reports(baseline, baseline))
    changed_text = comparison_report(question="另一个问题")
    assert compare_reports(baseline, changed_text)["matched_question_count"] == 0
    changed_ref = comparison_report(referenced=("p1",))
    result = compare_reports(baseline, changed_ref)
    assert result["matched_question_count"] == 0
    assert len(result["excluded_shared_questions"]) == 1
    assert result["matched_question_micro"]["paragraph_set"]["delta"]["precision"] is None


def test_report_pairing_rejects_duplicate_identity_and_warns_changed_input_fingerprints():
    baseline = comparison_report()
    duplicate = copy.deepcopy(baseline)
    duplicate["topics"][0]["questions"].append(copy.deepcopy(duplicate["topics"][0]["questions"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        compare_reports(baseline, duplicate)
    changed = comparison_report(); changed["sample_input_sha256"] = {"fixture": "changed"}
    comparison = compare_reports(baseline, changed)
    assert comparison["sample_input_sha256_matches"] is False
    assert comparison["comparison_input_warning"]
    assert "输入指纹发生变化" in comparison_markdown(comparison)


def test_cli_rejects_overwriting_frozen_baseline_before_loading_database(tmp_path):
    baseline = tmp_path / "frozen.json"
    baseline.write_text(json.dumps(comparison_report()), encoding="utf-8")
    frozen = baseline.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_samples.py"
    result = subprocess.run([sys.executable, str(script), "--baseline", str(baseline), "--output", str(baseline),
                             "--database", str(tmp_path / "does-not-exist.sqlite3")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "cannot be an output path" in result.stderr
    assert baseline.read_bytes() == frozen


def test_rejected_candidate_ledger_does_not_call_same_paragraph_a_semantic_error():
    candidates = [{"id": "a", "raw_doc_id": "d1", "raw_para_id": "p1", "quote": "同段背景"},
                  {"id": "b", "raw_doc_id": "d1", "raw_para_id": "p2", "quote": "支撑事实"}]
    selection = {"applied": True, "assessments": [
        {"evidence_id": "a", "relation": "background", "reason": "未陈述问题断言", "equivalent_to": None},
        {"evidence_id": "b", "relation": "support", "reason": "直接回答断言", "equivalent_to": None}]}
    gold = [fact("同段另一个事实", "p1")]
    report = selection_audit(selection, candidates, [candidates[1]], gold)
    assert report["rejected_records"] == 1
    assert report["rejected_reference_paragraph_filter"]["paragraph_set"]["matched"] == 1
    assert report["rejected_reference_paragraph_filter"]["strict_quote_set"]["matched"] == 0
    assert report["candidate_dispositions"][0]["retained"] is False
    assert report["candidate_dispositions"][0]["relation"] == "background"
    assert "quote" not in report["candidate_dispositions"][0]
    assert "not proof" in report["note"]
    assert selection_audit(None, None, [], gold)["available"] is False
