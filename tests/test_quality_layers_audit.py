"""Read-only source-window layer audit with synthetic persisted revisions."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from scripts.audit_quality_layers import (additional_retention, audit_revision, candidate_anchor_audit,
                                         final_reference_layers, layer_stats, ro_connect)


TEXT = "abcdefghij"


def window(identity, start, end, *, paragraph="p1", views=0):
    return {"id": identity, "paragraph_id": paragraph, "raw_doc_id": "d1", "raw_para_id": paragraph,
            "start_cp": start, "end_cp": end, "quote": TEXT[start:end], "fact_des": "synthetic " + identity,
            "interpretation_origin": "model", "additional_interpretations": [
                {"fact_des": f"synthetic extra interpretation {index}", "argument": {}, "entities": []} for index in range(views)]}


def fixture_database(tmp_path, *, historical=False):
    path = tmp_path / "sources.sqlite3"
    rows = [window("a", 0, 4, views=2), window("b", 2, 6), window("c", 6, 8, views=1),
            window("d", 1, 3, paragraph="p2")]
    result = {"origin_run_id": "origin-run", "evidence": deepcopy(rows[:3]),
        "candidate_evidence": deepcopy(rows),
        "final": {"supporting_evidence_ids": ["a"], "conflicting_evidence_ids": [],
                  "judgments": [{"evidence_ids": ["b"]}], "conflicts": []},
        "evidence_selection": {"applied": True, "candidate_count": 4, "selected_count": 3, "rejected_count": 1},
        "source_coverage": {"strategy": "full_source_scan", "complete": True, "batch_count": 2,
                            "completed_batches": 2, "total_paragraphs": 2, "scanned_paragraphs": 2}}
    if historical:
        result.pop("candidate_evidence")
    snapshot = {"question_results": {"internal-q": result}}
    connection = sqlite3.connect(path)
    try:
        connection.executescript("CREATE TABLE documents(id TEXT,content TEXT); CREATE TABLE paragraphs(id TEXT,text TEXT); "
            "CREATE TABLE graph_revisions(id TEXT,topic_id TEXT,state TEXT); CREATE TABLE questions(id TEXT,topic_id TEXT,raw_question_id TEXT,question TEXT); "
            "CREATE TABLE revision_artifacts(revision_id TEXT,payload_json TEXT); "
            "CREATE TABLE runs(id TEXT,model_config_json TEXT,status TEXT,created_at TEXT,completed_at TEXT);")
        connection.execute("INSERT INTO documents VALUES ('d1',?)", (TEXT,))
        connection.executemany("INSERT INTO paragraphs VALUES (?,?)", [("p1", TEXT), ("p2", TEXT)])
        connection.execute("INSERT INTO graph_revisions VALUES ('fixed-ready','t1','ready')")
        connection.execute("INSERT INTO questions VALUES ('internal-q','t1','q_003','Synthetic question')")
        connection.execute("INSERT INTO revision_artifacts VALUES ('fixed-ready',?)", (json.dumps(snapshot),))
        config = {"prompt_version": "foundation-v8" if historical else "trusted-analysis-json-v13", "prompt_sha256": "fixed-code-hash",
                  "model_name": "explicit-synthetic-provider", "api_key": "PRIVATE_CONFIG_SENTINEL_DO_NOT_REPORT"}
        connection.execute("INSERT INTO runs VALUES ('origin-run',?,'succeeded','created','completed')", (json.dumps(config),))
        connection.commit()
    finally:
        connection.close()
    return path


def test_readonly_connection_rejects_writes_and_closes_the_connection(tmp_path):
    path = fixture_database(tmp_path)
    original = path.read_bytes()
    with ro_connect(path) as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM paragraphs")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    assert path.read_bytes() == original


def test_overlap_counts_are_bound_to_source_paragraph_and_codepoint_geometry():
    rows = [window("a", 0, 4), window("b", 2, 6), window("adjacent", 6, 8),
            window("other-source", 1, 3, paragraph="p2")]
    before = deepcopy(rows)
    result = layer_stats(rows, paragraphs={"p1": TEXT, "p2": TEXT})
    assert result["window_records"] == result["valid_source_window_records"] == 4
    assert result["same_paragraph_overlapping_window_pairs"] == 1
    assert result["windows_in_same_paragraph_overlap"] == 2
    assert result["paragraphs_with_multiple_windows"] == 1
    assert result["sum_window_codepoints"] == 12 and result["union_source_codepoints"] == 10
    assert result["overlapping_duplicate_codepoints"] == 2
    assert rows == before


def test_invalid_source_span_remains_counted_and_cannot_inflate_overlap():
    row = window("bad", 0, 4)
    row["quote"] = "WXYZ"
    result = layer_stats([window("a", 0, 4), row], paragraphs={"p1": TEXT})
    assert result["window_records"] == 2 and result["valid_source_window_records"] == 1
    assert result["same_paragraph_overlapping_window_pairs"] == 0
    assert result["invalid_window_records"] == [{"row_index": 1, "evidence_id": "bad", "check": "source_span"}]


def test_same_source_exact_duplicate_and_nested_spans_are_explicit_separate_counts():
    rows = [window("a", 0, 8), window("b", 0, 8), window("inside", 2, 5)]
    result = layer_stats(rows, paragraphs={"p1": TEXT}, details_limit=1)
    assert result["same_paragraph_overlapping_window_pairs"] == 3
    assert result["nested_window_pairs"] == 3 and result["identical_span_pairs"] == 1
    assert result["windows_in_same_paragraph_overlap"] == 3
    assert result["overlap_pair_details_truncated"] is True
    assert len(result["overlap_pair_details"]) == 1


def test_fixed_revision_reports_layers_additional_annotations_and_uncited_retained_windows(tmp_path):
    path = fixture_database(tmp_path)
    original = path.read_bytes()
    report = audit_revision(path, topic_id="t1", revision_id="fixed-ready", question_id="q_003")
    row = report["questions"][0]
    assert {key: value["window_records"] for key, value in row["layers"].items()} == {
        "candidate": 4, "selected": 3, "final_direct_linked": 1, "final_any_arbitration_linked": 2}
    assert row["selected_without_direct_final_reference_count"] == 2
    assert row["selected_without_any_arbitration_reference_count"] == 1
    assert row["selected_without_any_arbitration_reference_ids"] == ["c"]
    assert row["layers"]["selected"]["windows_with_additional_interpretations"] == 2
    assert row["additional_interpretation_retention"]["candidate_additional_views_still_present_in_selected"] == 3
    assert row["additional_interpretation_retention"]["candidate_additional_windows_direct_final_linked"] == 1
    assert row["origin_run_id"] == "origin-run" and row["question_id"] == "q_003"
    assert row["origin_version"]["prompt_version"] == "trusted-analysis-json-v13"
    assert report["database_mode"] == "ro+query_only"
    assert report["gold_read"] is report["model_called"] is False
    assert report["source_unchanged"] is True and path.read_bytes() == original
    assert "PRIVATE_CONFIG_SENTINEL_DO_NOT_REPORT" not in json.dumps(report)
    assert TEXT not in json.dumps(report)


def test_missing_historical_ledger_is_unknown_instead_of_selected_reconstruction(tmp_path):
    path = fixture_database(tmp_path, historical=True)
    report = audit_revision(path, topic_id="t1", revision_id="fixed-ready")
    row = report["questions"][0]
    assert row["layers"]["candidate"]["available"] is False
    assert row["layers"]["candidate"]["window_records"] is None
    assert row["layers"]["selected"]["window_records"] == 3
    assert row["additional_interpretation_retention"]["available"] is False
    assert row["selected_ids_absent_from_candidate_ledger"] is None
    assert row["origin_version"]["prompt_version"] == "foundation-v8"


@pytest.mark.parametrize("scope", [
    {"topic_id": "wrong", "revision_id": "fixed-ready"},
    {"topic_id": "t1", "revision_id": "unknown"},
    {"topic_id": "t1", "revision_id": "fixed-ready", "question_id": "not-in-revision"},
])
def test_audit_requires_an_explicit_compatible_ready_revision_and_present_question(tmp_path, scope):
    path = fixture_database(tmp_path)
    with pytest.raises(ValueError):
        audit_revision(path, **scope)


def test_final_direct_reference_is_distinct_from_judgment_and_conflict_record_references():
    direct, all_ids, groups = final_reference_layers({"supporting_evidence_ids": ["a"], "conflicting_evidence_ids": [],
        "judgments": [{"evidence_ids": ["b"]}], "conflicts": [{"supporting_evidence_ids": ["a"], "conflicting_evidence_ids": ["c"]}]})
    assert direct == {"a"} and all_ids == {"a", "b", "c"}
    assert groups == {"direct_conclusion": ["a"], "judgment": ["b"], "conflict_record": ["a", "c"]}


def test_cli_prints_without_output_and_never_overwrites_a_frozen_report(tmp_path):
    database = fixture_database(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/audit_quality_layers.py"
    base = [sys.executable, str(script), "--database", str(database), "--topic-id", "t1", "--revision", "fixed-ready"]
    printed = subprocess.run(base, capture_output=True, text=True)
    assert printed.returncode == 0, printed.stderr
    assert json.loads(printed.stdout)["fixed_revision_id"] == "fixed-ready"
    output = tmp_path / "frozen-audit.json"
    output.write_text("immutable historical report", encoding="utf-8")
    original = output.read_bytes()
    rejected = subprocess.run([*base, "--output", str(output)], capture_output=True, text=True)
    assert rejected.returncode == 1 and output.read_bytes() == original
    fresh = tmp_path / "new-audit.json"
    written = subprocess.run([*base, "--output", str(fresh)], capture_output=True, text=True)
    assert written.returncode == 0, written.stderr
    assert json.loads(fresh.read_text(encoding="utf-8"))["source_unchanged"] is True


def test_module_has_no_runtime_or_model_initialization():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", "import sys; import scripts.audit_quality_layers; assert 'app' not in sys.modules"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_derived_fact_ids_require_exact_parent_geometry_and_do_not_mutate_candidates():
    parent = window("parent", 0, 8, views=2)
    child = {**deepcopy(parent), "id": "new-fact", "source_evidence_id": "parent", "fact_view_id": "F-version"}
    before = deepcopy(parent)
    report = candidate_anchor_audit([parent], [child])
    assert report["derived_fact_records"] == 1
    assert report["selected_without_valid_candidate_anchor"] == []
    assert parent == before
    child["quote"] = "changed!"
    invalid = candidate_anchor_audit([parent], [child])
    assert invalid["derived_fact_records"] == 0
    assert invalid["selected_parent_geometry_mismatch"] == ["new-fact"]
    child["source_evidence_id"] = "invented-parent"
    assert candidate_anchor_audit([parent], [child])["selected_without_valid_candidate_anchor"] == ["new-fact"]


def test_additional_annotation_promoted_to_verified_fact_row_is_retained_without_inventing_siblings():
    parent = window("parent", 0, 8, views=2)
    accepted = {**deepcopy(parent), "id": "new-fact", "source_evidence_id": "parent",
                **deepcopy(parent["additional_interpretations"][1])}
    accepted.pop("additional_interpretations")
    report = additional_retention([parent], [accepted], {"new-fact"}, {"new-fact"})
    assert report["candidate_additional_views_still_present_in_selected"] == 1
    assert len(report["candidate_additional_views_not_present_in_selected"]) == 1
    assert report["candidate_additional_windows_direct_final_linked"] == 1
    assert report["candidate_additional_windows_still_selected"] == 1
    accepted["end_cp"] -= 1
    invalid = additional_retention([parent], [accepted], {"new-fact"}, {"new-fact"})
    assert invalid["candidate_additional_views_still_present_in_selected"] == 0
    assert invalid["candidate_additional_windows_still_selected"] == 0


def test_derived_fact_parentage_remains_unknown_without_historical_ledger():
    child = {**window("fact", 0, 4), "source_evidence_id": "unknown"}
    report = candidate_anchor_audit(None, [child])
    assert report["available"] is False
    assert report["derived_fact_records"] is None
