"""Real durable event projections; no model calls or competition answers."""

from test_repository import create, setup_repo
from fastapi.testclient import TestClient
import pytest

from app.main import create_app


def started(tmp_path):
    repo, topic = setup_repo(tmp_path)
    run = create(repo, topic)
    repo.claim()
    return repo, run["id"], run["question_ids"][0]


def test_failed_parallel_branch_does_not_erase_completed_sibling(tmp_path):
    repo, rid, qid = started(tmp_path)
    for stage in ("retrieve", "extract", "verify", "freeze", "cross_validation_analyst"):
        repo.event(rid, "node_completed", {"node": stage, "question_id": qid})
    repo.event(rid, "node_started", {"node": "authority_analyst", "question_id": qid})
    details = {"check": "ungrounded_judgment", "judgment_index": 0, "field_path": "judgments[0].evidence_ids", "evidence_count": 0, "prerequisite_count": 0}
    repo.event(rid, "tool_failed", {"node": "authority_analyst", "question_id": qid, "call_id": "a", "code": "validation_failed", "message": "判断必须具有证据或已声明的前提。", "details": details})
    repo.finish_failed(rid, "validation_failed", "判断必须具有证据或已声明的前提。", stage="authority_analyst", details={"question_id": qid, **details})
    activity = repo.activity(rid)
    assert activity["progress"]["completed_count"] == 5
    assert activity["progress"]["total_count"] == 9
    assert activity["progress"]["active_stages"] == []
    failed = activity["progress"]["failed_stages"]
    assert len(failed) == 1 and failed[0]["stage"] == "authority_analyst"
    assert {"question_id": qid, "stage": "cross_validation_analyst"} in activity["progress"]["completed_stages"]
    assert activity["tool_calls"][0]["details"] == details


def test_later_success_clears_failed_stage_but_preserves_actual_failed_attempt(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_failed", {"node": "authority_analyst", "question_id": qid, "call_id": "initial", "code": "validation_failed"})
    repo.event(rid, "model_repair_started", {"node": "authority_analyst", "question_id": qid, "attempt": 1})
    repo.event(rid, "tool_started", {"node": "authority_analyst", "question_id": qid, "call_id": "repair", "name": "authority_analyst_repair", "kind": "model"})
    repo.event(rid, "tool_completed", {"node": "authority_analyst", "question_id": qid, "call_id": "repair"})
    repo.event(rid, "model_repair_completed", {"node": "authority_analyst", "question_id": qid, "attempt": 1})
    repo.event(rid, "node_completed", {"node": "authority_analyst", "question_id": qid})
    activity = repo.activity(rid)
    assert activity["progress"]["failed_stages"] == []
    assert activity["progress"]["completed_count"] == 1
    assert [call["status"] for call in activity["tool_calls"]] == ["failed", "completed"]
    assert activity["tool_calls"][1]["name"] == "authority_analyst_repair"


def test_not_executed_tool_is_neither_failure_nor_completed_work(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "node_started", {"node": "source_bundle_selection", "question_id": qid})
    repo.event(rid, "tool_started", {"node": "source_bundle_selection", "question_id": qid, "call_id": "over-quota", "name": "read_source_bundles"})
    repo.event(rid, "tool_deferred", {"node": "source_bundle_selection", "question_id": qid, "call_id": "over-quota", "summary": "未执行：读取额度保护。"})
    activity = repo.activity(rid)
    assert activity["tool_calls"][0]["status"] == "deferred"
    assert activity["tool_calls"][0]["completed_at"]
    assert activity["progress"]["failed_stages"] == []
    assert activity["progress"]["completed_stages"] == []
    assert activity["progress"]["active_stages"] == [{"question_id": qid, "stage": "source_bundle_selection"}]


def test_public_diagnostics_keep_only_fixed_fields_and_bounded_counts(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_failed", {"node": "extract", "question_id": qid, "call_id": "unsafe", "code": "validation_failed", "message": 'Bearer private-provider-body', "details": {"check": "schema", "judgment_index": 0, "candidate_index": -1, "evidence_count": True, "field_path": "private.provider.body", "response": "private-body", "question_id": "private-model-id", "repair_attempt": 1, "repair_exhausted": True}})
    call = repo.activity(rid)["tool_calls"][0]
    assert "message" not in call
    assert call["details"] == {"check": "schema", "judgment_index": 0, "repair_attempt": 1, "repair_exhausted": True}


def test_paused_stage_is_interrupted_not_failed(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "node_started", {"node": "extract", "question_id": qid})
    repo.event(rid, "tool_started", {"node": "extract", "question_id": qid, "call_id": "extract"})
    repo.pause(rid)
    repo.finish_failed(rid, "cancelled", "已取消", cancelled=True)
    activity = repo.activity(rid)
    assert activity["run"]["view_status"] == "paused"
    assert activity["progress"]["failed_stages"] == []
    assert activity["progress"]["paused_stages"] == [{"question_id": qid, "stage": "extract"}]
    assert activity["tool_calls"][0]["status"] == "cancelled"


def test_failure_stage_survives_when_only_run_error_records_it(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.finish_failed(rid, "validation_failed", "判断没有依据。", stage="authority_analyst", details={"check": "ungrounded_judgment", "question_id": qid})
    assert repo.activity(rid)["progress"]["failed_stages"] == [{"question_id": qid, "stage": "authority_analyst", "code": "validation_failed", "message": "判断没有依据。", "details": {"check": "ungrounded_judgment"}}]


def test_failed_repair_call_does_not_mark_running_parent_terminal(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "node_started", {"node": "authority_analyst", "question_id": qid})
    repo.event(rid, "tool_started", {"node": "authority_analyst", "question_id": qid, "call_id": "parent", "name": "authority_analyst"})
    repo.event(rid, "tool_started", {"node": "authority_analyst", "question_id": qid, "call_id": "repair", "name": "authority_analyst_repair"})
    repo.event(rid, "tool_failed", {"node": "authority_analyst", "question_id": qid, "call_id": "repair", "code": "validation_failed"})
    activity = repo.activity(rid)
    assert activity["progress"]["failed_stages"] == []
    assert activity["progress"]["active_stages"] == [{"question_id": qid, "stage": "authority_analyst"}]
    assert [call["status"] for call in activity["tool_calls"]] == ["running", "failed"]


def test_schema_diagnostic_projection_limits_paths_types_and_count(tmp_path):
    repo, rid, qid = started(tmp_path)
    details = {"schema_errors": [
        {"loc": ["judgments", 0, "statement"], "type": "string_type", "input": "private-body"},
        {"loc": ["evidence", 1, "argument", "时间"], "type": "missing"},
        {"loc": ["private-field", True, -1, {"private": "body"}], "type": "private-type"},
        {"loc": ["conclusion"], "type": "missing"},
    ]}
    repo.event(rid, "tool_failed", {"node": "authority_analyst", "question_id": qid, "call_id": "schema", "code": "model_output_invalid", "details": details})
    repo.finish_failed(rid, "model_output_invalid", "模型输出结构不符合要求。", stage="authority_analyst", details=details)
    projected = repo.activity(rid)
    safe = {"schema_errors": [
        {"loc": ["judgments", 0, "statement"], "type": "string_type"},
        {"loc": ["evidence", 1, "argument", "时间"], "type": "missing"},
        {"loc": ["<field>", "<field>", "<field>", "<field>"], "type": "schema"},
    ]}
    assert projected["tool_calls"][0]["details"] == safe
    assert projected["progress"]["failed_stages"][0]["details"] == safe


def test_http_activity_replays_real_batch_and_precision_summaries(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "node_started", {"node": "extract", "question_id": qid})
    batch = {"batch_index": 2, "batch_count": 5, "paragraph_count": 18, "candidate_count": 7,
             "codepoint_count": 8000, "provider_body": "private", "reasoning": "private"}
    for kind in ("tool_started", "tool_completed"):
        repo.event(rid, kind, {"node": "extract", "question_id": qid, "call_id": "batch-2",
                               "name": "evidence_batch_extraction", "kind": "model", "engine": "LangChain", "summary": batch})
    repo.event(rid, "source_coverage_progress", {"question_id": qid, "strategy": "full_source_scan", "batch_count": 5,
                                                "total_paragraphs": 90, "completed_batches": 2, "scanned_paragraphs": 36, "complete": False})
    selection = {"candidate_count": 30, "selected_count": 19, "rejected_count": 11, "manual_preserved_count": 0,
                 "quote": "private", "assessments": [{"reason": "private"}]}
    for kind in ("tool_started", "tool_completed"):
        repo.event(rid, kind, {"node": "verify", "question_id": qid, "call_id": "precision",
                               "name": "evidence_precision_review", "kind": "model", "engine": "LangChain", "summary": selection})
    with TestClient(create_app(repo.settings)) as client:
        response = client.get(f"/api/runs/{rid}/activity?after=2")
        assert response.status_code == 200
        result = response.json()
        batch_call, precision_call = result["tool_calls"]
        assert batch_call["summary"] == "第 2 / 5 批 · 18 段原文 · 7 条候选"
        assert precision_call["summary"] == "30 条候选 · 19 条精筛保留 · 11 条精筛排除 · 0 条人工保留"
        assert batch_call["status"] == precision_call["status"] == "completed"
        assert batch_call["kind"] == precision_call["kind"] == "model"
        assert all("private" not in call["summary"] for call in result["tool_calls"])
        assert result["progress"]["completed_count"] == 0  # Paid batch completion is not whole-stage completion.
        assert result["progress"]["active_stages"] == [{"question_id": qid, "stage": "extract"}]
        assert all(event["seq"] > 2 for event in result["events"])
    with TestClient(create_app(repo.settings)) as restarted:
        assert restarted.get(f"/api/runs/{rid}/activity").json()["tool_calls"] == result["tool_calls"]


def test_activity_batch_summary_drops_invalid_batch_positions_and_noninteger_counts(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_started", {"node": "extract", "question_id": qid, "call_id": "batch",
                                    "name": "evidence_batch_extraction",
                                    "summary": {"batch_index": 0, "batch_count": 2, "paragraph_count": True,
                                                "candidate_count": -1, "selected_count": "private"}})
    with TestClient(create_app(repo.settings)) as client:
        assert client.get(f"/api/runs/{rid}/activity").json()["tool_calls"][0]["summary"] == ""


@pytest.mark.parametrize("check", ["evidence_selection", "source_catalog", "source_catalog_limit", "source_selection", "unknown_source_selection"])
def test_http_activity_projects_new_fixed_checks_without_model_body(tmp_path, check):
    repo, rid, qid = started(tmp_path)
    details = {"check": check, "field_path": "assessments[0].reason", "reason": "private body",
               "selected_evidence_ids": ["private identity"], "source_text": "private source"}
    repo.event(rid, "tool_failed", {"node": "verify", "question_id": qid, "call_id": "review",
                                   "name": "evidence_precision_review", "code": "model_output_invalid", "details": details})
    repo.finish_failed(rid, "model_output_invalid", "证据精筛完整性校验失败。", stage="verify", details={"question_id": qid, **details})
    with TestClient(create_app(repo.settings)) as client:
        result = client.get(f"/api/runs/{rid}/activity").json()
        expected = {"check": check, "field_path": "assessments[0].reason"}
        assert result["tool_calls"][0]["details"] == expected
        assert result["progress"]["failed_stages"][0]["details"] == expected


def test_http_activity_selection_schema_locations_are_safe_paths_only(tmp_path):
    repo, rid, qid = started(tmp_path)
    details = {"check": "evidence_selection", "field_path": "selected_evidence_ids[2]", "schema_errors": [
        {"loc": ["selected_evidence_ids", 2], "type": "string_type", "input": "private id"},
        {"loc": ["assessments", 0, "equivalent_to"], "type": "missing", "msg": "private body"},
        {"loc": ["assessments", 1, "reason"], "type": "string_type", "input": "private reason"},
    ]}
    repo.event(rid, "tool_failed", {"node": "verify", "question_id": qid, "call_id": "review", "code": "model_output_invalid", "details": details})
    with TestClient(create_app(repo.settings)) as client:
        result = client.get(f"/api/runs/{rid}/activity").json()["tool_calls"][0]["details"]
        assert result == {"check": "evidence_selection", "field_path": "selected_evidence_ids[2]", "schema_errors": [
            {"loc": ["selected_evidence_ids", 2], "type": "string_type"},
            {"loc": ["assessments", 0, "equivalent_to"], "type": "missing"},
            {"loc": ["assessments", 1, "reason"], "type": "string_type"}]}


@pytest.mark.parametrize("path", ["assessments[0].private_response", "assessments[1000000].reason", "assessments[0].reason.provider_body"])
def test_activity_selection_field_paths_reject_unbounded_or_unknown_segments(tmp_path, path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_failed", {"node": "verify", "question_id": qid, "call_id": "review", "code": "model_output_invalid",
                                   "details": {"check": "evidence_selection", "field_path": path}})
    assert repo.activity(rid)["tool_calls"][0]["details"] == {"check": "evidence_selection"}


def test_http_activity_interval_schema_retains_only_new_field_locations(tmp_path):
    repo, rid, qid = started(tmp_path)
    details = {"check": "source_catalog_id", "field_path": "source_catalog_id", "source_catalog_id": "private-binding",
               "schema_errors": [
                   {"loc": ["source_catalog_id"], "type": "literal_error", "input": "private-binding"},
                   {"loc": ["evidence", 1, "source_start_id"], "type": "missing", "msg": "private-start"},
                   {"loc": ["evidence", 1, "source_end_id"], "type": "literal_error", "input": "private-end"}]}
    repo.event(rid, "tool_failed", {"node": "extract", "question_id": qid, "call_id": "interval", "code": "model_output_invalid", "details": details})
    with TestClient(create_app(repo.settings)) as client:
        assert client.get(f"/api/runs/{rid}/activity").json()["tool_calls"][0]["details"] == {
            "check": "source_catalog_id", "field_path": "source_catalog_id", "schema_errors": [
                {"loc": ["source_catalog_id"], "type": "literal_error"},
                {"loc": ["evidence", 1, "source_start_id"], "type": "missing"},
                {"loc": ["evidence", 1, "source_end_id"], "type": "literal_error"}]}


@pytest.mark.parametrize("issue_check", ["unknown_source_start", "unknown_source_end", "cross_paragraph_interval", "reversed_source_interval", "invalid_source_interval", "source_catalog_mismatch"])
def test_http_activity_interval_issues_are_bounded_safe_fixed_codes(tmp_path, issue_check):
    repo, rid, qid = started(tmp_path)
    details = {"check": "source_interval_selection", "invalid_candidate_count": 5, "candidate_index": 1,
               "field_path": "candidates[1].source_end_id", "source_intervals": [{"first_id": "private-start"}],
               "selection_issues": [
                   {"check": issue_check, "candidate_index": 1, "field_path": "candidates[1].source_end_id", "source_end_id": "private-end"},
                   {"check": "unknown_source_start", "candidate_index": 2, "field_path": "candidates[2].private_response", "reason": "private reason"},
                   {"check": "invalid_source_interval", "candidate_index": True, "field_path": "private-body"},
                   {"check": "cross_paragraph_interval", "candidate_index": 3, "field_path": "candidates[3].source_start_id"}]}
    repo.event(rid, "tool_failed", {"node": "extract", "question_id": qid, "call_id": "interval", "code": "model_output_invalid", "details": details})
    repo.finish_failed(rid, "model_output_invalid", "原文区间选择无效。", stage="extract", details={"question_id": qid, **details})
    with TestClient(create_app(repo.settings)) as client:
        result = client.get(f"/api/runs/{rid}/activity").json()
        expected = {"check": "source_interval_selection", "candidate_index": 1, "invalid_candidate_count": 5,
                    "field_path": "candidates[1].source_end_id", "selection_issues": [
                        {"check": issue_check, "candidate_index": 1, "field_path": "candidates[1].source_end_id"},
                        {"check": "unknown_source_start", "candidate_index": 2},
                        {"check": "invalid_source_interval"}]}
        assert result["tool_calls"][0]["details"] == expected
        assert result["progress"]["failed_stages"][0]["details"] == expected


def test_activity_interval_issue_whitelist_rejects_unknown_or_structured_codes(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_failed", {"node": "extract", "question_id": qid, "call_id": "interval", "code": "model_output_invalid",
                                   "details": {"check": "source_interval_selection", "invalid_candidate_count": True,
                                               "selection_issues": [{"check": "private-response"}, {"check": {"body": "private"}}]}})
    assert repo.activity(rid)["tool_calls"][0]["details"] == {"check": "source_interval_selection"}


def test_http_activity_window_schema_retains_safe_location_without_window_alias(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_failed", {"node": "extract", "question_id": qid, "call_id": "window", "code": "model_output_invalid",
                                   "details": {"check": "schema", "field_path": "evidence[2].source_window_id", "source_window_id": "private-window",
                                               "schema_errors": [{"loc": ["evidence", 2, "source_window_id"], "type": "literal_error", "input": "private-window"}]}})
    with TestClient(create_app(repo.settings)) as client:
        assert client.get(f"/api/runs/{rid}/activity").json()["tool_calls"][0]["details"] == {
            "check": "schema", "field_path": "evidence[2].source_window_id",
            "schema_errors": [{"loc": ["evidence", 2, "source_window_id"], "type": "literal_error"}]}


@pytest.mark.parametrize("issue_check", ["unknown_source_window", "invalid_source_window_catalog", "invalid_source_window"])
def test_http_activity_window_selection_issues_project_only_fixed_codes_and_locations(tmp_path, issue_check):
    repo, rid, qid = started(tmp_path)
    details = {"check": "source_window_selection", "invalid_candidate_count": 1, "candidate_index": 2,
               "field_path": "candidates[2].source_window_id", "source_window_catalog": [{"id": "private-window", "quote": "private-source"}],
               "selection_issues": [{"check": issue_check, "candidate_index": 2, "field_path": "candidates[2].source_window_id",
                                     "source_window_id": "private-window", "reason": "private-model-body"}]}
    repo.event(rid, "tool_failed", {"node": "extract", "question_id": qid, "call_id": "window", "code": "model_output_invalid", "details": details})
    repo.finish_failed(rid, "model_output_invalid", "所选原文窗口无效。", stage="extract", details={"question_id": qid, **details})
    with TestClient(create_app(repo.settings)) as client:
        result = client.get(f"/api/runs/{rid}/activity").json()
        expected = {"check": "source_window_selection", "invalid_candidate_count": 1, "candidate_index": 2,
                    "field_path": "candidates[2].source_window_id",
                    "selection_issues": [{"check": issue_check, "candidate_index": 2, "field_path": "candidates[2].source_window_id"}]}
        assert result["tool_calls"][0]["details"] == expected
        assert result["progress"]["failed_stages"][0]["details"] == expected


def test_http_activity_window_catalog_integrity_failure_has_fixed_public_check(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_failed", {"node": "extract", "question_id": qid, "call_id": "window", "code": "source_integrity_failed",
                                   "details": {"check": "source_window_catalog", "source_window_catalog": "private-source"}})
    with TestClient(create_app(repo.settings)) as client:
        assert client.get(f"/api/runs/{rid}/activity").json()["tool_calls"][0]["details"] == {"check": "source_window_catalog"}


def test_new_precision_usage_keeps_safe_operation_without_relabeling_historical_unknown(tmp_path):
    repo, rid, qid = started(tmp_path)
    old_record = {"operation": "unknown", "usage_source": "provider", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    repo.record_usage(rid, qid, [old_record])
    repo.finish_failed(rid, "model_output_invalid", "旧精筛协议未通过校验。", stage="verify")
    old_usage = repo.run(rid)["usage"]
    next_run = create(repo, repo.run(rid)["topic_id"])
    repo.claim()
    repo.record_usage(next_run["id"], qid, [
        {"operation": "select_evidence", "usage_source": "provider", "input_tokens": 20, "output_tokens": 7, "total_tokens": 27,
         "reasoning": "private-body", "model": "private-model"},
        {"operation": {"body": "private"}, "usage_source": "provider", "total_tokens": 3},
    ])
    with TestClient(create_app(repo.settings)) as client:
        usage = client.get(f"/api/runs/{next_run['id']}/activity").json()["usage"][qid]
        assert usage[0] == {"operation": "select_evidence", "model": "explicit-test-provider", "usage_source": "provider",
                            "input_tokens": 20, "output_tokens": 7, "total_tokens": 27, "cost": None}
        assert usage[1]["operation"] == "unknown"
        assert client.get(f"/api/runs/{rid}/activity").json()["usage"] == old_usage


def test_http_activity_precision_decision_schema_locations_use_normalized_candidate_positions(tmp_path):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_failed", {"node": "verify", "question_id": qid, "call_id": "decision-map", "code": "model_output_invalid",
                                   "details": {"check": "schema", "field_path": "decisions[2].equivalent_to", "candidate_catalog_id": "private-binding",
                                               "schema_errors": [
                                                   {"loc": ["decisions", 2, "equivalent_to"], "type": "literal_error", "input": "private-ref"},
                                                   {"loc": ["candidate_catalog_id"], "type": "literal_error", "input": "private-binding"},
                                                   {"loc": ["decisions", "private-record-id", "reason"], "type": "missing"}]}})
    with TestClient(create_app(repo.settings)) as client:
        assert client.get(f"/api/runs/{rid}/activity").json()["tool_calls"][0]["details"] == {
            "check": "schema", "field_path": "decisions[2].equivalent_to", "schema_errors": [
                {"loc": ["decisions", 2, "equivalent_to"], "type": "literal_error"},
                {"loc": ["candidate_catalog_id"], "type": "literal_error"},
                {"loc": ["decisions", "<field>", "reason"], "type": "missing"}]}


@pytest.mark.parametrize("issue_check", ["precision_duplicate_self", "precision_duplicate_target_not_retained", "precision_unexpected_equivalent_to", "precision_decision_coverage"])
def test_http_activity_precision_decisions_project_only_fixed_issues(tmp_path, issue_check):
    repo, rid, qid = started(tmp_path)
    details = {"check": "precision_decisions", "candidate_index": 2, "invalid_candidate_count": 1,
               "field_path": "decisions[2].equivalent_to", "decisions": {"private-id": {"reason": "private-body"}},
               "selection_issues": [{"check": issue_check, "candidate_index": 2, "field_path": "decisions[2].equivalent_to",
                                     "record": {"reason": "private-body"}, "equivalent_to": "private-id"}]}
    repo.event(rid, "tool_failed", {"node": "verify", "question_id": qid, "call_id": "decision-map", "code": "model_output_invalid", "details": details})
    repo.finish_failed(rid, "model_output_invalid", "精筛决定映射未通过校验。", stage="verify", details={"question_id": qid, **details})
    with TestClient(create_app(repo.settings)) as client:
        result = client.get(f"/api/runs/{rid}/activity").json()
        expected = {"check": "precision_decisions", "candidate_index": 2, "invalid_candidate_count": 1,
                    "field_path": "decisions[2].equivalent_to",
                    "selection_issues": [{"check": issue_check, "candidate_index": 2, "field_path": "decisions[2].equivalent_to"}]}
        assert result["tool_calls"][0]["details"] == expected
        assert result["progress"]["failed_stages"][0]["details"] == expected


@pytest.mark.parametrize("check,count_name", [("precision_duplicate_json_key", "duplicate_key_count"), ("precision_candidate_catalog", "invalid_candidate_count")])
def test_activity_precision_map_summary_diagnostics_never_expose_keys_or_catalog_records(tmp_path, check, count_name):
    repo, rid, qid = started(tmp_path)
    repo.event(rid, "tool_failed", {"node": "verify", "question_id": qid, "call_id": "decision-map", "code": "model_output_invalid",
                                   "details": {"check": check, count_name: 2, "duplicate_keys": ["private-key"], "candidate_catalog": "private-record"}})
    assert repo.activity(rid)["tool_calls"][0]["details"] == {"check": check, count_name: 2}
