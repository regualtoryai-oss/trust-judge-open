"""Separate semantic-evaluation process, synthetic gold and mocked HTTP only."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import httpx
import pytest

from scripts.evaluate_semantics import (
    HTTPPairJudge, INPUT_FORMAT, JudgeFailure, PROMPT_VERSION, EvaluationInputError,
    evaluate_frozen, freeze_sqlite_input, maximum_matching, validate_verdict, validate_wire_verdict, WIRE_SCHEMA,
)


def fact(identity="f1", *, quote="设备已复位。", para="p1", **changes):
    return {"fact_id": identity, "doc_id": "d1", "para_id": para, "quote": quote,
            "fact_des": "设备已复位。", "argument": {"主体": "设备", "时间": "当时"}, "entities": ["设备"], **changes}


def fixture(*, submitted=None, gold=None, refs=None):
    return {"format": INPUT_FORMAT, "scope": {"topic_id": "t1", "question_id": "q_003", "question": "设备是否复位？"},
        "submission": {"revision_id": "fixed-revision-one", "source_fingerprint_sha256": "immutable-source-hash",
                       "snapshot_sha256": "snapshot-hash", "origin_run_id": "generated-without-gold",
                       "evidence": [fact()] if submitted is None else submitted,
                       "known_runtime_usage": [{"total_tokens": 45}]},
        "reference": {"facts": [fact()] if gold is None else gold,
            "question_references": ([{"side": "supporting_facts", "reference": {"fact_id": "f1"}}] if refs is None else refs),
            "source_files_sha256": {"evidence_samples.json": "frozen-reference-hash"}},
        "source_paragraphs": [{"doc_id": "d1", "para_id": "p1", "text": "设备已复位。其他背景。"},
                              {"doc_id": "d1", "para_id": "p2", "text": "设备已复位。"}]}


def verdict(kind="match", **changes):
    return {"verdict": kind, "same_fact": "yes", "subject": "compatible", "time": "compatible",
            "value": "not_applicable", "statistical_scope": "compatible",
            "submitted_quote_supports_fact": "yes", "reference_quote_supports_fact": "yes",
            "reason": "同一设备、相同时点，原文明确复位。", **changes}


class Judge:
    identity = {"provider": "synthetic-test-only", "model": "explicit-injected-judge"}

    def __init__(self, values=None):
        self.calls, self.values = [], list(values or [])

    def __call__(self, payload):
        self.calls.append(deepcopy(payload))
        value = self.values.pop(0) if self.values else verdict()
        if isinstance(value, Exception):
            raise value
        return {"verdict": value, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}


def test_same_paragraph_is_only_an_upper_bound_and_cannot_force_semantic_success():
    data = fixture(submitted=[fact(quote="其他背景。", fact_des="设备周边背景。")])
    judge = Judge([verdict("no_match", same_fact="no", reason="背景段没有陈述复位状态。")])
    report = evaluate_frozen(data, judge=judge)
    assert report["paragraph_filter_upper_bound"]["precision"] == 1
    assert report["semantic_development_proxy_confirmed"]["precision"] == 0
    assert report["pairs"][0]["verdict"]["same_fact"] == "no"
    assert report["official_score"] is report["topic_wide_section_2_2_score"] is False
    assert "question_scoped" in report["evaluation_scope"]


def test_same_document_and_paragraph_filter_precedes_every_model_call():
    data = fixture(submitted=[fact(), fact("other-para", para="p2"), fact("other-doc", doc_id="d2")])
    judge = Judge()
    report = evaluate_frozen(data, judge=judge)
    assert len(judge.calls) == report["counts"]["eligible_same_paragraph_pairs"] == 1
    assert report["semantic_development_proxy_confirmed"]["precision"] == pytest.approx(1 / 3)
    assert report["unmatched_submitted_indices"] == [1, 2]


def test_question_reference_union_does_not_duplicate_evidence_but_keeps_relation_audit():
    refs = [{"side": side, "reference": {"fact_id": "f1"}} for side in ("supporting_facts", "conflicting_facts")]
    report = evaluate_frozen(fixture(refs=refs), judge=Judge())
    assert report["counts"]["referenced_gold_evidence_records"] == 1
    assert report["denominator_audit"]["raw_relation_reference_occurrences"] == 2
    assert report["denominator_audit"]["duplicate_relation_references_not_additional_evidence"] == 1
    assert report["semantic_development_proxy_confirmed"]["recall"] == 1


def test_original_duplicate_records_remain_in_both_full_denominators_and_match_one_to_one():
    data = fixture(submitted=[fact(), fact(), None], gold=[fact(), fact()])
    report = evaluate_frozen(data, judge=Judge())
    assert report["counts"]["submitted_records"] == 3
    assert report["counts"]["referenced_gold_evidence_records"] == 2
    assert report["semantic_development_proxy_confirmed"]["matches"] == 2
    assert report["semantic_development_proxy_confirmed"]["precision"] == pytest.approx(2 / 3)
    assert any(row["code"] == "duplicate_original_gold_fact_id" for row in report["issues"])
    assert any(row["code"] == "invalid_fact_location" for row in report["issues"])


def test_missing_and_malformed_reference_records_remain_unknown_denominator_placeholders():
    refs = [{"side": "supporting_facts", "reference": {"fact_id": "f1"}},
            {"side": "conflicting_facts", "reference": {"fact_id": "missing"}},
            {"side": "supporting_facts", "reference": {}}]
    data = fixture(gold=[fact(), {"quote": "缺少编号的原始gold行"}], refs=refs)
    report = evaluate_frozen(data, judge=Judge())
    assert report["counts"]["referenced_gold_evidence_records"] == 3
    assert report["semantic_development_proxy_confirmed"]["recall"] == pytest.approx(1 / 3)
    assert report["denominator_audit"]["unscopable_original_gold_rows"] == 1
    assert report["denominator_audit"]["missing_reference_placeholders_counted"] == 1
    assert report["denominator_audit"]["malformed_reference_placeholders_counted"] == 1
    assert report["unmatched_reference_indices"] == [1, 2]


def test_uncertainty_and_supplier_failure_are_not_success_and_known_usage_is_preserved():
    data = fixture(submitted=[fact(), fact(quote="其他背景。")])
    judge = Judge([verdict("uncertain", time="uncertain", reason="时点未说明，不能补全。"),
                   JudgeFailure("supplier_output_invalid", usage={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10})])
    report = evaluate_frozen(data, judge=judge)
    assert report["status"] == "incomplete_supplier_failure"
    assert report["semantic_development_proxy_confirmed"]["matches"] == 0
    assert report["semantic_development_proxy_possible_including_unknowns"]["matches"] == 1
    assert report["counts"]["uncertain_or_unavailable_pairs"] == 2
    usage = report["known_usage"]["new_evaluation_requests"]
    assert usage["request_records"] == 2 and usage["total_tokens"]["known_sum"] == 25
    assert report["supplier_failures"][0]["code"] == "supplier_output_invalid"


def test_transport_failure_retains_unknown_usage_instead_of_fabricating_zero_consumption():
    report = evaluate_frozen(fixture(), judge=Judge([JudgeFailure("supplier_transport_failed")]))
    usage = report["known_usage"]["new_evaluation_requests"]
    assert usage["request_records"] == 1
    assert usage["total_tokens"] == {"known_sum": 0, "unavailable_records": 1}
    assert report["semantic_development_proxy_confirmed"]["matches"] == 0


def test_quote_location_anomaly_is_not_repaired_or_dropped_from_denominator():
    judge = Judge()
    data = fixture(submitted=[fact(quote="“不存在原文的改写”")])
    report = evaluate_frozen(data, judge=judge)
    assert judge.calls == []
    assert report["pairs"][0]["status"] == "source_or_fact_anomaly"
    assert report["counts"]["submitted_records"] == 1
    assert report["paragraph_filter_upper_bound"]["matches"] == 1
    assert report["semantic_development_proxy_confirmed"]["matches"] == 0


def test_no_execute_plan_does_not_construct_a_model_or_claim_success():
    report = evaluate_frozen(fixture())
    assert report["status"] == "planned_not_called"
    assert report["counts"]["eligible_same_paragraph_pairs"] == 1
    assert report["pairs"][0]["status"] == "not_called"
    assert report["semantic_development_proxy_confirmed"]["matches"] == 0
    assert report["known_usage"]["new_evaluation_requests"]["request_records"] == 0


def test_maximum_matching_finds_augmenting_path_instead_of_greedy_underestimate():
    matches = maximum_matching(2, 2, [(0, 0), (0, 1), (1, 0)])
    assert matches == [{"submitted_index": 1, "reference_index": 0}, {"submitted_index": 0, "reference_index": 1}]
    assert maximum_matching(0, 0, []) == []


@pytest.mark.parametrize("change", [
    {"same_fact": "no"}, {"time": "incompatible"}, {"value": "uncertain"},
    {"subject": "unknown"}, {"statistical_scope": "uncertain"},
    {"submitted_quote_supports_fact": "no"}, {"reference_quote_supports_fact": "uncertain"},
    {"reason": ""}, {"subjective_score": .9},
])
def test_verdict_contract_forbids_scores_and_inconsistent_matches(change):
    with pytest.raises(ValueError):
        validate_verdict(verdict(**change))


def test_cache_avoids_repeat_payment_and_binds_complete_pair_prompt_and_judge_identity(tmp_path, monkeypatch):
    data, cache = fixture(), tmp_path / "separate-eval-cache"
    first = Judge(); baseline = evaluate_frozen(data, judge=first, cache_dir=cache)
    second = Judge(); cached = evaluate_frozen(data, judge=second, cache_dir=cache)
    assert len(first.calls) == 1 and second.calls == []
    assert cached["pairs"][0]["cache_hit"] is True
    assert cached["known_usage"]["new_evaluation_requests"]["request_records"] == 0
    assert cached["known_usage"]["reused_unique_cached_decisions_original_usage"]["total_tokens"]["known_sum"] == 15
    changed = deepcopy(data); changed["submission"]["evidence"][0]["argument"]["时间"] = "稍后"
    third = Judge(); evaluate_frozen(changed, judge=third, cache_dir=cache)
    assert len(third.calls) == 1
    fourth = Judge(); fourth.identity = {**Judge.identity, "model": "different-evaluator"}
    evaluate_frozen(data, judge=fourth, cache_dir=cache)
    assert len(fourth.calls) == 1
    monkeypatch.setattr("scripts.evaluate_semantics.PROMPT_VERSION", PROMPT_VERSION + "-changed")
    fifth = Judge(); current = evaluate_frozen(data, judge=fifth, cache_dir=cache)
    assert len(fifth.calls) == 1
    assert current["pairs"][0]["cache_key"] != baseline["pairs"][0]["cache_key"]


def test_cache_tampering_is_explicit_unknown_and_never_paid_or_silently_accepted(tmp_path):
    cache = tmp_path / "cache"
    evaluate_frozen(fixture(), judge=Judge(), cache_dir=cache)
    path = next(cache.glob("*.json"))
    value = json.loads(path.read_text(encoding="utf-8")); value["verdict"]["reason"] = "changed"
    path.write_text(json.dumps(value), encoding="utf-8")
    judge = Judge(); report = evaluate_frozen(fixture(), judge=judge, cache_dir=cache)
    assert judge.calls == []
    assert report["supplier_failures"][0]["code"] == "semantic_cache_invalid"
    assert report["semantic_development_proxy_confirmed"]["matches"] == 0


def test_cache_binds_actual_prompt_bytes_and_explicit_occurrence_coordinates(tmp_path, monkeypatch):
    data, cache = fixture(), tmp_path / "cache"
    first = Judge(); baseline = evaluate_frozen(data, judge=first, cache_dir=cache)
    changed = deepcopy(data)
    changed["submission"]["evidence"][0].update(start_cp=0, end_cp=6)
    second = Judge(); coordinate_report = evaluate_frozen(changed, judge=second, cache_dir=cache)
    assert len(second.calls) == 1
    assert second.calls[0]["submitted"]["start_cp"] == 0
    assert coordinate_report["pairs"][0]["cache_key"] != baseline["pairs"][0]["cache_key"]
    monkeypatch.setattr("scripts.evaluate_semantics.JUDGE_INSTRUCTIONS", "Changed independent evaluator instructions.")
    third = Judge(); new_prompt = evaluate_frozen(data, judge=third, cache_dir=cache)
    assert len(third.calls) == 1
    assert new_prompt["prompt_sha256"] != baseline["prompt_sha256"]


def test_eval_cache_cannot_be_written_inside_runtime_ingestion_directory():
    runtime = Path(__file__).resolve().parents[1] / ".runtime/forbidden-gold-eval-cache"
    with pytest.raises(EvaluationInputError, match="cannot_overwrite_input_or_enter_runtime"):
        evaluate_frozen(fixture(), judge=Judge(), cache_dir=runtime)
    assert not runtime.exists()


def test_frozen_inputs_cannot_be_mutated_by_the_independent_judge():
    data = fixture(); before = deepcopy(data)
    judge = Judge()
    original_call = judge.__class__.__call__

    class MutatingJudge(Judge):
        def __call__(self, payload):
            result = original_call(self, payload)
            payload["reference"]["fact_des"] = "mutated"
            return result

    report = evaluate_frozen(data, judge=MutatingJudge())
    assert data == before and report["input_unchanged"] is True


def database_fixture(tmp_path):
    database, sample = tmp_path / "runtime.sqlite3", tmp_path / "samples"
    sample.mkdir()
    snapshot = {"question_results": {"internal-q": {"origin_run_id": "raw-only-run", "usage": [{"total_tokens": 99}],
        "evidence": [{**fact(), "raw_doc_id": "d1", "raw_para_id": "p1"}]}}}
    with sqlite3.connect(database) as connection:
        connection.executescript("CREATE TABLE documents(id TEXT,content TEXT); CREATE TABLE paragraphs(id TEXT,text TEXT); "
            "CREATE TABLE graph_revisions(id TEXT,topic_id TEXT,state TEXT); CREATE TABLE questions(id TEXT,topic_id TEXT,raw_question_id TEXT,question TEXT); "
            "CREATE TABLE revision_artifacts(revision_id TEXT,payload_json TEXT); CREATE TABLE run_events(type TEXT,data_json TEXT);")
        connection.execute("INSERT INTO documents VALUES ('d1','设备已复位。')")
        connection.execute("INSERT INTO paragraphs VALUES ('p1','设备已复位。')")
        connection.execute("INSERT INTO graph_revisions VALUES ('fixed-revision','t1','ready')")
        connection.execute("INSERT INTO questions VALUES ('internal-q','t1','q_003','设备是否复位？')")
        connection.execute("INSERT INTO revision_artifacts VALUES (?,?)", ("fixed-revision", json.dumps(snapshot)))
    values = {"raw_documents.json": [{"doc_id": "d1", "content": "设备已复位。", "paragraphs": [{"para_id": "p1", "char_start": 0, "char_end": 6}]}],
              "evidence_samples.json": [fact(fact_des="EVAL_ONLY_GOLD_SENTINEL")],
              "conclusions.json": [{"question_id": "q_003", "supporting_facts": [{"fact_id": "f1"}], "conflicting_facts": []}]}
    for name, value in values.items():
        (sample / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return database, sample


def test_sqlite_fixed_revision_is_read_only_and_gold_never_enters_runtime_sources_events_or_artifacts(tmp_path):
    database, sample = database_fixture(tmp_path)
    frozen_bytes = database.read_bytes()
    source_bytes = {path.name: path.read_bytes() for path in sample.iterdir()}
    artifact = freeze_sqlite_input(database, topic_id="t1", revision_id="fixed-revision", question_id="q_003", sample_dir=sample)
    assert artifact["submission"]["revision_id"] == "fixed-revision"
    assert artifact["reference"]["facts"][0]["fact_des"] == "EVAL_ONLY_GOLD_SENTINEL"
    report = evaluate_frozen(artifact, judge=Judge())
    assert report["runtime_database_write_performed"] is False
    assert database.read_bytes() == frozen_bytes
    assert b"EVAL_ONLY_GOLD_SENTINEL" not in database.read_bytes()
    assert {path.name: path.read_bytes() for path in sample.iterdir()} == source_bytes
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM run_events").fetchone()[0] == 0
    with pytest.raises(EvaluationInputError, match="semantic_revision_not_ready_for_topic"):
        freeze_sqlite_input(database, topic_id="t1", revision_id="other", question_id="q_003", sample_dir=sample)


def test_module_import_never_loads_runtime_repository_provider_or_workflow():
    script = ("import sys; import scripts.evaluate_semantics; "
              "assert not any(x in sys.modules for x in ('app.repository','app.agents.provider','app.workflows.analysis'))")
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_http_judge_uses_separate_request_and_never_reports_secret_or_provider_body():
    key = "TEST_AUTH_SECRET_DO_NOT_PERSIST"
    seen = []

    def failure(request):
        seen.append(request)
        assert request.headers["Authorization"] == "Bearer " + key
        return httpx.Response(429, json={"error": key + " PROVIDER_PRIVATE_BODY"})

    judge = HTTPPairJudge(model="synthetic-model", provider="deepseek", base_url="https://eval.invalid/v1", api_key=key,
                          transport=httpx.MockTransport(failure))
    try:
        report = evaluate_frozen(fixture(), judge=judge)
    finally:
        judge.close()
    rendered = json.dumps(report)
    assert key not in rendered and "PROVIDER_PRIVATE_BODY" not in rendered
    assert report["supplier_failures"][0]["http_status"] == 429
    assert len(seen) == 1
    payload = json.loads(seen[0].content)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}


def test_http_invalid_paid_output_keeps_usage_after_an_earlier_success():
    outputs = [{key: value for key, value in verdict().items() if key != "verdict"}, {"score": 1}]

    def transport(request):
        return httpx.Response(200, json={"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(outputs.pop(0))}}]})

    judge = HTTPPairJudge(model="test", provider="deepseek", base_url="https://eval.invalid", api_key="transient",
                          transport=httpx.MockTransport(transport))
    try:
        report = evaluate_frozen(fixture(submitted=[fact(), fact(quote="其他背景。")]), judge=judge)
    finally:
        judge.close()
    assert report["known_usage"]["new_evaluation_requests"]["total_tokens"]["known_sum"] == 20
    assert report["known_usage"]["new_evaluation_requests"]["request_records"] == 2
    assert report["supplier_failures"][0]["code"] == "supplier_output_invalid"
    assert report["counts"]["judged_pairs"] == 1


@pytest.mark.parametrize("changes,expected", [
    ({}, "match"), ({"time": "uncertain"}, "uncertain"),
    ({"subject": "incompatible"}, "no_match"), ({"same_fact": "no", "time": "uncertain"}, "no_match"),
    ({"submitted_quote_supports_fact": "no"}, "no_match"),
])
def test_v2_wire_derives_verdict_conservatively_instead_of_asking_for_a_redundant_label(changes, expected):
    wire = {key: value for key, value in verdict(**changes).items() if key != "verdict"}
    assert validate_wire_verdict(wire)["verdict"] == expected
    assert "verdict" not in WIRE_SCHEMA["required"]
    assert "verdict" not in WIRE_SCHEMA["properties"]


@pytest.mark.parametrize("kind,expected_check,expected_field", [
    ("missing", "missing_fields", None), ("enum", "enum_value", "subject"),
    ("extra", "extra_fields", None), ("object", "object_type", None),
    ("reason", "reason_too_long", "reason"), ("json", "json_parse", None),
])
def test_v2_http_safe_contract_diagnostics_never_include_raw_values_or_extra_field_names(kind, expected_check, expected_field):
    wire = {key: value for key, value in verdict().items() if key != "verdict"}
    sentinel = "PRIVATE_PROVIDER_BODY_OR_VALUE_DO_NOT_REPORT"
    if kind == "missing":
        wire.pop("time")
    elif kind == "enum":
        wire["subject"] = sentinel
    elif kind == "extra":
        wire[sentinel] = sentinel
    elif kind == "object":
        wire = [sentinel]
    elif kind == "reason":
        wire["reason"] = sentinel * 30
    content = "{broken " + sentinel if kind == "json" else json.dumps(wire)

    def transport(request):
        system = json.loads(request.content)["messages"][0]["content"]
        assert "JSON Schema:" in system and '"additionalProperties": false' in system
        return httpx.Response(200, json={"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    judge = HTTPPairJudge(model="test", provider="deepseek", base_url="https://eval.invalid", api_key="transient",
                          transport=httpx.MockTransport(transport))
    try:
        report = evaluate_frozen(fixture(), judge=judge)
    finally:
        judge.close()
    details = report["supplier_failures"][0]["details"]
    assert details["contract_check"] == expected_check
    assert details.get("field_path") == expected_field
    assert sentinel not in json.dumps(report)
    assert report["known_usage"]["new_evaluation_requests"]["total_tokens"]["known_sum"] == 10
    assert report["semantic_development_proxy_confirmed"]["matches"] == 0


def test_cli_refuses_to_overwrite_an_existing_frozen_v1_report(tmp_path):
    input_path, output = tmp_path / "frozen-input.json", tmp_path / "frozen-v1-report.json"
    input_path.write_text(json.dumps(fixture(), ensure_ascii=False), encoding="utf-8")
    output.write_text('{"prompt_version":"same-paragraph-four-dimensions-v1"}', encoding="utf-8")
    before = output.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_semantics.py"
    result = subprocess.run([sys.executable, str(script), "--input", str(input_path), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "semantic_output_exists_choose_new_frozen_report_path" in result.stderr
    assert output.read_bytes() == before


def test_cli_plan_only_writes_separate_report_and_preserves_frozen_artifact(tmp_path):
    input_path, output = tmp_path / "frozen.json", tmp_path / "planned.json"
    input_path.write_text(json.dumps(fixture(), ensure_ascii=False), encoding="utf-8")
    before = input_path.read_bytes()
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_semantics.py"
    result = subprocess.run([sys.executable, str(script), "--input", str(input_path), "--output", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "planned_not_called"
    assert input_path.read_bytes() == before
    rejected = subprocess.run([sys.executable, str(script), "--input", str(input_path), "--output", str(input_path)],
                              capture_output=True, text=True)
    assert rejected.returncode == 1
    assert input_path.read_bytes() == before


def closed_cache_fixture(root, values=None, *, votes=1):
    from scripts import evaluate_semantics as semantic

    cache = root / "judge-cache"
    report = evaluate_frozen(fixture(), judge=Judge(values), cache_dir=cache, votes=votes)
    semantic._seal_cache(cache, report)
    return cache, report


def test_explicit_closed_import_reuses_exact_identity_and_preserves_all_source_bytes(tmp_path):
    first, _ = closed_cache_fixture(tmp_path / "first")
    second, _ = closed_cache_fixture(tmp_path / "second")
    original = {str(path): path.read_bytes() for root in (first, second) for path in root.rglob("*.json")}
    judge = Judge()
    result = evaluate_frozen(fixture(), judge=judge, import_cache_dirs=[first, second])
    pair = result["pairs"][0]
    assert judge.calls == [] and pair["cache_hit"] is True
    assert len(pair["cache_sources"]) == 2
    assert len(result["closed_cache_imports"]) == 2 and result["imported_cache_guards"]
    assert result["known_usage"]["new_evaluation_requests"]["request_records"] == 0
    assert result["known_usage"]["reused_unique_cached_decisions_original_usage"]["request_records"] == 1
    assert {path: Path(path).read_bytes() for path in original} == original
    changed = Judge(); changed.identity = {**Judge.identity, "model": "different-family-placeholder"}
    result = evaluate_frozen(fixture(), judge=changed, import_cache_dirs=[first, second])
    assert len(changed.calls) == 1 and result["pairs"][0]["cache_hit"] is False


def test_same_key_conflicting_closed_labels_remain_unknown_and_never_trigger_paid_tiebreak(tmp_path):
    first, _ = closed_cache_fixture(tmp_path / "first")
    second, _ = closed_cache_fixture(tmp_path / "second", [verdict("no_match", same_fact="no")])
    judge = Judge()
    result = evaluate_frozen(fixture(), judge=judge, import_cache_dirs=[second, first])
    assert judge.calls == [] and result["supplier_failures"] == []
    assert result["status"] == "completed_with_uncertainty"
    assert result["semantic_development_proxy_confirmed"]["matches"] == 0
    assert result["semantic_development_proxy_possible_including_unknowns"]["matches"] == 1
    conflict = result["cache_conflicts"][0]
    assert conflict["status"] == "cache_conflict" and conflict["verdict"] is None
    assert {r["verdict"]["verdict"] for r in conflict["cache_sources"]} == {"match", "no_match"}
    assert len(conflict["cache_sources"]) == 2
    assert result["known_usage"]["new_evaluation_requests"]["request_records"] == 0
    reverse = evaluate_frozen(fixture(), judge=Judge(), import_cache_dirs=[first, second])
    assert reverse["semantic_development_proxy_confirmed"] == result["semantic_development_proxy_confirmed"]


def test_identical_labels_with_different_reasons_keep_all_original_reasons_without_choosing_one(tmp_path):
    first, _ = closed_cache_fixture(tmp_path / "first")
    second, _ = closed_cache_fixture(tmp_path / "second", [verdict(reason="不同的短依据，不能静默替换旧标签。")])
    result = evaluate_frozen(fixture(), judge=Judge(), import_cache_dirs=[first, second])
    assert result["cache_conflicts"] == []
    pair = result["pairs"][0]
    assert pair["reason_variant_count"] == 2 and pair["verdict"]["verdict"] == "match"
    assert len({r["verdict_sha256"] for r in pair["cache_sources"]}) == 2
    assert pair["verdict"]["reason"] not in {r["verdict"]["reason"] for r in pair["cache_sources"]}


def test_same_final_verdict_with_different_dimension_labels_remains_a_cache_conflict(tmp_path):
    first, _ = closed_cache_fixture(tmp_path / "first")
    second, _ = closed_cache_fixture(tmp_path / "second", [verdict(value="compatible")])
    result = evaluate_frozen(fixture(), judge=Judge(), import_cache_dirs=[first, second])
    assert len(result["cache_conflicts"]) == 1 and result["cache_conflicts"][0]["label_conflict"] is False
    assert result["pairs"][0]["verdict"] is None


def test_closed_cache_import_requires_terminal_proof_and_rejects_unbound_entries(tmp_path):
    cache = tmp_path / "open-cache"
    evaluate_frozen(fixture(), judge=Judge(), cache_dir=cache)
    with pytest.raises(EvaluationInputError, match="closed_report_required"):
        evaluate_frozen(fixture(), judge=Judge(), import_cache_dirs=[cache])
    (tmp_path / "report.json").write_text(json.dumps({"format": "semantic-development-proxy-v1",
        "status": "planned_not_called", "input_unchanged": True}), encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="report_not_closed"):
        evaluate_frozen(fixture(), judge=Judge(), import_cache_dirs=[cache])
    (tmp_path / "report.json").unlink()
    from scripts import evaluate_semantics as semantic
    semantic._seal_cache(cache, evaluate_frozen(fixture(), judge=Judge(), cache_dir=cache))
    original = next(path for path in cache.glob("*.json") if len(path.stem) == 64)
    data = json.loads(original.read_text()); data["payload_sha256"] = "f" * 64
    original.write_text(json.dumps(data), encoding="utf-8")
    judge = Judge()
    with pytest.raises(EvaluationInputError, match="not_bound_to_closed_report"):
        evaluate_frozen(fixture(), judge=judge, import_cache_dirs=[cache])
    assert judge.calls == []


@pytest.mark.parametrize("case_field", ["cases", "component_cases"])
def test_closed_cache_can_be_bound_by_existing_comparison_report_without_rewriting_legacy_files(tmp_path, case_field):
    cache = tmp_path / "comparison/judge-cache/case"
    report = evaluate_frozen(fixture(), judge=Judge(), cache_dir=cache)
    summary = {"status": "completed", "measured": True, case_field: [{"report": report}]}
    (tmp_path / "comparison/comparison-report.json").write_text(json.dumps(summary), encoding="utf-8")
    result = evaluate_frozen(fixture(), judge_identity=Judge.identity, import_cache_dirs=[cache.parent])
    assert result["semantic_development_proxy_confirmed"]["matches"] == 1
    assert result["known_usage"]["new_evaluation_requests"]["request_records"] == 0


@pytest.mark.parametrize("labels,expected", [
    (["match", "match", "no_match"], "match"),
    (["no_match", "match", "no_match"], "no_match"),
    (["match", "no_match", "uncertain"], "uncertain"),
    (["uncertain", "uncertain", "match"], "uncertain"),
])
def test_explicit_three_votes_use_only_two_label_majority_and_keep_each_vote_usage(labels, expected):
    values = [verdict(label, same_fact={"match": "yes", "no_match": "no", "uncertain": "uncertain"}[label]) for label in labels]
    judge = Judge(values)
    report = evaluate_frozen(fixture(), judge=judge, votes=3)
    assert len(judge.calls) == 3 and judge.calls[0] == judge.calls[1] == judge.calls[2]
    assert report["format"] == "semantic-development-proxy-v2-majority3"
    pair = report["pairs"][0]
    assert pair["verdict"]["verdict"] == expected
    assert [vote["verdict"]["verdict"] for vote in pair["votes"]] == labels
    assert len({vote["cache_key"] for vote in pair["votes"]}) == 3
    assert [vote["sampling"]["sample_index"] for vote in pair["votes"]] == [1, 2, 3]
    assert all(vote["new_request_usage"]["total_tokens"] == 15 for vote in pair["votes"])
    assert report["known_usage"]["new_evaluation_requests"]["total_tokens"]["known_sum"] == 45
    assert pair["voting"]["majority_obtained"] is (len(set(labels)) != 3)
    assert report["sampling_policy"]["independent_model_families"] is False


def test_single_vote_and_three_vote_cache_identities_are_strictly_separate_and_complete_set_reuses(tmp_path):
    cache = tmp_path / "cache"
    single = evaluate_frozen(fixture(), judge=Judge(), cache_dir=cache)
    single_key = single["pairs"][0]["cache_key"]
    before = (cache / (single_key + ".json")).read_bytes()
    judge = Judge(); triple = evaluate_frozen(fixture(), judge=judge, cache_dir=cache, votes=3)
    assert len(judge.calls) == 3
    assert single_key not in {row["cache_key"] for row in triple["pairs"][0]["votes"]}
    assert (cache / (single_key + ".json")).read_bytes() == before
    again = Judge(); reused = evaluate_frozen(fixture(), judge=again, cache_dir=cache, votes=3)
    assert again.calls == [] and reused["pairs"][0]["cache_hit"] is True
    assert reused["known_usage"]["new_evaluation_requests"]["request_records"] == 0
    assert reused["known_usage"]["reused_unique_cached_decisions_original_usage"]["total_tokens"]["known_sum"] == 45
    only_three = tmp_path / "only-three"
    evaluate_frozen(fixture(), judge=Judge(), cache_dir=only_three, votes=3)
    fresh_single = Judge(); evaluate_frozen(fixture(), judge=fresh_single, cache_dir=only_three)
    assert len(fresh_single.calls) == 1


def test_three_vote_cache_only_import_never_constructs_judge_and_incomplete_set_is_visible(tmp_path):
    cache, previous = closed_cache_fixture(tmp_path / "closed", votes=3)
    reused = evaluate_frozen(fixture(), judge_identity=Judge.identity, import_cache_dirs=[cache], votes=3)
    assert reused["pairs"][0]["cache_hit"] is True
    assert reused["known_usage"]["new_evaluation_requests"]["request_records"] == 0
    assert reused["semantic_development_proxy_confirmed"]["matches"] == 1
    key = previous["pairs"][0]["votes"][2]["cache_key"]
    (cache / (key + ".json")).unlink()
    partial = evaluate_frozen(fixture(), judge_identity=Judge.identity, import_cache_dirs=[cache], votes=3)
    assert partial["pairs"][0]["voting"]["judged_samples"] == 2
    assert partial["status"] == "completed_with_uncertainty"
    assert partial["pairs"][0]["votes"][2]["status"] == "not_called"
    assert partial["known_usage"]["new_evaluation_requests"]["request_records"] == 0
    next_judge = Judge()
    finished = evaluate_frozen(fixture(), judge=next_judge, import_cache_dirs=[cache], cache_dir=tmp_path / "new", votes=3)
    assert len(next_judge.calls) == 1 and finished["pairs"][0]["voting"]["judged_samples"] == 3


def test_three_votes_preserve_failed_request_usage_and_do_not_invent_an_extra_vote():
    judge = Judge([verdict(), JudgeFailure("supplier_output_invalid", usage={"total_tokens": 8}), verdict()])
    report = evaluate_frozen(fixture(), judge=judge, votes=3)
    assert len(judge.calls) == 3 and report["status"] == "incomplete_supplier_failure"
    assert report["pairs"][0]["verdict"]["verdict"] == "match"
    assert report["pairs"][0]["voting"]["verdict_counts"] == {"match": 2, "no_match": 0, "uncertain": 0}
    assert report["supplier_failures"][0]["sample_index"] == 2
    assert report["known_usage"]["new_evaluation_requests"]["request_records"] == 3
    assert report["known_usage"]["new_evaluation_requests"]["total_tokens"]["known_sum"] == 38


def test_three_http_samples_keep_temperature_zero_and_one_usage_record_per_request():
    seen = []
    wire = {key: value for key, value in verdict().items() if key != "verdict"}

    def transport(request):
        body = json.loads(request.content); seen.append(body)
        assert body["temperature"] == 0
        return httpx.Response(200, json={"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(wire)}}]})

    judge = HTTPPairJudge(model="mock-family", provider="deepseek", base_url="https://eval.invalid", api_key="mock-only",
                          transport=httpx.MockTransport(transport))
    try:
        result = evaluate_frozen(fixture(), judge=judge, votes=3)
    finally:
        judge.close()
    assert len(seen) == 3 and seen[0] == seen[1] == seen[2]
    assert result["known_usage"]["new_evaluation_requests"]["request_records"] == 3
    assert result["known_usage"]["new_evaluation_requests"]["total_tokens"]["known_sum"] == 30


@pytest.mark.parametrize("changed", ["payload", "judge", "prompt"])
def test_three_slot_cache_binding_includes_complete_payload_judge_and_prompt(tmp_path, monkeypatch, changed):
    data, cache = fixture(), tmp_path / "cache"
    first = evaluate_frozen(data, judge=Judge(), cache_dir=cache, votes=3)
    judge = Judge()
    if changed == "payload":
        data["submission"]["evidence"][0]["argument"]["时间"] = "另一时点"
    elif changed == "judge":
        judge.identity = {**Judge.identity, "model": "another-family-configured-explicitly"}
    else:
        monkeypatch.setattr("scripts.evaluate_semantics.JUDGE_INSTRUCTIONS", "不同评判提示词。")
    second = evaluate_frozen(data, judge=judge, cache_dir=cache, votes=3)
    assert len(judge.calls) == 3
    assert not ({r["cache_key"] for r in first["pairs"][0]["votes"]} & {r["cache_key"] for r in second["pairs"][0]["votes"]})


def test_cache_writes_are_exclusive_even_if_a_conflicting_label_appears_during_request(tmp_path):
    from scripts import evaluate_semantics as semantic

    cache = tmp_path / "cache"

    class RacingJudge(Judge):
        def __call__(self, payload):
            response = super().__call__(payload)
            digest = semantic.fingerprint(payload)
            key = semantic._decision_key(self.identity, digest)
            old = verdict("no_match", same_fact="no")
            entry = {"cache_key": key, "payload_sha256": digest, "judge": self.identity,
                "prompt_version": semantic.PROMPT_VERSION, "prompt_sha256": semantic.prompt_sha256(),
                "verdict": old, "verdict_sha256": semantic.fingerprint(old), "usage_at_creation": {"total_tokens": 123}}
            cache.mkdir(); (cache / (key + ".json")).write_text(json.dumps(entry), encoding="utf-8")
            return response

    judge = RacingJudge()
    result = evaluate_frozen(fixture(), judge=judge, cache_dir=cache)
    stored = json.loads(next(cache.glob("*.json")).read_text())
    assert stored["verdict"]["verdict"] == "no_match"
    assert result["pairs"][0]["status"] == "cache_conflict"
    assert len(result["cache_conflicts"][0]["cache_sources"]) == 2
    assert result["known_usage"]["new_evaluation_requests"]["total_tokens"]["known_sum"] == 15


def test_closed_cache_import_is_read_only_and_requires_nonsecret_explicit_identity_without_judge(tmp_path):
    cache, _ = closed_cache_fixture(tmp_path / "closed")
    with pytest.raises(EvaluationInputError, match="explicit_judge_identity"):
        evaluate_frozen(fixture(), import_cache_dirs=[cache])
    with pytest.raises(EvaluationInputError, match="overlaps_imported"):
        evaluate_frozen(fixture(), judge=Judge(), cache_dir=cache / "writes", import_cache_dirs=[cache])


def test_cli_votes3_cache_only_uses_explicit_identity_and_no_runtime_model_settings(tmp_path):
    cache, _ = closed_cache_fixture(tmp_path / "closed", votes=3)
    source, identity, output = tmp_path / "input.json", tmp_path / "judge.json", tmp_path / "result.json"
    source.write_text(json.dumps(fixture()), encoding="utf-8")
    identity.write_text(json.dumps(Judge.identity), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_semantics.py"
    result = subprocess.run([sys.executable, "-B", str(script), "--input", str(source), "--output", str(output),
        "--votes3", "--judge-identity", str(identity), "--import-closed-cache", str(cache)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["format"] == "semantic-development-proxy-v2-majority3"
    assert report["known_usage"]["new_evaluation_requests"]["request_records"] == 0
    assert report["pairs"][0]["cache_hit"] is True
