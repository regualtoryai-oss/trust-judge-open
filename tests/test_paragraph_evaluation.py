"""Synthetic, offline tests; no sample answers are put into app/runtime inputs."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import evaluate_paragraphs as p


def fact(identity="g1", para="d#p1", *, quote="甲", content="甲", doc="d"):
    return {"fact_id": identity, "doc_id": doc, "para_id": para, "quote": quote,
            "fact_des": content, "argument": {"主体": "甲", "时间": None, "数值口径": None}, "entities": ["甲"]}


def fixture(gold=None, submitted=None, *, support=None, conflict=None):
    gold = deepcopy(gold if gold is not None else [fact()])
    submitted = deepcopy(submitted if submitted is not None else [fact("s1")])
    keys = sorted({p.location(row) for row in gold + submitted if p.location(row) is not None})
    source = [{"doc_id": key[0], "para_id": key[1], "text": "甲乙丙"} for key in keys]
    q = {"question_id": "q1", "question": "甲的说法是否成立？", "supporting_facts": support if support is not None else [{"fact_id": row["fact_id"]} for row in gold],
         "conflicting_facts": conflict or []}
    sq = {"question_id": "q1", "question": q["question"], "supporting_facts": [{"evidence_index": i} for i in range(len(submitted))], "conflicting_facts": []}
    return {"topics": [{"topic_id": "t", "evidence": gold, "conclusions": [q], "source_paragraphs": source}],
            "submissions": [{"topic_id": "t", "evidence": submitted, "conclusions": [sq], "status": "succeeded",
                             "evaluated_question_ids": ["q1"], "source_scope": [{"doc_id": k[0], "para_id": k[1]} for k in keys]}],
            "semantic_artifacts": []}


def verdict(label):
    return {"verdict": label, "same_fact": {"match": "yes", "no_match": "no", "uncertain": "uncertain"}[label],
            **{name: "compatible" for name in p.DIMENSIONS}, "submitted_quote_supports_fact": "yes",
            "reference_quote_supports_fact": "yes", "reason": "仅用于合成fixture的已闭合标签。"}


def closed_entry(bundle, labels):
    topic, sub = bundle["topics"][0], bundle["submissions"][0]
    q = topic["conclusions"][0]
    artifact = {"format": "semantic-eval-input-v1", "scope": {"topic_id": "internal", "question_id": "q1", "question": q["question"]},
                "source_paragraphs": deepcopy(topic["source_paragraphs"]), "submission": {"evidence": deepcopy(sub["evidence"])},
                "reference": {"facts": deepcopy(topic["evidence"]), "question_references": [{"side": "supporting_facts", "reference": {"fact_id": row["fact_id"]}} for row in topic["evidence"]]}}
    judge = {"provider": "synthetic", "model": "fixture", "protocol": "no_network"}
    source = {p.location(row): row["text"] for row in artifact["source_paragraphs"]}
    pairs = []
    for (li, ri), label in labels.items():
        left, right = artifact["submission"]["evidence"][li], artifact["reference"]["facts"][ri]
        key = p.location(left)
        payload = {"question": q["question"], "source_paragraph": {"doc_id": key[0], "para_id": key[1], "text": source[key]},
                   "submitted": p.project_fact(left), "reference": p.project_fact(right)}
        digest = p.fingerprint(payload)
        pairs.append({"submitted_index": li, "reference_index": ri, "status": "judged", "verdict": verdict(label),
                      "submitted_record_sha256": p.fingerprint(left), "reference_record_sha256": p.fingerprint(right),
                      "payload_sha256": digest, "cache_key": p.fingerprint({"prompt_version": p.SEMANTIC_PROMPT,
                          "prompt_sha256": p.SEMANTIC_PROMPT_SHA, "judge": judge, "payload_sha256": digest})})
    report = {"format": "semantic-development-proxy-v1", "status": "completed", "input_unchanged": True,
              "frozen_input_sha256": p.fingerprint(artifact), "prompt_version": p.SEMANTIC_PROMPT,
              "prompt_sha256": p.SEMANTIC_PROMPT_SHA, "judge": judge, "pairs": pairs}
    return {"topic_id": "t", "artifact": artifact, "report": report}


def topic_raw(report, scope="full_topic_reference"):
    return report["topic_reports"][0][scope]["raw_records"]


def test_id_overlap_is_only_a_and_missing_semantics_is_unknown_not_zero():
    bundle = fixture([fact(content="完全不同的事实")], [fact("s1", quote="乙", content="另一断言")])
    result = p.evaluate_bundle(bundle); row = topic_raw(result)
    assert row["A_id_only"]["metrics"]["f1"] == 1
    b = row["B_existing_semantics"]
    assert b["point_metrics"] is None and b["status"] == "partial_unknown"
    assert b["confirmed_lower_bound"]["matches"] == 0 and b["possible_upper_bound"]["matches"] == 1
    assert result["new_model_calls"] == 0


def test_markdown_reports_actual_scope_for_new_bundle_without_legacy_pilot_claim():
    bundle = fixture()
    unexecuted = deepcopy(bundle["topics"][0]["conclusions"][0])
    unexecuted["question_id"] = "q2"
    bundle["topics"][0]["conclusions"].append(unexecuted)
    bundle["topics"][0]["source_paragraphs"].append({"doc_id": "d", "para_id": "d#p2", "text": "乙"})
    markdown = p.comparison_markdown({"new-run": p.evaluate_bundle(bundle)})
    assert "| new-run | 1 | 1/2 | 1/2 |" in markdown
    assert "已声明来源范围不等于模型实际阅读量" in markdown
    assert "只执行6道题" not in markdown and "读取144段" not in markdown


def test_raw_duplicates_get_maximum_one_to_one_not_cartesian_credit_and_unique_is_separate():
    result = p.compare_rows("t", [fact("s1"), fact("s2"), fact("s3")], [fact("g1"), fact("g2")])
    raw = result["raw_records"]["A_id_only"]["metrics"]
    assert raw["matches"] == 2 and raw["submitted_records"] == 3 and raw["reference_records"] == 2
    assert result["unique_paragraphs"]["A_id_only"]["metrics"]["f1"] == 1
    assert result["denominator_audit"]["reference"]["duplicate_paragraph_groups"][0]["row_indices"] == [0, 1]


def test_maximum_matching_is_not_greedy():
    assert len(p.maximum_matching(2, 2, [(0, 0), (0, 1), (1, 0)])) == 2


def test_closed_four_dimension_match_is_required_and_known_rejection_cannot_be_rescued_by_a():
    bundle = fixture(); bundle["semantic_artifacts"] = [closed_entry(bundle, {(0, 0): "no_match"})]
    row = topic_raw(p.evaluate_bundle(bundle))
    assert row["A_id_only"]["metrics"]["matches"] == 1
    b = row["B_existing_semantics"]
    assert b["point_metrics"]["matches"] == 0 and b["possible_upper_bound"]["matches"] == 0
    bundle["semantic_artifacts"] = [closed_entry(bundle, {(0, 0): "match"})]
    assert topic_raw(p.evaluate_bundle(bundle))["B_existing_semantics"]["point_metrics"]["matches"] == 1


def test_unknown_label_and_disagreeing_closed_labels_stay_unknown():
    bundle = fixture()
    bundle["semantic_artifacts"] = [closed_entry(bundle, {(0, 0): "uncertain"})]
    assert topic_raw(p.evaluate_bundle(bundle))["B_existing_semantics"]["point_metrics"] is None
    bundle["semantic_artifacts"] = [closed_entry(bundle, {(0, 0): "match"}), closed_entry(bundle, {(0, 0): "no_match"})]
    b = topic_raw(p.evaluate_bundle(bundle))["B_existing_semantics"]
    assert b["point_metrics"] is None and b["confirmed_lower_bound"]["matches"] == 0


@pytest.mark.parametrize("attack", ["artifact", "pair", "question", "source", "missing_declared_source", "rubric", "contradictory"])
def test_closed_semantic_bindings_reject_corruption(attack):
    bundle = fixture(); entry = closed_entry(bundle, {(0, 0): "match"}); bundle["semantic_artifacts"] = [entry]
    if attack == "artifact": entry["artifact"]["submission"]["evidence"][0]["fact_des"] = "改写"
    elif attack == "pair": entry["report"]["pairs"][0]["payload_sha256"] = "0" * 64
    elif attack == "question":
        entry["artifact"]["scope"]["question"] = "另一个问题"
        entry["report"]["frozen_input_sha256"] = p.fingerprint(entry["artifact"])
    elif attack == "source":
        entry["artifact"]["source_paragraphs"][0]["text"] = "假原文"
        entry["report"]["frozen_input_sha256"] = p.fingerprint(entry["artifact"])
    elif attack == "missing_declared_source":
        entry["artifact"]["source_paragraphs"] = []
        entry["report"]["frozen_input_sha256"] = p.fingerprint(entry["artifact"])
    elif attack == "rubric": entry["report"]["prompt_sha256"] = "f" * 64
    else: entry["report"]["pairs"][0]["verdict"]["time"] = "incompatible"
    with pytest.raises(p.ParagraphEvaluationError): p.evaluate_bundle(bundle)


def test_bare_semantic_label_maps_cannot_create_b_matches():
    left, right = fact("s"), fact("g")
    bare = {("t", p.fingerprint(p.project_fact(left)), p.fingerprint(p.project_fact(right))): [{"question_id": "q1", "verdict": "match"}]}
    with pytest.raises(p.ParagraphEvaluationError, match="bare_semantic"):
        p.compare_rows("t", [left], [right], semantic_index=bare)


def test_complete_reference_and_selected_scope_denominators_are_not_conflated():
    bundle = fixture([fact("g1"), fact("g2", "d#p2")], [fact("s1")])
    bundle["submissions"][0]["source_scope"] = [{"doc_id": "d", "para_id": "d#p1"}]
    q2 = deepcopy(bundle["topics"][0]["conclusions"][0]); q2["question_id"] = "q2"
    bundle["topics"][0]["conclusions"].append(q2)
    result = p.evaluate_bundle(bundle)
    assert topic_raw(result)["A_id_only"]["metrics"]["reference_records"] == 2
    assert topic_raw(result, "within_source_scope_reference")["A_id_only"]["metrics"]["reference_records"] == 1
    assert result["coverage"]["full_reference_questions"] == 2 and result["coverage"]["executed_questions"] == 1
    assert result["question_aggregates"]["all_reference_questions"]["question_denominator"] == 2
    assert result["question_reports"][1]["supporting_facts"]["full_question_reference"]["raw_records"]["A_id_only"]["metrics"]["reference_records"] == 2


def test_support_precision_reference_scale_weighting_differs_from_micro():
    gold = [fact("g1", "d#p1"), *[fact(f"g{i}", f"d#p{i}") for i in range(2, 5)]]
    submitted = [fact(f"s{i}", f"d#p{i}") for i in [1, *range(5, 14), 2]]
    bundle = fixture(gold, submitted)
    q1 = bundle["topics"][0]["conclusions"][0]; q1["supporting_facts"] = [{"fact_id": "g1"}]
    q2 = {"question_id": "q2", "question": "第二问", "supporting_facts": [{"fact_id": f"g{i}"} for i in range(2, 5)], "conflicting_facts": []}
    bundle["topics"][0]["conclusions"].append(q2)
    sub = bundle["submissions"][0]; sub["evaluated_question_ids"].append("q2")
    sub["conclusions"][0]["supporting_facts"] = [{"evidence_index": i} for i in range(10)]
    sub["conclusions"].append({"question_id": "q2", "question": "第二问", "supporting_facts": [{"evidence_index": 10}], "conflicting_facts": []})
    result = p.evaluate_bundle(bundle)
    agg = result["question_aggregates"]["executed_questions_only"]["supporting_facts"]["full_question_reference"]["raw_records"]["A_id_only"]
    assert agg["micro"]["precision"] == pytest.approx(2 / 11)
    assert agg["reference_scale_weighted"]["precision"] == pytest.approx(.775)
    assert agg["reference_scale_weighted"]["recall"] == agg["micro"]["recall"] == .5


def test_topic_namespace_prevents_cross_topic_id_rescue():
    bundle = fixture([fact("g1")], [fact("s1", "d#p2")])
    other = deepcopy(bundle["topics"][0]); other["topic_id"] = "other"; other["evidence"] = [fact("g2", "d#p2")]
    bundle["topics"].append(other)
    result = p.evaluate_bundle(bundle)
    a = result["evidence_aggregates"]["full_topic_reference"]["raw_records"]["A_id_only"]["micro"]
    assert a["matches"] == 0 and a["reference_records"] == 2


def test_missing_reference_id_and_duplicate_relation_occurrences_are_not_silently_dropped():
    bundle = fixture(support=[{"fact_id": "g1"}, {"fact_id": "g1"}, {"fact_id": "missing"}])
    result = p.evaluate_bundle(bundle)
    q = result["question_reports"][0]["supporting_facts"]["full_question_reference"]
    assert q["raw_records"]["A_id_only"]["metrics"]["reference_records"] == 3
    assert q["unique_paragraphs"]["A_id_only"]["metrics"]["reference_records"] == 2
    assert any(x["code"] == "unresolved_relation_counted_in_raw_denominator" for x in result["issues"])


def test_failed_case_preserves_denominators_and_empty_precision_is_null():
    bundle = fixture(); sub = bundle["submissions"][0]
    sub.update(status="failed", evidence=[], conclusions=[])
    report = p.evaluate_bundle(bundle); a = topic_raw(report)["A_id_only"]["metrics"]
    assert a["precision"] is None and a["recall"] == 0 and a["reference_records"] == 1
    assert report["coverage"]["executed_questions"] == 1
    sub["evidence"] = [fact("bad")]
    with pytest.raises(p.ParagraphEvaluationError, match="failed_submission"):
        p.evaluate_bundle(bundle)


def test_relation_question_cannot_reuse_other_questions_closed_labels():
    bundle = fixture(); bundle["semantic_artifacts"] = [closed_entry(bundle, {(0, 0): "match"})]
    q2 = deepcopy(bundle["topics"][0]["conclusions"][0]); q2.update(question_id="q2", question="另一个问题")
    bundle["topics"][0]["conclusions"].append(q2)
    sub = bundle["submissions"][0]; sub["evaluated_question_ids"].append("q2")
    sq = deepcopy(sub["conclusions"][0]); sq.update(question_id="q2", question="另一个问题"); sub["conclusions"].append(sq)
    result = p.evaluate_bundle(bundle)
    assert result["question_reports"][0]["supporting_facts"]["full_question_reference"]["raw_records"]["B_existing_semantics"]["point_metrics"]["matches"] == 1
    assert result["question_reports"][1]["supporting_facts"]["full_question_reference"]["raw_records"]["B_existing_semantics"]["point_metrics"] is None


def test_inputs_remain_byte_equivalent_and_module_has_no_runtime_network_imports(tmp_path):
    bundle = fixture(); before = json.dumps(bundle, sort_keys=True)
    p.evaluate_bundle(bundle)
    assert json.dumps(bundle, sort_keys=True) == before
    root = Path(__file__).resolve().parents[1]
    code = "import sys; from scripts import evaluate_paragraphs; assert not any(k=='app' or k.startswith('app.') for k in sys.modules); assert not any(k in sys.modules for k in ('httpx','requests','openai','socket','sqlite3'))"
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_writes_new_directory_only_and_reports_zero_calls(tmp_path):
    source = tmp_path / "bundle.json"; source.write_text(json.dumps(fixture()), encoding="utf-8")
    output = tmp_path / "report"
    p.main(["--bundle", str(source), "--output", str(output)])
    assert json.loads((output / "report.json").read_text())["new_model_calls"] == 0
    with pytest.raises(p.ParagraphEvaluationError, match="new_output"):
        p.main(["--bundle", str(source), "--output", str(output)])


def test_literal_gold_anomaly_stays_in_denominator_and_unlocated_not_silently_discarded():
    gold = fact(quote="源文不存在")
    bundle = fixture([gold])
    report = p.evaluate_bundle(bundle)
    assert topic_raw(report)["A_id_only"]["metrics"]["reference_records"] == 1
    audit = report["topic_reports"][0]["literal_quote_audit"]["reference"]
    assert audit["literal_in_declared_paragraph"] == 0 and audit["rows_removed_or_repaired"] == 0
    unlocated = fact(); unlocated.pop("para_id")
    row = p.compare_rows("t", [], [unlocated])["raw_records"]["A_id_only"]["metrics"]
    assert row["reference_records"] == 1 and row["recall"] == 0


def build_closed_disk_fixture(tmp_path):
    """Only synthetic files, including a real-format bound standard projection."""
    sample_root = tmp_path / "samples"; sample = sample_root / "t"; sample.mkdir(parents=True)
    base = tmp_path / "comparison"; base.mkdir()
    run = tmp_path / "run"; folder = run / "case1"; folder.mkdir(parents=True)
    plans = tmp_path / "plans"; plans.mkdir()

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return p.file_descriptor(path)

    gold = fact(); args = {name: "" for name in ("时间", "主体", "地点", "内容", "影响", "数值口径")}; args["主体"] = "甲"
    standard = [{"doc_id": "d", "para_id": "d#p1", "quote": "甲", "fact_des": "甲", "argument": args, "entities": ["甲"]}]
    b = fixture([gold], standard)
    files = {"evidence_samples.json": [gold], "conclusions.json": b["topics"][0]["conclusions"],
             "raw_documents.json": [{"doc_id": "d", "content": "甲乙丙", "paragraphs": [{"para_id": "d#p1", "char_start": 0, "char_end": 3}]}]}
    sourcehashes = {name: write(sample / name, value)["sha256"] for name, value in files.items()}
    entry = closed_entry(b, {(0, 0): "match"})
    entry["artifact"]["reference"]["source_files_sha256"] = sourcehashes
    entry["report"]["frozen_input_sha256"] = p.fingerprint(entry["artifact"])
    payload = {"question": {"question": "甲的说法是否成立？"}, "paragraphs": [{"id": "p-internal", "raw_doc_id": "d", "raw_para_id": "d#p1", "text": "甲乙丙"}]}
    inp = write(plans / "input.json", payload)
    plan = {"cases": [{"case_id": "case1", "title": "t", "question_id": "q1", "input_file": "input.json", "input_sha256": inp["sha256"]}]}
    planpath = plans / "plan.json"; pd = write(planpath, plan)
    evidence = [{"id": "e1", "paragraph_id": "p-internal", "raw_doc_id": "d", "raw_para_id": "d#p1", "start_cp": 0, "end_cp": 1,
                 "quote": "甲", "fact_des": "甲", "argument": args, "entities": ["甲"]}]
    result = {"evidence": evidence, "final": {"supporting_evidence_ids": ["e1"], "conflicting_evidence_ids": []}}
    rd = write(folder / "result.json", result); sd = write(folder / "standard_evidence.json", standard)
    td = write(folder / "terminal.json", {"status": "succeeded", "input_sha256": inp["sha256"], "artifacts": {"result": rd, "standard_evidence": sd}})
    write(run / "final-receipt.json", {"plan_sha256": pd["sha256"], "app_sources_unchanged": True, "no_gold_read": True,
                                      "cases": [{"case_id": "case1", "status": "succeeded", "terminal": td}]})
    row = {"case_id": "case1", "comparison_id": "v17-case1", "run_label": "v17", "artifact": entry["artifact"]}
    write(base / "comparison-plan.json", {"plan_sha256": pd["sha256"], "runs": [{"label": "v17", "path": str(run)}], "cases": [row]})
    write(base / "comparison-report.json", {"status": "completed", "measured": True, "plan_sha256": pd["sha256"],
        "cases": [{"case_id": "case1", "run_label": "v17", "report": entry["report"]}]})
    write(base / "frozen-evaluation-inputs/v17-case1.json", entry["artifact"])
    return base, planpath, sample_root, run


def test_closed_pilot_adapter_binds_original_schema_final_relations_and_hashes(tmp_path):
    base, plan, samples, run = build_closed_disk_fixture(tmp_path)
    bundle, guards = p.make_closed_pilot_bundle(base, plan, samples, "v17")
    result = p.evaluate_bundle(bundle)
    q = result["question_reports"][0]["supporting_facts"]["full_question_reference"]["raw_records"]
    assert q["B_existing_semantics"]["confirmed_lower_bound"]["matches"] == 1
    assert len(guards) >= 10 and all(p.file_descriptor(g["file"])["sha256"] == g["sha256"] for g in guards)
    # An unchanged scored evidence file cannot authorize changed final links/result.
    changed = run / "case1/result.json"; data = p.load(changed); data["final"]["supporting_evidence_ids"] = []
    changed.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(p.ParagraphEvaluationError, match="sha256"):
        p.make_closed_pilot_bundle(base, plan, samples, "v17")


def test_closed_pilot_adapter_requires_terminal_marker_before_any_case_read(tmp_path):
    base, plan, samples, run = build_closed_disk_fixture(tmp_path)
    (run / "final-receipt.json").unlink()
    with pytest.raises(p.ParagraphEvaluationError, match="final_receipt"):
        p.make_closed_pilot_bundle(base, plan, samples, "v17")


def test_closed_pilot_adapter_detects_gold_source_hash_change(tmp_path):
    base, plan, samples, _ = build_closed_disk_fixture(tmp_path)
    evidence = samples / "t/evidence_samples.json"; rows = p.load(evidence); rows[0]["fact_des"] = "已改"
    evidence.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(p.ParagraphEvaluationError, match="sha256"):
        p.make_closed_pilot_bundle(base, plan, samples, "v17")
