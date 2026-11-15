"""Synthetic sample benchmark evaluation; explicit fake judge only, no HTTP."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time

import httpx
import pytest

from scripts import score_sample_benchmark as scoring
from scripts import evaluate_semantics as semantic
from app.domain.snapshot import build_snapshot


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def fact(identity="f1", **changes):
    return {"id": identity, "fact_id": identity, "doc_id": "d1", "para_id": "p1", "quote": "设备已复位。", "fact_des": "设备已复位。",
            "argument": {"时间": None, "主体": "设备", "数值口径": None}, "entities": ["设备"], **changes}


def canonical_fact(identity="f1", **changes):
    """A verified runtime record, distinct from the intentionally minimal gold."""
    row = fact(identity, **changes)
    return {**row, "paragraph_id": "p1", "raw_doc_id": row["doc_id"], "raw_para_id": row["para_id"],
            "start_cp": 0, "end_cp": len(row["quote"]), "source_group_id": "source-d1",
            "source_group_basis": "same_document_only_not_complete_origin_detection",
            "grade_score": None, "source_level": "unknown"}


def question_snapshot(topic_id, internal_id, raw_id, question, revision, *, evidence=None, **final_changes):
    """Use the publication builder so fixtures exercise the actual export contract."""
    evidence = [canonical_fact()] if evidence is None else deepcopy(evidence)
    final = {"decision": "asserted", "conclusion": "设备已复位。", "judgments": [], "conflicts": [],
             "supporting_evidence_ids": [evidence[0]["id"]], "conflicting_evidence_ids": [], **final_changes}
    frozen_id = "frozen-" + internal_id
    analyses = [{"role": role, "evidence_snapshot_id": frozen_id,
                 **{key: deepcopy(value) for key, value in final.items() if key != "conflicts"}}
                for role in ("authority_analyst", "cross_validation_analyst")]
    result = {"origin_run_id": "raw-only-" + internal_id, "usage": [{"total_tokens": 7}],
              "evidence": evidence, "final": final, "analyses": analyses, "evidence_snapshot_id": frozen_id}
    return build_snapshot({"id": topic_id}, [{"id": internal_id, "topic_id": topic_id,
        "raw_question_id": raw_id, "question": question}], {internal_id: result},
        revision_id=revision, graph_revision=1, run_id="raw-only-" + internal_id)


class FakeJudge:
    identity = {"provider": "synthetic-test-only", "model": "independent-test-judge"}

    def __init__(self, *, uncertain=False):
        self.calls, self.closed, self.uncertain = [], False, uncertain

    def __call__(self, payload):
        self.calls.append(deepcopy(payload))
        return {"verdict": {"verdict": "uncertain" if self.uncertain else "match", "same_fact": "yes", "subject": "compatible",
            "time": "uncertain" if self.uncertain else "not_applicable", "value": "not_applicable", "statistical_scope": "not_applicable",
            "submitted_quote_supports_fact": "yes", "reference_quote_supports_fact": "yes", "reason": "仅依据固定原段，未知时点不补全。"},
            "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}}

    def close(self):
        self.closed = True


class FakeConclusionJudge:
    identity = {"provider": "synthetic-test-only", "model": "conclusion-judge"}

    def __init__(self, label="full"):
        self.label, self.calls, self.closed = label, [], False

    def __call__(self, payload):
        self.calls.append(deepcopy(payload))
        return {"verdict": {"label": self.label, "reason": "方向、事实和口径按独立合成参考核对。", "source_refs": [{"doc_id": "d1", "para_id": "p1"}]},
                "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24}}

    def close(self): self.closed = True


def fixture(tmp_path, *, topics=6, questions_per_topic=3, succeeded=1, gold=None, refs=None):
    sample = tmp_path / "original-samples"
    sample.mkdir()
    database = tmp_path / "runtime.sqlite3"
    plan = {"format": scoring.PLAN_FORMAT, "source_bundle_sha256": "a" * 64,
            "total_topics": topics, "total_questions": topics * questions_per_topic, "topics": []}
    rows = []
    with sqlite3.connect(database) as connection:
        connection.executescript("CREATE TABLE documents(id TEXT,content TEXT); CREATE TABLE paragraphs(id TEXT,text TEXT); "
            "CREATE TABLE graph_revisions(id TEXT,topic_id TEXT,state TEXT); CREATE TABLE questions(id TEXT,topic_id TEXT,raw_question_id TEXT,question TEXT); "
            "CREATE TABLE revision_artifacts(revision_id TEXT,payload_json TEXT);")
        connection.execute("INSERT INTO documents VALUES ('d1','设备已复位。')")
        connection.execute("INSERT INTO paragraphs VALUES ('p1','设备已复位。')")
        for t in range(topics):
            title, key, tid = f"专题{t}", f"topic-{t}", f"internal-topic-{t}"
            directory = sample / title
            directory.mkdir()
            questions, conclusions = [], []
            for q in range(questions_per_topic):
                raw_qid, iqid, rev = f"q_{q+1:03d}", f"internal-{t}-{q}", f"fixed-{t}-{q}"
                question = f"专题{t}设备第{q+1}项是否复位？"
                questions.append({"raw_question_id": raw_qid, "question_sha256": hashlib.sha256(question.encode()).hexdigest()})
                conclusions.append({"question_id": raw_qid, "question": question, "conclusion": "ORIGINAL_GOLD_EVAL_ONLY",
                    "supporting_facts": deepcopy(refs) if refs is not None else [{"fact_id": "f1"}], "conflicting_facts": []})
                success = len(rows) < succeeded
                connection.execute("INSERT INTO questions VALUES (?,?,?,?)", (iqid, tid, raw_qid, question))
                if success:
                    snapshot = question_snapshot(tid, iqid, raw_qid, question, rev)
                    connection.execute("INSERT INTO graph_revisions VALUES (?,?,?)", (rev, tid, "ready"))
                    connection.execute("INSERT INTO revision_artifacts VALUES (?,?)", (rev, json.dumps(snapshot)))
                rows.append({"topic_key": key, "title": title, "database": str(database.resolve()), "topic_id": tid, "raw_question_id": raw_qid,
                    "internal_question_id": iqid, "run_id": f"run-{t}-{q}", "status": "succeeded" if success else "failed",
                    "ready_revision_id": rev if success else None, "error": None if success else {"code": "validation_failed", "message": "PRIVATE_PROVIDER_BODY"}})
            dump(directory / "raw_documents.json", [{"doc_id": "d1", "title": "独立合成材料", "content": "设备已复位。", "paragraphs": [{"para_id": "p1", "char_start": 0, "char_end": 6}]}])
            dump(directory / "evidence_samples.json", deepcopy(gold) if gold is not None else [fact()])
            dump(directory / "conclusions.json", conclusions)
            plan["topics"].append({"topic_key": key, "title": title, "questions": questions})
    plan_path, status_path = tmp_path / "benchmark-plan.json", tmp_path / "benchmark-status.json"
    dump(plan_path, plan)
    status = {"format": scoring.STATUS_FORMAT, "plan_sha256": scoring.bytes_sha(plan_path), "source_bundle_sha256": "a" * 64,
              "config_sha256": "b" * 64, "questions": rows}
    dump(status_path, status)
    return plan_path, status_path, sample, database


def test_all_18_including_17_failures_remain_in_reference_macro_and_micro_denominators(tmp_path):
    plan, status, sample, _ = fixture(tmp_path)
    prepared = scoring.prepare_benchmark(plan, status, sample)
    judge = FakeJudge()
    report = scoring.score_prepared(prepared, judge=judge, execute=True)
    assert len(judge.calls) == 2  # one question pair + independent all-topic export pair
    assert len(report["questions"]) == 18
    agg = report["aggregate"]
    assert agg["completion"]["completion_rate"] == pytest.approx(1/18)
    assert agg["confirmed"]["micro"]["recall"] == pytest.approx(1/18)
    assert agg["confirmed"]["micro"]["precision"] == 1
    macro = agg["confirmed"]["macro_all_questions"]
    assert macro["all_question_denominator"] == 18
    assert macro["precision"] is None and macro["precision_undefined_questions"] == 17
    assert macro["precision_zero_filled_operational"] == pytest.approx(1/18)
    assert macro["recall"] == pytest.approx(1/18)
    assert agg["confirmed"]["worst_question_f1_zero_filled"] == 0
    failed = report["questions"][1]
    assert failed["confirmed"]["precision"] is None and failed["confirmed"]["recall"] == 0
    assert "PRIVATE_PROVIDER_BODY" not in json.dumps(report)
    assert report["topic_level_formal_export_comparison"]["computed"] is True
    assert report["topic_level_formal_export_comparison"]["confirmed"]["recall"] == pytest.approx(1/6)
    assert report["topic_level_formal_export_comparison"]["original_evidence_rows_all_topics"] == 6


def test_same_raw_question_id_in_different_topics_is_not_collapsed(tmp_path):
    plan, status, sample, _ = fixture(tmp_path, topics=2, questions_per_topic=1, succeeded=2)
    prepared = scoring.prepare_benchmark(plan, status, sample, expected_questions=2)
    judge = FakeJudge()
    report = scoring.score_prepared(prepared, judge=judge, execute=True)
    assert len(report["questions"]) == 2 and len(judge.calls) == 4
    assert len({q["case_id"] for q in report["questions"]}) == 2
    assert len(report["by_topic"]) == 2
    assert all(q["ready_revision_id"].startswith("fixed-") for q in report["questions"])


def test_default_plan_never_constructs_settings_http_or_measured_scores_and_keeps_unknown_fields(tmp_path, monkeypatch):
    plan, status, sample, database = fixture(tmp_path)
    before = database.read_bytes()
    monkeypatch.setattr(scoring, "ANALYSIS_ROOT", tmp_path / "_analysis")
    def forbidden():
        pytest.fail("plan must not construct HTTP/Settings")
    monkeypatch.setattr(semantic.HTTPPairJudge, "from_runtime_settings", forbidden)
    path, report = scoring.write_benchmark(plan, status, sample, tmp_path / "_analysis" / "new-plan", judge_factory=forbidden)
    assert path.exists() and report["status"] == "planned_not_called"
    assert report["aggregate"]["confirmed"] is None
    assert report["new_evaluation_usage"]["request_records"] == 0
    assert report["conclusion_rubric"]["status"] == "frozen_before_execution"
    assert all(x["status"] == "not_judged" for x in report["conclusions_comparison"])
    frozen_path = next(p for p in (path.parent / "frozen-question-inputs").iterdir() if not p.name.endswith(".conclusion.json"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["submission"]["evidence"][0]["argument"]["时间"] is None
    assert database.read_bytes() == before and b"ORIGINAL_GOLD_EVAL_ONLY" not in before


def test_duplicate_missing_and_malformed_gold_remain_counted_and_literal_issue_is_not_repaired(tmp_path):
    gold = [fact(), fact(), fact("bad-quote", quote="原文不存在。"), {"quote": "unscopable"}]
    refs = [{"fact_id": "f1"}, {"fact_id": "missing"}, {}, {"fact_id": "bad-quote"}]
    plan, status, sample, database = fixture(tmp_path, topics=1, questions_per_topic=1, succeeded=0, gold=gold, refs=refs)
    original = {p: p.read_bytes() for p in sample.rglob("*.json")}
    prepared = scoring.prepare_benchmark(plan, status, sample, expected_questions=1)
    report = scoring.score_prepared(prepared, judge=FakeJudge(), execute=True)
    row = report["questions"][0]
    assert row["gold_audit"]["denominator"] == 5  # 2 duplicate f1 + bad quote + missing + malformed
    assert row["confirmed"]["referenced_gold_evidence_records"] == 5
    codes = {x["code"] for x in row["gold_audit"]["issues"]}
    assert {"duplicate_original_gold_fact_id", "missing_referenced_gold_fact", "invalid_referenced_fact_id", "unscopable_gold_fact_id"} <= codes
    assert len(row["gold_audit"]["literal_quote_issues"]) == 3
    assert {p: p.read_bytes() for p in original} == original
    assert b"unscopable" not in database.read_bytes()


def test_question_specific_reference_union_not_entire_topic_gold(tmp_path):
    plan, status, sample, _ = fixture(tmp_path, topics=1, questions_per_topic=1, succeeded=1, gold=[fact(), fact("unused", para_id="p2")])
    p = scoring.prepare_benchmark(plan, status, sample, expected_questions=1)
    r = scoring.score_prepared(p, judge=FakeJudge(), execute=True)
    assert r["questions"][0]["confirmed"]["referenced_gold_evidence_records"] == 1
    assert r["topic_level_formal_export_comparison"]["original_evidence_rows_all_topics"] == 2


def test_unknown_semantic_dimension_preserves_confirmed_possible_bounds_and_usage(tmp_path):
    plan, status, sample, _ = fixture(tmp_path, topics=1, questions_per_topic=1, succeeded=1)
    p = scoring.prepare_benchmark(plan, status, sample, expected_questions=1)
    r = scoring.score_prepared(p, judge=FakeJudge(uncertain=True), execute=True)
    assert r["aggregate"]["confirmed"]["micro"]["recall"] == 0
    assert r["aggregate"]["possible_including_unknowns"]["micro"]["recall"] == 1
    assert r["new_evaluation_usage"]["request_records"] == 2
    assert r["new_evaluation_usage"]["total_tokens"]["known_sum"] == 30


@pytest.mark.parametrize("mutation,code", [
    (lambda p,s: s["questions"].pop(), "question_set_mismatch"),
    (lambda p,s: s["questions"].append(deepcopy(s["questions"][0])), "question_key_mismatch"),
    (lambda p,s: s["questions"][0].update(status="made_up"), "unknown_runtime_status"),
    (lambda p,s: s["questions"][0].update(internal_question_id="another"), "internal_question_identity_mismatch"),
    (lambda p,s: s["questions"][0].update(ready_revision_id=None), "missing_ready_revision"),
    (lambda p,s: s.update(source_bundle_sha256="c"*64), "source_bundle_mismatch"),
])
def test_identity_or_inventory_gaps_fail_before_any_judge(tmp_path, mutation, code):
    plan, status, sample, _ = fixture(tmp_path)
    p,s = json.loads(plan.read_text()), json.loads(status.read_text())
    mutation(p,s)
    dump(status,s)
    with pytest.raises(scoring.BenchmarkError, match=code):
        scoring.prepare_benchmark(plan,status,sample)


def test_fixed_ready_revision_is_used_not_other_latest_question_result(tmp_path):
    plan, status, sample, database = fixture(tmp_path, topics=1, questions_per_topic=1, succeeded=1)
    with sqlite3.connect(database) as c:
        c.execute("INSERT INTO graph_revisions VALUES ('newer','internal-topic-0','ready')")
        c.execute("INSERT INTO revision_artifacts VALUES ('newer',?)", (json.dumps({"question_results": {"internal-0-0": {"evidence": []}}}),))
    p = scoring.prepare_benchmark(plan,status,sample,expected_questions=1)
    assert p["cases"][0]["artifact"]["submission"]["revision_id"] == "fixed-0-0"
    assert len(p["cases"][0]["artifact"]["submission"]["evidence"]) == 1


@pytest.mark.parametrize("location", ["runtime", "sample", "analysis-root", "existing"])
def test_output_path_rules_precede_judge_creation(tmp_path, monkeypatch, location):
    plan, status, sample, _ = fixture(tmp_path)
    analysis = tmp_path / "_analysis"
    monkeypatch.setattr(scoring,"ANALYSIS_ROOT",analysis)
    path = {"runtime": scoring.ROOT/".runtime"/"do-not-write-benchmark-test", "sample": sample/"new-output", "analysis-root": analysis, "existing": analysis/"exists"}[location]
    if location == "existing": path.mkdir(parents=True)
    called=[]
    def forbidden(): called.append(True);pytest.fail("preflight must precede HTTP")
    with pytest.raises(scoring.BenchmarkError):
        scoring.write_benchmark(plan,status,sample,path,execute=True,judge_factory=forbidden)
    assert not called


def test_nonterminal_execute_rejected_before_http_but_plan_allowed(tmp_path, monkeypatch):
    plan,status,sample,_=fixture(tmp_path)
    s=json.loads(status.read_text());s["questions"][1]["status"]="running";dump(status,s)
    monkeypatch.setattr(scoring,"ANALYSIS_ROOT",tmp_path/"_analysis")
    p=scoring.prepare_benchmark(plan,status,sample)
    assert scoring.score_prepared(p)["status"]=="planned_not_called"
    with pytest.raises(scoring.BenchmarkError,match="terminal_inventory"):
        scoring.write_benchmark(plan,status,sample,tmp_path/"_analysis"/"unfinished",execute=True,
            judge_factory=lambda:pytest.fail("no HTTP for nonterminal inventory"))


def test_explicit_execute_creates_only_independent_judge_closes_and_cache_avoids_repayment(tmp_path, monkeypatch):
    plan,status,sample,database=fixture(tmp_path,topics=1,questions_per_topic=1,succeeded=1)
    monkeypatch.setattr(scoring,"ANALYSIS_ROOT",tmp_path/"_analysis")
    before=database.read_bytes();cache=tmp_path/"_analysis"/"separate-cache"
    first,second=FakeJudge(),FakeJudge();con1,con2=FakeConclusionJudge(),FakeConclusionJudge()
    _,r1=scoring.write_benchmark(plan,status,sample,tmp_path/"_analysis"/"run-one",execute=True,cache_dir=cache,expected_questions=1,judge_factory=lambda:first,conclusion_judge_factory=lambda:con1)
    _,r2=scoring.write_benchmark(plan,status,sample,tmp_path/"_analysis"/"run-two",execute=True,cache_dir=cache,expected_questions=1,judge_factory=lambda:second,conclusion_judge_factory=lambda:con2)
    assert first.closed and second.closed
    assert len(first.calls)==2 and len(second.calls)==0
    assert len(con1.calls)==1 and len(con2.calls)==0 and con1.closed and con2.closed
    assert r1["new_evaluation_usage"]["request_records"]==3 and r2["new_evaluation_usage"]["request_records"]==0
    assert database.read_bytes()==before


@pytest.mark.parametrize("p,r,f,points,status", [
    (.9,.9,.9,7,"official_anchor_applied"),
    (.8,.8,.8,5,"official_anchor_applied"),
    (.7,.7,.7,3,"official_anchor_applied"),
    (.6,.6,.6,0,"official_anchor_applied"),
    (.75,1,6/7,None,"official_anchor_gap_f1_ge_085_but_precision_or_recall_lt_080"),
    (None,0,0,0,"official_anchor_applied"),
])
def test_official_seven_point_anchors_preserve_uncovered_high_f1_case(p,r,f,points,status):
    result=scoring.section_2_2_1_band({"precision":p,"recall":r,"f1":f})
    assert result["points"]==points and result["status"]==status


def test_actual_topic_all_reference_denominator_184_and_all_18_failed_conclusions_are_not_dropped(tmp_path):
    plan,status,sample,_=fixture(tmp_path,succeeded=0)
    for index,directory in enumerate(sorted(sample.iterdir())):
        count=31 if index<4 else 30
        dump(directory/"evidence_samples.json",[fact(f"f{j+1}") for j in range(count)])
    p=scoring.prepare_benchmark(plan,status,sample)
    pair,conclusion=FakeJudge(),FakeConclusionJudge()
    report=scoring.score_prepared(p,judge=pair,conclusion_judge=conclusion,execute=True)
    topic=report["topic_level_formal_export_comparison"]
    assert topic["original_evidence_rows_all_topics"]==184
    assert topic["confirmed"]["referenced_gold_evidence_records"]==184
    assert topic["confirmed"]["recall"]==0 and topic["section_2_2_1_confirmed_anchor"]["points"]==0
    assert report["conclusion_section_2_3_evaluation"]["original_conclusion_denominator"]==18
    assert report["conclusion_section_2_3_evaluation"]["confirmed_section_2_3_points"]==0
    assert report["conclusion_section_2_3_evaluation"]["possible_section_2_3_points"]==0
    assert len(pair.calls)==len(conclusion.calls)==0
    assert report["official_objective_50_total"] is None
    assert topic["section_2_2_2_fields"]["points"] is None
    assert report["subjective_40_points"]["points"] is None and report["optional_10_points"]["points"] is None


@pytest.mark.parametrize("label,confirmed,possible",[("full",15,15),("partial",7.5,7.5),("mismatch",0,0),("uncertain",0,15)])
def test_conclusion_official_weights_and_uncertainty_bounds(label,confirmed,possible,tmp_path):
    plan,status,sample,_=fixture(tmp_path,topics=1,questions_per_topic=1,succeeded=1)
    p=scoring.prepare_benchmark(plan,status,sample,expected_questions=1)
    result=scoring.evaluate_conclusions(p,judge=FakeConclusionJudge(label),execute=True)
    assert result["confirmed_section_2_3_points"]==confirmed
    assert result["possible_section_2_3_points"]==possible
    assert result["known_usage"]["new_evaluation_requests"]["request_records"]==1
    assert result["known_usage"]["new_evaluation_requests"]["total_tokens"]["known_sum"]==24


def test_one_successful_conclusion_keeps_other_17_failed_in_original_15_point_denominator(tmp_path):
    plan,status,sample,_=fixture(tmp_path)
    p=scoring.prepare_benchmark(plan,status,sample)
    judge=FakeConclusionJudge("full")
    result=scoring.evaluate_conclusions(p,judge=judge,execute=True)
    assert result["original_conclusion_denominator"]==18
    assert result["confirmed_section_2_3_points"]==pytest.approx(15/18)
    assert len(judge.calls)==1


def test_relations_rematch_each_side_instead_of_inheriting_wrong_whole_question_allocation(tmp_path):
    plan,status,sample,_=fixture(tmp_path,topics=1,questions_per_topic=1,succeeded=1,gold=[fact("f1"),fact("f2")],refs=[{"fact_id":"f2"}])
    conclusion_path=next(sample.iterdir())/"conclusions.json"
    conclusions=json.loads(conclusion_path.read_text(encoding="utf-8"))
    conclusions[0]["conflicting_facts"]=[{"fact_id":"f1"}]
    dump(conclusion_path,conclusions)
    p=scoring.prepare_benchmark(plan,status,sample,expected_questions=1)
    report=semantic.evaluate_frozen(p["cases"][0]["artifact"],judge=FakeJudge())
    assert report["confirmed_matching"]==[{"submitted_index":0,"reference_index":0}]
    assert report["reference_record_manifest"][0]["fact_id"]=="f1" # belongs to conflict side
    relation=scoring.evaluate_relations(p,[report],execute=True)
    assert relation["questions"][0]["supporting_facts"]["confirmed"]["matches"]==1 # f2 matched from frozen pair edge
    assert relation["questions"][0]["conflicting_facts"]["confirmed"]["matches"]==0
    assert relation["new_semantic_requests"]==0


def test_conclusion_judge_failure_preserves_known_usage_without_success(tmp_path):
    plan,status,sample,_=fixture(tmp_path,topics=1,questions_per_topic=1,succeeded=1)
    p=scoring.prepare_benchmark(plan,status,sample,expected_questions=1)
    class BadJudge(FakeConclusionJudge):
        def __call__(self,payload):
            raise semantic.JudgeFailure("supplier_output_invalid",usage={"total_tokens":19})
    r=scoring.evaluate_conclusions(p,judge=BadJudge(),execute=True)
    assert r["questions"][0]["status"]=="judge_failed"
    assert r["confirmed_section_2_3_points"]==0 and r["possible_section_2_3_points"]==15
    assert r["known_usage"]["new_evaluation_requests"]["total_tokens"]["known_sum"]==19


def test_conclusion_http_uses_official_rubric_and_never_exposes_auth_or_error_body(tmp_path):
    plan,status,sample,_=fixture(tmp_path,topics=1,questions_per_topic=1,succeeded=1)
    p=scoring.prepare_benchmark(plan,status,sample,expected_questions=1)
    payload=scoring._conclusion_payload(p["cases"][0])
    secret="MOCK_AUTH_DO_NOT_PERSIST"
    seen=[]
    def transport(request):
        seen.append(request)
        body=json.loads(request.content)
        assert request.headers["Authorization"]=="Bearer "+secret
        assert body["thinking"]=={"type":"disabled"}
        assert "2.3" in body["messages"][0]["content"]
        assert "补充参考未列但原文确有依据的限定不扣" in body["messages"][0]["content"]
        return httpx.Response(429,json={"error":secret+" PRIVATE_RESPONSE"})
    judge=scoring.HTTPConclusionJudge(model="synthetic",provider="deepseek",base_url="https://eval.invalid/v1",api_key=secret,transport=httpx.MockTransport(transport))
    try:
        result=scoring.evaluate_conclusions(p,judge=judge,execute=True)
    finally:judge.close()
    assert len(seen)==1
    rendered=json.dumps(result)
    assert secret not in rendered and "PRIVATE_RESPONSE" not in rendered
    assert result["questions"][0]["failure_code"]=="supplier_http_failed"


def test_execution_freeze_config_binding_is_verified_without_settings(tmp_path):
    plan,status,sample,_=fixture(tmp_path)
    s=json.loads(status.read_text())
    execution=status.parent/"execution-freeze.json"
    dump(execution,{"plan_sha256":s["plan_sha256"],"source_bundle_sha256":s["source_bundle_sha256"],"config_sha256":"different"})
    s.update(execution_freeze_path=str(execution),execution_freeze_sha256=scoring.bytes_sha(execution));dump(status,s)
    with pytest.raises(scoring.BenchmarkError,match="execution_freeze_identity_mismatch"):
        scoring.prepare_benchmark(plan,status,sample)


async def test_real_runner_ledger_and_normal_empty_snapshot_are_accepted_by_offline_plan(tmp_path):
    """Cross-module contract: real Repository+Worker, explicit fake provider only."""
    from scripts import run_sample_benchmark as runner
    from scripts.prepare_sample_inputs import prepare
    from app.config import Settings
    _,_,sample,_=fixture(tmp_path,topics=1,questions_per_topic=3,succeeded=0)
    raw_only=tmp_path/"prepared-raw-only"
    prepare(sample,raw_only)
    output=tmp_path/"real-runner"
    runner.prepare_plan(raw_only/"manifest.json",output)
    class Provider:
        premise_bank_enabled = False
        async def extract(self,*args):return []
        async def verify_evidence_scope(self,*args):pytest.fail("empty evidence must not trigger verification")
        async def review_fact_purpose(self,*args):pytest.fail("empty bank must not trigger purpose review")
        async def analyze(self,*args):return {"decision":"undetermined","conclusion":"材料不足。","judgments":[],"supporting_evidence_ids":[],"conflicting_evidence_ids":[]}
        async def arbitrate(self,*args):return {"decision":"undetermined","conclusion":"材料不足。","judgments":[],"supporting_evidence_ids":[],"conflicting_evidence_ids":[],"conflicts":[]}
    await runner.execute_plan(output,runner.file_sha(output/"benchmark-plan.json"),execute_authorized=True,
        settings_factory=lambda:Settings(_env_file=None,data_dir=tmp_path/"production-must-not-exist",model_name="",model_api_key="",model_premise_bank_enabled=False),
        provider_factory=lambda *_:Provider())
    p=scoring.prepare_benchmark(output/"benchmark-plan.json",output/"benchmark-status.json",sample,expected_questions=3)
    report=scoring.score_prepared(p)
    assert len(report["questions"])==3
    assert report["aggregate"]["completion"]["succeeded_questions"]==3
    assert report["topic_level_formal_export_comparison"]["topics"][0]["all_topic_questions_succeeded"] is True
    assert report["new_evaluation_usage"]["request_records"]==0
    assert report["execution_freeze_source"]["sha256"]==runner.file_sha(output/"execution-freeze.json")


def test_conclusion_payload_excludes_internal_audits_and_cannot_repair_exported_text(tmp_path):
    plan,status,sample,database=fixture(tmp_path,topics=1,questions_per_topic=1,succeeded=1)
    with sqlite3.connect(database) as connection:
        snapshot=question_snapshot("internal-topic-0", "internal-0-0", "q_001", "专题0设备第1项是否复位？", "fixed-0-0",
            conclusion="INTERNAL_MORE_COMPLETE_FINAL_SENTINEL", judgments=[{
                "id":"internal-judgment", "statement":"INTERNAL_MISSING_QUALIFIER_SENTINEL",
                "evidence_ids":["f1"], "prerequisite_judgment_ids":[]}])
        result=snapshot["question_results"]["internal-0-0"]
        result["evidence"][0].update(scope_verified=True, annotation_versions=[{"fact_des":"INTERNAL_AUDIT_LABEL_SENTINEL"}],
            source_counts={"trusted":17},scope_review={"approved":"INTERNAL_APPROVAL_SENTINEL"})
        result["evidence"][0]["argument"]["audit_grade"]="INTERNAL_ARGUMENT_AUDIT_SENTINEL"
        result["final"]["internal_conflict_audit"]=[{"basis":"INTERNAL_CONFLICT_BASIS_SENTINEL"}]
        snapshot["conclusions"][0]["conclusion"]="正式导出文本遗漏关键限定。"
        connection.execute("UPDATE revision_artifacts SET payload_json=? WHERE revision_id='fixed-0-0'",(json.dumps(snapshot),))
    prepared=scoring.prepare_benchmark(plan,status,sample,expected_questions=1)
    case=prepared["cases"][0]
    assert set(case["conclusion_comparison_plan"]["submitted_final"])=={"decision","supporting_evidence_ids","conflicting_evidence_ids"}
    payload=scoring._conclusion_payload(case)
    assert set(payload["submitted"])=={"export_conclusion","selected_evidence"}
    assert payload["submitted"]["export_conclusion"]["conclusion"]=="正式导出文本遗漏关键限定。"
    projected=payload["submitted"]["selected_evidence"][0]
    assert projected["id"]=="f1" and projected["quote"]=="设备已复位。"
    assert set(projected["argument"])=={"时间","主体","数值口径"}
    rendered=json.dumps(payload,ensure_ascii=False)
    for forbidden in ("scope_verified","annotation_versions","source_counts","scope_review","INTERNAL_","audit_grade"):
        assert forbidden not in rendered
    assert "不得替正式文本补写遗漏" in scoring.CONCLUSION_SYSTEM
    assert "formal-text-only-v2" in scoring.CONCLUSION_RUBRIC["version"]


def test_three_correct_conflict_facts_at_one_paragraph_expose_export_ceiling_and_separate_fact_diagnostic(tmp_path):
    clauses=["设备已复位。","机构甲提出异议。","机构乙提出异议。","机构丙提出异议。"]
    gold=[fact(identity,quote=quote,fact_des=quote) for identity,quote in zip(("support-gold","conflict-gold-1","conflict-gold-2","conflict-gold-3"),clauses)]
    plan,status,sample,database=fixture(tmp_path,topics=1,questions_per_topic=1,succeeded=1,gold=gold,refs=[{"fact_id":"support-gold"}])
    conclusion_path=next(sample.iterdir())/"conclusions.json"
    conclusions=json.loads(conclusion_path.read_text(encoding="utf-8"))
    conclusions[0]["conflicting_facts"]=[{"fact_id":f"conflict-gold-{i}"} for i in range(1,4)]
    dump(conclusion_path,conclusions)
    raw_path=conclusion_path.parent/"raw_documents.json"
    raw=json.loads(raw_path.read_text(encoding="utf-8"));raw[0]["content"]="".join(clauses);raw[0]["paragraphs"][0]["char_end"]=len(raw[0]["content"]);dump(raw_path,raw)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE documents SET content=? WHERE id='d1'",("".join(clauses),))
        connection.execute("UPDATE paragraphs SET text=? WHERE id='p1'",("".join(clauses),))
        # Multiple annotations retain the complete native paragraph quote. The
        # formal projection consolidates their location, not their factual IDs.
        evidence=[canonical_fact(identity,quote="".join(clauses),fact_des=quote)
                  for identity,quote in zip(("support-selected","conflict-selected-1","conflict-selected-2","conflict-selected-3"),clauses)]
        snapshot=question_snapshot("internal-topic-0", "internal-0-0", "q_001", "专题0设备第1项是否复位？", "fixed-0-0",
            evidence=evidence, supporting_evidence_ids=["support-selected"],
            conflicting_evidence_ids=[f"conflict-selected-{i}" for i in range(1,4)])
        connection.execute("UPDATE revision_artifacts SET payload_json=? WHERE revision_id='fixed-0-0'",(json.dumps(snapshot),))
    prepared=scoring.prepare_benchmark(plan,status,sample,expected_questions=1)
    case=prepared["cases"][0]
    assert len(case["formal_conclusion"]["conflicting_facts"])==1
    assert case["formal_conclusion"]["conflicting_facts"][0]["quote"]=="".join(clauses)
    assert all(row["quote"]=="".join(clauses) for row in case["artifact"]["submission"]["evidence"])
    class SameFactJudge(FakeJudge):
        def __call__(self,payload):
            result=super().__call__(payload)
            # Exact quote equality is not a semantic requirement: one native
            # paragraph can support several separate factual annotations.
            if payload["submitted"]["fact_des"]!=payload["reference"]["fact_des"]:
                result["verdict"].update(verdict="no_match",same_fact="no")
            return result
    judge=SameFactJudge()
    report=semantic.evaluate_frozen(case["artifact"],judge=judge)
    original_matching=deepcopy(report["confirmed_matching"])
    requests_before=len(judge.calls)
    relation=scoring.evaluate_relations(prepared,[report],execute=True)
    side=relation["questions"][0]["conflicting_facts"]
    literal=side["confirmed"]
    assert literal["matches"]==1 and literal["recall"]==pytest.approx(1/3)
    assert literal["granularity"]["has_export_reference_granularity_bottleneck"] is True
    assert literal["granularity"]["location_only_matching_upper_bound"]["recall"]==pytest.approx(1/3)
    diagnostic=side["internal_fact_id_diagnostic"]["confirmed"]
    assert diagnostic["matches"]==3 and diagnostic["recall"]==1
    assert diagnostic["formal_export_metric"] is False and diagnostic["official_score"] is False
    assert report["confirmed_matching"]==original_matching
    assert len(judge.calls)==requests_before and relation["new_semantic_requests"]==0
    assert relation["conflicting_list_proxy"]["confirmed"]["micro"]["matches"]==1
    assert relation["internal_fact_id_diagnostic_not_formal"]["conflicting_facts"]["confirmed"]["micro"]["matches"]==3


class BlockingPeakJudge(FakeJudge):
    """Explicit blocking fake; barrier proves three cases can actually overlap."""
    def __init__(self, *, synchronize_first_three=False, mixed=False):
        super().__init__()
        self.lock=threading.Lock()
        self.barrier=threading.Barrier(3) if synchronize_first_three else None
        self.active,self.peak,self.started=0,0,0
        self.mixed=mixed

    def __call__(self,payload):
        with self.lock:
            self.active+=1;self.started+=1
            sequence=self.started
            self.peak=max(self.peak,self.active)
        try:
            if self.barrier is not None and sequence<=3:
                self.barrier.wait(timeout=5)
            time.sleep(.01)
            response=super().__call__(payload)
            if self.mixed and "专题1设备第1项" in payload["question"]:
                raise semantic.JudgeFailure("supplier_output_invalid",usage={"input_tokens":6,"output_tokens":2,"total_tokens":8})
            if self.mixed and "第2项" in payload["question"]:
                response["verdict"].update(verdict="uncertain",time="uncertain")
            return response
        finally:
            with self.lock:self.active-=1


def semantic_results(report):
    return {"case_order":[q["case_id"] for q in report["questions"]],
        "question_metrics":[(q["confirmed"],q["possible"],q["counts"]) for q in report["questions"]],
        "aggregate":report["aggregate"],
        "question_matching":[(r["confirmed_matching"],r["possible_matching"],r["reference_record_manifest"]) for r in report["semantic_reports"]],
        "topic_matching":[(r["confirmed_matching"],r["possible_matching"],r["reference_record_manifest"]) for r in report["topic_level_formal_export_comparison"]["semantic_reports"]]}


def test_three_case_workers_are_bounded_and_results_follow_frozen_order_equal_to_serial(tmp_path):
    plan,status,sample,_=fixture(tmp_path,topics=2,questions_per_topic=3,succeeded=6)
    prepared=scoring.prepare_benchmark(plan,status,sample,expected_questions=6)
    serial,parallel=BlockingPeakJudge(),BlockingPeakJudge(synchronize_first_three=True)
    one=scoring.score_prepared(prepared,judge=serial,execute=True,judge_concurrency=1)
    three=scoring.score_prepared(prepared,judge=parallel,execute=True,judge_concurrency=3)
    assert serial.peak==1 and parallel.peak==3 and parallel.active==0
    assert semantic_results(one)==semantic_results(three)
    assert [q["case_id"] for q in three["questions"]]==[c["case_id"] for c in prepared["cases"]]
    assert three["new_evaluation_usage"]==one["new_evaluation_usage"]
    assert three["semantic_prompt_sha256"]==one["semantic_prompt_sha256"]==semantic.prompt_sha256()


def test_parallel_failures_uncertainty_and_runtime_failures_keep_original_denominators(tmp_path):
    plan,status,sample,_=fixture(tmp_path,topics=3,questions_per_topic=2,succeeded=4)
    prepared=scoring.prepare_benchmark(plan,status,sample,expected_questions=6)
    one=scoring.score_prepared(prepared,judge=BlockingPeakJudge(mixed=True),execute=True)
    parallel=BlockingPeakJudge(synchronize_first_three=True,mixed=True)
    three=scoring.score_prepared(prepared,judge=parallel,execute=True,judge_concurrency=3)
    assert parallel.peak==3
    assert semantic_results(one)==semantic_results(three)
    assert three["aggregate"]["confirmed"]["micro"]["referenced_gold_evidence_records"]==6
    assert three["aggregate"]["completion"]["failed_cancelled_superseded_questions"]==2
    assert any(r["supplier_failures"] for r in three["semantic_reports"])
    assert any(r["counts"]["uncertain_or_unavailable_pairs"] for r in three["semantic_reports"])


def test_parallel_identical_case_payloads_have_separate_cache_paths_without_changing_keys(tmp_path):
    plan,status,sample,_=fixture(tmp_path,topics=2,questions_per_topic=1,succeeded=2)
    prepared=scoring.prepare_benchmark(plan,status,sample,expected_questions=2)
    for case in prepared["cases"]:case["artifact"]["scope"]["question"]="同一完全相同的问题"
    cache=tmp_path/"case-scoped-cache"
    first=FakeJudge()
    one=scoring.score_prepared(prepared,judge=first,execute=True,cache_dir=cache,judge_concurrency=3)
    qreports=one["semantic_reports"]
    assert qreports[0]["pairs"][0]["cache_key"]==qreports[1]["pairs"][0]["cache_key"]
    qfiles=list((cache/"questions").glob("*/*.json"))
    tfiles=list((cache/"topics").glob("*/*.json"))
    assert len(qfiles)==2 and len(tfiles)==2
    assert len({p.parent for p in qfiles+tfiles})==4
    assert all(json.loads(p.read_text(encoding="utf-8"))["cache_key"]==p.stem for p in qfiles+tfiles)
    second=FakeJudge()
    cached=scoring.score_prepared(prepared,judge=second,execute=True,cache_dir=cache,judge_concurrency=3)
    assert len(second.calls)==0 and cached["new_evaluation_usage"]["request_records"]==0
    assert semantic_results(cached)==semantic_results(one)


def test_parallel_option_plan_still_has_no_workers_settings_or_http(tmp_path,monkeypatch):
    plan,status,sample,_=fixture(tmp_path)
    monkeypatch.setattr(scoring,"ANALYSIS_ROOT",tmp_path/"_analysis")
    def forbidden(*args,**kwargs):pytest.fail("plan must not construct workers/Settings/HTTP")
    monkeypatch.setattr(scoring,"ThreadPoolExecutor",forbidden)
    monkeypatch.setattr(semantic.HTTPPairJudge,"from_runtime_settings",forbidden)
    _,report=scoring.write_benchmark(plan,status,sample,tmp_path/"_analysis"/"parallel-option-plan",judge_concurrency=3,judge_factory=forbidden,conclusion_judge_factory=forbidden)
    assert report["new_evaluation_usage"]["request_records"]==0
    assert report["evaluation_concurrency"]["effective_workers"]==0
    assert report["status"]=="planned_not_called"


def test_parallel_progress_lines_are_complete_and_conclusions_begin_after_case_pools_join(tmp_path,monkeypatch):
    plan,status,sample,_=fixture(tmp_path,topics=1,questions_per_topic=3,succeeded=3)
    monkeypatch.setattr(scoring,"ANALYSIS_ROOT",tmp_path/"_analysis")
    pair=BlockingPeakJudge(synchronize_first_three=True)
    class AfterCasesConclusionJudge(FakeConclusionJudge):
        def __call__(self,payload):
            assert pair.active==0
            return super().__call__(payload)
    conclusion=AfterCasesConclusionJudge()
    factories=[]
    def pair_factory():factories.append("pair");return pair
    def conclusion_factory():factories.append("conclusion");return conclusion
    path,report=scoring.write_benchmark(plan,status,sample,tmp_path/"_analysis"/"parallel-progress",execute=True,
        judge_factory=pair_factory,conclusion_judge_factory=conclusion_factory,expected_questions=3,judge_concurrency=3)
    progress=[json.loads(line) for line in (path.parent/"evaluation-progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert factories==["pair","conclusion"] and pair.closed and conclusion.closed
    assert pair.peak==3
    assert [e["phase"] for e in progress].count("question_evidence")==3
    assert [e["phase"] for e in progress].count("topic_evidence")==1
    assert [e["phase"] for e in progress].count("conclusion")==3
    assert progress[0]["phase"]=="start" and progress[-1]["phase"]=="complete"
    assert len(report["questions"])==3


@pytest.mark.parametrize("workers",[0,-1,True,33,1.5])
def test_invalid_judge_concurrency_is_rejected_before_factory(tmp_path,monkeypatch,workers):
    plan,status,sample,_=fixture(tmp_path)
    monkeypatch.setattr(scoring,"ANALYSIS_ROOT",tmp_path/"_analysis")
    with pytest.raises(scoring.BenchmarkError,match="judge_concurrency"):
        scoring.write_benchmark(plan,status,sample,tmp_path/"_analysis"/"invalid-workers",execute=True,judge_concurrency=workers,
            judge_factory=lambda:pytest.fail("invalid concurrency must not construct HTTP"))


def test_colliding_case_cache_paths_are_rejected_before_any_judge_call(tmp_path,monkeypatch):
    plan,status,sample,_=fixture(tmp_path,topics=2,questions_per_topic=1,succeeded=2)
    prepared=scoring.prepare_benchmark(plan,status,sample,expected_questions=2)
    judge=FakeJudge()
    monkeypatch.setattr(scoring,"fingerprint",lambda value:"c"*64)
    with pytest.raises(scoring.BenchmarkError,match="case_cache_paths_overlap"):
        scoring.score_prepared(prepared,judge=judge,execute=True,cache_dir=tmp_path/"cache",judge_concurrency=3)
    assert judge.calls==[]
