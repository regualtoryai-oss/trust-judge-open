"""Explicit synthetic lifecycle tests; no network or model-quality claims."""
from copy import deepcopy
import json
from zipfile import ZipFile

import pytest

from app.agents.provider import LangChainOpenAIProvider, ModelOutputInvalid
from app.config import Settings
from app.data.ingest import import_topic
from app.errors import ServiceError
from app.repository import Repository
from app.recovery import recover_completed_run
from app.worker import Worker
from app.workflows.analysis import AnalysisCancelled, run_analysis
from app.workflows.strict_workbench import PROMPT_VERSION, VERSION, compile_result, prepare_case, validate_result


Q = {"id": "question-internal", "raw_question_id": "q-test", "question": "甲设备批准已经确认还是尚未确认？"}
TEXT = "甲设备批准状态说明：" + "完整原文😀。" * 90 + "主管部门表示批准尚未确认。"


def sources():
    return [{"id": f"p{i}", "document_id": f"d{i}", "raw_doc_id": f"doc{i}", "raw_para_id": f"doc{i}#p1",
             "title": "甲设备批准说明", "text": TEXT, "source_credibility_json": "0.95",
             "raw_publish_time": "刚刚"} for i in range(2)]


def selection(raw):
    return {"hypotheses": [{"id": "H1", "claim": "甲设备批准已确认"}],
            "decisions": [{"p": row["paragraph_ref"], "relevance": 2, "stance": "qualification", "hypothesis_id": "H1"}
                          for row in raw["catalog"]]}


def annotations(raw):
    return {"annotations": [{"p": row["paragraph_ref"], "fact_des": "主管部门表示批准尚未确认。",
             "argument": {"时间": "", "主体": "主管部门", "地点": "", "内容": "批准尚未确认", "影响": "", "数值口径": ""},
             "entities": ["甲设备"]} for row in raw["catalog"]]}


class StrictFixture:
    strict_evidence_enabled = True
    strict_candidate_limit = 60
    paragraph_ranked_enabled = premise_bank_enabled = False
    prompt_version = PROMPT_VERSION
    repair_analysis = None
    model_name = "explicit-synthetic-provider"
    _record_usage = LangChainOpenAIProvider._record_usage

    def __init__(self, *, invalid_annotation=False, pause_after_selection=None):
        self.calls = []
        self.inputs = []
        self.invalid_annotation = invalid_annotation
        self.pause_after_selection = pause_after_selection

    def record(self, stage):
        self.calls.append(stage)
        self._record_usage(stage, {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})

    async def strict_selection(self, raw):
        self.record("strict_selection")
        if self.pause_after_selection is not None:
            self.pause_after_selection["paused"] = True
        return selection(raw)

    async def strict_annotation(self, raw, chosen):
        self.record("strict_annotation")
        value = annotations(raw)
        if self.invalid_annotation:
            value["annotations"][0]["argument"]["时间"] = "凭空时点"
        return value

    def answer(self, evidence):
        refs = [row["id"] for row in evidence]
        return {"decision": "undetermined", "conclusion": "主管部门表示批准尚未确认。", "supporting_evidence_ids": refs,
                "conflicting_evidence_ids": [], "judgments": [{"id": "J1", "statement": "原文保留未确认限定。",
                "evidence_ids": refs, "prerequisite_judgment_ids": []}] if refs else []}

    async def analyze(self, role, question, evidence, snapshot_id):
        self.record(role); self.inputs.append((role, deepcopy(evidence), snapshot_id))
        return self.answer(evidence)

    async def arbitrate(self, question, evidence, analyses):
        self.record("arbitrate")
        assert {row["role"] for row in analyses} == {"authority_analyst", "cross_validation_analyst"}
        return {**self.answer(evidence), "conflicts": []}


def setup(tmp_path, *, strict=True, questions=1):
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="",
                        model_strict_evidence_enabled=strict)
    repo = Repository(settings)
    docs = [{"doc_id": row["raw_doc_id"], "title": row["title"], "content": row["text"], "source_credibility": .95,
             "paragraphs": [{"para_id": row["raw_para_id"], "char_start": 0, "char_end": len(row["text"])}]} for row in sources()]
    topic = import_topic(settings.database_path, docs, [{"question_id": f"q{i}", "question": Q["question"]} for i in range(questions)], "合成严格工作台")
    return repo, topic["topic_id"]


def create(repo, topic):
    body = {"base_revision_id": repo.topic(topic)["head_revision_id"], "question_ids": [q["id"] for q in repo.questions(topic)],
            "mode": "initial", "include_baseline": False}
    return repo.create_run(topic, body, "strict-fixture", model_available=True)


def test_full_parents_mirrors_fields_and_proof_are_replayable_without_atomic_split():
    raw = prepare_case(Q, sources()); out = compile_result(Q, sources(), raw, selection(raw), annotations(raw))
    assert len(out["evidence"]) == 2
    assert all(row["quote"] == TEXT and row["end_cp"] == len(TEXT) for row in out["evidence"])
    assert len({row["source_group_id"] for row in out["evidence"]}) == 1
    assert all(row["grade_score"] == .95 and row["annotation_semantics_verified"] is False for row in out["evidence"])
    assert validate_result(out, Q, sources()) == out
    changed = deepcopy(out); changed["evidence"][0]["fact_des"] = "已批准"
    with pytest.raises(ValueError): validate_result(changed, Q, sources())


async def test_actual_worker_strict_nodes_two_independent_analysts_publication_usage_and_export(tmp_path):
    repo, topic = setup(tmp_path); provider = StrictFixture(); worker = Worker(repo, provider)
    queued = create(repo, topic); await worker.execute(repo.claim()); done = repo.run(queued["id"])
    assert done["status"] == "succeeded", done["error"]
    assert provider.calls == ["strict_selection", "strict_annotation", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert provider.inputs[0][1:] == provider.inputs[1][1:]
    activity = repo.activity(done["id"])
    assert activity["progress"]["completed_count"] == activity["progress"]["total_count"] == 7
    assert {row["stage"] for row in activity["progress"]["completed_stages"]} == {
        "strict_selection", "strict_annotation", "freeze", "authority_analyst", "cross_validation_analyst", "arbitrate", "publish"}
    assert sum(row["total_tokens"] for rows in done["usage"].values() for row in rows) == 60
    result = next(iter(repo.snapshot(topic, done["target_revision_id"])["question_results"].values()))
    assert result["verification_protocol"] == VERSION and len(result["analyses"]) == 2
    job = repo.create_export(topic, done["target_revision_id"], "strict-export"); assert worker.export_one()
    assert repo.export(job["id"])["status"] == "succeeded"
    with ZipFile(repo.settings.export_dir / (job["id"] + ".zip")) as archive:
        rows = json.loads(archive.read("evidence.json")); assert len(rows) == 2 and all(row["quote"] == TEXT for row in rows)


async def test_invalid_annotation_fails_once_preserves_usage_and_no_publication(tmp_path):
    repo, topic = setup(tmp_path); provider = StrictFixture(invalid_annotation=True); queued = create(repo, topic)
    await Worker(repo, provider).execute(repo.claim()); done = repo.run(queued["id"])
    assert done["status"] == "failed" and done["error"]["stage"] == "strict_annotation"
    assert provider.calls == ["strict_selection", "strict_annotation"]
    assert sum(row["total_tokens"] for rows in done["usage"].values() for row in rows) == 24
    assert repo.topic(topic)["published_revision_id"] is None


async def test_checkpoint_resume_reuses_selection_and_runs_only_remaining_actual_nodes(tmp_path):
    paused = {"paused": False}; provider = StrictFixture(pause_after_selection=paused); events = []
    kwargs = {"checkpoint_path": tmp_path / "checkpoint.sqlite3", "cancel_check": lambda: paused["paused"],
              "emit": lambda kind, data: events.append((kind, data))}
    with pytest.raises(AnalysisCancelled):
        await run_analysis(Q, sources(), provider, thread_id="first", **kwargs)
    assert provider.calls == ["strict_selection"]
    paused["paused"] = False; provider.pause_after_selection = None
    result = await run_analysis(Q, sources(), provider, thread_id="second", resume_from_thread_id="first", **kwargs)
    assert provider.calls.count("strict_selection") == 1
    assert len(result["analyses"]) == 2 and "strict_selection" in result["execution_provenance"]["reused_nodes"]
    assert not {"extract", "retrieve", "verify"}.intersection(result["execution_provenance"]["reused_nodes"])


async def test_identical_frozen_snapshot_can_be_reused_without_changing_paid_model_input(tmp_path):
    events = []
    first = StrictFixture()
    await run_analysis(Q, sources(), first, thread_id="paid-parent", cancel_check=lambda:False,
        checkpoint_path=tmp_path / "checkpoints.sqlite3", emit=lambda kind,data: events.append((kind,data)))
    bound = [data for kind,data in events if kind == "node_completed" and data.get("node") == "freeze"][0]
    second = StrictFixture()
    second.reusable_frozen_snapshot = {"source_run_id":"paid-parent", **{key:bound[key]
        for key in ("evidence_snapshot_id","snapshot_sha256")}}
    await run_analysis(Q, sources(), second, thread_id="new-run", cancel_check=lambda:False,
        checkpoint_path=tmp_path / "checkpoints.sqlite3", emit=lambda kind,data: events.append((kind,data)))
    assert {row[2] for row in second.inputs} == {bound["evidence_snapshot_id"]}
    assert any(kind == "frozen_snapshot_reused" for kind,data in events)
    third = StrictFixture()
    third.reusable_frozen_snapshot = {**second.reusable_frozen_snapshot, "snapshot_sha256":"0" * 64}
    with pytest.raises(ModelOutputInvalid) as error:
        await run_analysis(Q, sources(), third, thread_id="changed-run", cancel_check=lambda:False,
            checkpoint_path=tmp_path / "checkpoints.sqlite3", emit=lambda *args:None)
    assert error.value.stage == "freeze"
    assert third.calls == ["strict_selection","strict_annotation"]


def test_strict_mode_limits_one_question_and_keeps_legacy_default(tmp_path):
    assert Settings(_env_file=None).model_strict_evidence_enabled is False
    repo, topic = setup(tmp_path, questions=2)
    with pytest.raises(ServiceError) as error: create(repo, topic)
    assert error.value.code == "strict_question_limit"


async def test_completed_checkpoint_recovery_keeps_strict_proof_without_new_calls(tmp_path, monkeypatch):
    repo, topic = setup(tmp_path); provider = StrictFixture(); queued = create(repo, topic)
    original = repo.publish
    monkeypatch.setattr(repo, "publish", lambda *args: (_ for _ in ()).throw(ValueError("synthetic post-graph failure")))
    await Worker(repo, provider).execute(repo.claim())
    assert repo.run(queued["id"])["error"]["stage"] == "publish"
    calls = list(provider.calls); monkeypatch.setattr(repo, "publish", original)
    recovered = await recover_completed_run(repo, queued["id"], "strict-completed-recovery")
    assert recovered["status"] == "succeeded" and provider.calls == calls
    result = next(iter(repo.snapshot(topic, recovered["target_revision_id"])["question_results"].values()))
    assert result["verification_protocol"] == VERSION and result["strict_evidence_proof"]


async def test_strict_question_can_inherit_other_published_questions_original_protocol(tmp_path):
    from test_paragraph_ranked_workflow import RankedProvider
    repo, topic = setup(tmp_path, strict=False, questions=2)
    repo.settings = repo.settings.model_copy(update={"model_paragraph_ranked_enabled": True})
    previous = create(repo, topic); await Worker(repo, RankedProvider()).execute(repo.claim())
    before = repo.run(previous["id"])
    assert before["status"] == "succeeded", before["error"]
    repo.settings = repo.settings.model_copy(update={"model_paragraph_ranked_enabled": False, "model_strict_evidence_enabled": True})
    qid = repo.questions(topic)[0]["id"]
    body = {"base_revision_id": before["target_revision_id"], "question_ids": [qid], "mode": "retry", "include_baseline": False}
    queued = repo.create_run(topic, body, "strict-one-new-question", model_available=True)
    await Worker(repo, StrictFixture()).execute(repo.claim()); done = repo.run(queued["id"])
    assert done["status"] == "succeeded", done["error"]
    results = repo.snapshot(topic, done["target_revision_id"])["question_results"]
    assert {row["verification_protocol"] for row in results.values()} == {VERSION, "paragraph-ranked-source-v19"}
    assert repo.formal_submission_files(topic, done["target_revision_id"])
