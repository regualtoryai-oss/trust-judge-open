"""Benchmark lifecycle/isolation tests; explicit doubles, never model accuracy."""
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from scripts import run_sample_benchmark as runner
from scripts.prepare_sample_inputs import prepare
from app.config import Settings
from app.agents.provider import ModelInvocationFailed, _usage_records


def fixture_plan(tmp_path, *, topics=2):
    samples = tmp_path / "samples"
    sentinel = "OFFLINE_GOLD_MUST_NOT_ENTER_RUNTIME"
    for i in range(topics):
        folder = samples / f"合成专题{i}"
        folder.mkdir(parents=True)
        text = "机构公告：材料尚待核实。🦊"
        documents = [{"doc_id": "doc_001", "title": "合成工程材料", "content": text,
                      "paragraphs": [{"para_id": "doc_001#p1", "char_start": 0, "char_end": len(text)}]}]
        questions = [{"question_id": f"q_{q:03d}", "question": f"问题{q}是否有充分依据？", "conclusion": sentinel,
                      "supporting_facts": [{"fact_id": sentinel}]} for q in range(1, 4)]
        (folder / "raw_documents.json").write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
        (folder / "conclusions.json").write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
        (folder / "evidence_samples.json").write_text(sentinel, encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepare(samples, prepared)
    output = tmp_path / "benchmark"
    runner.prepare_plan(prepared / "manifest.json", output)
    return output, runner.file_sha(output / "benchmark-plan.json"), sentinel


class ExplicitProvider:
    premise_bank_enabled = False

    def __init__(self, calls, *, fail_second=False):
        self.calls, self.fail_second = calls, fail_second

    def account(self, operation, question):
        self.calls[(question["raw_question_id"], operation)] += 1
        _usage_records.get().append({"operation": operation, "usage_source": "provider",
                                    "input_tokens": 7, "output_tokens": 3, "total_tokens": 10})

    async def extract(self, question, paragraphs):
        self.account("extract", question)
        if self.fail_second and question["raw_question_id"] == "q_002":
            raise ModelInvocationFailed("Synthetic deliberate transport failure", stage="extract")
        return []

    async def verify_evidence_scope(self, question, evidence):
        raise AssertionError("empty extraction must not invent a verification request")

    async def review_fact_purpose(self, question, bank, ids):
        raise AssertionError("empty bank must not invent a purpose request")

    @staticmethod
    def answer():
        return {"decision": "undetermined", "conclusion": "材料不足，仍待核实。", "judgments": [],
                "supporting_evidence_ids": [], "conflicting_evidence_ids": []}

    async def analyze(self, role, question, evidence, snapshot_id):
        self.account(role, question)
        return self.answer()

    async def arbitrate(self, question, evidence, analyses):
        self.account("arbitrate", question)
        return {**self.answer(), "conflicts": []}


def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "production-do-not-touch", model_name="", model_api_key="", model_premise_bank_enabled=False)


def test_plan_is_settings_free_raw_only_and_preserves_source(tmp_path, monkeypatch):
    import app.config

    def forbidden(*args, **kwargs):
        raise AssertionError("Settings must not be instantiated during planning")

    monkeypatch.setattr(app.config, "Settings", forbidden)
    output, sha, sentinel = fixture_plan(tmp_path)
    plan = runner.validate_plan(output, sha)
    assert plan["total_topics"] == 2 and plan["total_questions"] == 6
    assert not list(output.rglob("*.sqlite3"))
    for path in output.rglob("*.json"):
        assert sentinel not in path.read_text(encoding="utf-8")
    assert plan["settings_read"] is plan["gold_answers_read"] is False


def test_plan_rejects_gold_even_when_importer_could_project_it(tmp_path):
    output, _, sentinel = fixture_plan(tmp_path)
    prepared = tmp_path / "prepared"
    manifest = runner.read_json(prepared / "manifest.json")
    path = prepared / manifest["topics"][0]["input_file"]
    payload = runner.read_json(path)
    payload["questions"][0]["conclusion"] = sentinel
    runner.write_json(path, payload)
    manifest["topics"][0]["sha256"] = runner.file_sha(path)
    runner.write_json(prepared / "manifest.json", manifest)
    with pytest.raises(ValueError, match="gold fields"):
        runner.prepare_plan(prepared / "manifest.json", tmp_path / "gold-rejected")
    assert not (tmp_path / "gold-rejected").exists()


def test_plan_refuses_existing_directory_input_traversal_and_source_hash_change(tmp_path):
    output, sha, _ = fixture_plan(tmp_path)
    with pytest.raises(ValueError, match="new empty"):
        runner.prepare_plan(tmp_path / "prepared/manifest.json", output)
    with pytest.raises(ValueError, match="escapes"):
        runner.contained(output, "../production/business.sqlite3")
    plan = runner.read_json(output / "benchmark-plan.json")
    path = output / plan["topics"][0]["input_file"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="raw input changed"):
        runner.validate_plan(output, sha)


async def test_execution_requires_authorization_before_settings(tmp_path):
    output, sha, _ = fixture_plan(tmp_path)
    def forbidden():
        pytest.fail("unauthorized execution read Settings")
    with pytest.raises(ValueError, match="execute-authorized"):
        await runner.execute_plan(output, sha, settings_factory=forbidden)


async def test_real_repository_worker_failure_continues_and_no_automatic_rerun(tmp_path):
    output, sha, sentinel = fixture_plan(tmp_path)
    original = settings(tmp_path)
    calls = Counter()
    def provider_factory(_settings, budget):
        return ExplicitProvider(calls, fail_second=True)
    result = await runner.execute_plan(output, sha, execute_authorized=True,
        settings_factory=lambda: original, provider_factory=provider_factory)
    assert result["all_questions_terminal"]
    assert result["status_counts"] == {"succeeded": 4, "failed": 2}
    assert len(result["questions"]) == 6
    assert calls[("q_002", "extract")] == 2
    assert calls[("q_003", "arbitrate")] == 2
    assert not original.data_dir.exists()
    frozen_attempts = {r["run_id"] for r in result["questions"]}
    before = dict(calls)
    second = await runner.execute_plan(output, sha, execute_authorized=True,
        settings_factory=lambda: original, provider_factory=provider_factory)
    assert dict(calls) == before
    assert {r["run_id"] for r in second["questions"]} == frozen_attempts
    for row in second["questions"]:
        terminal = runner.read_json(row["terminal_path"])
        assert terminal["id"] == row["run_id"]
        assert terminal["status"] == row["status"]
        assert row["usage"]  # failed usage is not erased
        if row["status"] == "failed":
            assert row["error"]["code"] == "model_request_failed"
            assert row["ready_revision_id"] is None and row["snapshot_path"] is None
        else:
            snapshot = runner.read_json(row["snapshot_path"])
            assert snapshot["revision_id"] == row["ready_revision_id"]
            assert row["internal_question_id"] in snapshot["question_results"]
            assert runner.read_json(row["question_result_path"])["origin_run_id"] == row["run_id"]
        with sqlite3.connect(Path(row["database"]).as_uri() + "?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 3
            for table, column in (("documents", "content"), ("questions", "question"), ("revision_artifacts", "payload_json")):
                assert not any(sentinel in value for (value,) in db.execute(f"SELECT {column} FROM {table}"))
    assert all(t["export"] is None and not t["export_readiness"]["can_export"] for t in result["topics"])


async def test_all_success_has_normal_formal_export_and_fixed_revisions(tmp_path):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    calls = Counter()
    result = await runner.execute_plan(output, sha, execute_authorized=True, settings_factory=lambda: settings(tmp_path),
                                       provider_factory=lambda *_: ExplicitProvider(calls))
    assert result["status_counts"] == {"succeeded": 3}
    assert len({r["ready_revision_id"] for r in result["questions"]}) == 3
    export = result["topics"][0]["export"]
    assert export["status"] == "succeeded" and Path(export["artifact_path"]).is_file()
    from zipfile import ZipFile
    with ZipFile(export["artifact_path"]) as archive:
        conclusions = json.loads(archive.read("conclusions.json"))
    assert {q["question_id"] for q in conclusions} == {"q_001", "q_002", "q_003"}


async def test_changed_config_or_bundle_never_reuses_existing_attempts(tmp_path, monkeypatch):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    original = settings(tmp_path)
    calls = Counter()
    await runner.execute_plan(output, sha, execute_authorized=True, settings_factory=lambda: original,
                              provider_factory=lambda *_: ExplicitProvider(calls))
    before = dict(calls)
    with pytest.raises(ValueError, match="config, version"):
        await runner.execute_plan(output, sha, execute_authorized=True,
            settings_factory=lambda: original.model_copy(update={"model_max_output_tokens": 12345}),
            provider_factory=lambda *_: ExplicitProvider(calls))
    assert dict(calls) == before
    monkeypatch.setattr(runner, "source_bundle", lambda: {"bundle_sha256": "changed", "files": {}})
    with pytest.raises(ValueError, match="source bundle changed"):
        await runner.execute_plan(output, sha, execute_authorized=True, settings_factory=lambda: pytest.fail("read settings"))


async def test_execution_freeze_is_separate_immutable_and_status_bound(tmp_path):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    calls = Counter()
    result = await runner.execute_plan(output, sha, execute_authorized=True, settings_factory=lambda: settings(tmp_path),
                                       provider_factory=lambda *_: ExplicitProvider(calls))
    freeze = output / "execution-freeze.json"
    frozen_bytes = freeze.read_bytes()
    assert result["execution_freeze_sha256"] == hashlib.sha256(frozen_bytes).hexdigest()
    assert runner.read_json(freeze)["config_sha256"] == result["config_sha256"]
    await runner.execute_plan(output, sha, execute_authorized=True, settings_factory=lambda: settings(tmp_path),
                              provider_factory=lambda *_: ExplicitProvider(calls))
    assert freeze.read_bytes() == frozen_bytes
    payload = runner.read_json(freeze)
    payload["http_concurrency"] = 99
    runner.write_json(freeze, payload)
    with pytest.raises(ValueError, match="execution config, version"):
        await runner.execute_plan(output, sha, execute_authorized=True, settings_factory=lambda: settings(tmp_path),
                                  provider_factory=lambda *_: pytest.fail("provider constructed after freeze mutation"))


async def test_interrupted_running_attempt_preserved_and_later_questions_continue(tmp_path):
    output, sha, _ = fixture_plan(tmp_path, topics=1)
    original = settings(tmp_path)
    plan = runner.read_json(output / "benchmark-plan.json")
    from app.repository import Repository
    from app.data.import_service import ImportService
    repo = Repository(runner.isolated_settings(original, output, plan["topics"][0]))
    service = ImportService(repo)
    preview = service.preview({"entries": [{"name": "input.json", "content": (output / plan["topics"][0]["input_file"]).read_text(encoding="utf-8")} ]})
    tid = service.commit({"preview_id": preview["preview_id"], "fingerprint": preview["fingerprint"]})["topic_id"]
    qid = repo.questions(tid)[0]["id"]
    run = repo.create_run(tid, {"question_ids": [qid], "base_revision_id": None, "mode": "recompute", "include_baseline": False},
                          "benchmark:" + sha + ":q_001", model_available=True)
    assert repo.claim()["id"] == run["id"]
    calls = Counter()
    result = await runner.execute_plan(output, sha, execute_authorized=True, settings_factory=lambda: original,
                                       provider_factory=lambda *_: ExplicitProvider(calls))
    first = result["questions"][0]
    assert first["run_id"] == run["id"] and first["status"] == "failed"
    assert first["error"]["code"] == "worker_interrupted"
    assert calls[("q_001", "extract")] == 0
    assert result["status_counts"] == {"failed": 1, "succeeded": 2}


def test_collision_guard_precedes_repository_access(tmp_path):
    original = settings(tmp_path)
    output = tmp_path / "benchmark"
    topic = {"data_dir": "topics/topic_01/runtime"}
    same = original.model_copy(update={"data_dir": output / topic["data_dir"]})
    with pytest.raises(ValueError, match="production storage"):
        runner.isolated_settings(same, output, topic)
    assert not output.exists()


async def test_global_request_permits_cover_actual_bound_request_and_release_on_failure():
    budget = runner.SharedRequestBudget(3)
    entered, active, peak = [], 0, 0
    class RequestTarget:
        async def _request(self, identifier):
            nonlocal active, peak
            entered.append(identifier)
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                if identifier == 4:
                    raise RuntimeError("test")
                return identifier
            finally:
                active -= 1
    objects = [budget.wrap(RequestTarget()) for _ in range(6)]
    results = await asyncio.gather(*(objects[i % 6]._request(i) for i in range(18)), return_exceptions=True)
    assert sorted(entered) == list(range(18)) and sum(isinstance(r, RuntimeError) for r in results) == 1
    assert peak == budget.peak == 3 and active == budget.active == 0
    assert budget.calls == 18 and budget.report()["wire_http_attempts"] is None


def test_known_tokens_do_not_turn_unknown_into_zero():
    result = runner.usage_summary({"qid": [{"input_tokens": 7, "output_tokens": None, "total_tokens": None}]})
    assert result["input_tokens"] == {"known_sum": 7, "unknown_records": 0}
    assert result["total_tokens"] == {"known_sum": 0, "unknown_records": 1}
