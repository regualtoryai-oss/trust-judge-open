"""Regression for real P22 multi-literal time annotation; no paid inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.agents.lc_bundle_transport import LCBundleWorkbenchProvider
from app.repository import Repository
from app.worker import Worker
from app.workflows.bundle_workbench import _time_literal_grounded, compile_result
from test_context_workbench import _import
from test_lc_context_case import raw_case
from test_lc_local_rank_integration import model  # noqa: F401
from test_paragraph_quality_workbench import settings_for, patch_quality_http


def test_joined_time_requires_every_exact_original_span_and_preserves_legacy_rule():
    quote = "10:58，下午时段。"
    row = {"quote": quote, "argument": {"时间": "10:58；下午时段"}, "field_spans": {"argument": {"时间": [
        {"text": "10:58", "start_cp": 0, "end_cp": 5}, {"text": "下午时段", "start_cp": 6, "end_cp": 10}]}}}
    assert _time_literal_grounded(row)
    changed = deepcopy(row); changed["field_spans"]["argument"]["时间"][1]["start_cp"] = 5
    assert not _time_literal_grounded(changed)
    changed = deepcopy(row); changed["argument"]["时间"] += "；明年"
    assert not _time_literal_grounded(changed)
    changed = deepcopy(row); changed.pop("field_spans")
    assert not _time_literal_grounded(changed)
    changed["argument"]["时间"] = "10:58"
    assert _time_literal_grounded(changed)


@pytest.mark.asyncio
async def test_disjoint_time_survives_real_worker_projection_publication_and_export(tmp_path, monkeypatch, model):
    settings = settings_for(tmp_path / "runtime", model, compact=True, local_rank=False)
    repo = Repository(settings)
    topic = _import(repo, raw_case(["10:58首轮记录启动，随后下午时段仍待核验。", "另一条程序记录尚未确认。 "]))
    question = repo.questions(topic)[0]
    provider = LCBundleWorkbenchProvider(settings)
    observed = patch_quality_http(monkeypatch, provider, bad="disjoint_time")
    task = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "disjoint-time", model_available=True)
    await Worker(repo, provider).execute(repo.claim())
    run = repo.run(task["run_id"])
    assert run["status"] == "succeeded", run["error"]
    assert len(observed) == 4
    exported = repo.formal_submission_files(topic, run["target_revision_id"])["evidence.json"]
    assert any(row["argument"]["时间"] == "10:58；下午时段" for row in exported)


def test_original_real_v4_sealed_response_compiles_unchanged_if_available():
    directory = Path(__file__).resolve().parents[1] / ".runtime/strict-episodes/e1d9df1e-9554-4771-840a-fd2b536b131c/source-bundle"
    if not directory.is_dir():
        pytest.skip("Local closed real-run files are not part of the portable synthetic suite")
    raw = json.loads((directory / "raw-case.json").read_text("utf8"))
    response = json.loads((directory / "provider-response.json").read_text("utf8"))
    before = deepcopy(response)
    question = {"id": "offline-real-source-regression", "raw_question_id": raw["question"]["question_id"],
        "question": raw["question"]["question"]}
    compiled = compile_result(question, raw["paragraphs"], raw["documents"], response, excluded_spans=[])
    assert len(compiled["evidence"]) == len(response["artifact"]["evidence"])
    assert any(row["raw_doc_id"] == "doc_075" and row["raw_para_id"] == "doc_075#p4"
        and row["argument"]["时间"] == "10:58；下午时段" for row in compiled["evidence"])
    assert response == before  # original model fields, quote, source proof and hashes unchanged
