"""Explicit synthetic full workbench lifecycle, not paid quality evidence."""
from copy import deepcopy
from io import BytesIO
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient
import pytest

from app.main import create_app
from app.data.ingest import import_topic
from app.worker import Worker
from test_bundle_run_config import bundle_settings
import test_bundle_workbench as fixtures


def setup(tmp_path, monkeypatch):
    settings = bundle_settings(tmp_path / "runtime")
    provider = fixtures.BundleFixture(None)
    provider.source_bundle_config = settings.source_bundle_config
    app = create_app(settings, provider)
    repo = app.state.repo
    pieces = ["经营方表示甲设备已经批准。", "主管部门表示甲设备尚未批准。", "公开材料保留未确认限定。"]
    offset = 0
    parents = []
    for index, piece in enumerate(pieces):
        parents.append({"para_id": f"opaque-{index}", "char_start": offset, "char_end": offset + len(piece)})
        offset += len(piece) + 1
    imported = import_topic(settings.database_path, [{"doc_id": "synthetic-only", "title": "合成审批记录",
        "content": "\n".join(pieces), "source_credibility": .9, "paragraphs": parents}],
        [{"question_id": "q-synthetic", "question": "甲设备批准状态是否已经核实？"}], "合成完整生命周期")
    topic = imported["topic_id"]
    question = repo.questions(topic)[0]
    monkeypatch.setattr(fixtures, "Q", question)
    provider.response, _ = fixtures.response_fixture(tmp_path / "proof", repo.source_documents(topic), repo.paragraphs(topic))
    return app, repo, topic, question, provider


@pytest.mark.asyncio
async def test_actual_api_queue_worker_publication_preview_and_zip(tmp_path, monkeypatch):
    app, repo, topic, question, provider = setup(tmp_path, monkeypatch)
    with TestClient(app) as client:
        created = client.post(f"/api/topics/{topic}/runs", headers={"Idempotency-Key": "explicit-fixture-only"},
            json={"question_ids": [question["id"]], "base_revision_id": None, "include_baseline": False, "mode": "initial"})
        assert created.status_code == 202, created.text
        rid = created.json()["run_id"]
        worker = Worker(repo, provider)
        await worker.execute(repo.claim())
        run = client.get(f"/api/runs/{rid}").json()
        assert run["status"] == "succeeded", run["error"]
        revision = run["target_revision_id"]
        assert set(provider.calls) == {"source_bundle_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"}
        selected_input = provider.inputs[0][1]
        assert selected_input[2] == repo.paragraphs(topic)
        assert "paragraph_id" in selected_input[2][0]  # actual repository alias retained
        ready = client.get(f"/api/topics/{topic}/revisions/{revision}/export-readiness").json()
        assert ready["can_export"]
        preview = client.get(f"/api/topics/{topic}/revisions/{revision}/submission-preview").json()
        exported = preview["files"]["evidence.json"]
        assert len(exported) == 1
        assert exported[0]["para_id"] == "opaque-0"
        assert exported[0]["quote"] == "经营方表示甲设备已经批准。\n主管部门表示甲设备尚未批准。"
        job = client.post(f"/api/topics/{topic}/exports", headers={"Idempotency-Key": "fixture-zip"}, json={"revision_id": revision})
        assert job.status_code == 202, job.text
        assert worker.export_one()
        status = client.get(f'/api/exports/{job.json()["id"]}').json()
        assert status["status"] == "succeeded", status
        download = client.get(status["download_url"])
        assert download.status_code == 200
        with ZipFile(BytesIO(download.content)) as archive:
            assert json.loads(archive.read("evidence.json")) == exported
            assert json.loads(archive.read("conclusions.json")) == preview["files"]["conclusions.json"]


@pytest.mark.asyncio
async def test_tampered_native_proof_cannot_publish_or_export(tmp_path, monkeypatch):
    app, repo, topic, question, provider = setup(tmp_path, monkeypatch)
    response = deepcopy(provider.response)
    response["proof"]["native_requests"][0]["sha256"] = "0" * 64
    provider.response = response
    queued = repo.create_run(topic, {"question_ids": [question["id"]], "base_revision_id": None,
        "include_baseline": False, "mode": "initial"}, "tampered-synthetic-only", model_available=True)
    await Worker(repo, provider).execute(repo.claim())
    run = repo.run(queued["id"])
    assert run["status"] == "failed"
    assert provider.calls == ["source_bundle_selection"]
    assert repo.topic(topic)["published_revision_id"] is None
    assert repo.all("SELECT id FROM export_jobs") == []
