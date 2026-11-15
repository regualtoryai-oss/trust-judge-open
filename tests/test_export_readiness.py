"""Synthetic export lifecycle checks; no models or competition answers are used."""

from copy import deepcopy
from io import BytesIO
import json
from uuid import uuid4
from zipfile import ZipFile

from fastapi.testclient import TestClient
import pytest

from app.errors import ServiceError
from app.main import create_app
from app.utils import dumps, now
from app.worker import Worker
from test_repository import create, publish, setup_repo


def path(topic, revision, suffix="export-readiness"):
    return f"/api/topics/{topic}/revisions/{revision}/{suffix}"


def test_partial_revision_explains_missing_questions_and_downloads_nonformal_json(tmp_path):
    repo, topic = setup_repo(tmp_path, question_count=2)
    first, second = repo.questions(topic)
    run, snapshot = publish(repo, topic, selected=[first["id"]])
    revision = run["target_revision_id"]
    with TestClient(create_app(repo.settings)) as client:
        ready = client.get(path(topic, revision))
        assert ready.status_code == 200
        data = ready.json()
        assert (data["completed_questions"], data["total_questions"], data["can_export"]) == (1, 2, False)
        assert data["missing_questions"] == [second]
        assert data["blockers"][0]["code"] == "export_incomplete_topic"
        assert data["existing_export"] is None
        response = client.get(data["preview_url"])
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        assert "NONFORMAL" in response.headers["content-disposition"]
        preview = response.json()
        assert preview["formal_submission"] is False and preview["partial"] is True
        assert preview["scope"] == {"completed_questions": 1, "total_questions": 2,
                                    "completed_question_ids": [first["raw_question_id"]],
                                    "missing_question_ids": [second["raw_question_id"]]}
        assert set(preview["files"]) == {"evidence.json", "conclusions.json", "confidence.json", "conflicts.json"}
        assert [c["question_id"] for c in preview["files"]["conclusions.json"]] == [first["raw_question_id"]]
        rejected = client.post(f"/api/topics/{topic}/exports", json={"revision_id": revision}, headers={"Idempotency-Key": "partial"})
        assert rejected.status_code == 409
        assert rejected.json()["detail"]["code"] == "export_incomplete_topic"
    assert repo.all("SELECT id FROM export_jobs") == []
    assert repo.snapshot(topic, revision) == snapshot


def test_complete_revision_resumes_same_scope_job_and_zip_matches_preview(tmp_path):
    repo, topic = setup_repo(tmp_path, question_count=2)
    run, _ = publish(repo, topic)
    revision = run["target_revision_id"]
    with TestClient(create_app(repo.settings)) as client:
        ready = client.get(path(topic, revision)).json()
        assert ready["can_export"] and ready["missing_questions"] == []
        preview = client.get(ready["preview_url"]).json()
        assert preview["partial"] is False and preview["formal_submission"] is False
        response = client.post(f"/api/topics/{topic}/exports", json={"revision_id": revision}, headers={"Idempotency-Key": "complete"})
        assert response.status_code == 202
        job = response.json()
        assert client.get(path(topic, revision)).json()["existing_export"]["id"] == job["id"]
        with repo.connection() as db:
            db.execute("UPDATE export_jobs SET status='running' WHERE id=?", (job["id"],))
        assert client.get(path(topic, revision)).json()["existing_export"]["status"] == "running"
        with repo.connection() as db:
            db.execute("UPDATE export_jobs SET status='queued' WHERE id=?", (job["id"],))
        assert Worker(repo).export_one()
        existing = client.get(path(topic, revision)).json()["existing_export"]
        assert existing["id"] == job["id"] and existing["status"] == "succeeded"
        response = client.get(existing["download_url"])
        assert response.status_code == 200
        with ZipFile(BytesIO(response.content)) as archive:
            assert json.loads(archive.read("manifest.json"))["revision_id"] == revision
            assert all(json.loads(archive.read(name)) == value for name, value in preview["files"].items())


def test_old_complete_export_is_never_advertised_as_new_partial_revision(tmp_path):
    repo, topic = setup_repo(tmp_path, question_count=2)
    old, _ = publish(repo, topic)
    old_revision = old["target_revision_id"]
    job = repo.create_export(topic, old_revision, "old-complete")
    assert Worker(repo).export_one()
    new, _ = publish(repo, topic, selected=[repo.questions(topic)[0]["id"]])
    new_revision = new["target_revision_id"]
    ready = repo.export_readiness(topic, new_revision)
    assert ready["revision_id"] == new_revision and ready["published_revision_id"] == new_revision
    assert not ready["can_export"] and ready["existing_export"] is None
    old_ready = repo.export_readiness(topic, old_revision)
    assert old_ready["can_export"] and old_ready["existing_export"]["id"] == job["id"]
    assert old_ready["revision_id"] == old_revision and old_ready["published_revision_id"] == new_revision
    assert repo.submission_preview(topic, new_revision)["scope"]["completed_questions"] == 1
    assert repo.submission_preview(topic, old_revision)["scope"]["completed_questions"] == 2


@pytest.mark.parametrize("state", ["building", "failed"])
def test_unpublished_versions_have_no_preview_or_readiness(tmp_path, state):
    repo, topic = setup_repo(tmp_path)
    run = create(repo, topic)
    if state == "failed":
        repo.claim()
        repo.finish_failed(run["id"], "synthetic_failure", "显式合成测试失败")
    with TestClient(create_app(repo.settings)) as client:
        for suffix in ("export-readiness", "submission-preview"):
            response = client.get(path(topic, run["target_revision_id"], suffix))
            assert response.status_code == 409
            assert response.json()["detail"]["code"] == "revision_not_ready"


def test_foreign_or_missing_revision_cannot_expose_export_results(tmp_path):
    repo, topic = setup_repo(tmp_path)
    other_repo, other_topic = setup_repo(tmp_path / "other")
    run, _ = publish(other_repo, other_topic)
    with TestClient(create_app(repo.settings)) as client:
        for revision in (run["target_revision_id"], str(uuid4())):
            for suffix in ("export-readiness", "submission-preview"):
                response = client.get(path(topic, revision, suffix))
                assert response.status_code == 404
                assert response.json()["detail"]["code"] == "not_found"


def test_failed_export_can_retry_with_new_key_and_does_not_masquerade_as_download(tmp_path):
    repo, topic = setup_repo(tmp_path)
    run, _ = publish(repo, topic)
    revision = run["target_revision_id"]
    first = repo.create_export(topic, revision, "first")
    with repo.connection() as db:
        db.execute("UPDATE export_jobs SET status='failed',error_json=? WHERE id=?", (dumps({"code": "worker_interrupted"}), first["id"]))
    assert repo.export_readiness(topic, revision)["existing_export"] is None
    assert repo.create_export(topic, revision, "first")["status"] == "failed"
    second = repo.create_export(topic, revision, "second")
    assert second["id"] != first["id"]
    assert Worker(repo).export_one()
    assert repo.export_readiness(topic, revision)["existing_export"]["id"] == second["id"]
    with TestClient(create_app(repo.settings)) as client:
        response = client.get(f"/api/exports/{first['id']}/download")
        assert response.status_code == 409 and response.json()["detail"]["code"] == "export_not_ready"


def test_missing_successful_archive_is_not_advertised_and_can_be_regenerated(tmp_path):
    repo, topic = setup_repo(tmp_path)
    run, _ = publish(repo, topic)
    revision = run["target_revision_id"]
    first = repo.create_export(topic, revision, "first")
    assert Worker(repo).export_one()
    (repo.settings.export_dir / (first["id"] + ".zip")).unlink()
    ready = repo.export_readiness(topic, revision)
    assert ready["can_export"] and ready["existing_export"] is None
    with TestClient(create_app(repo.settings)) as client:
        response = client.get(f"/api/exports/{first['id']}/download")
        assert response.status_code == 404 and response.json()["detail"]["code"] == "artifact_missing"
    second = repo.create_export(topic, revision, "new-artifact")
    assert Worker(repo).export_one()
    assert repo.export_readiness(topic, revision)["existing_export"]["id"] == second["id"]


@pytest.mark.parametrize("corruption", ["quote", "raw_id", "reference", "graph", "scope"])
def test_preview_and_formal_export_both_reject_integrity_corruption(tmp_path, monkeypatch, corruption):
    repo, topic = setup_repo(tmp_path)
    run, original = publish(repo, topic)
    revision = run["target_revision_id"]
    invalid = deepcopy(original)
    if corruption == "quote":
        invalid["evidence"][0]["quote"] = "源文不存在的内容"
    elif corruption == "raw_id":
        invalid["evidence"][0]["raw_para_id"] = "other_para"
    elif corruption == "reference":
        invalid["conclusions"][0]["supporting_evidence_ids"] = ["missing-evidence"]
    elif corruption == "graph":
        invalid["edges"].append({"from_node_id": "missing-node", "to_node_id": invalid["nodes"][-1]["id"], "relation": "supports"})
    else:
        invalid["scope"]["is_complete_topic"] = False
    # Simulate an internal corrupt artifact without changing the immutable
    # database or asserting that source data can be edited through the API.
    monkeypatch.setattr(repo, "snapshot", lambda *_: invalid)
    ready = repo.export_readiness(topic, revision)
    assert not ready["can_export"] and ready["blockers"][0]["code"] == "export_validation_failed"
    for action in (lambda: repo.submission_preview(topic, revision),
                   lambda: repo.create_export(topic, revision, "bad-export")):
        with pytest.raises(ServiceError) as caught:
            action()
        assert caught.value.code == "export_validation_failed"
        assert "源文不存在" not in caught.value.message
    assert repo.all("SELECT id FROM export_jobs") == []


def test_worker_rechecks_legacy_queued_partial_export(tmp_path):
    repo, topic = setup_repo(tmp_path, question_count=2)
    run, _ = publish(repo, topic, selected=[repo.questions(topic)[0]["id"]])
    export_id = str(uuid4())
    # This models a job queued by the old API before synchronous eligibility
    # existed. The worker must still refuse to produce a formal artifact.
    with repo.connection() as db:
        db.execute("INSERT INTO export_jobs(id,topic_id,revision_id,status,adapter_version,idempotency_key,request_sha256,created_at) VALUES (?,?,?,'queued','competition-v0.2','legacy',?,?)", (export_id, topic, run["target_revision_id"], "0" * 64, now()))
    assert Worker(repo).export_one()
    result = repo.export(export_id)
    assert result["status"] == "failed" and result["download_url"] is None
    assert not (repo.settings.export_dir / (export_id + ".zip")).exists()
