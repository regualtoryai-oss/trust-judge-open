"""Synthetic failures only: fallback retrieval is not published as an answer."""
from copy import deepcopy

from fastapi.testclient import TestClient

from app.main import create_app
from app.domain.failure_appendix import build_failure_appendix
from tests.test_repository import setup_repo, create, publish


def test_failed_api_download_retains_all_questions_and_usage_without_publication(tmp_path):
    repo, topic_id = setup_repo(tmp_path, question_count=2)
    run = create(repo, topic_id)
    repo.claim()
    qids = [q["id"] for q in repo.questions(topic_id)]
    repo.record_usage(run["id"], qids[0], [{"operation":"extract","input_tokens":5,"output_tokens":2,"total_tokens":7}])
    repo.finish_failed(run["id"], "model_insufficient_balance", "余额不足", stage="extract")
    before = deepcopy(repo.run(run["id"]))
    with TestClient(create_app(repo.settings)) as client:
        response = client.get(f"/api/runs/{run['id']}/failure-appendix")
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment;")
    result = response.json()
    assert result["formal_submission"] is False
    assert result["original_run_status"] == "failed"
    assert result["original_usage"] == before["usage"]
    assert result["extra_model_calls"] == result["extra_model_tokens"] == 0
    assert len(result["questions"]) == 2
    for q in result["questions"]:
        assert q["model_generated_conclusion"] is False
        assert q["supporting_facts"] == q["conflicting_facts"] == []
        p = q["retrieval_candidates"][0]
        assert p["quote"] == repo.paragraphs(topic_id)[0]["text"]
        assert p["end_cp"] == len(p["quote"]) and p["start_cp"] == 0
        assert p["verification_status"] == "unverified_not_support"
    assert repo.run(run["id"]) == before
    assert repo.topic(topic_id)["published_revision_id"] is None


def test_nonfailed_task_cannot_download_or_replace_valid_output(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    run, snapshot = publish(repo, topic_id)
    with TestClient(create_app(repo.settings)) as client:
        response = client.get(f"/api/runs/{run['id']}/failure-appendix")
    assert response.status_code == 409
    assert repo.snapshot(topic_id, run["target_revision_id"]) == snapshot


def test_changed_manifest_rejects_old_failure_appendix(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    run = create(repo, topic_id)
    repo.finish_failed(run["id"], "model_request_failed", "请求失败")
    with repo.connection() as db:
        db.execute("UPDATE topics SET source_manifest_sha256=? WHERE id=?", ("changed", topic_id))
    with TestClient(create_app(repo.settings)) as client:
        response = client.get(f"/api/runs/{run['id']}/failure-appendix")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "failure_appendix_source_changed"


def test_excluded_paragraph_is_not_returned_and_no_match_stays_empty(tmp_path):
    repo, topic_id = setup_repo(tmp_path)
    run = create(repo, topic_id)
    repo.finish_failed(run["id"], "model_request_failed", "请求失败")
    paragraph = repo.paragraphs(topic_id)[0]
    excluded = [[paragraph["id"], 0, len(paragraph["text"]), paragraph["text"]]]
    result = build_failure_appendix(repo.run(run["id"]), repo.questions(topic_id), [paragraph],
        source_manifest_sha256=repo.topic(topic_id)["source_manifest_sha256"], excluded_spans=excluded)
    assert result["questions"][0]["retrieval_candidates"] == []
    questions = repo.questions(topic_id)
    questions[0]["question"] = "unrelatedzebraszz"
    result = build_failure_appendix(repo.run(run["id"]), questions, [paragraph], source_manifest_sha256="fixture")
    assert result["questions"][0]["retrieval_candidates"] == []
