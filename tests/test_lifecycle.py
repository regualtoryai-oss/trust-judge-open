"""Durable controls and real checkpoint continuation with synthetic providers."""
import asyncio
from copy import deepcopy
import time

from fastapi.testclient import TestClient
import pytest

from app.errors import ServiceError
from app.main import create_app
from app.repository import Repository
from app.utils import dumps, loads
from app.worker import Worker
from test_integration import IntegrationProvider, running_client, start_run, wait_terminal
from test_recovery import failed_source
from test_repository import create, setup_repo


def test_retry_inherits_only_user_overlays_not_recovery_or_continuation_controls(tmp_path):
    repo, topic = setup_repo(tmp_path)
    source = create(repo, topic)
    overlays = {"excluded": [], "manual_evidence": [], "interpretation_overrides": {}}
    with repo.connection() as db:
        db.execute("UPDATE run_options SET options_json=? WHERE run_id=?", (dumps({
            "overlays": overlays, "include_baseline": True,
            "checkpoint_recovery": {"new_model_calls": 0},
            "resume_from_run_id": "private-source", "resume_checkpoint_question_ids": ["private-question"],
        }), source["id"]))
    repo.cancel(source["id"])
    retry = create(repo, topic)
    options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (retry["id"],))["options_json"])
    assert options == {"overlays": overlays, "include_baseline": False}


def test_queued_pause_resume_has_explicit_unstarted_lineage_and_idempotency(tmp_path):
    repo, topic = setup_repo(tmp_path)
    source = create(repo, topic)
    paused = repo.pause(source["id"])
    assert paused["status"] == "cancelled" and paused["view_status"] == "paused"
    assert paused["resumable"] and repo.claim() is None
    continued = repo.resume(source["id"], "resume-once", model_available=True)
    assert continued["id"] != source["id"]
    assert continued["resume_mode"] == "start_unstarted"
    assert continued["resumed_from_run_id"] == source["id"]
    assert repo.resume(source["id"], "resume-once", model_available=True)["id"] == continued["id"]
    assert len(repo.runs(topic)) == 2
    assert repo.run(source["id"])["view_status"] == "paused"
    event = next(e for e in repo.events(continued["id"]) if e["type"] == "resume_queued")
    assert event["data"]["checkpoint_question_ids"] == []


def test_resume_guards_config_and_head_before_creating_a_new_run(tmp_path):
    repo, topic = setup_repo(tmp_path)
    source = create(repo, topic)
    repo.pause(source["id"])
    with repo.connection() as db:
        db.execute("UPDATE runs SET model_config_json='{}' WHERE id=?", (source["id"],))
    blocked = repo.run(source["id"])
    assert not blocked["resumable"] and "流程" in blocked["resume_blocked_reason"]
    with pytest.raises(ServiceError) as caught:
        repo.resume(source["id"], "bad-config", model_available=True)
    assert caught.value.code == "resume_not_available"
    assert len(repo.runs(topic)) == 1
    with repo.connection() as db:
        db.execute("UPDATE runs SET model_config_json=? WHERE id=?", (dumps(repo.config()), source["id"]))
    next_run = create(repo, topic)
    repo.pause(next_run["id"])
    with pytest.raises(ServiceError) as caught:
        repo.resume(source["id"], "stale", model_available=True)
    assert caught.value.code == "resume_not_available"
    assert len(repo.runs(topic)) == 2


def test_activity_and_topic_history_survive_api_recreation_and_cursor(tmp_path):
    repo, topic = setup_repo(tmp_path)
    source = create(repo, topic)
    repo.claim()
    question = source["question_ids"][0]
    repo.event(source["id"], "node_started", {"node": "retrieve", "question_id": question})
    repo.event(source["id"], "tool_started", {"call_id": "local-1", "name": "retrieve", "kind": "local", "engine": "BM25/jieba", "question_id": question, "summary": "检索原文"})
    repo.event(source["id"], "tool_completed", {"call_id": "local-1", "name": "retrieve", "elapsed_ms": 12, "summary": "检索 2 段"})
    repo.event(source["id"], "node_completed", {"node": "retrieve", "question_id": question})
    with TestClient(create_app(repo.settings, provider=IntegrationProvider())) as client:
        history = client.get(f"/api/topics/{topic}/runs").json()
        assert history[0]["id"] == source["id"]
        result = client.get(f"/api/runs/{source['id']}/activity?after=2").json()
        assert all(e["seq"] > 2 for e in result["events"])
        assert result["progress"]["completed_count"] == 1
        assert result["progress"]["total_count"] == 9
        assert result["tool_calls"][0]["engine"] == "BM25/jieba"
        assert result["tool_calls"][0]["status"] == "completed"
        assert result["tool_calls"][0]["elapsed_ms"] == 12
    # A second API process sees the same durable events, no new task is run.
    with TestClient(create_app(repo.settings, provider=IntegrationProvider())) as client:
        assert client.get(f"/api/runs/{source['id']}/activity").json()["tool_calls"] == result["tool_calls"]


def test_pause_running_http_does_not_publish_and_retains_checkpoint(tmp_path):
    provider = IntegrationProvider(block_extraction=True)
    with running_client(tmp_path, provider) as (client, settings, topic, question):
        source = start_run(client, topic, question)
        assert provider.extraction_started.wait(5)
        response = client.post(f"/api/runs/{source['id']}/pause")
        assert response.status_code == 202
        assert response.json()["view_status"] in {"pausing", "paused"}
        paused = wait_terminal(client, f"/api/runs/{source['id']}")
        assert paused["view_status"] == "paused" and paused["error"]["code"] == "paused"
        assert client.get(f"/api/topics/{topic}").json()["published_revision_id"] is None
        assert settings.checkpoint_path.is_file()
        provider.release_extraction.set()
        resumed = client.post(f"/api/runs/{source['id']}/resume", headers={"Idempotency-Key": "resume-http"})
        assert resumed.status_code == 202, resumed.text
        replay = client.post(f"/api/runs/{source['id']}/resume", headers={"Idempotency-Key": "resume-http"})
        assert replay.status_code == 202 and replay.json()["id"] == resumed.json()["id"]
        ready = wait_terminal(client, f"/api/runs/{resumed.json()['id']}")
        assert ready["status"] == "succeeded", ready
        activity = client.get(f"/api/runs/{ready['id']}/activity").json()
        assert any(e["type"] == "checkpoint_reused" for e in activity["events"])
        assert activity["progress"]["completed_count"] == activity["progress"]["total_count"] == 8


async def test_completed_checkpoint_can_resume_through_regular_worker_without_new_calls(tmp_path):
    repo, source, topic, question, provider, output = await failed_source(tmp_path)
    before_calls, before_usage = deepcopy(provider.calls), deepcopy(repo.run(source)["usage"])
    created = repo.resume(source, "resume-complete", model_available=True)
    assert created["resume_mode"] == "checkpoint"
    await Worker(repo, provider).execute(repo.claim())
    ready = repo.run(created["id"])
    assert ready["status"] == "succeeded", ready
    assert ready["usage"] == {question: []}
    assert provider.calls == before_calls
    assert repo.run(source)["status"] == "failed"
    assert repo.run(source)["usage"] == before_usage
    artifacts = repo.snapshot(topic, ready["target_revision_id"])
    result = artifacts["question_results"][question]
    assert result["evidence_snapshot_id"] != output["evidence_snapshot_id"]
    assert result["execution_provenance"]["source_thread_id"] == source + ":" + question
    states = repo.all("SELECT provenance_json FROM node_states WHERE revision_id=?", (ready["target_revision_id"],))
    assert all(loads(row["provenance_json"])["reused"] for row in states)
    assert all(loads(row["provenance_json"])["run_id"] == source for row in states)


async def test_repeated_pause_keeps_ancestor_checkpoint_if_new_task_has_not_forked_it(tmp_path):
    repo, source, topic, question, provider, output = await failed_source(tmp_path)
    before = deepcopy(provider.calls)
    first = repo.resume(source, "first", model_available=True)
    repo.pause(first["id"])
    second = repo.resume(first["id"], "second", model_available=True)
    options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (second["id"],))["options_json"])
    assert options["resume_checkpoint_sources"] == {question: source}
    await Worker(repo, provider).execute(repo.claim())
    assert repo.run(second["id"])["status"] == "succeeded"
    assert provider.calls == before


def test_resume_rejects_a_model_failure_if_checkpoint_is_missing(tmp_path):
    repo, topic = setup_repo(tmp_path)
    source = create(repo, topic)
    repo.claim()
    repo.event(source["id"], "node_started", {"node": "extract", "question_id": source["question_ids"][0]})
    repo.finish_failed(source["id"], "model_request_failed", "合成请求失败")
    assert not repo.run(source["id"])["resumable"]
    with pytest.raises(ServiceError) as caught:
        repo.resume(source["id"], "bad", model_available=True)
    assert caught.value.code == "resume_not_available"
    assert len(repo.runs(topic)) == 1


def test_unreadable_checkpoint_does_not_break_history_or_claim_recovery(tmp_path):
    repo, topic = setup_repo(tmp_path)
    source = create(repo, topic)
    repo.claim()
    repo.finish_failed(source["id"], "worker_interrupted", "合成中断")
    repo.settings.checkpoint_path.write_bytes(b"synthetic invalid checkpoint marker")
    before = repo.settings.checkpoint_path.read_bytes()
    status = repo.run(source["id"])
    assert status["status"] == "failed" and not status["resumable"]
    assert "存储" in status["resume_blocked_reason"]
    assert repo.runs(topic)[0]["id"] == source["id"]
    assert repo.settings.checkpoint_path.read_bytes() == before


def test_activity_has_safe_count_summary_and_paused_model_tool_is_cancelled(tmp_path):
    repo, topic = setup_repo(tmp_path)
    source = create(repo, topic)
    repo.claim()
    repo.event(source["id"], "tool_started", {"call_id": "retrieve", "name": "retrieve", "kind": "local"})
    repo.event(source["id"], "tool_completed", {"call_id": "retrieve", "name": "retrieve", "summary": {"paragraph_count": 3, "provider_body": "must not reach tool summary"}})
    repo.event(source["id"], "tool_started", {"call_id": "extract", "name": "extract", "kind": "model"})
    repo.event(source["id"], "tool_failed", {"call_id": "extract", "name": "extract", "code": "cancelled"})
    calls = repo.activity(source["id"])["tool_calls"]
    assert calls[0]["summary"] == "3 段原文"
    assert calls[1]["status"] == "cancelled"
