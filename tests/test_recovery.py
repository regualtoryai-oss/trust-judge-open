"""Explicit completed-checkpoint publication recovery using synthetic data only."""

from copy import deepcopy

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import pytest

from app.agents.provider import LangChainOpenAIProvider, ModelOutputInvalid
from app.data.ingest import import_topic
from app.errors import ServiceError
from app.recovery import recover_completed_run
from app.repository import Repository
from app.worker import Worker, WorkerLease
from app.workflows.analysis import run_analysis
from test_integration import IntegrationProvider, seed_runtime


class MeteredProvider(IntegrationProvider):
    async def extract(self, question, paragraphs):
        LangChainOpenAIProvider._record_usage(
            self, "extract", {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
        )
        return await super().extract(question, paragraphs)


async def failed_source(tmp_path, *, complete=True, question_mismatch=False, stage="snapshot"):
    settings, topic_id, question_id = seed_runtime(tmp_path)
    repo = Repository(settings)
    original_config = repo.config
    repo.config = lambda: {**original_config(), "strict_scope_enabled": False, "premise_bank_enabled": False}
    created = repo.create_run(topic_id, {
        "question_ids": [question_id], "base_revision_id": None,
        "mode": "initial", "include_baseline": False,
    }, "source", model_available=True)
    claimed = repo.claim()
    assert claimed["id"] == created["run_id"]
    question = repo.questions(topic_id)[0]
    if question_mismatch:
        question = {**question, "question": question["question"] + "（篡改题干）"}
    provider = MeteredProvider(bad_quote=not complete)
    try:
        result = await run_analysis(
            question, repo.paragraphs(topic_id), provider,
            test_only_legacy_precision=True, checkpoint_path=settings.checkpoint_path,
            thread_id=created["run_id"] + ":" + question_id,
            cancel_check=lambda: False,
            emit=lambda kind, data: repo.event(created["run_id"], kind, data),
            usage_callback=lambda records: repo.record_usage(created["run_id"], question_id, records),
        )
    except ModelOutputInvalid:
        assert not complete
        result = None
    # Simulate the older domain/publication boundary rejecting an otherwise
    # completed graph. The incomplete fixture deliberately lies about the stage
    # to ensure recovery independently checks checkpoint completion.
    repo.finish_failed(created["run_id"], "snapshot_validation_failed", "合成旧版业务校验失败", stage=stage)
    return repo, created["run_id"], topic_id, question_id, provider, result


def assert_single_failed_source(repo, source_id, topic_id):
    assert repo.all("SELECT id FROM runs") == [{"id": source_id}]
    assert repo.run(source_id)["status"] == "failed"
    assert repo.topic(topic_id)["head_revision_id"] == repo.run(source_id)["target_revision_id"]
    assert repo.topic(topic_id)["published_revision_id"] is None


async def test_completed_checkpoint_recovery_publishes_with_provenance_and_no_new_usage(tmp_path, monkeypatch):
    repo, source_id, topic_id, question_id, provider, original = await failed_source(tmp_path)
    before_usage = deepcopy(repo.run(source_id)["usage"])
    before_calls = deepcopy(provider.calls)

    def no_provider_construction(*args, **kwargs):
        raise AssertionError("Completed recovery must never construct a model provider")

    monkeypatch.setattr(LangChainOpenAIProvider, "__init__", no_provider_construction)
    recovered = await recover_completed_run(repo, source_id, "recover-complete")
    assert recovered["status"] == "succeeded", recovered
    assert recovered["run_id"] != source_id
    assert recovered["base_revision_id"] == repo.run(source_id)["target_revision_id"]
    assert recovered["usage"] == {question_id: []}
    assert recovered["recovery"]["source_run_id"] == source_id
    assert recovered["recovery"]["new_model_calls"] == 0
    assert repo.run(source_id)["status"] == "failed"
    assert repo.run(source_id)["usage"] == before_usage
    assert before_usage[question_id][0]["total_tokens"] == 18
    assert provider.calls == before_calls
    topic = repo.topic(topic_id)
    assert topic["published_revision_id"] == recovered["target_revision_id"]
    assert repo.revision(topic_id, recovered["target_revision_id"])["state"] == "ready"
    assert repo.revision(topic_id, repo.run(source_id)["target_revision_id"])["state"] == "failed"
    snapshot = repo.snapshot(topic_id, recovered["target_revision_id"])
    result = snapshot["question_results"][question_id]
    expected_provenance = {
        "kind": "completed_checkpoint_reuse", "origin_run_id": source_id,
        "checkpoint_id": recovered["recovery"]["checkpoint_ids"][question_id],
    }
    assert result["origin_run_id"] == source_id
    assert result["execution_provenance"] == expected_provenance
    assert result["usage"] == []
    assert result["evidence"] == original["evidence"]
    assert all(output["execution_provenance"] == expected_provenance for output in snapshot["agent_outputs"])
    for evidence in snapshot["evidence"]:
        paragraph = repo.paragraph(evidence["paragraph_id"])
        assert paragraph["text"][evidence["start_cp"]:evidence["end_cp"]] == evidence["quote"]
        assert (evidence["raw_doc_id"], evidence["raw_para_id"]) == (paragraph["raw_doc_id"], paragraph["raw_para_id"])
    events = repo.events(recovered["run_id"])
    reused = [event["data"] for event in events if event["type"] == "checkpoint_reused"]
    assert len(reused) == 1 and reused[0]["source_run_id"] == source_id and reused[0]["new_model_calls"] == 0
    assert all(event["type"] not in {"node_started", "node_completed"} for event in events)
    # Idempotency is checked before the source-head guard: the successful
    # recovery is now the head, yet an exact replay returns the same result.
    repeated = await recover_completed_run(repo, source_id, "recover-complete")
    assert repeated == recovered
    assert len(repo.all("SELECT id FROM runs")) == 2


async def test_incomplete_checkpoint_refused_before_any_new_run_or_revision(tmp_path):
    repo, source_id, topic_id, _, _, _ = await failed_source(tmp_path, complete=False)
    before_usage = deepcopy(repo.run(source_id)["usage"])
    with pytest.raises(ServiceError) as failure:
        await recover_completed_run(repo, source_id, "incomplete")
    assert failure.value.status == 409
    assert failure.value.code == "recovery_checkpoint_incomplete"
    assert_single_failed_source(repo, source_id, topic_id)
    assert len(repo.all("SELECT id FROM graph_revisions")) == 1
    assert repo.run(source_id)["usage"] == before_usage
    # Every rejection must release the worker lease for subsequent explicit work.
    lease = WorkerLease(repo.settings.data_dir / "worker.lock")
    lease.acquire()
    lease.close()


async def test_live_worker_lease_refuses_recovery_without_claiming_or_recovering_anything(tmp_path):
    repo, source_id, topic_id, _, _, _ = await failed_source(tmp_path)
    lease = WorkerLease(repo.settings.data_dir / "worker.lock")
    lease.acquire()
    try:
        with pytest.raises(ServiceError) as failure:
            await recover_completed_run(repo, source_id, "busy-worker")
        assert failure.value.status == 409
        assert failure.value.code == "recovery_worker_active"
        assert_single_failed_source(repo, source_id, topic_id)
    finally:
        lease.close()


async def test_recovered_checkpoint_must_still_match_immutable_original_quote(tmp_path):
    repo, source_id, topic_id, question_id, _, _ = await failed_source(tmp_path)
    # Deliberately corrupt this isolated test checkpoint while retaining a
    # plausible same-length quote and all completed analysis channels.
    async with AsyncSqliteSaver.from_conn_string(str(repo.settings.checkpoint_path)) as saver:
        stored = await saver.aget_tuple({"configurable": {"thread_id": source_id + ":" + question_id}})
        checkpoint = deepcopy(stored.checkpoint)
        evidence = checkpoint["channel_values"]["evidence"][0]
        original = evidence["quote"]
        evidence["quote"] = original.replace("人", "名")
        assert evidence["quote"] != original and len(evidence["quote"]) == len(original)
        await saver.aput(stored.config, checkpoint, stored.metadata, {})
    with pytest.raises(ServiceError) as failure:
        await recover_completed_run(repo, source_id, "bad-source-quote")
    assert failure.value.code == "recovery_source_invalid"
    assert_single_failed_source(repo, source_id, topic_id)


@pytest.mark.parametrize("mismatch", ["config", "manifest", "question", "stage"])
async def test_checkpoint_reuse_requires_matching_config_input_question_and_failure_stage(tmp_path, mismatch):
    repo, source_id, topic_id, _, _, _ = await failed_source(
        tmp_path, question_mismatch=mismatch == "question", stage="verify" if mismatch == "stage" else "snapshot",
    )
    if mismatch == "config":
        repo.settings.model_max_output_tokens += 1
    elif mismatch == "manifest":
        with repo.connection() as db:
            db.execute("UPDATE graph_revisions SET input_manifest_sha256=? WHERE id=?", (
                "0" * 64, repo.run(source_id)["target_revision_id"],
            ))
    with pytest.raises(ServiceError) as failure:
        await recover_completed_run(repo, source_id, "mismatch")
    assert failure.value.status == 409
    assert failure.value.code == {
        "config": "recovery_config_mismatch", "manifest": "recovery_input_mismatch",
        "question": "recovery_question_mismatch", "stage": "recovery_source_not_completed",
    }[mismatch]
    assert_single_failed_source(repo, source_id, topic_id)


async def test_unrelated_queued_job_prevents_recovery_and_is_never_claimed(tmp_path):
    repo, source_id, topic_id, _, _, _ = await failed_source(tmp_path)
    text = "另一专题通报失联9人。"
    imported = import_topic(repo.settings.database_path, [{
        "doc_id": "other-doc", "title": "另一合成专题", "content": text,
        "paragraphs": [{"para_id": "p1", "char_start": 0, "char_end": len(text)}],
    }], [{"question_id": "other-question", "question": "另一通报失联人数？"}], "另一合成专题")
    other_topic = imported["topic_id"]
    other = repo.create_run(other_topic, {
        "question_ids": [repo.questions(other_topic)[0]["id"]], "base_revision_id": None,
        "mode": "initial", "include_baseline": False,
    }, "other-queued", model_available=True)
    with pytest.raises(ServiceError) as failure:
        await recover_completed_run(repo, source_id, "queue-busy")
    assert failure.value.code == "recovery_queue_busy"
    assert len(repo.all("SELECT id FROM runs")) == 2
    assert repo.run(other["run_id"])["status"] == "queued"
    assert repo.run(source_id)["status"] == "failed"
    assert repo.topic(topic_id)["head_revision_id"] == repo.run(source_id)["target_revision_id"]
    assert [event["type"] for event in repo.events(other["run_id"])] == ["run_queued"]


async def test_recovery_key_cannot_be_reused_for_a_different_operation(tmp_path):
    repo, source_id, topic_id, _, _, _ = await failed_source(tmp_path)
    # The source's own key is already bound to an ordinary initial run.
    with pytest.raises(ServiceError) as failure:
        await recover_completed_run(repo, source_id, "source")
    assert failure.value.code == "idempotency_conflict"
    assert_single_failed_source(repo, source_id, topic_id)


async def test_ordinary_worker_refuses_queued_recovery_after_explicit_process_interruption(tmp_path):
    repo, source_id, topic_id, question_id, _, _ = await failed_source(tmp_path)
    original_usage = deepcopy(repo.run(source_id)["usage"])
    recovery = repo.create_run(topic_id, {
        "question_ids": [question_id], "base_revision_id": repo.run(source_id)["target_revision_id"],
        "mode": "retry", "include_baseline": False,
    }, "interrupted-cli-recovery", model_available=True, options={
        "include_baseline": False,
        "checkpoint_recovery": {"kind": "completed_checkpoint_reuse", "source_run_id": source_id, "new_model_calls": 0},
    })
    # Model the CLI disappearing just after its durable queue insert. A normal
    # worker may claim the entry, but must never turn it into a paid model run.
    claimed = repo.claim()
    assert claimed["id"] == recovery["run_id"]
    provider = MeteredProvider()
    await Worker(repo, provider).execute(claimed)
    failed = repo.run(recovery["run_id"])
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "checkpoint_recovery_requires_cli"
    assert not provider.calls
    assert failed["usage"] is None
    assert repo.topic(topic_id)["published_revision_id"] is None
    assert repo.run(source_id)["status"] == "failed"
    assert repo.run(source_id)["usage"] == original_usage
    assert repo.all("SELECT id FROM agent_executions WHERE run_id=?", (recovery["run_id"],)) == []
