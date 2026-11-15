"""Cost-only ancestry across schema changes; all counters/providers synthetic."""
from copy import deepcopy

import pytest

from app.agents.provider import ModelOutputInvalid
from app.config import Settings
from app.data.ingest import import_topic
from app.repository import Repository
from app.worker import Worker
from app.workflows.strict_workbench import PROMPT_VERSION


class SeedOnlyProvider:
    strict_evidence_enabled = True
    strict_candidate_limit = 60
    paragraph_ranked_enabled = premise_bank_enabled = False
    prompt_version = PROMPT_VERSION

    def __init__(self):
        self.seeded = []

    def seed_prior_usage(self, records):
        self.seeded.append(deepcopy(records))


def setup(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="",
                        model_strict_evidence_enabled=True)
    repo = Repository(settings)
    text = "甲设备批准尚未确认。"
    docs = [{"doc_id": "d1", "title": "甲设备", "content": text,
             "paragraphs": [{"para_id": "p1", "char_start": 0, "char_end": len(text)}]}]
    questions = [{"question_id": "q1", "question": "甲设备是否已经批准？"},
                 {"question_id": "q2", "question": "乙设备是否已经批准？"}]
    topic = import_topic(settings.database_path, docs, questions, "合成费用链专题")["topic_id"]
    return repo, topic, [row["id"] for row in repo.questions(topic)]


def queued(repo, topic, qid, key):
    return repo.create_run(topic, {"base_revision_id": repo.topic(topic)["head_revision_id"], "question_ids": [qid],
                           "mode": "initial", "include_baseline": False}, key, model_available=True)


def prior(repo, topic, qid, key, *, unknown=False, amount=12):
    run = queued(repo, topic, qid, key); repo.claim()
    repo.record_usage(run["id"], qid, [{"operation": "strict_selection", "usage_source": "unavailable" if unknown else "provider",
        "input_tokens": None if unknown else amount-2, "output_tokens": None if unknown else 2,
        "total_tokens": None if unknown else amount}])
    repo.finish_failed(run["id"], "synthetic_contract_failure", "显式合成费用链失败。", stage="strict_selection")
    return run["id"]


def intercept(monkeypatch):
    invocations = []
    async def stop_before_any_model(*args, **kwargs):
        invocations.append(kwargs)
        raise ModelOutputInvalid("显式测试：在任何模型调用前结束。", stage="strict_selection", code="synthetic_stop")
    monkeypatch.setattr("app.worker.run_analysis", stop_before_any_model)
    return invocations


async def test_cost_parent_deduplicates_transitive_profile_parents_and_never_reuses_checkpoint(tmp_path, monkeypatch):
    repo, topic, qids = setup(tmp_path); first = prior(repo, topic, qids[0], "first", amount=12)
    repo.settings = repo.settings.model_copy(update={"strict_evidence_cost_parent_run_ids": [first]})
    second = prior(repo, topic, qids[0], "second", amount=18)
    repo.settings = repo.settings.model_copy(update={"strict_evidence_cost_parent_run_ids": [second, first, second],
        "strict_evidence_max_requests": 6, "strict_evidence_token_stop": 60000})
    calls = intercept(monkeypatch); provider = SeedOnlyProvider()
    current = queued(repo, topic, qids[0], "fresh-changed-schema"); await Worker(repo, provider).execute(repo.claim())
    assert len(provider.seeded) == 1 and len(provider.seeded[0]) == 2
    assert sum(row["total_tokens"] for row in provider.seeded[0]) == 30
    assert len(calls) == 1 and calls[0]["resume_from_thread_id"] is None
    events = [row["data"] for row in repo.events(current["id"]) if row["type"] == "cost_lineage_seeded"]
    assert len(events) == 1 and events[0]["prior_run_ids"] == sorted([first, second])
    assert events[0]["prior_known_tokens"] == 30 and events[0]["prior_calls"] == 2 and events[0]["checkpoint_results_reused"] is False
    assert repo.run(current["id"])["error"]["code"] == "synthetic_stop"
    config = repo.config()
    assert config["strict_evidence_max_requests"] == 6 and config["strict_evidence_token_stop"] == 60000


@pytest.mark.parametrize("bad", ["other_question", "source_manifest", "unknown_usage"])
async def test_cost_parent_binding_or_unknown_usage_blocks_before_provider_or_graph(tmp_path, monkeypatch, bad):
    repo, topic, qids = setup(tmp_path)
    parent = prior(repo, topic, qids[1] if bad == "other_question" else qids[0], "parent", unknown=bad == "unknown_usage")
    if bad == "source_manifest":
        with repo.connection() as db:
            db.execute("UPDATE graph_revisions SET input_manifest_sha256=? WHERE id=(SELECT target_revision_id FROM runs WHERE id=?)", ("f"*64, parent))
    repo.settings = repo.settings.model_copy(update={"strict_evidence_cost_parent_run_ids": [parent]})
    calls = intercept(monkeypatch); provider = SeedOnlyProvider(); current = queued(repo, topic, qids[0], "blocked-child")
    await Worker(repo, provider).execute(repo.claim())
    assert provider.seeded == [] and calls == []
    expected = {"other_question": "strict_cost_parent_binding", "source_manifest": "strict_cost_parent_source_changed",
                "unknown_usage": "strict_cost_parent_usage_unknown"}[bad]
    assert repo.run(current["id"])["error"]["code"] == expected


def test_cost_controls_preserve_defaults_and_validate_explicit_profile_limits():
    settings = Settings(_env_file=None)
    assert settings.strict_evidence_cost_parent_run_ids == []
    assert settings.strict_evidence_max_requests == 5 and settings.strict_evidence_token_stop == 40000
    with pytest.raises(ValueError): Settings(_env_file=None, strict_evidence_cost_parent_run_ids=[""])
    with pytest.raises(ValueError): Settings(_env_file=None, strict_evidence_max_requests=7)
    assert Settings(_env_file=None, strict_evidence_token_stop=75000).strict_evidence_token_stop == 75000
    assert Settings(_env_file=None, strict_evidence_token_stop=120000).strict_evidence_token_stop == 120000
    with pytest.raises(ValueError): Settings(_env_file=None, strict_evidence_token_stop=120001)
