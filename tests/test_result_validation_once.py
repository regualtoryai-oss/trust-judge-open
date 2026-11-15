"""Request-local replay deduplication over synthetic SQLite results only."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL, build_snapshot
from app.errors import ServiceError
from app.repository import Repository, dumps
from test_repository import candidate, create, publish, setup_repo


def spy_protocol(repo, monkeypatch):
    spy = Mock(wraps=repo._validate_result_protocol)
    monkeypatch.setattr(repo, "_validate_result_protocol", spy)
    return spy


def claimed(repo, topic):
    run = create(repo, topic)
    assert repo.claim()["id"] == run["id"]
    return run


def test_publish_replays_identical_fresh_and_snapshot_result_once_in_original_transaction(tmp_path, monkeypatch):
    repo, topic = setup_repo(tmp_path); run = claimed(repo, topic)
    snapshot, fresh = candidate(repo, topic, run)
    original = repo._validate_result_protocol; transactions = []
    def audited(*args, **kwargs):
        transactions.append(kwargs["db"].in_transaction)
        return original(*args, **kwargs)
    spy = Mock(side_effect=audited)
    monkeypatch.setattr(repo, "_validate_result_protocol", spy)
    repo.publish(run["id"], snapshot, fresh)
    assert spy.call_count == 1
    assert spy.call_args.args[1:] == (run["id"], topic, run["question_ids"][0])
    assert transactions == [True]
    assert repo.run(run["id"])["status"] == "succeeded"
    assert repo.run(run["id"])["source_bundle_complete_grading"] is False
    assert repo.run(run["id"])["source_bundle_hybrid_retrieval"] is False


def test_each_submission_request_rechecks_protocol_and_never_reuses_across_exports(tmp_path, monkeypatch):
    repo, topic = setup_repo(tmp_path); run, _snapshot = publish(repo, topic)
    spy = spy_protocol(repo, monkeypatch)
    first = repo.submission_projection(topic, run["target_revision_id"])
    assert spy.call_count == 1 and spy.call_args.kwargs["db"] is None
    assert repo.submission_projection(topic, run["target_revision_id"]) == first
    assert spy.call_count == 2
    # A new dependency failure at the same complete validation boundary must
    # be observed by the next request, even for the same immutable revision.
    spy.side_effect = ValueError("synthetic-proof-dependency-changed")
    with pytest.raises(ServiceError) as error:
        repo.submission_projection(topic, run["target_revision_id"])
    assert error.value.code == "export_validation_failed" and spy.call_count == 3


def test_inherited_result_keeps_its_actual_origin_and_fresh_current_origin_runs_first(tmp_path, monkeypatch):
    repo, topic = setup_repo(tmp_path, question_count=2)
    old_run, old_snapshot = publish(repo, topic)
    q1, q2 = repo.questions(topic)
    run = create(repo, topic, selected=[q1["id"]]); assert repo.claim()["id"] == run["id"]
    _partial, fresh = candidate(repo, topic, run)
    results = deepcopy(old_snapshot["question_results"]); results.update(fresh)
    revision = repo.revision(topic, run["target_revision_id"])
    snapshot = build_snapshot(repo.topic(topic), [q1, q2], results, revision_id=revision["id"],
        graph_revision=revision["revision_no"], run_id=run["id"])
    snapshot["scope"] = {"completed_question_ids": list(results), "all_question_count": 2, "is_complete_topic": True}
    spy = spy_protocol(repo, monkeypatch)
    repo.publish(run["id"], snapshot, fresh)
    assert [(call.args[1], call.args[3]) for call in spy.call_args_list] == [
        (run["id"], q1["id"]), (old_run["id"], q2["id"])]
    spy.reset_mock()
    repo.submission_projection(topic, run["target_revision_id"])
    assert {(call.args[1], call.args[3]) for call in spy.call_args_list} == {
        (run["id"], q1["id"]), (old_run["id"], q2["id"])}
    assert spy.call_count == 2


def test_changing_fresh_origin_cannot_escape_validation_under_the_current_frozen_run(tmp_path, monkeypatch):
    repo, topic = setup_repo(tmp_path); old_run, _ = publish(repo, topic)
    config = repo.config()
    # An explicit synthetic frozen configuration requiring a protocol that
    # the legacy result does not have; no provider is created or invoked.
    repo.config = lambda: {**config, "strict_evidence_enabled": True}
    run = claimed(repo, topic); snapshot, fresh = candidate(repo, topic, run)
    qid = run["question_ids"][0]
    fresh[qid]["origin_run_id"] = old_run["id"]
    snapshot["question_results"][qid]["origin_run_id"] = old_run["id"]
    spy = spy_protocol(repo, monkeypatch)
    with pytest.raises(ValueError, match="strict protocol required"):
        repo.publish(run["id"], snapshot, fresh)
    assert spy.call_count == 1 and spy.call_args.args[1] == run["id"]
    assert repo.revision(topic, run["target_revision_id"])["state"] == "building"
    assert repo.topic(topic)["published_revision_id"] == old_run["target_revision_id"]


def test_changed_fresh_payload_still_fails_snapshot_equality_before_replay(tmp_path, monkeypatch):
    repo, topic = setup_repo(tmp_path); run = claimed(repo, topic)
    snapshot, fresh = candidate(repo, topic, run)
    fresh[run["question_ids"][0]]["final"]["conclusion"] += "被替换内容。"
    spy = spy_protocol(repo, monkeypatch)
    with pytest.raises(ValueError, match="changed a fresh question result"):
        repo.publish(run["id"], snapshot, fresh)
    assert spy.call_count == 0


@pytest.mark.parametrize("change", ["origin", "question", "payload"])
def test_request_cache_does_not_reuse_different_authorization_or_result_digest(change):
    repo = object.__new__(Repository)
    repo._validate_result_protocol = Mock()
    marker = object(); validate = repo._request_result_validator("topic", db=marker)
    result = {"final": {"conclusion": "原有内容。"}, "proof": {"dependency_sha256": "a" * 64}}
    validate(result, "origin", "question")
    validate(deepcopy(result), "origin", "question")
    assert repo._validate_result_protocol.call_count == 1
    origin, question, changed = "origin", "question", deepcopy(result)
    if change == "origin": origin = "other-origin"
    elif change == "question": question = "other-question"
    else: changed["proof"]["dependency_sha256"] = "b" * 64
    validate(changed, origin, question)
    assert repo._validate_result_protocol.call_count == 2
    assert repo._validate_result_protocol.call_args.kwargs["db"] is marker
    repo._request_result_validator("topic", db=marker)(result, "origin", "question")
    assert repo._validate_result_protocol.call_count == 3


def test_failed_validation_is_never_cached():
    repo = object.__new__(Repository)
    repo._validate_result_protocol = Mock(side_effect=[ValueError("dependency"), None])
    validate = repo._request_result_validator("topic")
    with pytest.raises(ValueError): validate({}, "origin", "question")
    validate({}, "origin", "question")
    assert repo._validate_result_protocol.call_count == 2


@pytest.mark.parametrize("feature", ["complete_grading", "hybrid_retrieval"])
@pytest.mark.parametrize("change", [None, "missing_mode", "wrong_type", "policy", "unfrozen", "no_context", "missing_policy"])
def test_derived_context_mode_is_bound_to_frozen_context_policy(monkeypatch, change, feature):
    from app.workflows import bundle_workbench
    repo = object.__new__(Repository)
    policy = {"version": "synthetic-" + feature + "-v1"}
    policy_key = feature + "_policy"
    mode = {"front_framework": "langchain", "context_passages": True,
        feature: True, policy_key: deepcopy(policy)}
    proof = {"format": "langchain-source-bundle-workbench-proof-v1", "context_passages": True,
        "source_bundle_protocol": SOURCE_CONTEXT_PROTOCOL, feature: True, policy_key: deepcopy(policy)}
    protocol = SOURCE_CONTEXT_PROTOCOL
    if change == "missing_mode": proof.pop(feature)
    elif change == "wrong_type": proof[feature] = 1
    elif change == "policy": proof[policy_key]["version"] = "changed"
    elif change == "unfrozen": mode.pop(feature)
    elif change == "missing_policy": mode.pop(policy_key); proof.pop(policy_key)
    elif change == "no_context":
        mode.pop("context_passages"); proof.pop("context_passages"); protocol = SOURCE_BUNDLE_PROTOCOL
        proof["source_bundle_protocol"] = protocol
    cfg = {"source_bundle_enabled": True, "source_bundle_protocol": protocol,
        "prompt_version": "trusted-analysis-langchain-source-bundles-v1", "source_strategy": "langchain_source_book", "source_bundle_config": mode}
    origin = {"topic_id": "t", "question_ids_json": dumps(["q"]), "model_config_json": dumps(cfg)}
    repo.one = Mock(side_effect=lambda sql, *_a, **_kw: {"id": "q", "question": "合成原题？"} if "FROM questions" in sql else origin)
    repo.source_documents = Mock(return_value=[]); repo.paragraphs = Mock(return_value=[])
    result = {"source_bundle_protocol": protocol, "evidence": [], "source_bundle_proof": {"response": {"proof": proof}}}
    source_replay = Mock(); monkeypatch.setattr(bundle_workbench, "validate_result", source_replay)
    if change is None:
        repo._validate_result_protocol(result, "run", "t", "q")
        assert source_replay.call_count == 1
    else:
        with pytest.raises(ValueError, match=feature.replace("_", " ")):
            repo._validate_result_protocol(result, "run", "t", "q")
        assert source_replay.call_count == 0


def test_public_run_reports_frozen_hybrid_and_complete_modes_without_importing_new_ranker(tmp_path):
    repo, topic = setup_repo(tmp_path)
    original = repo.config()
    repo.config = lambda: {**original, "source_bundle_config": {"hybrid_retrieval": True, "complete_grading": True,
        "hybrid_retrieval_policy": {"version": "synthetic-hybrid-v1"},
        "complete_grading_policy": {"version": "synthetic-complete-v1"}}}
    run = create(repo, topic)
    assert run["source_bundle_hybrid_retrieval"] is True
    assert run["source_bundle_complete_grading"] is True
