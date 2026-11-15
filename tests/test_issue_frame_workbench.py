"""Directional scope integration using synthetic text and explicit local mocks.

These are transport/publication contracts, not real UI or semantic quality
evidence. No reference files, account calls, or historical answers are read.
"""
from copy import deepcopy
from hashlib import sha256
import json
from unittest.mock import Mock

import httpx
import pytest

from app.agents.schemas import ArbitrationOutput
from app.agents.scoped_arbitration import (
    ISSUE_FRAME_VERSION, CONCLUSION_FRAME_VERSION, ScopedArbitrationError, canonical_scoped_arbitration,
    validate_scoped_result,
)
from app.agents import strict_transport as transport
from app.config import Settings
from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL, build_snapshot, competition_files
from app.repository import Repository, dumps
from scripts import score_workbench_real_episode as scorer
from scripts.score_bundle_workbench_real import _selection_mode
from test_snapshot import fixture as snapshot_fixture
from test_strict_request_projection import input_payload


def v2_wire(question, aliases):
    return {"decision": "undetermined", "conclusion": "通报记载的口径尚待核验，应保留现有两方材料及其限定。",
        "issue_frame": {"question_anchor": question["question"], "proposition": "现有通报采用所列失联人数口径。",
            "claim_kind": "attribution", "scope": "仅限所给通报及其时点，不合并其他统计范围。"},
        "judgments": [], "conflicts": [],
        "relations": {alias: ("QS" if i == 0 else "QO" if i == 1 else "U")
            for i, alias in enumerate(aliases)}}


def remapped_scope(question, aliases, mapping):
    final, scope = canonical_scoped_arbitration(v2_wire(question, aliases), aliases,
        question=question, issue_frame=True, conclusion_aligned=False)
    final, scope = transport._remap(final, mapping), transport._remap(scope, mapping)
    scope["evidence_order_sha256"] = scorer.canonical_digest(scope["evidence_order"])
    return final, scope


def publication_fixture():
    topic, questions, results = snapshot_fixture()
    question = questions[0]; result = results[question["id"]]
    first = result["evidence"][0]
    first["id"] = "z-evidence-service-id"
    second = deepcopy(first)
    second.update(id="a-evidence-service-id", paragraph_id="second-original-paragraph",
        raw_doc_id="doc_002", raw_para_id="doc_002#p1", source_group_id="source_2")
    # Model E order deliberately differs from public/service UUID ordering.
    result["evidence"] = [first, second]
    mapping = {"E1": first["id"], "E2": second["id"]}
    final, scope = remapped_scope(question, list(mapping), mapping)
    result.update(final=final, evidence_relation_scope=scope)
    for analysis in result["analyses"]:
        analysis.update({key: deepcopy(value) for key, value in final.items() if key != "conflicts"})
    return topic, questions, results


def build(topic, questions, results):
    return build_snapshot(topic, questions, results, revision_id="synthetic-revision",
        graph_revision=1, run_id="explicit-synthetic-run")


def test_v2_snapshot_and_competition_export_preserve_reverse_qualification_and_model_ledger():
    topic, questions, results = publication_fixture()
    result = results[questions[0]["id"]]
    result["evidence"].sort(key=lambda row: row["id"])
    before = deepcopy(results)
    snapshot = build(topic, questions, results)
    files = competition_files(snapshot)
    scope = snapshot["conclusions"][0]["evidence_relation_scope"]
    assert scope["evidence_order"] == ["z-evidence-service-id", "a-evidence-service-id"]
    assert scope["evidence_order_sha256"] == scorer.canonical_digest(scope["evidence_order"])
    assert scope["qualification_support_evidence_ids"] == ["z-evidence-service-id"]
    assert scope["qualification_oppose_evidence_ids"] == ["a-evidence-service-id"]
    conclusion = files["conclusions.json"][0]
    assert [row["doc_id"] for row in conclusion["supporting_facts"]] == ["doc_001"]
    assert [row["doc_id"] for row in conclusion["conflicting_facts"]] == ["doc_002"]
    assert results == before


@pytest.mark.parametrize("boundary", ["build", "export"])
def test_snapshot_boundaries_reject_question_text_drift_even_when_anchor_still_occurs(boundary):
    topic, questions, results = publication_fixture()
    if boundary == "build":
        questions[0]["question"] += " 请说明时点。"
        with pytest.raises(ScopedArbitrationError, match="question_binding_changed"):
            build(topic, questions, results)
    else:
        snapshot = build(topic, questions, results)
        snapshot["conclusions"][0]["question"] += " 请说明时点。"
        with pytest.raises(ScopedArbitrationError, match="question_binding_changed"):
            competition_files(snapshot)


def repo_gate(issue, scope, final, question, *, front_issue=None):
    repo = object.__new__(Repository)
    cfg = {"source_bundle_enabled": True, "source_bundle_protocol": SOURCE_BUNDLE_PROTOCOL,
        "prompt_version": "trusted-analysis-langchain-source-bundles-v1", "source_strategy": "langchain_source_book",
        "source_bundle_config": {"front_framework": "langchain", **({"issue_frame": True} if issue else {})}}
    proof = {"format": "langchain-source-bundle-workbench-proof-v1"}
    if front_issue is None: front_issue = issue
    if front_issue: proof["issue_frame"] = front_issue
    origin = {"topic_id": "t", "question_ids_json": dumps([question["id"]]), "model_config_json": dumps(cfg)}
    repo.one = Mock(side_effect=lambda sql, *_args, **_kwargs: deepcopy(question) if "FROM questions" in sql else origin)
    repo.source_documents = Mock(return_value=[]); repo.paragraphs = Mock(return_value=[])
    result = {"source_bundle_protocol": SOURCE_BUNDLE_PROTOCOL, "evidence": [], "final": final,
        "evidence_relation_scope": scope, "source_bundle_proof": {"response": {"proof": proof}}}
    return repo, result


def test_publication_uses_trusted_database_question_and_not_self_declared_scope_hash(monkeypatch):
    from app.workflows import bundle_workbench
    question = {"id": "q", "question": "项目是否获批准？"}
    final, scope = remapped_scope(question, [], {})
    repo, result = repo_gate(True, scope, final, question)
    source_replay = Mock(side_effect=lambda value, *_args: value)
    monkeypatch.setattr(bundle_workbench, "validate_result", source_replay)
    repo._validate_result_protocol(result, "run", "t", "q")
    assert source_replay.call_args.args[1] == question
    changed = {**question, "question": question["question"] + " 请说明时点。"}
    repo, result = repo_gate(True, scope, final, changed)
    with pytest.raises(ScopedArbitrationError, match="question_binding_changed"):
        repo._validate_result_protocol(result, "run", "t", "q")


def test_v2_scope_cannot_be_introduced_under_frozen_v1_run_or_front_proof(monkeypatch):
    from app.workflows import bundle_workbench
    question = {"id": "q", "question": "项目是否获批准？"}
    final, scope = remapped_scope(question, [], {})
    repo, result = repo_gate(False, scope, final, question)
    monkeypatch.setattr(bundle_workbench, "validate_result", lambda value, *_args: value)
    with pytest.raises(ValueError, match="issue frame|directional"):
        repo._validate_result_protocol(result, "run", "t", "q")


@pytest.mark.parametrize("change", ["proof_missing", "public_missing", "public_false", "frozen_false", "integer_mode"])
def test_offline_selection_mode_requires_frozen_front_and_public_issue_flag(change):
    proof = {"issue_frame": True}; frozen = {"issue_frame": True}; run = {"source_bundle_issue_frame": True}
    if change == "proof_missing": proof.pop("issue_frame")
    elif change == "public_missing": run.pop("source_bundle_issue_frame")
    elif change == "public_false": run["source_bundle_issue_frame"] = False
    elif change == "frozen_false": frozen.pop("issue_frame")
    elif change == "integer_mode": proof["issue_frame"] = 1
    with pytest.raises(ValueError, match="frozen_selection_mode_changed"):
        _selection_mode(proof, frozen, run)


@pytest.mark.asyncio
async def test_v3_shared_native_request_equals_http_mock_and_scoper_replays_mapped_ledger(tmp_path, monkeypatch):
    """One explicit mock tail call; no front/real-workbench claim is made."""
    payload = input_payload()
    settings = Settings(_env_file=None, data_dir=tmp_path, model_provider="deepseek", model_name="deepseek-v4-pro",
        model_api_key="explicit-offline-http-mock", model_source_bundle_enabled=True, source_bundle_front_framework="langchain",
        strict_evidence_max_requests=10, strict_evidence_balanced_enabled=True,
        source_bundle_compact_analyses_enabled=True, source_bundle_issue_frame_enabled=True)
    provider = transport.StrictWorkbenchProvider(settings)
    observed = []
    def handle(request):
        body = json.loads(request.content); observed.append(body)
        view = json.loads(body["messages"][-1]["content"])
        aliases = [row["id"] for row in view["evidence"]]
        output = v2_wire(view["question"], aliases)
        output["issue_frame"].pop("proposition")
        output["issue_frame"].update(alternatives={"X": "", "Y": ""}, alternatives_relation="not_binary")
        output["relations"] = {key: {"assessment": role + ("_supporting_limit" if role == "QS" else "_limiting_counterclaim" if role == "QO" else "_background"), "side": "neither"}
            for key, role in output["relations"].items()}
        return httpx.Response(200, json={"id": "explicit-local-mock", "object": "chat.completion", "created": 1,
            "model": "deepseek-v4-pro", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "tool_calls": [{"id": "mock-v2-call", "type": "function",
                    "function": {"name": "ScopedArbitrationWireOutput", "arguments": json.dumps(output, ensure_ascii=False)}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
    monkeypatch.setattr(transport.httpx, "AsyncHTTPTransport", lambda **_kwargs: httpx.MockTransport(handle))
    class ExplicitAudit(transport.AuditedTransport, httpx.MockTransport):
        pass
    monkeypatch.setattr(transport, "AuditedTransport", ExplicitAudit)
    monkeypatch.setattr(transport, "count_wire", lambda body, directory, ordinal: (100,
        transport.freeze(directory / f"request-{ordinal:02d}-wire.json", body)))
    prepared = transport.prepare_strict_request("arbitrate", "保留完整来源与两路判断。", payload, ArbitrationOutput,
        balanced_scoped_enabled=True, compact_analyses_enabled=True, issue_frame_enabled=True)
    planned = transport.build_strict_request_payload(prepared, max_output_tokens=provider.output_limits["arbitrate"])
    output = await provider._request("arbitrate", "保留完整来源与两路判断。", payload, ArbitrationOutput)
    assert len(observed) == 1 and observed[0] == planned["body"]
    assert json.loads(planned["sdk_preview_body_json"]) == observed[0]
    assert len(output["conflicting_evidence_ids"]) == 1
    scope = provider.arbitration_scope
    assert scope["format"] == CONCLUSION_FRAME_VERSION
    assert scope["evidence_order_sha256"] == scorer.canonical_digest(scope["evidence_order"])
    # Publication sorts service IDs independently from the model E ledger.
    validate_scoped_result(scope, output, sorted(scope["evidence_order"]), question=payload["question"])
    response_path = provider.directory / "request-01-response.json"
    response = json.loads(response_path.read_text("utf8"))
    projection = json.loads((provider.directory / "request-01-relation-projection.json").read_text("utf8"))
    sources = {name: sha256((transport.ROOT / name).read_bytes()).hexdigest()
        for name in ("app/agents/scoped_arbitration.py", "app/agents/schemas.py")}
    native = {"logical_operations": ["arbitrate"], "relation_projections": [projection],
        "response_file_sha256": [sha256(response_path.read_bytes()).hexdigest()], "runtime_sources_sha256": sources}
    result = {"verification_protocol": "strict-workbench-evidence-balanced-v2", "evidence": payload["evidence"],
        "final": output, "evidence_relation_scope": scope}
    audit = scorer.verify_scoped_relations(result, observed, [response], native, question_internal_id=payload["question"]["id"])
    assert audit["visible_actual_model_wire_bound"] is True and audit["semantic_target_verified"] is False
    changed = deepcopy(result); changed["evidence_relation_scope"]["evidence_order"].reverse()
    changed["evidence_relation_scope"]["evidence_order_sha256"] = scorer.canonical_digest(changed["evidence_relation_scope"]["evidence_order"])
    with pytest.raises(ValueError, match="differ_from_actual_model_projection"):
        scorer.verify_scoped_relations(changed, observed, [response], native)
