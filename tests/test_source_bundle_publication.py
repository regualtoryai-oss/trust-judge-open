"""Synthetic source/publication contracts; no model, reference answers or quality claims."""
from copy import deepcopy
from typing import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
import pytest

from app.config import Settings
from app.data.ingest import import_topic
from app.domain.material_eligibility import apply_workflow_material_eligibility, validate_workflow_material_eligibility
from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL, build_snapshot, competition_files
from app.domain.source_bundles import SOURCE_BUNDLE_CORE_FIELDS, assemble_source_bundle, build_source_bundles
from app.errors import ServiceError
from app.recovery import recover_completed_run
from app.repository import Repository


def fixture(tmp_path, *, enabled=True, cutoff=False):
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="",
                        model_premise_bank_enabled=False, model_strict_evidence_enabled=False,
                        model_paragraph_ranked_enabled=False)
    pieces = ["😀甲方声明：已完成检查。", "限定：结果仍未最终确认。", "补充记录。"]
    content, paragraphs = "前缀|", []
    for index, piece in enumerate(pieces):
        start = len(content)
        content += piece
        paragraphs.append({"para_id": ["原段-Z", "opaque-A", "另一个原段"][index], "char_start": start, "char_end": len(content)})
        if index < len(pieces) - 1:
            content += ["\r\n<真实分隔>", "\n\n"][index]
    question = "材料是否表明检查已完成且结果最终确定？"
    if cutoff:
        question = "仅依据截至2026-10-20T12:00:00+08:00已发布的材料，" + question
    imported = import_topic(settings.database_path, [{"doc_id": "原文档", "title": "合成原始记录", "content": content,
        "publish_time": "2026-10-19T12:00:00+08:00", "paragraphs": paragraphs}],
        [{"question_id": "q_001", "question": question}], "合成范围发布测试")
    repo = Repository(settings)
    original_config = repo.config
    repo.config = lambda: {**original_config(), "strict_scope_enabled": False,
        **({"source_bundle_protocol": SOURCE_BUNDLE_PROTOCOL} if enabled else {})}
    topic_id = imported["topic_id"]
    return repo, topic_id, repo.questions(topic_id)[0]


def result_for(repo, topic_id, question, *, ranges=((0, 1),), summaries=None):
    documents, parents = repo.source_documents(topic_id), repo.paragraphs(topic_id)
    by_id = {row["id"]: row for row in parents}
    ordered = [by_id[member["id"]] for doc in documents for member in doc["paragraph_manifest"]]
    evidence = []
    for position, (start, end) in enumerate(ranges):
        bundle = assemble_source_bundle(documents, parents, ordered[start]["id"], ordered[end]["id"])
        evidence.append({**bundle, "id": "bundle-evidence-" + bundle["bundle_id"] + f"-{position}", "fact_des": summaries[position] if summaries else "记录检查完成并保留未最终确认的限定。",
            "argument": {"时间": "", "主体": "甲方", "地点": "", "内容": "检查完成，结果未最终确认", "影响": "", "数值口径": ""},
            "entities": ["甲方"], "source_group_id": "same-original-document", "source_group_basis": "same_document_not_independent_votes",
            "grade_score": None, "source_level": "unknown", "annotation_semantics_verified": False})
    qualified, material = apply_workflow_material_eligibility(question, parents, evidence,
        source_documents=documents, source_bundle_protocol=SOURCE_BUNDLE_PROTOCOL)
    ids = [row["id"] for row in qualified]
    final = {"decision": "asserted", "conclusion": "材料记载检查完成，但结果仍未最终确认。",
        "judgments": [{"id": "range-judgment", "statement": "原始材料同时保留陈述和限定", "evidence_ids": ids, "prerequisite_judgment_ids": []}],
        "supporting_evidence_ids": ids, "conflicting_evidence_ids": [], "conflicts": []}
    analyses = [{"role": role, "evidence_snapshot_id": "synthetic-frozen-bundle", **deepcopy(final)}
        for role in ("authority_analyst", "cross_validation_analyst")]
    return {"source_bundle_protocol": SOURCE_BUNDLE_PROTOCOL, "evidence": qualified, "material_eligibility": material,
        "analyses": analyses, "final": final, "evidence_snapshot_id": "synthetic-frozen-bundle", "usage": []}


def create_and_build(repo, topic_id, question, result):
    run = repo.create_run(topic_id, {"question_ids": [question["id"]], "base_revision_id": None,
        "mode": "initial", "include_baseline": False}, "synthetic-source-bundle", model_available=True)
    assert repo.claim()["id"] == run["run_id"]
    snapshot = build_snapshot(repo.topic(topic_id), [question], {question["id"]: result},
        revision_id=run["target_revision_id"], graph_revision=1, run_id=run["run_id"],
        source_documents=repo.source_documents(topic_id), source_paragraphs=repo.paragraphs(topic_id))
    snapshot["scope"] = {"completed_question_ids": [question["id"]], "all_question_count": 1, "is_complete_topic": True}
    return run, snapshot


def test_document_range_publishes_and_exports_exact_slice_with_all_source_fields(tmp_path):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question)
    original = deepcopy(result)
    run, snapshot = create_and_build(repo, topic_id, question, result)
    repo.publish(run["run_id"], snapshot, {question["id"]: result})
    stored = repo.snapshot(topic_id, run["target_revision_id"])
    row = stored["evidence"][0]
    for field in SOURCE_BUNDLE_CORE_FIELDS:
        assert row[field] == original["evidence"][0][field]
        assert stored["question_results"][question["id"]]["evidence"][0][field] == row[field]
        assert next(n for n in stored["nodes"] if n["kind"] == "evidence")["content"][field] == row[field]
    assert row["span_scope"] == "document" and row["start_cp"] > 0
    assert "\r\n<真实分隔>" in row["quote"]
    assert repo.run(run["run_id"])["status"] == "succeeded"
    files = repo.formal_submission_files(topic_id, run["target_revision_id"])
    assert files["evidence.json"][0]["quote"] == row["quote"]
    assert files["conclusions.json"][0]["supporting_facts"][0]["quote"] == row["quote"]
    assert files["confidence.json"][0]["object_ref"]["quote"] == row["quote"]
    assert repo.paragraph(row["paragraph_id"])["text"] != row["quote"]
    assert result == original


def test_same_first_parent_different_end_ranges_are_rejected_without_widening_or_duplicate_fact(tmp_path):
    repo, topic_id, question = fixture(tmp_path)
    with pytest.raises(ValueError, match="eligibility_source_bundle_contract"):
        result_for(repo, topic_id, question, ranges=((0, 1), (0, 2)))
    assert repo.topic(topic_id)["published_revision_id"] is None
    assert repo.all("SELECT id FROM evidence") == []
    assert repo.all("SELECT id FROM runs") == []


def test_same_range_annotations_consolidate_into_one_official_anchor(tmp_path):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question, ranges=((0, 1), (0, 1)), summaries=["记录检查完成。", "保留结果尚未最终确认的限定。"])
    run, snapshot = create_and_build(repo, topic_id, question, result)
    repo.publish(run["run_id"], snapshot, {question["id"]: result})
    files = repo.formal_submission_files(topic_id, run["target_revision_id"])
    assert len(files["evidence.json"]) == 1
    assert len({(row["doc_id"], row["para_id"]) for row in files["evidence.json"]}) == 1
    assert {row["quote"] for row in files["evidence.json"]} == {row["quote"] for row in result["evidence"]}
    assert files["evidence.json"][0]["fact_des"] == "记录检查完成。；保留结果尚未最终确认的限定。"
    assert len(files["conclusions.json"][0]["supporting_facts"]) == 1
    assert len([row for row in files["confidence.json"] if row["object_type"] == "evidence"]) == 1


def test_anchor_range_collision_is_rejected_across_separately_valid_question_results(tmp_path):
    from app.domain.snapshot import validate_source_bundle_rows
    repo, topic_id, question = fixture(tmp_path)
    narrow = result_for(repo, topic_id, question, ranges=((0, 1),))
    wider = result_for(repo, topic_id, question, ranges=((0, 2),))
    with pytest.raises(ValueError, match="one original anchor"):
        validate_source_bundle_rows(narrow["evidence"] + wider["evidence"], protocol=SOURCE_BUNDLE_PROTOCOL,
            source_documents=repo.source_documents(topic_id), source_paragraphs=repo.paragraphs(topic_id))


def test_explicit_bundle_protocol_retains_legacy_local_evidence_projection(tmp_path):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question)
    bundle = result["evidence"][0]
    parent = repo.paragraph(repo.source_documents(topic_id)[0]["paragraph_manifest"][-1]["id"])
    legacy = {key: deepcopy(value) for key, value in bundle.items() if key not in SOURCE_BUNDLE_CORE_FIELDS and key != "program_material_eligibility"}
    legacy.update(id="legacy-local-evidence", paragraph_id=parent["id"], raw_doc_id=parent["raw_doc_id"],
        raw_para_id=parent["raw_para_id"], quote=parent["text"], start_cp=0, end_cp=len(parent["text"]))
    result["evidence"], result["material_eligibility"] = apply_workflow_material_eligibility(question, repo.paragraphs(topic_id),
        [*result["material_eligibility"]["input_evidence"], legacy], source_documents=repo.source_documents(topic_id),
        source_bundle_protocol=SOURCE_BUNDLE_PROTOCOL)
    result["final"]["supporting_evidence_ids"].append(legacy["id"])
    run, snapshot = create_and_build(repo, topic_id, question, result)
    repo.publish(run["run_id"], snapshot, {question["id"]: result})
    files = repo.formal_submission_files(topic_id, run["target_revision_id"])
    assert {row["quote"] for row in files["evidence.json"]} == {bundle["quote"], parent["text"]}


@pytest.mark.parametrize("field,value", [
    ("quote", "人为拼接的正文"), ("start_cp", 0), ("source_paragraph_ids", ["missing-middle"]),
    ("document_content_sha256", "0" * 64), ("paragraph_manifest_sha256", "0" * 64), ("bundle_id", "changed"),
])
def test_bundle_source_tampering_is_rejected_before_publication(tmp_path, field, value):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question)
    result["evidence"][0][field] = value
    with pytest.raises(ValueError):
        create_and_build(repo, topic_id, question, result)
    assert repo.topic(topic_id)["published_revision_id"] is None


def test_explicit_protocol_and_full_raw_documents_are_mandatory(tmp_path):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question)
    result.pop("source_bundle_protocol")
    with pytest.raises(ValueError, match="explicit source bundle protocol"):
        create_and_build(repo, topic_id, question, result)
    result["source_bundle_protocol"] = SOURCE_BUNDLE_PROTOCOL
    with pytest.raises(ValueError, match="original documents"):
        build_snapshot(repo.topic(topic_id), [question], {question["id"]: result}, revision_id="x", graph_revision=1,
            run_id="r", source_paragraphs=repo.paragraphs(topic_id))


def test_legacy_run_cannot_self_enable_document_citations(tmp_path):
    repo, topic_id, question = fixture(tmp_path, enabled=False)
    result = result_for(repo, topic_id, question)
    run, snapshot = create_and_build(repo, topic_id, question, result)
    with pytest.raises(ValueError, match="legacy run"):
        repo.publish(run["run_id"], snapshot, {question["id"]: result})
    assert repo.topic(topic_id)["published_revision_id"] is None
    assert repo.all("SELECT id FROM evidence") == []


def test_authentic_read_context_is_not_a_final_citation(tmp_path):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question)
    read_context = build_source_bundles(repo.source_documents(topic_id), repo.paragraphs(topic_id),
        [repo.paragraphs(topic_id)[0]["id"]], before=0, after=1)[0]
    result["evidence"][0].update(read_context)
    with pytest.raises(ValueError, match="read context"):
        create_and_build(repo, topic_id, question, result)


def test_publication_cutoff_replays_bundle_members_and_bound_document_manifest(tmp_path):
    repo, topic_id, question = fixture(tmp_path, cutoff=True)
    result = result_for(repo, topic_id, question)
    assert result["material_eligibility"]["applied"] is True
    # The importer currently records a broad datetime precision. Do not promote
    # that existing unknown case merely because the document is a source bundle.
    assert result["material_eligibility"]["counts"]["unknown"] == 1
    parents = repo.paragraphs(topic_id)
    for parent in parents:
        parent["time_precision"] = "second"  # Explicit synthetic metadata, not a program inference.
    qualified, audit = apply_workflow_material_eligibility(question, parents, result["material_eligibility"]["input_evidence"],
        source_documents=repo.source_documents(topic_id), source_bundle_protocol=SOURCE_BUNDLE_PROTOCOL)
    result.update(evidence=qualified, material_eligibility=audit)
    assert audit["counts"]["qualified"] == 1
    validate_workflow_material_eligibility(result, question, parents, source_documents=repo.source_documents(topic_id))
    changed = deepcopy(result)
    changed["material_eligibility"]["source_documents_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="eligibility_workflow_audit_replay"):
        validate_workflow_material_eligibility(changed, question, parents, source_documents=repo.source_documents(topic_id))


def test_export_rechecks_ready_bundle_against_actual_document_hash(tmp_path, monkeypatch):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question)
    run, snapshot = create_and_build(repo, topic_id, question, result)
    repo.publish(run["run_id"], snapshot, {question["id"]: result})
    corrupt_source_projection = repo.source_documents(topic_id)
    corrupt_source_projection[0]["content"] = "非法篡改"
    monkeypatch.setattr(repo, "source_documents", lambda _topic_id: deepcopy(corrupt_source_projection))
    with pytest.raises(ServiceError) as error:
        repo.formal_submission_files(topic_id, run["target_revision_id"])
    assert error.value.code == "export_validation_failed"
    assert repo.revision(topic_id, run["target_revision_id"])["state"] == "ready"


class CompletedBundleState(TypedDict, total=False):
    question: dict
    evidence: list
    authority_analysis: dict
    cross_analysis: dict
    final: dict
    evidence_snapshot_id: str
    source_bundle_protocol: str
    material_eligibility: dict


async def completed_checkpoint(repo, topic_id, question, result):
    run = repo.create_run(topic_id, {"question_ids": [question["id"]], "base_revision_id": None,
        "mode": "initial", "include_baseline": False}, "synthetic-completed-checkpoint", model_available=True)
    assert repo.claim()["id"] == run["run_id"]
    channels = {"question": question, "evidence": result["evidence"], "authority_analysis": result["analyses"][0],
        "cross_analysis": result["analyses"][1], "final": result["final"], "evidence_snapshot_id": result["evidence_snapshot_id"],
        "source_bundle_protocol": result["source_bundle_protocol"], "material_eligibility": result["material_eligibility"]}
    graph = StateGraph(CompletedBundleState)
    graph.add_node("synthetic_completed_result", lambda _state: deepcopy(channels))
    graph.add_edge(START, "synthetic_completed_result")
    graph.add_edge("synthetic_completed_result", END)
    async with AsyncSqliteSaver.from_conn_string(str(repo.settings.checkpoint_path)) as saver:
        await graph.compile(checkpointer=saver).ainvoke({}, {"configurable": {"thread_id": run["run_id"] + ":" + question["id"]}})
    repo.finish_failed(run["run_id"], "snapshot_validation_failed", "合成发布边界失败", stage="publish")
    return run


async def test_completed_bundle_checkpoint_recovers_without_changing_source_scope_or_usage(tmp_path):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question)
    run = await completed_checkpoint(repo, topic_id, question, result)
    recovered = await recover_completed_run(repo, run["run_id"], "explicit-bundle-recovery")
    assert recovered["status"] == "succeeded"
    assert recovered["recovery"]["new_model_calls"] == 0
    assert recovered["usage"] == {question["id"]: []}
    assert repo.run(run["run_id"])["status"] == "failed"
    stored = repo.snapshot(topic_id, recovered["target_revision_id"])
    assert stored["evidence"] == result["evidence"]
    assert stored["question_results"][question["id"]]["source_bundle_protocol"] == SOURCE_BUNDLE_PROTOCOL
    assert repo.formal_submission_files(topic_id, recovered["target_revision_id"])["evidence.json"][0]["quote"] == result["evidence"][0]["quote"]


async def test_completed_but_source_tampered_checkpoint_cannot_recover(tmp_path):
    repo, topic_id, question = fixture(tmp_path)
    result = result_for(repo, topic_id, question)
    result["evidence"][0]["document_content_sha256"] = "0" * 64
    run = await completed_checkpoint(repo, topic_id, question, result)
    with pytest.raises(ServiceError) as error:
        await recover_completed_run(repo, run["run_id"], "reject-bundle-recovery")
    assert error.value.code == "recovery_validation_failed"
    assert len(repo.all("SELECT id FROM runs")) == 1
    assert repo.topic(topic_id)["published_revision_id"] is None
