"""Isolated source-context publication contracts; no model or reference IO.

The source replay boundary is explicitly stubbed where marked. These tests
verify downstream geometry/coverage/authorization, not native delivery itself.
"""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.domain.material_eligibility import apply_workflow_material_eligibility, validate_workflow_material_eligibility
from app.domain.snapshot import (SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL, build_snapshot,
    competition_files, validate_source_bundle_rows)
from app.domain.source_bundles import assemble_source_bundle
from app.domain.source_passages import assemble_source_passage
from app.errors import ServiceError
from app.repository import Repository, dumps
from app.workflows import bundle_workbench as workbench
from test_source_bundles import corpus


QUESTION = {"id": "question", "raw_question_id": "q", "question": "项目审批状态是否已核实？"}


def sources():
    docs, parents = corpus(["背景😀。\r\n项目尚未批准，须审核。后续原文尚未核实。", "短段完整。", "未交付原段。"],
        gaps=["|真实非换行gap|", "\r\n"], prefix="文前😀|")
    for row in parents:
        row.update(content_sha256=docs[0]["content_sha256"], title="项目状态", source_credibility_json="0.9")
    return docs, parents


def annotation(source):
    return {**deepcopy(source), "bundle_purpose": "citation_source", "fact_des": source["quote"],
        "argument": {key: source["quote"] if key == "内容" else "" for key in ("时间", "主体", "地点", "内容", "影响", "数值口径")},
        "entities": []}


def compiled(monkeypatch, docs, parents, *, selected=None, delivered=None, complete=None, proof=None):
    partial = assemble_source_passage(docs, parents, parents[0]["id"], 1, 2)
    full = assemble_source_passage(docs, parents, parents[1]["id"])
    delivered = [partial, full] if delivered is None else delivered
    selected = [partial] if selected is None else selected
    touched = list(dict.fromkeys(pid for row in delivered for pid in row["source_paragraph_ids"]))
    scope = {"exposed_paragraph_ids": touched, "returned_complete_paragraph_ids": [parents[1]["id"]] if complete is None else complete,
        "delivered_source_passages": [{**deepcopy(row), "source_window_id": f"C{i+1}", "source_context_id": f"program-context-{i+1}"}
            for i, row in enumerate(delivered)]}
    artifact = {"evidence": [annotation(row) for row in selected]}
    response = {"proof": {"format": "langchain-source-bundle-workbench-proof-v1", "context_passages": True,
        "source_bundle_protocol": SOURCE_CONTEXT_PROTOCOL} if proof is None else proof}
    # Native/source replay belongs to the separate LC proof tests. It is not
    # fabricated or counted as verified by this downstream isolation fixture.
    monkeypatch.setattr(workbench, "_replay", lambda *_args: (deepcopy(artifact), deepcopy(scope)))
    return workbench.compile_result(QUESTION, parents, docs, response), response, scope


def add_final(result, question=QUESTION):
    result = deepcopy(result)
    ids = [row["id"] for row in result["evidence"]]
    final = {"decision": "undetermined", "conclusion": "原文保留尚待核实的程序限定。", "judgments": [],
        "supporting_evidence_ids": ids, "conflicting_evidence_ids": [], "conflicts": []}
    result.update(final=final, analyses=[{"role": role, "evidence_snapshot_id": "frozen", **{key: deepcopy(value)
        for key, value in final.items() if key != "conflicts"}} for role in ("authority_analyst", "cross_validation_analyst")],
        evidence_snapshot_id="frozen", usage=[], origin_run_id="trusted-origin")
    return result


def snapshot(result, docs, parents):
    return build_snapshot({"id": "topic"}, [QUESTION], {QUESTION["id"]: add_final(result)},
        revision_id="revision", graph_revision=1, run_id="trusted-origin", source_documents=docs, source_paragraphs=parents)


def test_context_compiler_and_export_keep_exact_mid_parent_without_claiming_full_read(monkeypatch):
    docs, parents = sources(); result, _response, _scope = compiled(monkeypatch, docs, parents)
    source = result["evidence"][0]
    assert result["source_bundle_protocol"] == result["verification_protocol"] == SOURCE_CONTEXT_PROTOCOL
    assert result["source_bundle_proof"]["version"] == SOURCE_CONTEXT_PROTOCOL
    assert result["source_coverage"]["provided_full_paragraph_ids"] == [parents[1]["id"]]
    assert result["source_coverage"]["provided_full_paragraphs"] == 1
    assert result["source_coverage"]["partially_delivered_parent_ids"] == [parents[0]["id"]]
    assert result["source_coverage"]["delivered_source_passage_count"] == 2
    assert source["local_start_cp"] > 0 and source["start_cp"] > parents[0]["document_start_cp"]
    assert workbench.validate_result(result, QUESTION, parents, docs) == result
    built = snapshot(result, docs, parents)
    exported = competition_files(built, source_documents=docs, source_paragraphs=parents)["evidence.json"][0]
    assert exported["quote"] == source["quote"] != parents[0]["text"]
    assert exported["para_id"] == parents[0]["raw_para_id"]
    Repository._validate_snapshot_evidence_projection(built)


@pytest.mark.parametrize("protocol", [None, SOURCE_BUNDLE_PROTOCOL])
def test_old_protocol_cannot_smuggle_even_a_full_parent_with_new_geometry(protocol):
    docs, parents = sources()
    passage = assemble_source_passage(docs, parents, parents[0]["id"])
    with pytest.raises(ValueError, match="explicit source context protocol"):
        validate_source_bundle_rows([passage], protocol=protocol, source_documents=docs, source_paragraphs=parents)


def test_new_protocol_allows_original_full_bundle_without_reinterpreting_its_boundaries(monkeypatch):
    docs, parents = sources()
    bundle = assemble_source_bundle(docs, parents, parents[1]["id"], parents[2]["id"])
    result, _, _ = compiled(monkeypatch, docs, parents, selected=[bundle], delivered=[bundle],
        complete=[row["id"] for row in parents[1:]])
    assert "source_geometry" not in result["evidence"][0]
    assert result["evidence"][0]["quote"] == bundle["quote"]
    assert result["source_coverage"]["provided_full_paragraphs"] == 2


@pytest.mark.parametrize("change", ["old_flag", "wrong_protocol", "missing_protocol", "unknown_geometry", "local_cp", "sentence_hash", "parent_hash"])
def test_compile_fails_closed_on_geometry_or_frozen_mode_drift(monkeypatch, change):
    docs, parents = sources()
    passage = assemble_source_passage(docs, parents, parents[0]["id"], 1, 2)
    proof = {"format": "langchain-source-bundle-workbench-proof-v1", "context_passages": True,
        "source_bundle_protocol": SOURCE_CONTEXT_PROTOCOL}
    if change == "old_flag": proof.pop("context_passages"); proof.pop("source_bundle_protocol")
    elif change == "wrong_protocol": proof["source_bundle_protocol"] = SOURCE_BUNDLE_PROTOCOL
    elif change == "missing_protocol": proof.pop("source_bundle_protocol")
    elif change == "unknown_geometry": passage["source_geometry"] = "short-snippet"
    elif change == "local_cp": passage["local_start_cp"] += 1
    elif change == "sentence_hash": passage["sentence_manifest_sha256"] = "0" * 64
    elif change == "parent_hash": passage["parent_text_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        compiled(monkeypatch, docs, parents, selected=[passage], proof=proof)


def test_two_partial_deliveries_cannot_be_promoted_to_one_whole_parent_read_or_citation(monkeypatch):
    docs, parents = sources(); pid = parents[0]["id"]
    left = assemble_source_passage(docs, parents, pid, 0, 1)
    rest = assemble_source_passage(docs, parents, pid, 1, 3)
    assert left["quote"] + rest["quote"] == parents[0]["text"]
    result, _, _ = compiled(monkeypatch, docs, parents, selected=[left], delivered=[left, rest], complete=[])
    assert result["source_coverage"]["provided_full_paragraphs"] == 0
    with pytest.raises(ValueError, match="full_parent_read_claim_changed"):
        compiled(monkeypatch, docs, parents, selected=[left], delivered=[left, rest], complete=[pid])
    whole = assemble_source_passage(docs, parents, pid)
    with pytest.raises(ValueError, match="not_in_one_actual_delivery"):
        compiled(monkeypatch, docs, parents, selected=[whole], delivered=[left, rest], complete=[])


def test_selected_undelivered_context_or_different_range_under_same_anchor_is_rejected(monkeypatch):
    docs, parents = sources()
    first = assemble_source_passage(docs, parents, parents[0]["id"], 0, 1)
    with pytest.raises(ValueError, match="not_in_one_actual_delivery"):
        compiled(monkeypatch, docs, parents, selected=[first])
    second = assemble_source_passage(docs, parents, parents[0]["id"], 1, 2)
    with pytest.raises(ValueError, match="final_citation_scope"):
        compiled(monkeypatch, docs, parents, selected=[first, second], delivered=[first, second], complete=[])


def test_distinct_mirror_source_identities_survive_context_material_eligibility_and_replay():
    docs, parents = sources()
    mirror_docs, mirror_parents = corpus([parents[0]["text"]], doc_id="mirror-doc", raw_doc_id="mirror-raw")
    for row in mirror_parents: row.update(content_sha256=mirror_docs[0]["content_sha256"])
    docs += mirror_docs; parents += mirror_parents
    rows = [annotation(assemble_source_passage(docs, parents, pid, 1, 2)) for pid in (parents[0]["id"], mirror_parents[0]["id"])]
    for i, row in enumerate(rows): row["id"] = f"source-{i}"
    qualified, audit = apply_workflow_material_eligibility(QUESTION, parents, rows,
        source_documents=docs, source_bundle_protocol=SOURCE_CONTEXT_PROTOCOL)
    assert len(qualified) == 2 and qualified[0]["quote"] == qualified[1]["quote"]
    assert audit["source_bundle_protocol"] == SOURCE_CONTEXT_PROTOCOL
    validate_workflow_material_eligibility({"evidence": qualified, "material_eligibility": audit,
        "source_bundle_protocol": SOURCE_CONTEXT_PROTOCOL}, QUESTION, parents, source_documents=docs)


def test_mixed_old_and_context_questions_validate_separately_and_keep_exact_union(monkeypatch):
    docs, parents = sources(); new, _, _ = compiled(monkeypatch, docs, parents)
    old = deepcopy(new); bundle = annotation(assemble_source_bundle(docs, parents, parents[1]["id"], parents[1]["id"]))
    bundle.update(id="old-evidence", source_group_id="old-source", source_group_basis="same-document", grade_score=None)
    old.update(evidence=[bundle], source_bundle_protocol=SOURCE_BUNDLE_PROTOCOL)
    second_question = {**QUESTION, "id": "other-question", "raw_question_id": "q-other"}
    built = build_snapshot({"id": "topic"}, [QUESTION, second_question],
        {QUESTION["id"]: add_final(new), second_question["id"]: add_final(old)},
        revision_id="revision", graph_revision=1, run_id="run", source_documents=docs, source_paragraphs=parents)
    files = competition_files(built, source_documents=docs, source_paragraphs=parents)
    assert len(files["evidence.json"]) == 2
    built["question_results"][QUESTION["id"]]["source_bundle_protocol"] = SOURCE_BUNDLE_PROTOCOL
    with pytest.raises(ValueError, match="explicit source context protocol"):
        competition_files(built, source_documents=docs, source_paragraphs=parents)


@pytest.mark.parametrize("frozen_context", [False, True])
def test_repository_document_source_dispatch_is_authorized_by_frozen_origin_not_row(frozen_context):
    docs, parents = sources(); row = assemble_source_passage(docs, parents, parents[0]["id"], 1, 2)
    repo = object.__new__(Repository)
    config = {"source_bundle_protocol": SOURCE_CONTEXT_PROTOCOL if frozen_context else SOURCE_BUNDLE_PROTOCOL,
        "source_bundle_config": {"context_passages": True} if frozen_context else {}}
    repo.one = Mock(return_value={"model_config_json": dumps(config)})
    if frozen_context:
        repo._validate_document_source(row, "origin", docs, parents)
    else:
        with pytest.raises(ValueError, match="explicit source context protocol"):
            repo._validate_document_source(row, "origin", docs, parents)


def test_document_overlay_uses_real_parent_local_offsets_and_rejects_legacy_authorization():
    docs, parents = sources(); row = annotation(assemble_source_passage(docs, parents, parents[0]["id"], 1, 2)); row["id"] = "e"
    snap = {"topic_id": "t", "run_id": "new", "question_results": {"q": {"origin_run_id": "actual-context-origin", "evidence": [row]}}}
    repo = object.__new__(Repository); repo.paragraphs = Mock(return_value=parents); repo.source_documents = Mock(return_value=docs)
    config = {"source_bundle_protocol": SOURCE_CONTEXT_PROTOCOL, "source_bundle_config": {"context_passages": True}}
    repo.one = Mock(return_value={"model_config_json": dumps(config)})
    assert repo._document_overlay_spans(snap, row) == [(row["paragraph_id"], row["local_start_cp"], row["local_end_cp"], row["quote"])]
    assert repo.one.call_args.args[1] == ("actual-context-origin",)
    config["source_bundle_protocol"] = SOURCE_BUNDLE_PROTOCOL; config["source_bundle_config"] = {}
    repo.one.return_value = {"model_config_json": dumps(config)}
    with pytest.raises(ServiceError) as error: repo._document_overlay_spans(snap, row)
    assert error.value.status == 409


@pytest.mark.asyncio
async def test_context_protocol_flows_through_real_graph_and_snapshot_without_full_parent_promotion(tmp_path, monkeypatch):
    from app.workflows.analysis import run_analysis
    from test_bundle_workbench import BundleFixture
    docs, parents = sources(); _result, response, _scope = compiled(monkeypatch, docs, parents)
    class ExplicitContextFixture(BundleFixture):
        source_bundle_protocol = SOURCE_CONTEXT_PROTOCOL
        source_bundle_config = {"front_framework": "langchain", "context_passages": True}
        prompt_version = "trusted-analysis-langchain-source-bundles-v1"
    provider = ExplicitContextFixture(response)
    result = await run_analysis(QUESTION, parents, provider, source_documents=docs,
        checkpoint_path=tmp_path / "checkpoints.sqlite3", thread_id="context-isolation",
        cancel_check=lambda: False, emit=lambda *_args: None)
    assert set(provider.calls) == {"source_bundle_selection", "authority_analyst", "cross_validation_analyst", "arbitrate"}
    assert result["source_bundle_protocol"] == result["verification_protocol"] == SOURCE_CONTEXT_PROTOCOL
    assert result["source_coverage"]["provided_full_paragraphs"] == 1
    analyst_inputs = [data for role, data in provider.inputs if role.endswith("analyst")]
    assert analyst_inputs[0] == analyst_inputs[1]
    assert analyst_inputs[0][1][0]["quote"] != parents[0]["text"]
    exported = competition_files(snapshot(result, docs, parents), source_documents=docs, source_paragraphs=parents)
    assert exported["evidence.json"][0]["quote"] == result["evidence"][0]["quote"]


@pytest.mark.asyncio
async def test_graph_rejects_context_flag_protocol_disagreement_before_any_fixture_call(tmp_path, monkeypatch):
    from app.agents.provider import ModelNotConfigured
    from app.workflows.analysis import run_analysis
    from test_bundle_workbench import BundleFixture
    docs, parents = sources(); _result, response, _scope = compiled(monkeypatch, docs, parents)
    class InvalidContextFixture(BundleFixture):
        source_bundle_protocol = SOURCE_BUNDLE_PROTOCOL
        source_bundle_config = {"front_framework": "langchain", "context_passages": True}
        prompt_version = "trusted-analysis-langchain-source-bundles-v1"
    provider = InvalidContextFixture(response)
    with pytest.raises(ModelNotConfigured):
        await run_analysis(QUESTION, parents, provider, source_documents=docs,
            checkpoint_path=tmp_path / "checkpoints.sqlite3", thread_id="must-not-start",
            cancel_check=lambda: False, emit=lambda *_args: None)
    assert provider.calls == []


def test_actual_sqlite_publication_export_and_human_exclusion_use_frozen_context_geometry(tmp_path):
    """Source-only synthetic origin; no native/workbench proof is claimed."""
    from app.config import Settings
    from app.data.ingest import import_topic
    from app.repository import loads
    docs, parents = sources(); doc = docs[0]
    settings = Settings(_env_file=None, data_dir=tmp_path / "runtime", model_name="", model_api_key="")
    imported = import_topic(settings.database_path,
        [{"doc_id": doc["raw_doc_id"], "title": "合成原文", "content": doc["content"], "paragraphs": [
            {"para_id": row["raw_para_id"], "char_start": row["document_start_cp"], "char_end": row["document_end_cp"]}
            for row in parents]}], [{"question_id": "q", "question": QUESTION["question"]}], "显式源几何隔离测试")
    repo = Repository(settings); tid = imported["topic_id"]; question = repo.questions(tid)[0]
    original_config = repo.config
    repo.config = lambda: {**original_config(), "strict_scope_enabled": False,
        "source_bundle_protocol": SOURCE_CONTEXT_PROTOCOL, "source_bundle_config": {"context_passages": True}}
    documents, originals = repo.source_documents(tid), repo.paragraphs(tid)
    target = next(parent for parent in originals if parent["raw_para_id"] == parents[0]["raw_para_id"])
    passage = assemble_source_passage(documents, originals, target["id"], 1, 2)
    row = {**annotation(passage), "id": "synthetic-source-only-evidence", "source_group_id": "same-doc",
        "source_group_basis": "source identity is not independent voting", "grade_score": None}
    qualified, material = apply_workflow_material_eligibility(question, originals, [row], source_documents=documents,
        source_bundle_protocol=SOURCE_CONTEXT_PROTOCOL)
    result = add_final({"evidence": qualified, "material_eligibility": material, "source_bundle_protocol": SOURCE_CONTEXT_PROTOCOL})
    run = repo.create_run(tid, {"question_ids": [question["id"]], "base_revision_id": None,
        "mode": "initial", "include_baseline": False}, "source-only-synthetic", model_available=True)
    assert repo.claim()["id"] == run["run_id"]
    result["origin_run_id"] = run["run_id"]
    built = build_snapshot(repo.topic(tid), [question], {question["id"]: result}, revision_id=run["target_revision_id"],
        graph_revision=1, run_id=run["run_id"], source_documents=documents, source_paragraphs=originals)
    built["scope"] = {"completed_question_ids": [question["id"]], "all_question_count": 1, "is_complete_topic": True}
    repo.publish(run["run_id"], built, {question["id"]: result})
    assert repo.run(run["run_id"])["status"] == "succeeded"
    assert repo.run(run["run_id"])["source_bundle_context_passages"] is True
    files = repo.formal_submission_files(tid, run["target_revision_id"])
    assert files["evidence.json"][0]["quote"] == passage["quote"] != target["text"]
    preview = repo.preview_change(tid, run["target_revision_id"], [{"op": "exclude_evidence", "evidence_id": row["id"]}], "合成排除范围")
    queued = repo.commit_change(tid, {"preview_id": preview["preview_id"], "preview_content_hash": preview["content_hash"],
        "base_revision_id": run["target_revision_id"]}, "synthetic-change-only-no-worker", model_available=True)
    options = loads(repo.one("SELECT options_json FROM run_options WHERE run_id=?", (queued["run_id"],))["options_json"])
    assert options["overlays"]["excluded"] == [[row["paragraph_id"], row["local_start_cp"], row["local_end_cp"], row["quote"]]]
