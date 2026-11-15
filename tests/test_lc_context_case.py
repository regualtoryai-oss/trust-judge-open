"""Synthetic complete raw/window fixtures, no model, inference or answers."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from app.domain.context_candidates import build_context_plan
from app.domain.lc_context_case import (
    POLICY, PREPARED_FORMAT, SOURCE_BUNDLE_PROTOCOL, compile_context_business,
    prepare_context_case, selection_schema,
)
from app.domain.local_parent_rank import fingerprint, prepare_pairs
from app.domain.local_passage_rank import FORMAT as WINDOW_FORMAT
from app.domain.source_bundles import paragraph_manifest_sha256
from app.domain.source_passages import SOURCE_GEOMETRY, validate_source_passage


def raw_case(texts=None):
    texts = texts if texts is not None else [
        "".join(f"第{i}句并不代表最终结论，仍待进一步核实。" + "原始上下文😀" * 12 + "。\r\n" for i in range(16)),
        "材料已正式受理，但结果尚未公布。", "竞争说法也没有被最终证实。", ""]
    content, parents, manifest = "原始前缀😀|", [], []
    for index, text in enumerate(texts):
        start = len(content)
        content += text
        pid, raw_pid = f"opaque-parent-{99-index}", f"原段{index}"
        parents.append({"id": pid, "document_id": "doc-uuid", "raw_doc_id": "原文档", "raw_para_id": raw_pid,
            "text": text, "document_start_cp": start, "document_end_cp": len(content)})
        manifest.append({"id": pid, "raw_para_id": raw_pid, "start_cp": start, "end_cp": len(content),
                         "text_sha256": sha256(text.encode()).hexdigest()})
        content += "\r\n<原始间隔>\x00"
    return {"format": "frozen-source-bundle-case-v1",
        "question": {"question_id": "raw-question", "question": "材料已经确认还是仍待核验？"},
        "documents": [{"id": "doc-uuid", "raw_doc_id": "原文档", "title": "待核验来源标题，不是事实结论",
            "content": content, "content_sha256": sha256(content.encode()).hexdigest(),
            "paragraph_manifest": manifest, "paragraph_manifest_sha256": paragraph_manifest_sha256(manifest)}],
        "paragraphs": parents, "limits": {"max_read_calls": 2}}


def windows(raw):
    plan, payloads = prepare_pairs(raw, "a" * 64)
    by_query = {row["query_id"]: row["text"] for row in plan["queries"]}
    records = []
    for source in plan["records"]:
        row = {k: deepcopy(v) for k, v in source.items() if k != "query_id"}
        query = by_query[source["query_id"]]
        row.update(query_facet=source["query_id"], query=query,
            query_sha256=sha256(query.encode()).hexdigest(),
            maxlogit=float(int(source["pair_sha256"][:6], 16) / 100000))
        records.append(row)
    return {"format": WINDOW_FORMAT,
        "binding": {"raw_case_sha256": "f" * 64, "raw_case_canonical_sha256": fingerprint(raw),
            "rank_plan_sha256": "b" * 64, "proof": {"file": "synthetic-not-read", "sha256": "c" * 64},
            "model_manifest_sha256": "a" * 64, "formatter_config_sha256": "d" * 64},
        "rank_verification": {"rank_plan_sha256": "b" * 64, "proof_sha256": "c" * 64},
        "queries": plan["queries"], "records": records, "query_window_count": len(records),
        "unique_pair_count": len(payloads), "parent_count": len(raw["paragraphs"]),
        "all_original_windows_preserved": True, "selected_top32_filter_applied": False,
        "provider_calls": 0, "new_local_inference_calls": 0, "semantic_verified": False, "model_delivery_verified": False}


def prepare(raw=None, excluded_spans=()):
    raw = raw or raw_case()
    scores = windows(raw)
    digest = fingerprint(scores)
    plan = build_context_plan(raw, scores, full_window_scores_digest=digest)
    kwargs = {"context_plan": plan, "context_plan_sha256": fingerprint(plan),
              "window_scores": scores, "full_window_scores_digest": digest, "excluded_spans": excluded_spans}
    prepared = prepare_context_case(raw, **kwargs)
    return raw, prepared, kwargs


def output(prepared, choices=()):
    return {"source_catalog_id": prepared["source_catalog_id"], "ranked_candidates": [
        {"source_window_id": alias, "relevance_score": score, "use_type": use_type}
        for alias, score, use_type in choices]}


def same_parent_contexts(prepared):
    grouped = {}
    for row in prepared["catalog"]:
        grouped.setdefault(row["source_bundle"]["paragraph_id"], []).append(row)
    return next(group for group in grouped.values() if len(group) >= 2)


def test_model_view_is_only_actual_full_context_quotes_and_attributed_metadata():
    raw, prepared, kwargs = prepare()
    assert prepared["format"] == PREPARED_FORMAT
    assert prepared["source_bundle_protocol"] == SOURCE_BUNDLE_PROTOCOL
    assert prepared["source_geometry"] == SOURCE_GEOMETRY
    assert prepared["messages"][1]["content"] == json.dumps(prepared["model_input"], ensure_ascii=False,
        sort_keys=True, separators=(",", ":"))
    schema = selection_schema(prepared)
    assert schema.model_json_schema() == prepared["schema_json"]
    assert prepared["wire_tool"]["function"]["name"] == "LCSourceBundleRankedIDsV1"
    assert "完整原段P" not in json.dumps(prepared["schema_json"], ensure_ascii=False)
    assert [row["source_window_id"] for row in prepared["catalog"]] == [f"C{i+1}" for i in range(len(prepared["catalog"]))]
    assert prepared["scope"]["native_model_delivery_verified"] is False
    assert prepared["scope"]["local_window_coverage_is_native_reading"] is False
    assert prepared["scope"]["local_query_window_count"] == len(kwargs["window_scores"]["records"])
    assert prepared["source_binding"]["full_window_scores_digest"] == kwargs["full_window_scores_digest"]
    assert prepared["source_binding"]["context_plan_sha256"] == kwargs["context_plan_sha256"]
    for item, native in zip(prepared["catalog"], prepared["model_input"]["source_contexts"], strict=True):
        source = item["source_bundle"]
        assert validate_source_passage(source, raw["documents"], raw["paragraphs"]) == source
        assert native["id"] == item["source_window_id"] and native["quote"] == source["quote"]
        assert native["sentence_range"] == [source["start_sentence_index"], source["end_sentence_index"]]
        assert native["paragraph"] == item["paragraph_ref"]
        assert native["quote"] == raw["documents"][0]["content"][source["start_cp"]:source["end_cp"]]
    for key in ("ranked_contexts", "rrf_score", "maxlogit", "window_origins", "full_window_scores_digest"):
        assert key not in prepared["messages"][1]["content"]


def test_partial_contexts_do_not_claim_entire_parent_read_even_if_all_windows_were_scored():
    raw, prepared, _ = prepare()
    longest = raw["paragraphs"][0]
    assert longest["id"] in prepared["scope"]["exposed_paragraph_ids"]
    assert longest["id"] not in prepared["scope"]["returned_complete_paragraph_ids"]
    assert longest["id"] in prepared["scope"]["partially_provided_paragraph_ids"]
    for pid in prepared["scope"]["returned_complete_paragraph_ids"]:
        parent = next(row for row in raw["paragraphs"] if row["id"] == pid)
        assert any(row["paragraph_id"] == pid and row["quote"] == parent["text"]
                   and row["start_cp"] == parent["document_start_cp"] and row["end_cp"] == parent["document_end_cp"]
                   for row in prepared["scope"]["delivered_source_passages"])
    assert raw["paragraphs"][-1]["id"] in prepared["scope"]["unprovided_paragraph_ids"]


def test_single_parent_representative_keeps_own_scope_grade_and_type_no_merge_or_transfer():
    raw, prepared, kwargs = prepare()
    contexts = same_parent_contexts(prepared)
    first, second = contexts[:2]
    parsed = output(prepared, [(first["source_window_id"], 2, "counterclaim"),
                               (second["source_window_id"], 3, "direct_answer")])
    result = compile_context_business(raw, prepared, parsed, **kwargs)
    artifact = result["artifact"]
    assert artifact["raw_model_selection"] == parsed
    assert len(artifact["evidence"]) == 1
    evidence = artifact["evidence"][0]
    assert evidence["quote"] == second["source_bundle"]["quote"]
    assert evidence["source_window_id"] == second["source_window_id"]
    assert evidence["start_cp"] == second["source_bundle"]["start_cp"]
    assert evidence["end_cp"] == second["source_bundle"]["end_cp"]
    group = artifact["selection_audit"]["groups"][0]
    assert group["representative_relevance_score"] == 3
    assert group["representative_use_type"] == "direct_answer" and group["scope_merged"] is False
    coverage = artifact["selection_audit"]["counterclaim_input_coverage"]
    assert coverage["required"] is False
    assert coverage["discarded_same_parent_counterclaim_input_indices"] == [0]
    assert artifact["selection_audit"]["candidate_decisions"][0]["context_retained"] is False
    assert artifact["selection_audit"]["candidate_decisions"][1]["context_retained"] is True
    assert evidence["model_generated_annotations"] is False and evidence["entities"] == []
    assert evidence["fact_des"] in evidence["quote"] and evidence["argument"]["内容"] == evidence["fact_des"]
    assert all(value == "" for key, value in evidence["argument"].items() if key != "内容")
    assert artifact["publishable"] is False and artifact["semantic_verified"] is False


def test_same_parent_tie_and_duplicate_alias_keep_original_model_order_with_all_audit():
    raw, prepared, kwargs = prepare()
    contexts = same_parent_contexts(prepared)
    a, b = contexts[0]["source_window_id"], contexts[1]["source_window_id"]
    parsed = output(prepared, [(b, 3, "necessary_condition"), (a, 3, "counterclaim"), (b, 1, "background")])
    artifact = compile_context_business(raw, prepared, parsed, **kwargs)["artifact"]
    assert artifact["evidence"][0]["source_window_id"] == b
    assert artifact["raw_model_selection"] == parsed
    assert len(artifact["selection_audit"]["candidate_decisions"]) == 3
    assert artifact["selection_audit"]["groups"][0]["representative_input_index"] == 0
    assert artifact["selection_audit"]["counterclaim_input_coverage"]["required"] is False


def test_threshold_top10_and_counterclaim_replacement_use_retained_context_own_types():
    raw, prepared, kwargs = prepare(raw_case([f"独立原段{i}记录尚待核验。" for i in range(12)]))
    ids = [row["source_window_id"] for row in prepared["catalog"]]
    parsed = output(prepared, [(alias, 3, "direct_answer") for alias in ids[:10]] +
                    [(ids[10], 2, "counterclaim"), (ids[11], 1, "background")])
    artifact = compile_context_business(raw, prepared, parsed, **kwargs)["artifact"]
    actual = [row["source_window_id"] for row in artifact["evidence"]]
    assert len(actual) == POLICY["maximum_selected_anchors"] == 10
    assert actual == ids[:9] + [ids[10]]
    audit = artifact["selection_audit"]
    assert audit["counterclaim_input_coverage"]["replacement_applied"] is True
    assert audit["counterclaim_input_coverage"]["removed_source_window_id"] == ids[9]
    assert audit["counterclaim_input_coverage"]["is_final_conflict_label"] is False
    assert audit["groups"][-1]["selection_reason"] == "score_below_threshold"
    assert audit["original_candidate_count"] == 12 and audit["all_original_selections_preserved"] is True


@pytest.mark.parametrize("mutation", [
    lambda out: out.update(source_catalog_id="wrong"),
    lambda out: out.update(evidence=[]),
    lambda out: out["ranked_candidates"][0].update(source_window_id="C99999"),
    lambda out: out["ranked_candidates"][0].update(relevance_score=True),
    lambda out: out["ranked_candidates"][0].update(relevance_score=2.0),
    lambda out: out["ranked_candidates"][0].update(relevance_score="2"),
    lambda out: out["ranked_candidates"][0].update(quote="模型自写原文"),
    lambda out: out["ranked_candidates"][0].update(use_type="support"),
    lambda out: out["ranked_candidates"][0].update(use_type="background"),
])
def test_invalid_free_prose_unknown_contexts_and_inconsistent_grades_reject_whole_output(mutation):
    raw, prepared, kwargs = prepare(raw_case(["完整原始来源。"] ))
    parsed = output(prepared, [(prepared["catalog"][0]["source_window_id"], 2, "direct_answer")])
    mutation(parsed)
    with pytest.raises(ValueError, match="lc_context_"):
        compile_context_business(raw, prepared, parsed, **kwargs)


def test_empty_model_selection_is_valid_not_fabricated_support():
    raw, prepared, kwargs = prepare()
    artifact = compile_context_business(raw, prepared, output(prepared), **kwargs)["artifact"]
    assert artifact["evidence"] == [] and artifact["raw_model_selection"]["ranked_candidates"] == []
    assert artifact["selection_audit"]["retained_selection_count"] == 0


def test_identical_mirror_text_does_not_delete_original_source_identities():
    raw, prepared, kwargs = prepare(raw_case(["同一转述仍未证实。", "同一转述仍未证实。"] ))
    assert len(prepared["catalog"]) == 2
    choices = [(row["source_window_id"], 2, "necessary_condition") for row in prepared["catalog"]]
    evidence = compile_context_business(raw, prepared, output(prepared, choices), **kwargs)["artifact"]["evidence"]
    assert len(evidence) == 2
    assert evidence[0]["quote"] == evidence[1]["quote"]
    assert evidence[0]["paragraph_id"] != evidence[1]["paragraph_id"]
    assert evidence[0]["passage_id"] != evidence[1]["passage_id"]


def test_tampered_prepared_or_plan_or_incomplete_window_object_fail_without_fallback():
    raw, prepared, kwargs = prepare()
    altered = deepcopy(prepared)
    altered["model_input"]["source_contexts"][0]["quote"] += "伪造"
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_context_business(raw, altered, output(prepared), **kwargs)
    altered = deepcopy(kwargs)
    altered["context_plan"]["candidates"][0]["source_passage"]["quote"] += "伪造"
    altered["context_plan_sha256"] = fingerprint(altered["context_plan"])
    with pytest.raises(ValueError, match="plan_changed"):
        prepare_context_case(raw, **altered)
    altered = deepcopy(kwargs)
    altered["window_scores"]["records"].pop()
    altered["full_window_scores_digest"] = fingerprint(altered["window_scores"])
    with pytest.raises(ValueError, match="coverage_changed"):
        prepare_context_case(raw, **altered)
    with pytest.raises(ValueError, match="context_plan_digest"):
        prepare_context_case(raw, kwargs["context_plan"], "0" * 64,
            window_scores=kwargs["window_scores"], full_window_scores_digest=kwargs["full_window_scores_digest"])


def test_operator_exclusions_remove_contexts_from_native_view_bind_hash_and_keep_plan_intact():
    raw = raw_case(["第一原段尚未确认。", "第二原段不同说法。", "第三原段需要核验。"])
    first = raw["paragraphs"][0]
    excluded = [(first["id"], 0, len(first["text"]), first["text"])]
    _, original, _ = prepare(raw)
    raw, prepared, kwargs = prepare(raw, excluded_spans=excluded)
    assert first["id"] not in prepared["scope"]["exposed_paragraph_ids"]
    assert len(prepared["scope"]["operator_excluded_contexts"]) == 1
    assert prepared["scope"]["operator_exclusions"] == [list(excluded[0])]
    assert len(kwargs["context_plan"]["candidates"]) == 3
    assert prepared["source_catalog_id"] != original["source_catalog_id"]
    assert {row["paragraph_ref"] for row in prepared["catalog"]} == {"P2", "P3"}
    assert first["text"] not in prepared["messages"][1]["content"]
    with pytest.raises(ValueError, match="output_schema"):
        compile_context_business(raw, prepared, output(prepared, [("C3", 3, "direct_answer")]), **kwargs)
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_context_business(raw, prepared, output(prepared), **{**kwargs, "excluded_spans": ()})


def test_document_cp_exclusions_propagate_to_overlapping_original_parents():
    raw = raw_case(["首句。重叠原句仍未确认。尾句。", "unused"])
    first, second = raw["paragraphs"]
    offset = len("首句。")
    text = "重叠原句仍未确认。"
    second.update(text=text, document_start_cp=first["document_start_cp"] + offset,
                  document_end_cp=first["document_start_cp"] + offset + len(text))
    doc = raw["documents"][0]
    doc["paragraph_manifest"][1].update(start_cp=second["document_start_cp"], end_cp=second["document_end_cp"],
                                        text_sha256=sha256(text.encode()).hexdigest())
    doc["paragraph_manifest_sha256"] = paragraph_manifest_sha256(doc["paragraph_manifest"])
    excluded = [(first["id"], offset, offset + len(text), text)]
    _, prepared, kwargs = prepare(raw, excluded_spans=excluded)
    assert prepared["catalog"] == [] and prepared["model_input"]["source_contexts"] == []
    assert {row["paragraph_id"] for row in prepared["scope"]["operator_excluded_contexts"]} == {first["id"], second["id"]}
    assert compile_context_business(raw, prepared, output(prepared), **kwargs)["artifact"]["evidence"] == []
    with pytest.raises(ValueError, match="output_schema"):
        compile_context_business(raw, prepared, output(prepared, [("C1", 3, "direct_answer")]), **kwargs)
