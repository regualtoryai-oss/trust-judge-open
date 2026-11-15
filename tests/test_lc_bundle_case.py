"""Raw-only deterministic LC catalog tests; no model transport or reference data."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.agents.strict_transport import strict_tool
from app.domain.frozen_selection_inputs import canonical
from app.domain.lc_bundle_case import (
    prepare_lc_case, selection_schema, compile_business, rank_queries, rank_source_binding,
    build_rank_plan, validate_rank_plan, RANK_PRODUCER_HASHES, RANK_SELECTION,
)
from app.domain.source_bundle_tools import SourceBundleTools
from app.domain.source_bundles import paragraph_manifest_sha256, validate_source_bundle
from scripts.lc_bundle_case import decode, execute
from test_source_bundle_tool_bridge import raw_case


def case_with_paragraphs(count=8, *, long=False, numeric_index=None):
    raw = raw_case()
    content, paragraphs, manifest = "前缀😀|", [], []
    for index in range(count):
        text = "项目许可尚未批准，审批结果需要核实。" + ("完整长原段。" * 1000 if long else "")
        if index == numeric_index:
            text += "已投入50人。"
        pid, start = f"opaque-{99-index}", len(content)
        content += text
        row = {"id": pid, "document_id": "doc-internal", "raw_doc_id": "原文档", "raw_para_id": f"原段{index}",
               "document_start_cp": start, "document_end_cp": len(content), "text": text}
        paragraphs.append(row)
        manifest.append({"id": pid, "raw_para_id": row["raw_para_id"], "start_cp": start, "end_cp": len(content),
                         "text_sha256": sha256(text.encode()).hexdigest()})
        content += "\r\n<保留😀原始分隔>\n"
    raw["paragraphs"] = paragraphs
    raw["documents"][0].update(content=content, content_sha256=sha256(content.encode()).hexdigest(),
        paragraph_manifest=manifest, paragraph_manifest_sha256=paragraph_manifest_sha256(manifest))
    return raw


def output_for(prepared, window=None):
    window = window or prepared["catalog"][0]
    return {"source_catalog_id": prepared["source_catalog_id"], "evidence": [{"source_window_id": window["source_window_id"],
        "fact_des": "项目尚待核实。", "argument": {"时间": None, "主体": None, "地点": None,
            "内容": "项目尚待核实。", "影响": None, "数值口径": None}, "entities": []}]}


def retrieve_only(monkeypatch, ids):
    original = SourceBundleTools.search
    def search(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        value["results"] = [row for row in value["results"] if row["paragraph_id"] in ids]
        return value
    monkeypatch.setattr(SourceBundleTools, "search", search)


def test_all_candidates_are_directly_labeled_and_run_delivery_is_separate_from_five_parent_citations():
    raw = case_with_paragraphs()
    prepared = prepare_lc_case(raw)
    assert [len(run["paragraph_refs"]) for run in prepared["source_runs"]] == [8]
    assert len(prepared["catalog"]) == 30  # All 1-to-5 subranges, including old arbitrary chunk boundaries.
    actual_ids = [pid for run in prepared["source_runs"] for pid in run["source_paragraph_ids"]]
    assert actual_ids == [row["id"] for row in raw["paragraphs"]]
    assert prepared["scope"]["corpus_paragraph_count"] == 8
    assert prepared["scope"]["native_model_delivery_verified"] is False
    assert strict_tool(selection_schema(prepared)) == prepared["wire_tool"]
    assert prepared["wire_tool"]["function"]["name"] == prepared["function_name"]
    assert prepared["messages"][1]["content"] == canonical(prepared["model_input"]).decode()
    assert selection_schema(prepared).model_json_schema() == prepared["schema_json"]
    assert len(prepared["messages"]) == 2
    assert prepared["format"] == "lc-source-bundle-prepared-v2"
    for run, native in zip(prepared["source_runs"], prepared["model_input"]["source_runs"]):
        native_paragraphs = [segment for segment in native["segments"] if segment["kind"] == "paragraph"]
        assert [segment["id"] for segment in native_paragraphs] == run["paragraph_refs"]
        for original, segment in zip(run["paragraphs"], native_paragraphs):
            assert segment["text"] == next(r["text"] for r in raw["paragraphs"] if r["id"] == original["paragraph_id"])
            assert "start_cp" not in segment and "end_cp" not in segment
        assert len(native_paragraphs) == len(run["source_paragraph_ids"])
        assert all(segment["text"] == raw["documents"][0]["content"][segment["start_cp"]:segment["end_cp"]] for segment in run["segments"])
    gaps = [s for s in prepared["model_input"]["source_runs"][0]["segments"] if s["kind"] == "original_gap"]
    assert len(gaps) == 7 and all(s["text"] == "\r\n<保留😀原始分隔>\n" for s in gaps)
    assert "source_window_catalog" not in prepared["model_input"] and "source_books" not in prepared["model_input"]
    assert prepared["model_input"]["source_ranges"] == [row["source_window_id"] for row in prepared["catalog"]]
    assert all(row["source_window_id"].startswith("P") for row in prepared["catalog"])
    for row in prepared["catalog"]:
        source = row["source_bundle"]
        assert validate_source_bundle(source, raw["documents"], raw["paragraphs"]) == source
        run = next(run for run in prepared["source_runs"] if run["run_id"] == row["run_id"])
        assert set(source["source_paragraph_ids"]) <= set(run["source_paragraph_ids"])
        assert len(source["source_paragraph_ids"]) <= 5


def test_nonadjacent_candidates_never_import_unretrieved_neighbor_or_create_cross_run_window(monkeypatch):
    raw = case_with_paragraphs()
    wanted = {raw["paragraphs"][i]["id"] for i in [0, 2, 3, 6]}
    retrieve_only(monkeypatch, wanted)
    prepared = prepare_lc_case(raw)
    assert [len(run["paragraph_refs"]) for run in prepared["source_runs"]] == [1, 2, 1]
    assert set(prepared["scope"]["returned_complete_paragraph_ids"]) == wanted
    assert prepared["scope"]["corpus_paragraph_count"] == 8
    for row in prepared["catalog"]:
        assert set(row["source_bundle"]["source_paragraph_ids"]) <= wanted
        assert not {raw["paragraphs"][i]["id"] for i in [1, 4, 5, 7]} & set(row["source_bundle"]["source_paragraph_ids"])


def test_contiguous_run_covers_old_artificial_chunk_boundary_without_repeating_body_per_range():
    raw = case_with_paragraphs()
    prepared = prepare_lc_case(raw)
    crossing = next(window for window in prepared["catalog"] if window["source_window_id"] == "P5..P6")
    assert len(prepared["source_runs"]) == 1 and len(crossing["source_bundle"]["source_paragraph_ids"]) == 2
    compiled = compile_business(raw, prepared, output_for(prepared, crossing))
    a, b = raw["paragraphs"][4], raw["paragraphs"][5]
    assert compiled["artifact"]["evidence"][0]["quote"] == raw["documents"][0]["content"][a["document_start_cp"]:b["document_end_cp"]]
    assert len([s for s in prepared["model_input"]["source_runs"][0]["segments"] if s["kind"] == "paragraph"]) == 8


def test_overlapping_manifest_parents_keep_full_source_envelope_and_precise_native_boundaries():
    raw = raw_case()
    content = "前缀😀甲乙丙丁戊己\r\n<gap>庚辛壬癸末尾"
    intervals = [(2, 16), (4, 9), (16, 20)]
    paragraphs, manifest = [], []
    for index, (start, end) in enumerate(intervals):
        pid, text = f"overlap-{index}", content[start:end]
        paragraphs.append({"id": pid, "document_id": "doc-internal", "raw_doc_id": "原文档", "raw_para_id": f"重叠段{index}",
            "document_start_cp": start, "document_end_cp": end, "text": text})
        manifest.append({"id": pid, "raw_para_id": f"重叠段{index}", "start_cp": start, "end_cp": end,
                         "text_sha256": sha256(text.encode()).hexdigest()})
    raw["paragraphs"] = list(reversed(paragraphs))  # Original manifests, not input row order, define P.
    raw["question"]["question"] = "甲乙丙丁戊己与庚辛壬癸各自是什么？"
    raw["documents"][0].update(content=content, content_sha256=sha256(content.encode()).hexdigest(),
        paragraph_manifest=manifest, paragraph_manifest_sha256=paragraph_manifest_sha256(manifest))
    prepared = prepare_lc_case(raw)
    run = prepared["source_runs"][0]
    assert run["source_paragraph_ids"] == [p["id"] for p in paragraphs]
    assert (run["start_cp"], run["end_cp"]) == (2, 20)
    assert run["original_slice_sha256"] == sha256(content[2:20].encode()).hexdigest()
    pair = next(w for w in prepared["catalog"] if w["paragraph_refs"] == ["P1", "P2"])
    assert pair["source_bundle"]["end_cp"] == 16  # Preserve all of the first parent despite nested endpoint.
    assert pair["source_bundle"]["quote"] == content[2:16]
    native = prepared["model_input"]["source_runs"][0]
    native_paragraphs = [s for s in native["segments"] if s["kind"] == "paragraph"]
    assert [s["text"] for s in native_paragraphs] == [p["text"] for p in paragraphs]
    assert native_paragraphs[1]["overlaps_prior_source"] is True
    assert "".join(s["text"] for s in native["segments"]) != content[2:20]  # Honest overlay, never call this a concatenated quote.
    overlay = {}
    for segment in run["segments"]:
        for i, char in enumerate(segment["text"], segment["start_cp"]):
            assert overlay.get(i, char) == char
            overlay[i] = char
    assert "".join(overlay[i] for i in range(run["start_cp"], run["end_cp"])) == content[2:20]
    assert not run["contiguous_quote_payload_claimed"] and not prepared["model_input"]["contiguous_quote_payload_claimed"]


def test_different_documents_are_separate_runs_even_with_identical_original_text():
    raw = raw_case()
    second = deepcopy(raw["documents"][0])
    second.update(id="second-internal", raw_doc_id="独立文档标识")
    second_paragraphs = deepcopy(raw["paragraphs"])
    for row, item in zip(second_paragraphs, second["paragraph_manifest"]):
        row.update(id="second-" + row["id"], document_id=second["id"], raw_doc_id=second["raw_doc_id"])
        item["id"] = row["id"]
    second["paragraph_manifest_sha256"] = paragraph_manifest_sha256(second["paragraph_manifest"])
    raw["documents"].append(second)
    raw["paragraphs"].extend(second_paragraphs)
    prepared = prepare_lc_case(raw)
    assert len(prepared["source_runs"]) == 2 and prepared["scope"]["candidate_count"] == 6
    assert len({w["source_bundle"]["document_id"] for w in prepared["catalog"]}) == 2
    assert prepared["source_runs"][0]["original_slice_sha256"] == prepared["source_runs"][1]["original_slice_sha256"]
    assert prepared["source_runs"][0]["document_id"] != prepared["source_runs"][1]["document_id"]


def test_pool_has_whole_parents_not_preview_windows_and_no_long_parent_truncation():
    raw = case_with_paragraphs(1, long=True)
    prepared = prepare_lc_case(raw)
    paragraph = prepared["model_input"]["source_runs"][0]["segments"][0]
    assert paragraph["id"] == "P1" and paragraph["text"] == raw["paragraphs"][0]["text"]
    assert len(paragraph["text"]) > 5000
    assert len(prepared["retrieval"]["backend_result"]["results"][0]["excerpt_text"]) <= 80


def test_retrieval_limit_is_forty_for_larger_complete_raw_corpus():
    raw = case_with_paragraphs(50)
    prepared = prepare_lc_case(raw)
    assert prepared["scope"]["candidate_count"] == 40 and prepared["scope"]["corpus_paragraph_count"] == 50
    assert len(prepared["scope"]["returned_complete_paragraph_ids"]) == 40
    assert {pid for run in prepared["source_runs"] for pid in run["source_paragraph_ids"]} == set(prepared["retrieval"]["candidate_rank_order"])


def test_compile_program_supplies_exact_document_quote_and_preserves_unknown_metadata():
    raw = raw_case()
    prepared = prepare_lc_case(raw)
    window = next(row for row in prepared["catalog"] if len(row["paragraph_refs"]) == 2 and row["paragraph_refs"][0] == "P1")
    output = output_for(prepared, window)
    compiled = compile_business(raw, prepared, output)
    row = compiled["artifact"]["evidence"][0]
    assert row["quote"] == window["source_bundle"]["quote"]
    assert row["quote"].startswith(raw["paragraphs"][0]["text"]) and "\r\n<实际分隔>" in row["quote"]
    assert row["argument"]["时间"] == "" and output["evidence"][0]["argument"]["时间"] is None
    assert row["annotation_semantics_verified"] is False and row["native_model_delivery_verified"] is False
    assert compiled["scope"]["binding"] == prepared["source_binding"]
    assert compiled["artifact"]["publishable"] is False
    assert "proposed_roles" not in row


def test_direct_p16_id_sits_beside_its_own_text_and_cannot_borrow_next_paragraph_number():
    raw = case_with_paragraphs(18, numeric_index=16)
    prepared = prepare_lc_case(raw)
    target = next(row for row in prepared["catalog"] if row["source_window_id"] == "P16")
    native = [s for run in prepared["model_input"]["source_runs"] for s in run["segments"] if s["kind"] == "paragraph"]
    assert next(s for s in native if s["id"] == "P16")["text"] == raw["paragraphs"][15]["text"]
    assert "50人" in next(s for s in native if s["id"] == "P17")["text"]
    assert not any("W" in row["source_window_id"] for row in prepared["catalog"])
    output = output_for(prepared, target)
    output["evidence"][0]["fact_des"] = "项目已投入50人。"
    with pytest.raises(ValueError, match="annotation_number_not_in_source"):
        compile_business(raw, prepared, output)
    # The longer explicit range legitimately includes both complete parents and their real gap.
    output["evidence"][0]["source_window_id"] = "P16..P17"
    accepted = compile_business(raw, prepared, output)["artifact"]["evidence"][0]
    assert accepted["source_paragraph_ids"] == [p["id"] for p in raw["paragraphs"][15:17]]
    assert "50人" in accepted["quote"] and "\r\n<保留😀原始分隔>\n" in accepted["quote"]


def test_six_paragraph_and_unknown_cross_run_ranges_are_not_in_native_schema():
    prepared = prepare_lc_case(case_with_paragraphs())
    assert "P1..P6" not in prepared["model_input"]["source_ranges"]
    output = output_for(prepared)
    output["evidence"][0]["source_window_id"] = "P1..P6"
    with pytest.raises(ValueError, match="output_schema"):
        compile_business(case_with_paragraphs(), prepared, output)


def test_old_w_catalog_or_prepared_version_is_not_silently_reinterpreted_as_direct_p_ranges():
    raw, prepared = raw_case(), prepare_lc_case(raw_case())
    stale = deepcopy(prepared)
    stale["format"] = "lc-source-bundle-prepared-v1"
    with pytest.raises(ValueError, match="prepared_case_required"):
        selection_schema(stale)
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_business(raw, stale, output_for(prepared))
    output = output_for(prepared)
    output["evidence"][0]["source_window_id"] = "W1"
    with pytest.raises(ValueError, match="output_schema"):
        compile_business(raw, prepared, output)


@pytest.mark.parametrize("mutation", ["unknown_window", "wrong_catalog", "duplicate_window", "same_anchor", "missing_argument", "quote", "time", "number", "entity", "empty_content"])
def test_invalid_model_output_is_rejected_as_a_whole_without_repair(mutation):
    raw, prepared = raw_case(), prepare_lc_case(raw_case())
    output = output_for(prepared)
    row = output["evidence"][0]
    if mutation == "unknown_window":
        row["source_window_id"] = "W999999"
    elif mutation == "wrong_catalog":
        output["source_catalog_id"] = "Cwrong"
    elif mutation == "duplicate_window":
        output["evidence"].append(deepcopy(row))
    elif mutation == "same_anchor":
        other = next(w for w in prepared["catalog"] if w["source_bundle"]["paragraph_id"] == prepared["catalog"][0]["source_bundle"]["paragraph_id"] and w["source_window_id"] != row["source_window_id"])
        output["evidence"].append(output_for(prepared, other)["evidence"][0])
    elif mutation == "missing_argument":
        del row["argument"]["时间"]
    elif mutation == "quote":
        row["quote"] = "模型自行传入的源文不受信。"
    elif mutation == "time":
        row["argument"]["时间"] = "2026年"
    elif mutation == "number":
        row["fact_des"] = "已经批准99个项目。"
    elif mutation == "entity":
        row["entities"] = ["原文没有的机构"]
    else:
        row["argument"]["内容"] = ""
    original = deepcopy(output)
    with pytest.raises(ValueError, match="lc_bundle_"):
        compile_business(raw, prepared, output)
    assert output == original


def test_mutable_prepared_catalog_or_delivered_segment_is_never_authoritative():
    raw = raw_case()
    prepared = prepare_lc_case(raw)
    output = output_for(prepared)
    altered = deepcopy(prepared)
    altered["source_runs"][0]["segments"][0]["text"] += "篡改"
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_business(raw, altered, output)
    assert prepare_lc_case(raw) == prepared


def test_original_local_exclusion_only_blocks_intersecting_windows_and_never_cuts_source():
    raw = raw_case()
    row = raw["paragraphs"][0]
    excluded = [(row["id"], 0, 2, row["text"][:2])]
    prepared = prepare_lc_case(raw, excluded_spans=excluded)
    assert prepared["model_input"]["source_runs"][0]["segments"][0]["text"] == row["text"]
    assert all(row["id"] not in w["source_bundle"]["source_paragraph_ids"] for w in prepared["catalog"])
    assert row["id"] in prepared["scope"]["returned_complete_paragraph_ids"]
    assert compile_business(raw, prepared, {"source_catalog_id": prepared["source_catalog_id"], "evidence": []}, excluded_spans=excluded)["artifact"]["evidence"] == []
    with pytest.raises(ValueError, match="exclusion_original_span"):
        prepare_lc_case(raw, excluded_spans=[(row["id"], 0, 2, "错误")])


def test_raw_references_and_changed_source_bytes_fail_closed():
    raw = raw_case()
    raw["documents"][0]["reference_answer"] = "严禁进入运行库"
    with pytest.raises(ValueError, match="raw_only_inputs"):
        prepare_lc_case(raw)
    raw = raw_case()
    raw["documents"][0]["content"] += "篡改"
    with pytest.raises(ValueError, match="sha256"):
        prepare_lc_case(raw)


def test_cli_duplicate_keys_raw_case_hash_and_request_shape_are_strict(tmp_path):
    with pytest.raises(ValueError, match="duplicate_json_key"):
        decode(b'{"operation":"prepare","operation":"compile"}')
    raw = canonical(raw_case())
    with pytest.raises(ValueError, match="bound_raw_case_changed"):
        execute(raw, "0" * 64, {"operation": "prepare"})
    with pytest.raises(ValueError, match="request_shape"):
        execute(raw, sha256(raw).hexdigest(), {"operation": "prepare", "shell": "not a tool"})
    casefile = tmp_path / "raw.json"
    casefile.write_bytes(raw)
    script = Path(__file__).resolve().parents[1] / "scripts/lc_bundle_case.py"
    run = subprocess.run([sys.executable, "-I", "-B", str(script), str(casefile), sha256(raw).hexdigest()],
        input=b'{"operation":"prepare"}', capture_output=True, timeout=20)
    assert run.returncode == 0, run.stderr.decode("utf8", errors="replace")
    result = json.loads(run.stdout)
    assert result["result"] == prepare_lc_case(raw_case()) and result["model_calls"] == 0 and result["publishable"] is False
    output = output_for(result["result"])
    compiled = execute(raw, sha256(raw).hexdigest(), {"operation": "compile", "output": output})
    assert compiled["result"] == compile_business(raw_case(), result["result"], output)


def ranking_input(raw, score=None):
    queries = rank_queries(raw)
    return [{"paragraph_id": row["id"], "source_text_sha256": sha256(row["text"].encode()).hexdigest(),
        "facet_scores": [{"query_id": query["query_id"],
            "max_window_logit": score(index, qindex) if score else float(-index)}
            for qindex, query in enumerate(queries)]} for index, row in enumerate(raw["paragraphs"])]


def ranking_plan(raw, score=None):
    producer = {key: sha256(key.encode()).hexdigest() for key in RANK_PRODUCER_HASHES}
    producer["formatter_id"] = "original-full-question-literal-alternatives-body-bge-v1"
    producer["formatter_sha256"] = sha256(producer["formatter_id"].encode()).hexdigest()
    plan = build_rank_plan(raw, ranking_input(raw, score), producer)
    return plan, sha256(canonical(plan)).hexdigest()


def multi_document_case(counts=(12, 12, 12, 12)):
    result = case_with_paragraphs(1)
    result["documents"], result["paragraphs"] = [], []
    for index, count in enumerate(counts):
        source = case_with_paragraphs(count)
        doc = source["documents"][0]
        doc.update(id=f"doc-{index}", raw_doc_id=f"原文档{index}")
        for row, item in zip(source["paragraphs"], doc["paragraph_manifest"]):
            row.update(id=f"doc{index}-" + row["id"], document_id=doc["id"], raw_doc_id=doc["raw_doc_id"])
            item["id"] = row["id"]
        doc["paragraph_manifest_sha256"] = paragraph_manifest_sha256(doc["paragraph_manifest"])
        result["documents"].append(doc)
        result["paragraphs"].extend(source["paragraphs"])
    return result


def test_rank_queries_are_exact_literal_facets_and_full_source_binding_uses_every_original():
    raw = multi_document_case((5, 4))
    question = " 原说法A 还是 原说法B还是补充？ "
    raw["question"]["question"] = question
    assert rank_queries(raw) == [{"query_id": "F0", "text": question},
        {"query_id": "F1", "text": question + "\n重点核验： 原说法A "},
        {"query_id": "F2", "text": question + "\n重点核验： 原说法B还是补充？ "}]
    binding = rank_source_binding(raw)
    assert binding["paragraph_count"] == 9 and binding["document_count"] == 2
    assert binding["raw_case_canonical_sha256"] == sha256(canonical(raw)).hexdigest()
    raw["question"]["question"] = "没有比较词的问题？"
    assert len(rank_queries(raw)) == 1 and rank_source_binding(raw) != binding


def test_equal_weight_facet_rrf_is_scale_invariant_and_softcap_defers_instead_of_deleting():
    raw = multi_document_case()
    raw["question"]["question"] = "A还是B？"
    values = lambda index, facet: (index if facet == 1 else -index if facet == 0 else -(index % 11))
    plan, bound = ranking_plan(raw, values)
    receipt = validate_rank_plan(raw, plan, bound)
    rank = {row["id"]: i for i, row in enumerate(raw["paragraphs"])}
    scores = {pid: 0 for pid in rank}
    for facet in range(3):
        ordered = sorted(rank, key=lambda pid: (-values(rank[pid], facet), rank[pid]))
        for position, pid in enumerate(ordered, 1):
            scores[pid] += 1 / (60 + position)
    expected_rank = sorted(rank, key=lambda pid: (-scores[pid], rank[pid]))
    assert receipt["ranked_paragraph_ids"] == expected_rank
    by_id = {row["id"]: row for row in raw["paragraphs"]}
    first, deferred, counts = [], [], {}
    for pid in expected_rank:
        doc = by_id[pid]["document_id"]
        if counts.get(doc, 0) < 3:
            first.append(pid); counts[doc] = counts.get(doc, 0) + 1
        else:
            deferred.append(pid)
    assert plan["selected_paragraph_ids"] == (first + deferred)[:32]
    assert len(plan["parent_scores"]) == 48 and len(plan["selected_paragraph_ids"]) == 32
    assert set(receipt["ranked_paragraph_ids"]) == set(rank)
    transformed, transformed_sha = ranking_plan(raw, lambda i, f: values(i, f) * (1000000 if f == 1 else 2) + f * 100000)
    assert validate_rank_plan(raw, transformed, transformed_sha)["selected_paragraph_ids"] == plan["selected_paragraph_ids"]
    assert receipt["selection"] == RANK_SELECTION and receipt["semantic_verified"] is False


@pytest.mark.parametrize("mutation", ["score_hash", "nan", "inf", "bool", "duplicate_parent", "missing_parent",
    "unknown_parent", "missing_facet", "duplicate_facet", "source_hash", "selected", "query", "producer_path", "reference"])
def test_rank_plan_tampering_missing_scores_and_unknown_fields_fail_closed(mutation):
    raw = multi_document_case()
    raw["question"]["question"] = "A还是B？"
    plan, bound = ranking_plan(raw)
    if mutation == "score_hash":
        plan["parent_scores"][0]["facet_scores"][0]["max_window_logit"] -= .01
    elif mutation in {"nan", "inf", "bool"}:
        plan["parent_scores"][0]["facet_scores"][0]["max_window_logit"] = {"nan": float("nan"), "inf": float("inf"), "bool": True}[mutation]
    elif mutation == "duplicate_parent":
        plan["parent_scores"][1] = deepcopy(plan["parent_scores"][0])
    elif mutation == "missing_parent":
        plan["parent_scores"].pop()
    elif mutation == "unknown_parent":
        plan["parent_scores"][0]["paragraph_id"] = "not-an-original"
    elif mutation == "missing_facet":
        plan["parent_scores"][0]["facet_scores"].pop()
    elif mutation == "duplicate_facet":
        plan["parent_scores"][0]["facet_scores"][1] = deepcopy(plan["parent_scores"][0]["facet_scores"][0])
    elif mutation == "source_hash":
        plan["parent_scores"][0]["source_text_sha256"] = "0" * 64
    elif mutation == "selected":
        plan["selected_paragraph_ids"].reverse()
    elif mutation == "query":
        plan["queries"][1]["text"] = "删掉完整原题的查询"
    elif mutation == "producer_path":
        plan["producer"]["formatter_id"] = "D:/arbitrary/source.json"
    else:
        plan["reference_answers"] = []
    with pytest.raises(ValueError, match="lc_bundle_rank_"):
        prepare_lc_case(raw, rank_plan=plan, rank_plan_sha256=bound)


def test_rank_source_change_or_partial_plan_never_silently_falls_back_to_rrf():
    raw = case_with_paragraphs()
    plan, bound = ranking_plan(raw)
    changed = deepcopy(raw)
    changed["question"]["question"] += "附加条件"
    with pytest.raises(ValueError, match="rank_full_source_or_question_changed"):
        prepare_lc_case(changed, rank_plan=plan, rank_plan_sha256=bound)
    changed = deepcopy(raw)
    changed["paragraphs"] = changed["paragraphs"][:-1]
    with pytest.raises(ValueError):
        prepare_lc_case(changed, rank_plan=plan, rank_plan_sha256=bound)
    for kwargs in ({"rank_plan": plan}, {"rank_plan_sha256": bound}):
        with pytest.raises(ValueError, match="required_together"):
            prepare_lc_case(raw, **kwargs)
    assert prepare_lc_case(raw) == prepare_lc_case(raw, rank_plan=None, rank_plan_sha256=None)


def test_rank_branch_keeps_whole_raw_sources_stable_manifest_aliases_and_unknown_unprovided(monkeypatch):
    raw = case_with_paragraphs(40, long=True)
    # All score ties use raw row order; aliases must instead use the full original manifest order.
    raw["paragraphs"].reverse()
    plan, bound = ranking_plan(raw)
    original = deepcopy(raw)
    monkeypatch.setattr(SourceBundleTools, "search", lambda *_a, **_k: pytest.fail("rank plan must not fall back to BM25"))
    prepared = prepare_lc_case(raw, rank_plan=plan, rank_plan_sha256=bound)
    assert raw == original and prepared["scope"]["corpus_paragraph_count"] == 40
    assert len(prepared["scope"]["returned_complete_paragraph_ids"]) == 32
    assert len(prepared["scope"]["unprovided_paragraph_ids"]) == 8
    assert prepared["scope"]["unprovided_context"] == "unknown_not_irrelevant"
    native = [s for run in prepared["model_input"]["source_runs"] for s in run["segments"] if s["kind"] == "paragraph"]
    manifest = raw["documents"][0]["paragraph_manifest"]
    by_id = {row["id"]: row for row in raw["paragraphs"]}
    expected_aliases = {row["id"]: f"P{i+1}" for i, row in enumerate(manifest)}
    for run in prepared["source_runs"]:
        for segment in run["segments"]:
            if segment["kind"] == "paragraph":
                assert segment["ref"] == expected_aliases[segment["paragraph_id"]]
                assert segment["text"] == by_id[segment["paragraph_id"]]["text"]
                assert len(segment["text"]) > 5000
    assert {s["id"] for s in native} == set(expected_aliases[pid] for pid in plan["selected_paragraph_ids"])
    assert "max_window_logit" not in prepared["messages"][1]["content"] and "parent_scores" not in prepared["messages"][1]["content"]
    assert prepared["source_binding"]["rank_plan_sha256"] == bound
    output = output_for(prepared)
    compiled = compile_business(raw, prepared, output, rank_plan=plan, rank_plan_sha256=bound)
    assert compiled["scope"]["binding"] == prepared["source_binding"]
    assert compiled["artifact"]["evidence"][0]["quote"] == prepared["catalog"][0]["source_bundle"]["quote"]
    with pytest.raises(ValueError, match="rank_plan_and_sha256_required_together"):
        compile_business(raw, prepared, output)


def test_rank_citation_cannot_cross_an_unprovided_original_or_distinct_delivery_run():
    raw = case_with_paragraphs(40)
    # Lowest eight original parents are all interior; selecting a later parent cannot fill those gaps.
    low = {5, 6, 15, 16, 25, 26, 35, 36}
    plan, bound = ranking_plan(raw, lambda i, _: -100 if i in low else -i)
    prepared = prepare_lc_case(raw, rank_plan=plan, rank_plan_sha256=bound)
    assert len(prepared["source_runs"]) == 5
    assert "P5..P8" not in prepared["model_input"]["source_ranges"]
    assert "P10..P15" not in prepared["model_input"]["source_ranges"]
    assert all(len(w["source_bundle"]["source_paragraph_ids"]) <= 5 for w in prepared["catalog"])
    output = output_for(prepared)
    output["evidence"][0]["source_window_id"] = "P5..P8"
    with pytest.raises(ValueError, match="output_schema"):
        compile_business(raw, prepared, output, rank_plan=plan, rank_plan_sha256=bound)


def test_cli_rank_plan_and_verify_rank_are_explicit_bound_operations(monkeypatch):
    import app.domain.local_parent_rank as verifier
    raw = case_with_paragraphs()
    plan, bound = ranking_plan(raw)
    data, actual_sha = canonical(raw) + b"\n", sha256(canonical(raw) + b"\n").hexdigest()
    request = {"operation": "prepare", "rank_plan": plan, "rank_plan_sha256": bound}
    prepared = execute(data, actual_sha, request)["result"]
    compiled = execute(data, actual_sha, {**request, "operation": "compile", "output": output_for(prepared)})
    assert compiled["result"] == compile_business(raw, prepared, output_for(prepared), rank_plan=plan, rank_plan_sha256=bound)
    for broken in ({"operation": "prepare", "rank_plan": plan},
                   {"operation": "prepare", "rank_plan": None, "rank_plan_sha256": None},
                   {"operation": "verify_rank", "rank_plan": plan, "rank_plan_sha256": bound}):
        with pytest.raises(ValueError, match="request_shape"):
            execute(data, actual_sha, broken)
    calls = []
    def verified(raw_case, supplied, proof, *, expected_model_manifest_sha256, raw_case_sha256):
        calls.append((raw_case, supplied, proof, expected_model_manifest_sha256, raw_case_sha256))
        return {"format": "verified-test-only", "provider_calls": 0}
    monkeypatch.setattr(verifier, "verify_local_rank", verified)
    proof = {"file": "test-only-immutable-archive", "sha256": "a" * 64, "bytes": 12}
    checked = execute(data, actual_sha, {"operation": "verify_rank", "rank_plan": plan,
        "rank_plan_sha256": bound, "local_rank_proof": proof, "expected_model_manifest_sha256": "b" * 64})
    assert calls == [(raw, plan, proof, "b" * 64, actual_sha)]
    assert checked["result"] == {"format": "verified-test-only", "provider_calls": 0}
    assert checked["model_calls"] == 0 and actual_sha != rank_source_binding(raw)["raw_case_canonical_sha256"]


def ids_output(prepared, aliases=()):
    return {"source_catalog_id": prepared["source_catalog_id"], "source_window_ids": list(aliases)}


def test_selection_only_schema_has_only_bound_ids_and_default_protocol_is_unchanged():
    raw = raw_case()
    old = prepare_lc_case(raw)
    new = prepare_lc_case(raw, selection_only=True)
    assert prepare_lc_case(raw, selection_only=False) == old
    assert new["function_name"] == "LCSourceBundleIDsV1" and old["function_name"] == "LCSourceBundleSelectionV2"
    assert new["source_catalog_id"] != old["source_catalog_id"]
    assert new["source_binding"]["selection_protocol"] == "lc-source-bundle-ids-v1"
    assert new["scope"]["model_generated_annotations"] is False
    assert new["source_runs"] == old["source_runs"] and new["catalog"] == old["catalog"]
    assert new["wire_tool"] == strict_tool(selection_schema(new))
    schema = new["wire_tool"]["function"]["parameters"]
    assert set(schema["properties"]) == {"source_catalog_id", "source_window_ids"}
    assert set(schema["properties"]["source_window_ids"]["items"]["enum"]) == set(new["model_input"]["source_ranges"])
    for model_output, prepared, mode in [(output_for(new), new, True),
                                        (ids_output(old, ["P1"]), old, False)]:
        with pytest.raises(ValueError, match="output_schema"):
            compile_business(raw, prepared, model_output, selection_only=mode)
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_business(raw, new, ids_output(new, ["P1"]))
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_business(raw, old, output_for(old), selection_only=True)


def test_selection_only_merges_prefix_duplicates_and_keeps_full_original_choice_audit():
    raw = raw_case()
    prepared = prepare_lc_case(raw, selection_only=True)
    choices = ["P1", "P1..P2", "P1", "P1..P3", "P1..P2", "P2"]
    output = ids_output(prepared, choices)
    original = deepcopy(output)
    compiled = compile_business(raw, prepared, output, selection_only=True)
    artifact = compiled["artifact"]
    assert output == original and artifact["raw_model_selection"] == original
    assert [r["source_window_id"] for r in artifact["evidence"]] == ["P1..P3", "P2"]
    assert artifact["selection_audit"]["original_selection_count"] == 6
    assert artifact["selection_audit"]["retained_selection_count"] == 2
    audit = artifact["selection_audit"]["groups"][0]
    assert audit["input_indices"] == [0, 1, 2, 3, 4] and audit["input_source_window_ids"] == choices[:5]
    assert audit["retained_source_window_id"] == "P1..P3"
    row = artifact["evidence"][0]
    assert row["quote"] == prepared["catalog"][2]["source_bundle"]["quote"]
    assert row["source_paragraph_ids"] == [p["id"] for p in raw["paragraphs"]]
    assert "\r\n<实际分隔>" in row["quote"] and "😀" in row["quote"]
    assert row["fact_des"] == row["argument"]["内容"] == "项目申请已受理😀。"
    assert all(row["argument"][key] == "" for key in ["时间", "主体", "地点", "影响", "数值口径"])
    assert row["entities"] == [] and artifact["model_generated_annotations"] is False
    assert artifact["annotation_policy"] == "program_literal_lead_sentence_not_abstractive_summary"
    assert row["downstream_evaluation_basis"] == artifact["downstream_evaluation_basis"] == "complete_original_quote_required"
    assert artifact["native_model_delivery_verified"] is False and artifact["publishable"] is False


@pytest.mark.parametrize("text,expected", [
    ("\r\n  首句😀。\r\n后句不得丢出quote。", "首句😀。"),
    ("首行无句号\r\n第二行。", "首行无句号"),
    ("  没有句号的完整原文  ", "  没有句号的完整原文  "),
    ("Value is 3.14. Next statement.", "Value is 3.14."),
    ("Is the claim true? Later detail.", "Is the claim true?"),
    ("No! Later detail.", "No!"),
])
def test_selection_only_program_lead_is_an_exact_source_sentence_without_number_or_time_inference(text, expected):
    from app.domain.lc_bundle_case import _literal_lead_sentence
    assert _literal_lead_sentence(text) == expected and expected in text


def test_selection_only_unknown_ids_reject_the_whole_output_and_ranges_stay_bounded(monkeypatch):
    raw = case_with_paragraphs()
    retrieve_only(monkeypatch, {raw["paragraphs"][i]["id"] for i in [0, 2, 3, 4, 5, 6, 7]})
    prepared = prepare_lc_case(raw, selection_only=True)
    assert "P1..P2" not in prepared["model_input"]["source_ranges"]  # Manifest gap, despite compressed default P aliases.
    # Choose legal aliases by the bound catalog and then append one unknown. Nothing is silently dropped.
    for invalid in ["P1..P7", "W1", "missing", "P1..P2"]:
        with pytest.raises(ValueError, match="output_schema"):
            compile_business(raw, prepared, ids_output(prepared, ["P1", invalid]), selection_only=True)
    assert compile_business(raw, prepared, ids_output(prepared), selection_only=True)["artifact"]["evidence"] == []


def test_selection_only_defensively_rejects_incomparable_same_anchor_instead_of_unioning():
    from app.domain.lc_bundle_case import _compile_source_ids
    prepared = prepare_lc_case(case_with_paragraphs(), selection_only=True)
    altered = deepcopy(prepared)
    first = next(w for w in altered["catalog"] if w["source_window_id"] == "P1..P2")
    second = next(w for w in altered["catalog"] if w["source_window_id"] == "P1..P3")
    second["source_bundle"]["source_paragraph_ids"][1] = "a-nonprefix-original-id"
    with pytest.raises(ValueError, match="incomparable_same_start"):
        _compile_source_ids(altered, ids_output(altered, [first["source_window_id"], second["source_window_id"]]))


def test_selection_only_cli_mode_is_explicit_bool_and_rank_branch_keeps_source_binding():
    raw = case_with_paragraphs()
    plan, bound = ranking_plan(raw)
    data = canonical(raw)
    request = {"operation": "prepare", "selection_only": True, "rank_plan": plan, "rank_plan_sha256": bound}
    prepared = execute(data, sha256(data).hexdigest(), request)["result"]
    assert prepared["source_binding"]["selection_only"] is True and prepared["source_binding"]["rank_plan_sha256"] == bound
    output = ids_output(prepared, ["P1", "P1..P2"])
    result = execute(data, sha256(data).hexdigest(), {**request, "operation": "compile", "output": output})
    assert result["result"]["artifact"]["evidence"][0]["source_window_id"] == "P1..P2"
    for value in [1, "true", None]:
        with pytest.raises(ValueError, match="request_shape"):
            execute(data, sha256(data).hexdigest(), {"operation": "prepare", "selection_only": value})
        with pytest.raises(ValueError, match="boolean_required"):
            prepare_lc_case(raw, selection_only=value)


def graded_output(prepared, rows=()):
    return {"source_catalog_id": prepared["source_catalog_id"], "ranked_candidates": [
        {"source_window_id": alias, "relevance_score": score, "use_type": kind} for alias, score, kind in rows]}


def test_graded_ids_schema_and_policy_are_bound_without_changing_either_old_mode():
    raw = raw_case()
    ids = prepare_lc_case(raw, selection_only=True)
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    assert prepare_lc_case(raw, graded_selection=False) == prepare_lc_case(raw)
    assert prepare_lc_case(raw, selection_only=True, graded_selection=False) == ids
    assert prepared["function_name"] == "LCSourceBundleRankedIDsV1"
    assert prepared["source_catalog_id"] != ids["source_catalog_id"]
    assert prepared["source_runs"] == ids["source_runs"] and prepared["catalog"] == ids["catalog"]
    assert prepared["source_binding"]["graded_selection_policy"]["maximum_selected_anchors"] == 10
    assert prepared["scope"]["use_type_is_final_relation"] is False and prepared["scope"]["selection_limit"] == 10
    assert prepared["wire_tool"] == strict_tool(selection_schema(prepared))
    parameters = prepared["wire_tool"]["function"]["parameters"]
    assert set(parameters["properties"]) == {"source_catalog_id", "ranked_candidates"}
    row = parameters["properties"]["ranked_candidates"]["items"]
    assert set(row["properties"]) == {"source_window_id", "relevance_score", "use_type"}
    assert row["properties"]["relevance_score"]["enum"] == [0, 1, 2, 3]
    assert "source_window_ids" not in parameters["properties"]
    assert "只输出source_catalog_id和source_window_ids" not in prepared["messages"][0]["content"]
    with pytest.raises(ValueError, match="requires_selection_only"):
        prepare_lc_case(raw, graded_selection=True)
    for output, target, kwargs in [
        (ids_output(prepared, ["P1"]), prepared, {"selection_only": True, "graded_selection": True}),
        (graded_output(ids, [("P1", 3, "direct_answer")]), ids, {"selection_only": True}),
    ]:
        with pytest.raises(ValueError, match="output_schema"):
            compile_business(raw, target, output, **kwargs)


@pytest.mark.parametrize("mutation", ["bool", "float", "string", "out_of_range", "unknown_type", "unknown_id", "extra_annotation", "missing_type"])
def test_graded_candidate_types_and_closed_fields_reject_whole_output(mutation):
    raw = raw_case()
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    output = graded_output(prepared, [("P1", 3, "direct_answer")])
    row = output["ranked_candidates"][0]
    if mutation in {"bool", "float", "string", "out_of_range"}:
        row["relevance_score"] = {"bool": True, "float": 3.0, "string": "3", "out_of_range": 4}[mutation]
    elif mutation == "unknown_type":
        row["use_type"] = "officially_verified"
    elif mutation == "unknown_id":
        row["source_window_id"] = "P1..P99"
    elif mutation == "extra_annotation":
        row["fact_des"] = "不应生成的自由注释"
    else:
        row.pop("use_type")
    original = deepcopy(output)
    with pytest.raises(ValueError, match="output_schema"):
        compile_business(raw, prepared, output, selection_only=True, graded_selection=True)
    assert output == original


@pytest.mark.parametrize("use_type,score", [("background", 2), ("background", 3), ("unrelated", 2), ("unrelated", 3)])
def test_graded_background_cannot_be_promoted_to_necessary_evidence(use_type, score):
    raw = raw_case()
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    output = graded_output(prepared, [("P1", 3, "direct_answer"), ("P2", score, use_type)])
    with pytest.raises(ValueError, match="inconsistent_background_score"):
        compile_business(raw, prepared, output, selection_only=True, graded_selection=True)


def test_graded_top10_is_score_then_original_rank_with_audited_counterclaim_replacement():
    raw = case_with_paragraphs(14)
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    choices = [(f"P{i}", 3, "direct_answer") for i in range(1, 9)] + [
        ("P9", 2, "necessary_condition"), ("P10", 2, "necessary_condition"), ("P11", 1, "background"),
        ("P12", 2, "counterclaim"), ("P13", 2, "counterclaim"), ("P14", 0, "unrelated")]
    output = graded_output(prepared, choices)
    artifact = compile_business(raw, prepared, output, selection_only=True, graded_selection=True)["artifact"]
    assert artifact["raw_model_selection"] == output
    assert [row["source_window_id"] for row in artifact["evidence"]] == [f"P{i}" for i in range(1, 10)] + ["P12"]
    audit = artifact["selection_audit"]
    assert audit["original_candidate_count"] == 14 and audit["merged_anchor_count"] == 14
    assert audit["eligible_anchor_count"] == 12 and audit["retained_selection_count"] == 10
    assert audit["original_top10_source_window_ids"] == [f"P{i}" for i in range(1, 11)]
    coverage = audit["counterclaim_input_coverage"]
    assert coverage["required"] and coverage["replacement_applied"] and not coverage["already_in_top10"]
    assert coverage["inserted_source_window_id"] == "P12" and coverage["removed_source_window_id"] == "P10"
    assert not coverage["is_final_conflict_label"] and not artifact["use_type_is_final_relation"]
    reasons = {row["source_window_id"]: row["reason"] for row in audit["candidate_decisions"]}
    assert reasons["P10"] == "replaced_for_counterclaim_input_coverage"
    assert reasons["P13"] == "outside_score_rank_top10"
    assert reasons["P11"] == reasons["P14"] == "below_score_threshold_not_an_independent_adoption"
    for row in artifact["evidence"]:
        assert row["quote"] == next(w["source_bundle"]["quote"] for w in prepared["catalog"] if w["source_window_id"] == row["source_window_id"])
        assert row["fact_des"] == row["argument"]["内容"] and row["entities"] == []
        assert "use_type" not in row and "supporting_facts" not in row and "conflicting_facts" not in row


def test_graded_same_anchor_longest_merge_preserves_each_grade_and_counterclaim_identity():
    raw = raw_case()
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    choices = [("P1", 3, "direct_answer"), ("P1..P2", 2, "counterclaim"),
        ("P1..P3", 1, "background"), ("P1", 3, "direct_answer"), ("P2", 1, "background")]
    output = graded_output(prepared, choices)
    artifact = compile_business(raw, prepared, output, selection_only=True, graded_selection=True)["artifact"]
    assert artifact["raw_model_selection"] == output
    assert len(artifact["evidence"]) == 1 and artifact["evidence"][0]["source_window_id"] == "P1..P3"
    assert artifact["evidence"][0]["source_paragraph_ids"] == [row["id"] for row in raw["paragraphs"]]
    assert "\r\n<实际分隔>" in artifact["evidence"][0]["quote"]
    audit = artifact["selection_audit"]
    first = audit["groups"][0]
    assert first["input_indices"] == [0, 1, 2, 3] and first["representative_input_index"] == 0
    assert first["representative_use_type"] == "direct_answer" and first["qualified_counterclaim_input_indices"] == [1]
    assert audit["counterclaim_input_coverage"]["already_in_top10"]
    assert not audit["counterclaim_input_coverage"]["replacement_applied"]
    assert audit["candidate_decisions"][2]["score_eligible"] is False
    assert audit["candidate_decisions"][2]["source_anchor_retained"] is True
    assert all(artifact["evidence"][0]["argument"][key] == "" for key in ["时间", "主体", "地点", "影响", "数值口径"])


def test_graded_coverage_uses_highest_qualified_counterclaim_and_does_not_fill_low_scores():
    raw = case_with_paragraphs(14)
    prepared = prepare_lc_case(raw, selection_only=True, graded_selection=True)
    choices = [(f"P{i}", 3, "direct_answer") for i in range(1, 12)] + [
        ("P12", 2, "counterclaim"), ("P13", 3, "counterclaim"), ("P14", 1, "counterclaim")]
    artifact = compile_business(raw, prepared, graded_output(prepared, choices), selection_only=True, graded_selection=True)["artifact"]
    assert artifact["selection_audit"]["counterclaim_input_coverage"]["inserted_source_window_id"] == "P13"
    sparse = graded_output(prepared, [("P5", 2, "necessary_condition"), ("P4", 3, "direct_answer"),
        ("P3", 3, "direct_answer"), ("P1", 1, "counterclaim"), ("P2", 0, "unrelated")])
    sparse_artifact = compile_business(raw, prepared, sparse, selection_only=True, graded_selection=True)["artifact"]
    assert [row["source_window_id"] for row in sparse_artifact["evidence"]] == ["P4", "P3", "P5"]
    assert not sparse_artifact["selection_audit"]["counterclaim_input_coverage"]["required"]
    empty = compile_business(raw, prepared, graded_output(prepared), selection_only=True, graded_selection=True)
    assert empty["artifact"]["evidence"] == [] and empty["artifact"]["selection_audit"]["original_candidate_count"] == 0


def test_graded_cli_is_explicit_typed_and_does_not_reinterpret_prepared_or_old_output():
    raw = raw_case()
    data, raw_sha = canonical(raw), sha256(canonical(raw)).hexdigest()
    request = {"operation": "prepare", "selection_only": True, "graded_selection": True}
    prepared = execute(data, raw_sha, request)["result"]
    output = graded_output(prepared, [("P1", 3, "direct_answer")])
    artifact = execute(data, raw_sha, {**request, "operation": "compile", "output": output})["result"]["artifact"]
    assert artifact["graded_selection"] and artifact["raw_model_selection"] == output
    with pytest.raises(ValueError, match="requires_selection_only"):
        execute(data, raw_sha, {"operation": "prepare", "graded_selection": True})
    for value in [1, "true", None]:
        with pytest.raises(ValueError, match="request_shape"):
            execute(data, raw_sha, {**request, "graded_selection": value})
    altered = deepcopy(prepared)
    altered["source_binding"]["graded_selection_policy"]["maximum_selected_anchors"] = 9
    with pytest.raises(ValueError, match="prepared_contract_changed"):
        compile_business(raw, altered, output, selection_only=True, graded_selection=True)
