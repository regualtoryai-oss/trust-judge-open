from copy import deepcopy
from hashlib import sha256

import pytest

from app.domain.frozen_selection_inputs import canonical
from app.domain.source_bundle_tools import SourceBundleTools
from app.domain.source_bundles import assemble_source_bundle, paragraph_manifest_sha256


QUESTION = {"question_id": "opaque-question", "question": "项目是已经完成，还是尚未确认完成？"}


def corpus(pieces, *, doc_id="opaque-doc", raw_doc_id="原文甲", title="项目记录", gaps=None):
    gaps = gaps if gaps is not None else ["\r\n<原分隔😀>"] * max(0, len(pieces) - 1)
    content, rows, manifest = "前缀|", [], []
    for index, text in enumerate(pieces):
        start = len(content)
        content += text
        end = len(content)
        pid, raw_pid = f"{doc_id}-row-{index}", f"原始-{chr(937 - index)}"
        rows.append({"id": pid, "document_id": doc_id, "raw_doc_id": raw_doc_id,
            "raw_para_id": raw_pid, "text": text, "document_start_cp": start, "document_end_cp": end})
        manifest.append({"id": pid, "raw_para_id": raw_pid, "start_cp": start, "end_cp": end,
                         "text_sha256": sha256(text.encode()).hexdigest()})
        if index < len(gaps):
            content += gaps[index]
    content += "|尾部"
    doc = {"id": doc_id, "raw_doc_id": raw_doc_id, "title": title, "content": content,
           "content_sha256": sha256(content.encode()).hexdigest(), "paragraph_manifest": manifest,
           "paragraph_manifest_sha256": paragraph_manifest_sha256(manifest)}
    return [doc], rows


def expose_all(tools, rows):
    result = tools.search(QUESTION["question"], limit=60)
    assert {row["paragraph_id"] for row in result["results"]} == {row["id"] for row in rows}
    return result


def test_whole_corpus_search_does_not_accept_a_fixed_candidate_pool_or_cut_to_60():
    docs, rows = corpus([f"项目完成记录编号{index}" for index in range(75)])
    tools = SourceBundleTools(docs, rows, QUESTION)
    first = tools.search(QUESTION["question"], limit=60)
    assert len(first["results"]) == 60
    assert first["matching_parent_count"] == first["corpus_paragraph_count"] == 75
    assert first["more_matching_sources_available"] is True
    seen = [row["paragraph_id"] for row in first["results"]]
    second = tools.search(QUESTION["question"], limit=60, exclude_paragraph_ids=seen)
    assert len(second["results"]) == 15
    assert second["more_matching_sources_available"] is False
    assert set(seen).isdisjoint(row["paragraph_id"] for row in second["results"])
    assert len(tools.scope()["exposed_paragraph_ids"]) == 75
    assert tools.scope()["returned_complete_paragraph_ids"] == []


def test_title_is_a_bounded_preference_not_a_title_only_candidate_or_evidence_vote():
    docs, rows = corpus(["项目完成。"], title="项目进展")
    others, other_rows = corpus(["完全不同的无关信息。"], doc_id="other-doc", raw_doc_id="原文乙",
                               title=QUESTION["question"])
    tools = SourceBundleTools(docs + others, rows + other_rows, QUESTION)
    result = tools.search(QUESTION["question"], limit=60)
    assert [row["paragraph_id"] for row in result["results"]] == [rows[0]["id"]]
    assert tools.scope()["returned_complete_paragraph_ids"] == []


def test_navigation_previews_are_exact_contiguous_codepoints_and_not_read_evidence():
    text = "前置😀" * 40 + "项目已经完成。" + "后置🚀" * 40
    docs, rows = corpus([text, "项目完成。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    result = expose_all(tools, rows)
    for preview in result["results"]:
        original = next(row for row in rows if row["id"] == preview["paragraph_id"])
        assert preview["excerpt_text"] == original["text"][preview["excerpt_start_cp"]:preview["excerpt_end_cp"]]
        assert len(preview["excerpt_text"]) <= 80
        assert preview["navigation_only"] is True
        assert preview["eligible_for_citation"] is False
        assert preview["unread_context_state"] == "unknown_not_irrelevant"
    assert tools.scope()["returned_complete_paragraph_ids"] == []
    with pytest.raises(ValueError, match="not_completely_returned"):
        tools.cite(rows[1]["id"], rows[1]["id"])


def test_source_and_question_are_deep_copied_and_all_binding_outputs_detached():
    docs, rows = corpus(["项目已经完成。", "项目尚未确认。"])
    question = deepcopy(QUESTION)
    bound = deepcopy({"documents": docs, "paragraphs": rows, "question": question})
    tools = SourceBundleTools(docs, rows, question)
    original_binding = tools.scope()["binding"]
    assert original_binding["input_sha256"] == sha256(canonical(bound)).hexdigest()
    docs[0]["content"] = "外部篡改"
    rows[0]["text"] = "外部篡改"
    question["question"] = "外部篡改"
    preview = tools.search(QUESTION["question"], limit=60)
    assert all("外部篡改" not in row["excerpt_text"] for row in preview["results"])
    assert preview["original_question"] == QUESTION["question"]
    scope = tools.scope()
    scope["binding"]["input_sha256"] = "bad"
    scope["exposed_paragraph_ids"].clear()
    assert tools.scope()["binding"] == original_binding
    assert len(tools.scope()["exposed_paragraph_ids"]) == 2


@pytest.mark.parametrize("field", ["gold", "reference", "reference_answers", "fact_des", "argument", "supporting_facts", "Answer"])
def test_reference_or_generated_annotation_fields_rejected_recursively_including_tuples(field):
    docs, rows = corpus(["项目完成。"])
    docs[0]["metadata"] = ({"nested": {field: "不允许的离线答案"}},)
    with pytest.raises(ValueError, match="raw_only_inputs"):
        SourceBundleTools(docs, rows, QUESTION)


def test_entire_trusted_source_manifest_is_validated_even_for_unread_sources():
    docs, rows = corpus(["项目完成。", "尚未确认。"])
    rows[1]["text"] += "篡改"
    with pytest.raises(ValueError, match="paragraph_manifest_mismatch"):
        SourceBundleTools(docs, rows, QUESTION)


def test_read_returns_full_contiguous_document_slice_and_narrow_citation_proof():
    docs, rows = corpus(["项目尚未确认。", "项目在准备。", "项目已经完成😀。", "项目仍待核验🚀。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    read = tools.read([rows[1]["id"]], before=1, after=2)
    context = read["bundles"][0]
    assert context["quote"] == docs[0]["content"][rows[0]["document_start_cp"]:rows[-1]["document_end_cp"]]
    assert "\r\n<原分隔😀>" in context["quote"]
    assert context["bundle_purpose"] == "read_context"
    assert context["source_paragraph_ids"] == [row["id"] for row in rows]
    assert read["complete_original_slices"] is True
    assert read["native_model_delivery_verified"] is False
    assert read["source_bundle_set_sha256"] == sha256(canonical(read["bundles"])).hexdigest()
    assert read["source_codepoints"] == len(context["quote"])
    source = tools.cite(rows[2]["id"], rows[3]["id"])
    canonical_source = assemble_source_bundle(docs, rows, rows[2]["id"], rows[3]["id"])
    assert all(source[key] == value for key, value in canonical_source.items())
    assert source["paragraph_id"] == rows[2]["id"]
    assert source["start_cp"] == rows[2]["document_start_cp"]
    assert source["source_delivery"]["read_bundle_id"] == context["bundle_id"]
    assert source["source_delivery"]["read_result_sha256"] == sha256(canonical(read)).hexdigest()
    assert source["source_delivery"]["read_result_serialization"] == "canonical-json-utf8"
    assert source["bundle_purpose"] == "citation_source"
    assert source["adoption_or_publication_created"] is False
    assert tools.validate_citation(source) == source
    assert tools.scope()["model_calls"] == 0
    assert len(tools.scope()["complete_read_ranges"]) == 1
    assert "quote" not in tools.scope()["complete_read_ranges"][0]


def test_read_context_is_not_implicitly_a_final_citation():
    docs, rows = corpus(["项目已经完成。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    context = tools.read([rows[0]["id"]], before=0, after=0)["bundles"][0]
    with pytest.raises(ValueError, match="read_context_is_not_citation"):
        tools.validate_citation(context)


def test_unexposed_neighbor_is_read_completely_but_unread_neighbor_cannot_be_cited():
    docs, rows = corpus(["项目完成。", "限定条件。", "仍未知。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    result = tools.search(QUESTION["question"], limit=1)
    assert result["results"][0]["paragraph_id"] == rows[0]["id"]
    tools.read([rows[0]["id"]], before=0, after=1)
    assert tools.cite(rows[1]["id"], rows[1]["id"])["quote"] == rows[1]["text"]
    with pytest.raises(ValueError, match="not_completely_returned"):
        tools.cite(rows[1]["id"], rows[2]["id"])


def test_cross_document_or_reverse_range_is_rejected_after_complete_read():
    docs, rows = corpus(["项目完成甲。", "项目完成乙。"])
    docs2, rows2 = corpus(["项目完成丙。"], doc_id="second-doc", raw_doc_id="原文乙")
    tools = SourceBundleTools(docs + docs2, rows + rows2, QUESTION)
    expose_all(tools, rows + rows2)
    tools.read([rows[0]["id"], rows2[0]["id"]], before=0, after=1)
    with pytest.raises(ValueError, match="cross_document_range"):
        tools.cite(rows[0]["id"], rows2[0]["id"])
    with pytest.raises(ValueError, match="reverse_range"):
        tools.cite(rows[1]["id"], rows[0]["id"])
    assert tools.trace()[-1]["status"] == "rejected"


def test_separate_read_bursts_are_not_stitched_into_an_unreturned_bundle():
    docs, rows = corpus(["项目完成甲。", "项目完成乙。", "项目完成丙。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    tools.read([rows[0]["id"]], before=0, after=0)
    tools.read([rows[1]["id"]], before=0, after=1)
    assert len(tools.scope()["returned_complete_paragraph_ids"]) == 3
    assert tools.cite(rows[1]["id"], rows[2]["id"])["source_paragraph_ids"] == [row["id"] for row in rows[1:]]
    with pytest.raises(ValueError, match="not_completely_returned"):
        tools.cite(rows[0]["id"], rows[2]["id"])


def test_read_return_and_receipts_cannot_be_mutated_by_caller():
    docs, rows = corpus(["项目完成。", "项目待确认。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    read = tools.read([rows[0]["id"]], before=0, after=1)
    read["bundles"][0]["quote"] = "返回对象被外部修改"
    read["bundles"][0]["source_paragraph_ids"].clear()
    source = tools.cite(rows[1]["id"], rows[1]["id"])
    source["source_delivery"]["read_bundle_id"] = "伪造"
    with pytest.raises(ValueError, match="delivery_proof_mismatch"):
        tools.validate_citation(source)
    assert tools.cite(rows[1]["id"], rows[1]["id"])["quote"] == rows[1]["text"]
    trace = tools.trace()
    trace[0]["returned_paragraph_ids"].clear()
    assert tools.trace()[0]["returned_paragraph_ids"]


@pytest.mark.parametrize("cap_name,cap_value,error", [
    ("max_read_codepoints", 1, "complete_read_codepoint_budget_exceeded"),
    ("max_read_payload_bytes", 1, "complete_read_payload_byte_budget_exceeded"),
])
def test_complete_read_budgets_reject_whole_payload_without_silent_partial_exposure(cap_name, cap_value, error):
    docs, rows = corpus(["项目完成😀" * 20, "项目未完成🚀" * 20])
    tools = SourceBundleTools(docs, rows, QUESTION, **{cap_name: cap_value})
    expose_all(tools, rows)
    with pytest.raises(ValueError, match=error):
        tools.read([rows[0]["id"]], before=0, after=1)
    assert tools.scope()["returned_complete_paragraph_ids"] == []
    assert tools.scope()["complete_read_ranges"] == []
    assert tools.scope()["calls"]["read"] == 1
    assert tools.trace()[-1]["status"] == "rejected"


def test_identical_windows_deduplicated_explicitly_preserve_all_requested_anchors():
    docs, rows = corpus(["项目完成甲。", "项目完成乙。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    read = tools.read([row["id"] for row in rows], before=1, after=1)
    assert read["result_count"] == 1
    assert read["bundles"][0]["anchor_paragraph_ids"] == [row["id"] for row in rows]


def test_adjacent_anchors_preserve_overlapping_complete_bundles_and_unique_coverage():
    docs, rows = corpus(["项目完成甲😀。", "项目完成乙。", "项目完成丙。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    read = tools.read([rows[0]["id"], rows[1]["id"]], before=0, after=1)
    assert read["result_count"] == 2
    assert read["bundles"][0]["source_paragraph_ids"] == [rows[0]["id"], rows[1]["id"]]
    assert read["bundles"][1]["source_paragraph_ids"] == [rows[1]["id"], rows[2]["id"]]
    assert tools.scope()["returned_complete_paragraph_ids"] == [row["id"] for row in rows]
    assert tools.trace()[-1]["returned_paragraph_ids"] == [row["id"] for row in rows]
    assert read["source_codepoints"] == sum(len(b["quote"]) for b in read["bundles"])
    for bundle in read["bundles"]:
        assert bundle["quote"] == docs[0]["content"][bundle["start_cp"]:bundle["end_cp"]]
    assert tools.cite(rows[1]["id"], rows[2]["id"])["quote"] == read["bundles"][1]["quote"]
    # Seeing adjacent overlapping contexts does not license a newly stitched
    # source range that never appeared as one complete returned bundle.
    with pytest.raises(ValueError, match="not_completely_returned"):
        tools.cite(rows[0]["id"], rows[2]["id"])
    repeated = tools.read([rows[1]["id"]], before=0, after=0)
    assert repeated["bundles"][0]["quote"] == rows[1]["text"]
    assert repeated["delivery"]["new_original_paragraph_ids"] == []


def test_original_overlapping_coordinates_keep_distinct_manifest_ids():
    docs, rows = corpus(["项目完成甲。项目完成乙。"], gaps=[])
    row = deepcopy(rows[0])
    row.update(id="nested-opaque-id", raw_para_id="嵌套原始段", text="项目完成乙。",
               document_start_cp=rows[0]["document_start_cp"] + len("项目完成甲。"))
    rows.append(row)
    docs[0]["paragraph_manifest"].append({"id": row["id"], "raw_para_id": row["raw_para_id"],
        "start_cp": row["document_start_cp"], "end_cp": row["document_end_cp"],
        "text_sha256": sha256(row["text"].encode()).hexdigest()})
    docs[0]["paragraph_manifest_sha256"] = paragraph_manifest_sha256(docs[0]["paragraph_manifest"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    read = tools.read([rows[0]["id"]], before=0, after=1)
    assert read["bundles"][0]["source_paragraph_ids"] == [r["id"] for r in rows]
    assert tools.cite(row["id"], row["id"])["quote"] == "项目完成乙。"


@pytest.mark.parametrize("before,after", [(True, 0), (0, False), (-1, 0), (1, 4), (1.0, 0)])
def test_invalid_or_oversize_neighbor_window_is_rejected_not_truncated(before, after):
    docs, rows = corpus(["项目完成。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    with pytest.raises(ValueError, match="invalid_source_range"):
        tools.read([rows[0]["id"]], before=before, after=after)
    assert tools.scope()["returned_complete_paragraph_ids"] == []


def test_unknown_duplicate_unexposed_and_repeated_read_ids_have_honest_errors():
    docs, rows = corpus(["项目完成。"])
    for anchors, expected in [(["unknown"], "unexposed_or_unknown"),
                               ([rows[0]["id"]], "unexposed_or_unknown"),
                               ([rows[0]["id"], rows[0]["id"]], "unique_nonempty"),
                               ([], "unique_nonempty"), ("not-list", "unique_nonempty")]:
        tools = SourceBundleTools(docs, rows, QUESTION)
        with pytest.raises(ValueError, match=expected):
            tools.read(anchors, before=0, after=0)
        assert tools.trace()[-1]["status"] == "rejected"
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    first = tools.read([rows[0]["id"]], before=0, after=0)
    second = tools.read([rows[0]["id"]], before=0, after=0)
    assert second["bundles"] == first["bundles"]
    assert second["delivery"]["already_read_original_paragraph_ids"] == [rows[0]["id"]]
    assert second["delivery"]["new_original_paragraph_ids"] == []
    assert first["delivery"]["complete_delivery_index"] == 1
    assert second["delivery"]["complete_delivery_index"] == 2
    assert canonical(first) != canonical(second)
    receipt = tools.read([rows[0]["id"]], before=0, after=0)
    assert receipt["status"] == "reused_existing_delivery"
    assert receipt["new_delivery"] is False and receipt["complete_original_slices"] is False
    assert "quote" not in canonical(receipt).decode()
    assert tools.scope()["successful_read_calls"] == 2
    assert len(tools.scope()["complete_read_ranges"]) == 2
    assert tools.scope()["calls"]["read"] == 2


def test_cross_batch_natural_overlap_returns_whole_original_ranges_and_unique_coverage():
    docs, rows = corpus(["项目完成甲😀。", "项目完成乙。", "项目完成丙。", "项目未读丁。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    first = tools.read([rows[0]["id"]], before=0, after=1)
    second = tools.read([rows[1]["id"]], before=0, after=1)
    assert first["bundles"][0]["source_paragraph_ids"] == [rows[0]["id"], rows[1]["id"]]
    assert second["bundles"][0]["source_paragraph_ids"] == [rows[1]["id"], rows[2]["id"]]
    assert second["delivery"]["already_read_original_paragraph_ids"] == [rows[1]["id"]]
    assert second["delivery"]["new_original_paragraph_ids"] == [rows[2]["id"]]
    assert tools.scope()["returned_complete_paragraph_ids"] == [r["id"] for r in rows[:3]]
    for result in [first, second]:
        assert result["source_codepoints"] == sum(len(b["quote"]) for b in result["bundles"])
        for bundle in result["bundles"]:
            assert bundle["quote"] == docs[0]["content"][bundle["start_cp"]:bundle["end_cp"]]
        assert result["source_utf8_bytes"] == sum(len(b["quote"].encode()) for b in result["bundles"])
    assert tools.cite(rows[1]["id"], rows[2]["id"])["quote"] == second["bundles"][0]["quote"]
    with pytest.raises(ValueError, match="not_completely_returned"):
        tools.cite(rows[0]["id"], rows[2]["id"])
    before = tools.scope()
    receipt = tools.read([rows[1]["id"]], before=0, after=0)
    assert receipt["source_delivery_receipts"][0]["source_delivery"]["read_bundle_id"] == first["bundles"][0]["bundle_id"]
    assert receipt["read_allowance_consumed"] is False and receipt["new_delivery"] is False
    assert tools.scope() == before
    with pytest.raises(ValueError, match="read_call_limit_reached"):
        tools.read([rows[0]["id"]], before=0, after=2)  # Seen IDs cannot be stitched into a new larger range.
    with pytest.raises(ValueError, match="read_call_limit_reached"):
        tools.read([rows[3]["id"]], before=0, after=0)


@pytest.mark.parametrize("limit", [True, 0, 61, 1.5, "10"])
def test_search_limits_reject_invalid_types(limit):
    docs, rows = corpus(["项目完成。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    with pytest.raises(ValueError, match="invalid_search_arguments"):
        tools.search(QUESTION["question"], limit=limit)


def test_search_caps_exclusions_and_empty_results_never_claim_absence_or_truth():
    docs, rows = corpus(["项目完成。"])
    tools = SourceBundleTools(docs, rows, QUESTION, max_search_calls=2)
    result = tools.search("")
    assert result["results"] == []
    assert result["no_match_proves_irrelevance"] is False
    assert result["semantic_verified"] is False
    with pytest.raises(ValueError, match="invalid_search_exclusions"):
        tools.search("项目", exclude_paragraph_ids=["unknown"])
    with pytest.raises(ValueError, match="search_call_limit_reached"):
        tools.search("项目")
    assert tools.scope()["calls"]["search"] == 2


@pytest.mark.parametrize("field", ["quote", "start_cp", "end_cp", "source_paragraph_ids", "bundle_id", "document_content_sha256"])
def test_final_citation_source_fields_cannot_be_changed(field):
    docs, rows = corpus(["项目完成甲。", "项目完成乙。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    tools.read([rows[0]["id"]], before=0, after=1)
    citation = tools.cite(rows[0]["id"], rows[1]["id"])
    if field in ("start_cp", "end_cp"):
        citation[field] += 1
    elif field == "source_paragraph_ids":
        citation[field].reverse()
    else:
        citation[field] = "篡改"
    with pytest.raises(ValueError, match="source_bundle_"):
        tools.validate_citation(citation)


def test_source_only_validation_preserves_unverified_annotation_without_certifying_it():
    docs, rows = corpus(["项目完成。"])
    tools = SourceBundleTools(docs, rows, QUESTION)
    expose_all(tools, rows)
    tools.read([rows[0]["id"]], before=0, after=0)
    citation = tools.cite(rows[0]["id"], rows[0]["id"])
    citation.update(fact_des="尚未审核的描述", entities=["任意实体"], semantic_verified=True,
                    native_model_delivery_verified=True, adoption_or_publication_created=True)
    validated = tools.validate_citation(citation)
    assert validated["fact_des"] == citation["fact_des"]
    assert validated["semantic_verified"] is False
    assert validated["native_model_delivery_verified"] is False
    assert validated["adoption_or_publication_created"] is False
    validated["entities"].append("另一个")
    assert citation["entities"] == ["任意实体"]
