"""Complete-source planning must not become a hidden retrieval cutoff."""

from copy import deepcopy

import pytest

from app.domain.coverage import CoverageLimitExceeded, merge_exact_candidates, plan_source_batches
from app.domain.source_catalog import MAX_CATALOG_SPANS, MAX_SOURCE_CODEPOINTS, build_source_catalog


def para(identity, text, document="doc-a", **metadata):
    return {"id": identity, "text": text, "document_id": document, **metadata}


def planned_ids(plan):
    return [identity for batch in plan["batches"] for identity in batch["paragraph_ids"]]


def test_bm25_prioritizes_a_remote_match_without_discarding_zero_match_sources():
    sources = [para(f"noise-{index}", "山间河流正常流动。", f"document-{index}") for index in range(40)]
    sources.append(para("remote-relevant", "新能源设施完成复查。", "remote-document"))
    original = deepcopy(sources)
    plan = plan_source_batches("新能源设施复查", sources, max_codepoints=14)
    assert plan["batches"][0]["paragraph_ids"] == ["remote-relevant"]
    assert set(planned_ids(plan)) == {row["id"] for row in sources}
    assert len(planned_ids(plan)) == len(sources)
    assert plan["coverage"]["bm25_matched_paragraph_count"] == 1
    assert plan["coverage"]["zero_match_paragraph_count"] == 40
    assert plan["coverage"]["complete"] is True
    assert sources == original
    assert all(row is sources[next(index for index, value in enumerate(sources) if value["id"] == row["id"])]
               for batch in plan["batches"] for row in batch["paragraphs"])


def test_a_query_with_no_tokens_or_no_matches_still_scans_every_original_paragraph():
    sources = [para("one", "abc。", "a"), para("two", "def。", "b")]
    for question in ("不存在的内容", "的 和 是", "🦊"):
        plan = plan_source_batches(question, sources)
        assert planned_ids(plan) == ["one", "two"]
        assert plan["coverage"]["zero_match_paragraph_count"] == 2


def test_document_neighbors_remain_in_order_even_when_the_last_paragraph_matches():
    sources = [para("a1", "first。"), para("b1", "other。", "doc-b"),
               para("a2", "middle。"), para("a3", "target。")]
    plan = plan_source_batches("target", sources, max_codepoints=25)
    assert plan["batches"][0]["paragraph_ids"] == ["a1", "a2", "a3"]
    assert plan["batches"][0]["bm25_match_count"] == 1
    assert plan["batches"][1]["paragraph_ids"] == ["b1"]


def test_small_documents_share_batch_space_without_losing_sources_or_document_adjacency():
    sources = [para(f"p{index}", "原文甲。原文乙。", f"document-{index}") for index in range(40)]
    plan = plan_source_batches("不存在", sources)
    assert len(plan["batches"]) == 1  # Not forty model calls for forty tiny documents.
    assert planned_ids(plan) == [row["id"] for row in sources]
    assert plan["batches"][0]["catalog_spans"] == 80
    assert len(build_source_catalog(plan["batches"][0]["paragraphs"])) == 80


def test_both_budgets_are_measured_with_the_actual_immutable_catalog():
    sources = [para(f"p{index}", "🦊甲。乙！尾段\r\n") for index in range(8)]
    plan = plan_source_batches("不存在", sources, max_codepoints=25, max_catalog_spans=5)
    assert len(plan["batches"]) == 8  # Three catalog units per paragraph, so two cannot fit.
    for batch in plan["batches"]:
        catalog = build_source_catalog(batch["paragraphs"])
        assert len(catalog) == batch["catalog_spans"] <= 5
        assert sum(len(row["text"]) for row in batch["paragraphs"]) == batch["codepoints"] <= 25
        for row in batch["paragraphs"]:
            assert "".join(unit["quote"] for unit in catalog if unit["paragraph_id"] == row["id"]) == row["text"]
    assert plan["coverage"]["catalog_span_count"] == 24
    assert plan["coverage"]["codepoint_count"] == sum(len(row["text"]) for row in sources)


def test_codepoint_boundary_uses_python_codepoints_and_never_splits_a_paragraph():
    sources = [para("one", "🦊👩\u200d💻"), para("two", "e\u0301末")]
    plan = plan_source_batches("未知", sources, max_codepoints=4)
    assert [batch["codepoints"] for batch in plan["batches"]] == [4, 3]
    assert [batch["paragraph_ids"] for batch in plan["batches"]] == [["one"], ["two"]]


def test_equal_text_in_different_sources_and_repeated_occurrences_are_all_preserved():
    text = "原话。原话。"
    sources = [para("a", text, "doc-a"), para("b", text, "doc-b")]
    plan = plan_source_batches("原话", sources, max_codepoints=len(text))
    assert set(planned_ids(plan)) == {"a", "b"}
    assert len({batch["id"] for batch in plan["batches"]}) == 2
    assert plan["coverage"]["catalog_span_count"] == 4
    ids = [unit["id"] for batch in plan["batches"] for unit in build_source_catalog(batch["paragraphs"])]
    assert len(set(ids)) == 4


def test_whitespace_is_legal_source_text_and_unknown_document_identities_are_preserved():
    sources = [{"id": "one", "text": "  \n"}, {"id": "two", "text": "  \n"}]
    plan = plan_source_batches("不存在", sources)
    assert [batch["paragraph_ids"] for batch in plan["batches"]] == [["one", "two"]]
    assert plan["coverage"]["codepoint_count"] == 6


@pytest.mark.parametrize("text,kwargs,message", [
    ("甲" * (MAX_SOURCE_CODEPOINTS + 1), {}, "coverage_paragraph_codepoint_limit_exceeded"),
    ("甲。" * (MAX_CATALOG_SPANS + 1), {"max_catalog_spans": MAX_CATALOG_SPANS},
     "coverage_paragraph_catalog_limit_exceeded"),
], ids=["hard-codepoints", "hard-spans"])
def test_oversized_single_paragraph_fails_before_returning_partial_coverage(text, kwargs, message):
    sources = [para("small-first", "甲。"), para("too-large", text)]
    before = deepcopy(sources)
    with pytest.raises(CoverageLimitExceeded, match=message):
        plan_source_batches("甲", sources, **kwargs)
    assert sources == before


@pytest.mark.parametrize("text,kwargs", [
    ("甲" * 11, {"max_codepoints": 10}),
    ("甲。乙。丙。", {"max_catalog_spans": 2}),
    ("甲。" * 488, {}),
    ("🦊" * MAX_SOURCE_CODEPOINTS, {}),
    ("甲。" * MAX_CATALOG_SPANS, {}),
], ids=["soft-codepoints", "soft-spans", "dense-realistic", "exact-hard-codepoints", "exact-hard-spans"])
def test_paragraph_above_soft_target_runs_alone_up_to_exact_hard_limits(text, kwargs):
    large = para("large", text)
    sources = [para("first", "前。"), large, para("last", "后。")]
    before = deepcopy(sources)
    plan = plan_source_batches("不存在", sources, **kwargs)
    large_batch = next(batch for batch in plan["batches"] if "large" in batch["paragraph_ids"])
    assert large_batch["paragraphs"] == [large]
    assert large_batch["paragraphs"][0] is large
    assert large_batch["soft_budget_exceeded"] is True
    assert large_batch["codepoints"] <= MAX_SOURCE_CODEPOINTS
    assert large_batch["catalog_spans"] <= MAX_CATALOG_SPANS
    assert len(build_source_catalog(large_batch["paragraphs"])) == large_batch["catalog_spans"]
    assert len(planned_ids(plan)) == len(sources)
    assert set(planned_ids(plan)) == {"first", "large", "last"}
    assert plan["coverage"]["soft_budget_exceeded_batch_count"] == 1
    assert sources == before


@pytest.mark.parametrize("kwargs", [
    {"max_codepoints": 0}, {"max_codepoints": True}, {"max_codepoints": MAX_SOURCE_CODEPOINTS + 1},
    {"max_catalog_spans": 0}, {"max_catalog_spans": 1.5}, {"max_catalog_spans": MAX_CATALOG_SPANS + 1},
])
def test_configured_budget_cannot_overrun_catalog_or_use_boolean_as_integer(kwargs):
    with pytest.raises(ValueError, match="invalid_coverage_.*_budget"):
        plan_source_batches("甲", [para("one", "甲。")], **kwargs)


@pytest.mark.parametrize("sources,message", [
    ([para("same", "甲。"), para("same", "乙。", "doc-b")], "duplicate_coverage_paragraph"),
    ([para("one", "")], "invalid_coverage_paragraph"),
    ([para("", "甲。")], "invalid_coverage_paragraph"),
    ([para("one", "甲。", "")], "invalid_coverage_document_identity"),
    ([None], "invalid_coverage_paragraph"),
    ([para("one", "\ud800")], "invalid_source_unicode"),
])
def test_invalid_source_identity_or_geometry_fails_explicitly(sources, message):
    with pytest.raises(ValueError, match=message):
        plan_source_batches("甲", sources)


def test_empty_input_has_explicit_complete_zero_source_metadata():
    plan = plan_source_batches("甲", [])
    assert plan["batches"] == []
    assert plan["coverage"]["complete"] is True
    for field in ("paragraph_count", "planned_paragraph_count", "batch_count", "codepoint_count",
                  "catalog_span_count", "bm25_matched_paragraph_count", "zero_match_paragraph_count"):
        assert plan["coverage"][field] == 0


def test_ids_and_fingerprints_are_stable_and_bind_exact_source_question_and_plan_order():
    sources = [para("a1", "First sentence。"), para("a2", "Second sentence。"),
               para("b1", "Elsewhere。", "doc-b")]
    first = plan_source_batches("First", sources)
    duplicate = plan_source_batches("First", deepcopy(sources))
    assert first == duplicate
    reverse = plan_source_batches("First", list(reversed(sources)))
    assert first["coverage"]["source_fingerprint"] == reverse["coverage"]["source_fingerprint"]
    assert first["coverage"]["plan_fingerprint"] != reverse["coverage"]["plan_fingerprint"]
    changed_question = plan_source_batches("Elsewhere", sources)
    assert first["coverage"]["source_fingerprint"] == changed_question["coverage"]["source_fingerprint"]
    assert first["coverage"]["plan_fingerprint"] != changed_question["coverage"]["plan_fingerprint"]
    changed_text = deepcopy(sources)
    changed_text[0]["text"] += " "
    assert first["coverage"]["source_fingerprint"] != plan_source_batches("First", changed_text)["coverage"]["source_fingerprint"]
    changed_source = deepcopy(sources)
    changed_source[0]["document_id"] = "another-document"
    assert first["coverage"]["source_fingerprint"] != plan_source_batches("First", changed_source)["coverage"]["source_fingerprint"]
    changed_raw_identity = deepcopy(sources)
    changed_raw_identity[0]["raw_para_id"] = "raw-paragraph-9"
    assert first["coverage"]["source_fingerprint"] != plan_source_batches("First", changed_raw_identity)["coverage"]["source_fingerprint"]
    smaller_batches = plan_source_batches("First", sources, max_codepoints=20)
    assert first["coverage"]["plan_fingerprint"] != smaller_batches["coverage"]["plan_fingerprint"]


def candidate(**changes):
    return {"paragraph_id": "p1", "quote": "原话。", "start_cp": 0, "end_cp": 3,
            "fact_des": "待核验事实", "argument": {"主体": "甲", "时间": "未知"},
            "entities": ["甲"], **changes}


def test_merge_is_strictly_exact_and_retains_occurrences_provenance_and_different_facts():
    first = candidate()
    reordered = candidate(argument={"时间": "未知", "主体": "甲"})
    different_source = candidate(paragraph_id="p2")
    different_occurrence = candidate(start_cp=3, end_cp=6)
    different_fact = candidate(fact_des="另一项判断")
    different_annotation = candidate(entities=["甲", "乙"])
    overlap = candidate(quote="原话。后文。", end_cp=6)
    extra_provenance = candidate(document_id="new-source")
    batches = [[first, different_source, different_occurrence],
               [deepcopy(first), reordered, different_fact, different_annotation, overlap, extra_provenance]]
    before = deepcopy(batches)
    merged = merge_exact_candidates(batches)
    assert merged == [first, different_source, different_occurrence, different_fact,
                      different_annotation, overlap, extra_provenance]
    assert merged[0] is first
    assert batches == before


def test_merge_never_normalizes_literal_quotes_or_collapses_different_annotations():
    single = candidate(quote="‘原话’", end_cp=4)
    double = candidate(quote="“原话”", end_cp=4)
    spaced = candidate(fact_des="待核验事实 ")
    assert merge_exact_candidates([[single], [double], [spaced]]) == [single, double, spaced]


@pytest.mark.parametrize("batches", [None, [None], [[{}]], [[candidate(argument={"bad": float("nan")})]]])
def test_invalid_merge_records_fail_without_silent_drop(batches):
    with pytest.raises(ValueError):
        merge_exact_candidates(batches)
