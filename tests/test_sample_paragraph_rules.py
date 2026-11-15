"""Deterministic fixtures, never real reference answers or model calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
from zipfile import ZipFile

import pytest

from app.domain.paragraph_mirrors import (
    build_paragraph_mirror_index, character_similarity,
    expand_paragraph_mirror_candidates,
    find_literal_fragment_mirror_candidates,
)

_path = Path(__file__).resolve().parents[1] / "scripts/audit_sample_paragraph_rules.py"
_spec = importlib.util.spec_from_file_location("sample_paragraph_rule_audit_test", _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def paragraph(doc, para, text, **metadata):
    return {"raw_doc_id": doc, "raw_para_id": para, "id": doc + ":" + para, "text": text, **metadata}


def doc(doc_id="doc_016", texts=("原段五。", "邻接六。")):
    content = "\n".join(texts)
    result = {"doc_id": doc_id, "content": content, "paragraphs": [], "publish_time": "2026-07-XX", "source_credibility": 0.55}
    offset = 0
    for i, text in enumerate(texts, 5):
        result["paragraphs"].append({"para_id": f"{doc_id}#p{i}", "char_start": offset, "char_end": offset + len(text)})
        offset += len(text) + 1
    return result


def fact(document, quote=None, fact_id="f1", para_id=None):
    p = document["paragraphs"][0]
    return {"fact_id": fact_id, "doc_id": document["doc_id"], "para_id": para_id or p["para_id"],
        "quote": quote or document["content"][p["char_start"]:p["char_end"]], "char_start": 0,
        "char_end": len(quote) if quote else p["char_end"]}


def test_similarity_is_exact_unicode_indel_not_normalized_words():
    assert character_similarity("甲😀乙", "甲😀丙") == pytest.approx(2 / 3)
    assert character_similarity("A a", "aa") != 1
    assert character_similarity("", "a") == 0
    assert character_similarity("", "") == 1


def test_bitset_lcs_matches_independent_dynamic_programming():
    generator = random.Random(20261019)
    for _ in range(120):
        left = "".join(generator.choice("甲乙😀 \n") for _ in range(generator.randrange(15)))
        right = "".join(generator.choice("甲乙😀 \n") for _ in range(generator.randrange(15)))
        table = [0] * (len(right) + 1)
        for a in left:
            old = table[:]
            for j, b in enumerate(right, 1):
                table[j] = old[j - 1] + 1 if a == b else max(old[j], table[j - 1])
        expected = 2 * table[-1] / (len(left) + len(right)) if left or right else 1
        assert character_similarity(left, right) == expected


def test_exact_mirror_preserves_each_identity_and_ignores_annotations():
    rows = [paragraph("a", "a#p1", "同一原句", source_group_id="one", fact_des="DO_NOT_USE_GOLD", selected=True),
        paragraph("b", "b#p2", "同一原句", source_group_id="two")]
    before = deepcopy(rows)
    index = build_paragraph_mirror_index(rows)
    result = expand_paragraph_mirror_candidates(index, [rows[0]])
    assert [(r["doc_id"], r["para_id"]) for r in result] == [("b", "b#p2")]
    assert result[0]["origins"][0]["match"] == "exact"
    assert result[0]["candidate_only"] and result[0]["requires_independent_fact_verification"]
    assert index["evidence_created_or_deleted"] is False and index["independent_source_votes_added"] == 0
    assert "DO_NOT_USE_GOLD" not in json.dumps(index)
    assert rows == before


def test_similarity_components_do_not_allow_transitive_expansion():
    rows = [paragraph("a", "p1", "0000000000"), paragraph("b", "p2", "1000000000"), paragraph("c", "p3", "1100000000")]
    index = build_paragraph_mirror_index(rows)
    assert len(index["components"]) == 1 and len(index["similar_edges"]) == 2
    assert [r["doc_id"] for r in expand_paragraph_mirror_candidates(index, [rows[0]])] == ["b"]
    assert [r["doc_id"] for r in expand_paragraph_mirror_candidates(index, [rows[1]])] == ["a", "c"]


def test_different_source_group_never_deletes_or_collapses_locations():
    rows = [paragraph("a", "p1", "原句同文", source_group_id="shared"), paragraph("b", "p2", "原句同文", source_group_id="shared"), paragraph("c", "p3", "原句同文", source_group_id="different")]
    index = build_paragraph_mirror_index(rows)
    assert len(index["source_manifest"]) == 3
    assert len(expand_paragraph_mirror_candidates(index, [rows[0]])) == 2
    assert {r["doc_id"] for r in index["exact_groups"][0]["members"]} == {"a", "b", "c"}


def test_same_document_repetition_is_separate_from_cross_document_default():
    rows = [paragraph("a", "p1", "重复"), paragraph("a", "p2", "重复")]
    assert not build_paragraph_mirror_index(rows)["exact_groups"]
    index = build_paragraph_mirror_index(rows, cross_document_only=False)
    assert len(index["exact_groups"]) == 1
    assert len(expand_paragraph_mirror_candidates(index, [rows[0]])) == 1


def test_shared_fragment_is_independent_of_whole_paragraph_similarity_and_keeps_repeats():
    phrase = "双方明确同意建立独立事件沟通渠道。"
    rows = [paragraph("a", "p1", phrase + phrase), paragraph("b", "p2", "很长的不同背景" * 40 + phrase)]
    assert character_similarity(rows[0]["text"], rows[1]["text"]) < 0.9
    value = find_literal_fragment_mirror_candidates(rows, [phrase])
    assert value["candidate_only"] and value["independent_source_votes_added"] == 0
    occurrences = value["fragments"][0]["occurrences"]
    assert len(occurrences) == 3 and occurrences[1]["start_cp"] == len(phrase)
    for hit in occurrences:
        source = next(row for row in rows if row["raw_doc_id"] == hit["doc_id"])
        assert source["text"][hit["start_cp"]:hit["end_cp"]] == phrase
    assert find_literal_fragment_mirror_candidates(rows, ["没有出现在原文的另一个合法长片段"])["fragments"][0]["occurrences"] == []
    with pytest.raises(ValueError):
        find_literal_fragment_mirror_candidates(rows, ["太短"])


def test_shared_fragment_collection_distinguishes_declared_key_from_cross_paragraph_quote():
    phrase = "双方明确同意建立独立事件沟通渠道。"
    a = doc("a", (phrase,))
    b = doc("b", ("前段背景。", phrase))
    c = doc("c", ("不同评论背景" * 30 + phrase,))
    refs = [fact(a), fact(b, b["content"], fact_id="cross")]
    result = audit.audit_topic("synthetic", [a, b, c], refs, [], shared_fragments=[phrase])
    probe = result["shared_literal_fragment_probes"][0]
    assert probe["declared_location_collection"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}
    assert probe["literal_reference_quote_content_collection"] == {"numerator": 2, "denominator": 3, "rate": 2 / 3}
    second = next(row for row in probe["locations"] if row["doc_id"] == "b")
    assert second["para_id"] == "b#p6" and second["verified_gold_document_quote_spans_covering_fragment"][0]["declared_para_id"] == "b#p5"
    assert not second["verified_gold_document_quote_spans_covering_fragment"][0]["reference_identity_repaired"]


def test_determinism_anchors_and_bound_index_validation():
    rows = [paragraph("a", "p1", "同文"), paragraph("b", "p2", "同文")]
    index = build_paragraph_mirror_index(rows)
    assert index == build_paragraph_mirror_index(list(reversed(rows)))
    with pytest.raises(ValueError, match="unknown_or_duplicate"):
        expand_paragraph_mirror_candidates(index, [rows[0], rows[0]])
    with pytest.raises(ValueError, match="unknown_or_duplicate"):
        expand_paragraph_mirror_candidates(index, [{"doc_id": "outside", "para_id": "p"}])
    index["threshold"] = 0.1
    with pytest.raises(ValueError, match="binding"):
        expand_paragraph_mirror_candidates(index, [rows[0]])


@pytest.mark.parametrize("threshold", [0, -1, 1.1, True, float("nan"), "0.9"])
def test_invalid_similarity_policy_rejected(threshold):
    with pytest.raises(ValueError):
        build_paragraph_mirror_index([], threshold=threshold)


def test_duplicate_raw_identity_rejected_even_with_different_internal_ids():
    with pytest.raises(ValueError, match="duplicate_mirror_location"):
        build_paragraph_mirror_index([paragraph("a", "p1", "甲"), paragraph("a", "p1", "乙")])


def test_cross_paragraph_exact_join_newline_and_raw_slice_keep_original_id():
    document = doc()
    joined = fact(document, "原段五。\n邻接六。")
    matches = audit.cross_paragraph_matches(joined, document, "doc_016#p5")
    assert {r["method"] for r in matches} == {"joined_single_newline", "original_document_slice_preserving_gaps"}
    assert all(r["para_ids"] == ["doc_016#p5", "doc_016#p6"] and not r["reference_identity_changed"] for r in matches)
    plain = audit.cross_paragraph_matches(fact(document, "五。邻接六。"), document, "doc_016#p5")
    assert [r["method"] for r in plain] == ["concatenated_no_separator"]
    assert not audit.cross_paragraph_matches(fact(document, "邻接六。"), document, "doc_016#p5")


def test_cross_search_never_reaches_sixth_paragraph_or_repairs_start_identity():
    document = doc(texts=tuple(f"第{i}段" for i in range(6)))
    assert not audit.cross_paragraph_matches(fact(document, document["content"]), document, "doc_016#p5")
    assert not audit.cross_paragraph_matches(fact(document, "第1段\n第2段"), document, "doc_016#p5")


def test_audit_reference_multiplicity_not_deduplicated_and_mirror_rate_not_precision():
    a, b, c = doc("a", ("这是完全同一段原文",)), doc("b", ("这是完全同一段原文",)), doc("c", ("这是完全同一段原文",))
    references = [fact(a), fact(a, fact_id="f2"), fact(b, fact_id="f3")]
    results = audit.audit_topic("fixture", [a, b, c], references, [{"question_id": "q", "question": "fixture", "supporting_facts": [{"fact_id": "f1", "verdict_label": "full_support"}], "conflicting_facts": [{"fact_id": "f3", "resolution_type": "unresolved"}]}])
    assert results["F1_same_doc_para_reference_multiplicity"]["rows_per_location_histogram"] == {"1": 1, "2": 1}
    mirrors = results["F6_gold_mirror_location_collection"]
    assert mirrors["all_at_threshold"]["collection_rate"] == {"numerator": 2, "denominator": 4, "rate": 0.5}
    assert mirrors["source_key_proxy_not_semantic_precision_or_recall"]
    assert results["F9_conclusion_fields_and_labels"]["verdict_label_distribution"] == {"full_support": 1}
    assert results["F9_conclusion_fields_and_labels"]["resolution_type_distribution"] == {"unresolved": 1}
    assert results["F2_quote_position"]["equals_whole_paragraph"] == 3


@pytest.mark.parametrize(("value", "expected"), [(None, "null"), ("", "empty_string"), ("2026-07-XX", "partial_or_unknown_placeholder"), ("2026-02-30", "invalid_iso_calendar_value"), ("2026-10-19", "iso_date"), ("2026-10-19T12:00:00+08:00", "iso_datetime_with_timezone")])
def test_publish_time_categories_do_not_guess_missing_dates(value, expected):
    assert audit.publish_time_category(value) == expected


def test_offline_files_source_hashes_and_docx_excerpt_are_literal(tmp_path):
    source = tmp_path / "source";folder = source / "fixture";folder.mkdir(parents=True)
    document = doc()
    for name, data in [("raw_documents.json", [document]), ("evidence_samples.json", [fact(document)]), ("conclusions.json", [{"question_id": "q", "question": "原始问题", "conclusion": "REFERENCE_ONLY"}])]:
        (folder / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    official = tmp_path / "rubric.docx"
    with ZipFile(official, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>逐字按doc_id和para_id匹配</w:t></w:r></w:p></w:body></w:document>')
    output = tmp_path / "audit"
    result = audit.write_audit(source, output, [official])
    assert result["sources_unchanged"] and result["source_files_sha256_before"] == result["source_files_sha256_after"]
    assert result["model_calls"] == 0 and result["runtime_database_or_index_used"] is False
    excerpts = json.loads((output / "official-document-extracts.json").read_text(encoding="utf-8"))
    assert excerpts[0]["selected_paragraphs"][0]["text"] == "逐字按doc_id和para_id匹配"
    with pytest.raises(ValueError, match="new_outside"):
        audit.write_audit(source, source / "nested")
    with pytest.raises(ValueError, match="new_outside"):
        audit.write_audit(source, output)
