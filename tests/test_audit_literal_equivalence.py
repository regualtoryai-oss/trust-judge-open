import json
from pathlib import Path

import pytest

from scripts.audit_literal_equivalence import (
    audit, compare_literals, exact_fragments, one_to_one, occurrences,
)


def test_all_exact_repeated_occurrences_and_codepoint_coordinates():
    left, right = "😀甲乙丙甲乙丙", "甲乙丙/甲乙丙"
    fragments = exact_fragments(left, right, 3)
    assert {(tuple(f["left_span_cp"]), tuple(f["right_span_cp"])) for f in fragments} == {
        ((1, 4), (0, 3)), ((1, 4), (4, 7)), ((4, 7), (0, 3)), ((4, 7), (4, 7)),
    }
    assert all(left[slice(*f["left_span_cp"])] == right[slice(*f["right_span_cp"])] == f["literal"] for f in fragments)


def test_maximal_fragments_agree_with_independent_small_brute_force():
    # Exhaustively inspect starts rather than use the seeded implementation.
    left, right, minimum = "aababaabab", "babaababa", 2
    expected = set()
    for i in range(len(left)):
        for j in range(len(right)):
            common = 0
            while i + common < len(left) and j + common < len(right) and left[i + common] == right[j + common]:
                common += 1
            if common >= minimum and (not i or not j or left[i - 1] != right[j - 1]):
                expected.add((i, j, common))
    actual = exact_fragments(left, right, minimum)
    assert {(f["left_span_cp"][0], f["right_span_cp"][0], f["length_cp"]) for f in actual} == expected


def test_substrings_preserve_every_repeated_location():
    assert occurrences("甲甲甲", "甲甲") == [[0, 2], [1, 3]]
    assert occurrences("甲甲甲", "") == []


def test_whitespace_traditional_and_punctuation_are_hints_only():
    white = compare_literals("模型\n安全", "模型安全", 5)
    assert white["review_hints_only"]["whitespace_removed_equal"] is True
    assert white["whole_quote_equal"] is False and white["exact_fragments"] == []
    for left, right in [("安全監管", "安全监管"), ("达里奥·阿莫迪", "达里奥・阿莫迪")]:
        pair = compare_literals(left, right, 4)
        assert not pair["whole_quote_equal"]
        assert pair["semantic_equivalence"] == "unknown"


@pytest.mark.parametrize("prefix", ["不", "2025年", "2026年10%相反表述："])
def test_negation_time_value_overlap_never_promoted_to_same_fact(prefix):
    quote = "政府应该承担企业的全部法律责任。"
    pair = compare_literals(prefix + quote, quote, 8)
    assert pair["right_whole_quote_in_left"]
    assert pair["exact_fragments"]
    assert not pair["whole_quote_equal"]
    assert pair["semantic_equivalence"] == "unknown"
    assert pair["review_hints_only"]["accepted_for_certificate"] is False


def test_even_identical_quote_does_not_validate_annotation():
    pair = compare_literals("尚未确认。", "尚未确认。", 3)
    assert pair["whole_quote_equal"]
    assert pair["semantic_equivalence"] == "unknown"


def test_maximum_matching_reroutes_without_many_to_one_inflation():
    result = one_to_one(4, 2, [(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)])
    assert result["matches"] == 2 and result["eligible_edges"] == 5
    assert len({x[0] for x in result["one_to_one_pairs"]}) == 2
    assert len({x[1] for x in result["one_to_one_pairs"]}) == 2
    assert result["matched_left_fraction"] == .5


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def closed_fixture(tmp_path):
    first, second = "政府应评估模型。", "企业承担责任。"
    content = first + second
    doc = {"doc_id": "d", "content": content, "title": "Synthetic boundary fixture",
           "paragraphs": [{"para_id": "d#p1", "char_start": 0, "char_end": len(first)},
                          {"para_id": "d#p2", "char_start": len(first), "char_end": len(content)}]}
    source = [{"raw_doc_id": "d", "raw_para_id": p["para_id"], "document_start_cp": p["char_start"],
               "document_end_cp": p["char_end"], "text": content[p["char_start"]:p["char_end"]]} for p in doc["paragraphs"]]
    fact = {"fact_id": "f", "doc_id": "d", "para_id": "d#p1", "quote": content,
            "char_start": 0, "char_end": len(content), "fact_des": "Both original sentences."}
    conclusion = {"question_id": "q", "question": "What is stated?", "supporting_facts": [{"fact_id": "f"}], "conflicting_facts": [{"fact_id": "f"}]}
    submitted = [{"id": "s1", "raw_doc_id": "d", "raw_para_id": "d#p2", "quote": second, "start_cp": 0, "end_cp": len(second)},
                 {"id": "s2", "raw_doc_id": "d", "raw_para_id": "d#p2", "quote": second, "start_cp": 0, "end_cp": len(second)}]
    paths = [write(tmp_path / "source.json", source), write(tmp_path / "result.json", {"evidence": submitted}), tmp_path,
             write(tmp_path / "score.json", {"question": {"question_id": "q", "question": "What is stated?"}, "after": {"question_selection": {"A_id_only_proxy": {"matches": 0}}}})]
    write(tmp_path / "raw_documents.json", [doc])
    write(tmp_path / "evidence_samples.json", [fact])
    write(tmp_path / "conclusions.json", [conclusion])
    return paths


def test_bound_cross_paragraph_quote_complete_matrix_and_multiplicity(closed_fixture):
    report = audit(*closed_fixture, "q", 5)
    assert report["source_inventory"]["archived_paragraph_count"] == 2
    assert report["reference"][0]["quote_crosses_original_paragraph_boundary"]
    assert report["reference"][0]["overlapping_original_paragraph_ids"] == ["d#p1", "d#p2"]
    assert report["reference"][0]["literal_coordinate_verified"]
    assert report["record_accounting"]["reference_union_records"] == 1
    assert report["record_accounting"]["reference_relation_links"] == 2
    assert report["record_accounting"]["complete_pair_count"] == 2
    assert report["one_to_one_literal_coverage"]["whole_quote_containment_either_direction"]["matches"] == 1
    assert report["one_to_one_literal_coverage"]["whole_quote_containment_either_direction"]["left_records"] == 2
    assert report["one_to_one_unique_quote_group_literal_coverage"]["whole_quote_containment_either_direction"]["left_records"] == 1
    assert report["inputs_unchanged_after_read"]
    assert len(report["input_bindings"]) == 6
    assert all(len(d["sha256"]) == 64 and Path(d["path"]).exists() for d in report["input_bindings"].values())
    for pair in report["complete_pair_matrix"]:
        for fragment in pair["exact_fragments"]:
            assert fragment["left_document_span_cp"] == fragment["right_document_span_cp"]


def test_incorrect_explicit_coordinates_stay_invalid_no_silent_relocation(closed_fixture):
    result_path = closed_fixture[1]
    result = json.loads(result_path.read_text("utf-8"))
    for row in result["evidence"]:
        row["start_cp"] = 1
    write(result_path, result)
    report = audit(*closed_fixture, "q", 5)
    assert all(not row["literal_coordinate_verified"] for row in report["submitted"])
    assert all(not pair["both_original_quote_spans_verified"] for pair in report["complete_pair_matrix"])
    assert all(policy["matches"] == 0 for policy in report["one_to_one_literal_coverage"].values())


def test_incomplete_or_modified_source_archive_is_rejected(closed_fixture):
    source_path = closed_fixture[0]
    rows = json.loads(source_path.read_text("utf-8"))
    write(source_path, rows[:1])
    with pytest.raises(ValueError, match="complete_archive"):
        audit(*closed_fixture, "q")
    rows[0]["text"] += "修改"
    write(source_path, rows)
    with pytest.raises(ValueError, match="complete_archive"):
        audit(*closed_fixture, "q")


def test_score_question_mismatch_is_rejected(closed_fixture):
    score = json.loads(closed_fixture[3].read_text("utf-8"))
    score["question"]["question_id"] = "other"
    write(closed_fixture[3], score)
    with pytest.raises(ValueError, match="score_question_mismatch"):
        audit(*closed_fixture, "q")
