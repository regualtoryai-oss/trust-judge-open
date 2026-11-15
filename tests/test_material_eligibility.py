"""Invented questions and source metadata only; no gold/runtime/model access."""
from copy import deepcopy

import pytest

from app.domain.material_eligibility import Invalid, MaterialEligibilityCatalog, digest, extract_question_policy, parse_time_literal


QUESTION = "仅依据截至2038年5月9日18:00（北京时间）已发布的材料，某项目是否完成交付？"


def source(raw="2038-05-09T17:00:00+08:00", **extra):
    return {"id": "S-alpha", "raw_doc_id": "raw-alpha", "content_sha256": digest("原始导入正文"),
        "raw_publish_time": raw, **extra}


def ref(**extra):
    return {"source_id": "S-alpha", "raw_doc_id": "raw-alpha", **extra}


def assess(raw="2038-05-09T17:00:00+08:00", *, question=QUESTION, **metadata):
    return MaterialEligibilityCatalog(question, [source(raw, **metadata)]).assess(ref())


def test_publication_policy_has_literal_unicode_spans_and_no_inferred_context():
    policy = extract_question_policy(QUESTION)
    assert policy["mode"] == "publication_cutoff" and policy["comparison"] == "le"
    assert policy["cutoff_time"]["instant_utc"] == "2038-05-09T10:00:00+00:00"
    for key in ("policy_span", "cutoff_span", "operator_span"):
        span = policy[key]
        assert QUESTION[span["start_cp"]:span["end_cp"]] == span["text"]
    assert policy["operator_span"]["text"] == "截至"


@pytest.mark.parametrize("raw,status", [
    ("2038-05-09T17:59:59+08:00", "within_cutoff"),
    ("2038-05-09T18:00:00+08:00", "within_cutoff"),
    ("2038-05-09T18:00:01+08:00", "after_cutoff"),
    ("2038-05-10T09:00:00+08:00", "after_cutoff"),
    ("2038-05-10T00:15:00+14:00", "after_cutoff"),
    ("2038-05-08T23:30:00-12:00", "after_cutoff")])
def test_publication_cutoff_inclusive_comparison_uses_utc(raw, status):
    assert assess(raw)["status"] == status


def test_offsets_can_reverse_apparent_calendar_day_order():
    question = "仅依据截至2038-05-09T12:30:00Z已发布的资料，某项行为是否完成？"
    assert assess("2038-05-10T00:15:00+14:00", question=question)["status"] == "within_cutoff"
    assert assess("2038-05-09T23:30:00-12:00", question=question)["status"] == "after_cutoff"


def test_before_is_exclusive_not_silently_normalized_to_inclusive():
    question = "仅依据2038-05-09T10:00:00Z之前发表的材料，某项行为是否完成？"
    policy = extract_question_policy(question)
    assert policy["comparison"] == "lt" and policy["operator_span"]["text"] == "之前"
    assert assess("2038-05-09T10:00:00Z", question=question)["status"] == "excluded_at_cutoff"
    assert assess("2038-05-09T09:59:59Z", question=question)["status"] == "within_cutoff"


@pytest.mark.parametrize("raw,reason", [(None, "publication_missing"), ("", "publication_missing"),
    ("2038-05-09", "publication_date_only_uncertain"),
    ("2038-05-09T17:00:00", "publication_timezone_unknown"),
    ("5月9日17:00（北京时间）", "publication_year_unknown"),
    ("2038-05-09T17:00:00CST", "publication_unsupported_literal")])
def test_source_unknown_year_zone_or_date_precision_is_never_guessed(raw, reason):
    decision = assess(raw)
    assert decision["status"] == "unknown" and decision["reason"] == reason


@pytest.mark.parametrize("cutoff,reason", [
    ("5月9日18:00（北京时间）", "cutoff_year_unknown"),
    ("2038年5月9日18:00", "cutoff_timezone_unknown"),
    ("2038年5月9日（北京时间）", "cutoff_date_only_uncertain"),
    ("2038年2月30日18:00（北京时间）", "cutoff_invalid_calendar_time"),
    ("2038-05-09T18:00:00+25:00", "cutoff_invalid_timezone")])
def test_cutoff_unknown_even_if_all_source_metadata_could_supply_year_or_zone(cutoff, reason):
    question = "仅依据截至" + cutoff + "已发布的材料，项目是否完成？"
    decision = assess(question=question)
    assert decision["status"] == "unknown" and decision["reason"] == reason


def test_event_cutoff_allows_later_material_but_does_not_certify_event_time():
    question = "仅判断截至2038年5月9日18:00（北京时间）已经发生的事件，相关事项是什么？"
    decision = assess("2038-05-10T09:00:00+08:00", question=question)
    assert decision["mode"] == "event_cutoff" and decision["status"] == "not_applicable"
    assert not decision["publication_requirement_applies"]
    assert decision["event_time_assessment"] == "not_performed"


@pytest.mark.parametrize("question", [
    "截至2038年5月9日18:00，情况如何？",
    "请判断这些材料的说法。",
    "不限制材料发表时间，仅判断截至2038年5月9日18:00（北京时间）已经发生的事件。",
    "不仅依据截至2038年5月9日18:00（北京时间）已发布的材料，还参考其他资料。",
    "某媒体说‘仅依据截至2038年5月9日18:00（北京时间）已发布的材料’，这一方法合理吗？"])
def test_no_supported_explicit_scope_or_negated_or_quoted_policy_stays_unknown(question):
    policy = extract_question_policy(question)
    assert policy["mode"] == "unspecified"
    assert assess(question=question)["status"] == "unknown"


@pytest.mark.parametrize("remainder", [
    "判断截至2038-05-08T20:00:00Z已经发生的事项。",
    "但不受材料发布日期限制。",
    "允许使用截止后发布的材料。"])
def test_multiple_or_overridden_constraints_are_not_collapsed_into_first_cutoff(remainder):
    question = QUESTION + remainder
    policy = extract_question_policy(question)
    assert policy["extraction_status"] == "multiple_or_overridden_policies_unresolved"
    assert assess(question=question)["status"] == "unknown"


def test_old_event_retold_later_and_model_self_eligibility_cannot_backdate_source():
    catalog = MaterialEligibilityCatalog(QUESTION, [source("2038-05-10T09:00:00+08:00")])
    evidence = ref(raw_publish_time="2038-05-01T00:00:00Z", eligible=True,
        publication_eligible=True, event_time="2038-05-09T12:00:00+08:00", quote="先前已经完成，后来统一发布。")
    decision = catalog.assess(evidence)
    assert decision["status"] == "after_cutoff"
    assert decision["event_truth_assessment"] == decision["source_authority_assessment"] == "not_performed"
    assert "not_truth_or_trust_grade" in decision["meaning"]


def test_parsed_helper_cannot_fill_unknown_raw_zone_year_or_missing_value():
    for raw in (None, "2038-05-09T17:00:00", "5月9日17:00（北京时间）", "2038-05-09"):
        decision = assess(raw, parsed_publish_time="2038-05-09T17:00:00+08:00")
        assert decision["status"] == "unknown"


def test_conflicting_known_metadata_does_not_let_helper_or_raw_silently_win():
    decision = assess(parsed_publish_time="2038-05-10T17:00:00+08:00")
    assert decision["status"] == "unknown" and decision["reason"] == "publication_metadata_conflict"
    assert assess(parsed_publish_time="2038-05-09T09:00:00Z")["status"] == "within_cutoff"


def test_declared_low_precision_is_not_overridden_by_timestamp_shape():
    assert assess(time_precision="date")["status"] == "unknown"


def test_rebinding_question_or_source_metadata_invalidates_replayed_decision():
    catalog = MaterialEligibilityCatalog(QUESTION, [source()])
    decision = catalog.assess(ref())
    changed = MaterialEligibilityCatalog(QUESTION, [source("2038-05-10T09:00:00+08:00")])
    with pytest.raises(Invalid, match="eligibility_decision_replay_binding"):
        changed.verify(decision, ref())
    altered = deepcopy(decision)
    altered["status"] = "after_cutoff"
    altered["decision_binding_sha256"] = digest({k: v for k, v in altered.items() if k != "decision_binding_sha256"})
    with pytest.raises(Invalid, match="eligibility_decision_replay_binding"):
        catalog.verify(altered, ref())


def test_mutable_callers_cannot_change_source_catalog_or_policy_snapshot():
    original = source()
    catalog = MaterialEligibilityCatalog(QUESTION, [original])
    original["raw_publish_time"] = "2038-05-10T09:00:00+08:00"
    policy = catalog.policy
    policy["mode"] = "event_cutoff"
    assert catalog.assess(ref())["status"] == "within_cutoff"


@pytest.mark.parametrize("changes", [{"source_id": "missing"}, {"raw_doc_id": "other-document"}])
def test_source_identity_mismatch_is_rejected_instead_of_borrowing_early_metadata(changes):
    catalog = MaterialEligibilityCatalog(QUESTION, [source()])
    with pytest.raises(Invalid, match="eligibility_canonical_source_binding"):
        catalog.assess(ref(**changes))


def test_partition_preserves_all_facts_and_qualifies_neither_truth_nor_selection():
    sources = [source(), source("2038-05-10T09:00:00+08:00", id="S-late", raw_doc_id="raw-late"),
        source(None, id="S-unknown", raw_doc_id="raw-unknown")]
    facts = [{"id": "F" + str(i), "source": {"source_id": s["id"], "raw_doc_id": s["raw_doc_id"]},
              "annotation": {"fact_des": "保留原始注释"}} for i, s in enumerate(sources)]
    before = deepcopy(facts)
    result = MaterialEligibilityCatalog(QUESTION, sources).partition(facts)
    assert facts == before
    assert [len(result["buckets"][k]) for k in ("within_cutoff", "after_cutoff", "unknown")] == [1, 1, 1]
    assert result["all_original_facts_retained"] and result["classification_is_not_semantic_verification"]


def test_unsupported_iso_geometry_is_unknown_not_a_guessed_timestamp():
    assert parse_time_literal("2038-05-0917:00:00+08:00")["status"] == "unsupported_literal"


def test_duplicate_document_metadata_cannot_offer_an_earlier_alternative_source_timestamp():
    with pytest.raises(Invalid, match="eligibility_duplicate_document_metadata"):
        MaterialEligibilityCatalog(QUESTION, [source(), source("2038-05-01T00:00:00Z", id="another-source")])


@pytest.mark.parametrize("question", [
    "仅依据2038-05-09T12:00:00Z之前已发布的材料，同时仅依据2038-05-08T12:00:00Z之前已发布的材料，判断项目情况。",
    "仅依据2038-05-09T12:00:00Z之前已发布的材料，但也可以使用其后发表的材料，判断项目情况。"])
def test_second_before_policy_or_explicit_override_cannot_silently_use_first_cutoff(question):
    assert extract_question_policy(question)["extraction_status"] == "multiple_or_overridden_policies_unresolved"
    assert assess("2038-05-08T15:00:00Z", question=question)["status"] == "unknown"
