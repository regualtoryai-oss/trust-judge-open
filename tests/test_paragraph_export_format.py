from copy import deepcopy

import pytest

from app.domain.snapshot import build_snapshot, competition_files
from tests.test_snapshot import build, fixture


def test_extended_relation_fields_link_exact_exported_fact_and_keep_unknown_label_partial():
    topic, questions, results = fixture()
    snapshot = build(topic, questions, results)
    files = competition_files(snapshot)
    fact = files["evidence.json"][0]
    ref = files["conclusions.json"][0]["supporting_facts"][0]
    assert {key: ref[key] for key in ("doc_id", "para_id", "fact_id", "quote")} == {
        key: fact[key] for key in ("doc_id", "para_id", "fact_id", "quote")}
    assert ref["verdict_label"] == "partial_support"
    assert results[questions[0]["id"]]["final"]["decision"] == "asserted"


def test_full_support_only_from_explicit_valid_per_evidence_label():
    topic, questions, results = fixture()
    results[questions[0]["id"]]["final"]["evidence_verdict_labels"] = {"evidence-internal": "full_support"}
    assert competition_files(build(topic, questions, results))["conclusions.json"][0]["supporting_facts"][0]["verdict_label"] == "full_support"
    results[questions[0]["id"]]["final"]["evidence_verdict_labels"] = {"unknown": "full_support"}
    with pytest.raises(ValueError, match="verdict labels"):
        build(topic, questions, results)


def test_same_paragraph_views_merge_once_without_erasing_mirror_source():
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    peer = deepcopy(result["evidence"][0])
    peer.update(id="second-view", fact_des="另一事实说明", argument={"内容": "另一事实说明"})
    mirror = deepcopy(peer)
    mirror.update(id="mirror", paragraph_id="mirror-paragraph", raw_doc_id="mirror-doc", source_group_id="source_1")
    result["evidence"] += [peer, mirror]
    result["final"]["supporting_evidence_ids"] += [peer["id"], mirror["id"]]
    snapshot = build(topic, questions, results)
    before = deepcopy(snapshot)
    files = competition_files(snapshot)
    assert len(files["evidence.json"]) == 2
    assert len(files["conclusions.json"][0]["supporting_facts"]) == 2
    assert "另一事实说明" in files["evidence.json"][0]["fact_des"]
    assert files["confidence.json"][-1]["components"]["source_credibility"] == 0
    assert len(files["confidence.json"][-1]["inputs"]["support_groups"]) == 1
    assert snapshot == before


def test_disjoint_unavailable_source_is_rejected_instead_of_inventing_joined_quote():
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    peer = deepcopy(result["evidence"][0])
    peer.update(id="disjoint", start_cp=20, end_cp=24, quote="另有说明", fact_des="另有说明")
    result["evidence"].append(peer)
    with pytest.raises(ValueError, match="original continuous source"):
        build(topic, questions, results)


def test_confidence_quote_tamper_cannot_be_repaired_by_paragraph_projection():
    topic, questions, results = fixture()
    snapshot = build(topic, questions, results)
    snapshot["confidence"][0]["object_ref"]["quote"] = "不是原文"
    with pytest.raises(ValueError, match="absent evidence"):
        competition_files(snapshot)


def test_raw_source_context_expands_legacy_slice_without_changing_snapshot():
    topic, questions, results = fixture()
    snapshot = build(topic, questions, results)
    before = deepcopy(snapshot)
    sources = [{"id": "paragraph-internal", "raw_doc_id": "doc_001", "raw_para_id": "doc_001#p1", "text": "通报称：失联3人。"}]
    output = competition_files(snapshot, source_paragraphs=sources)
    assert output["evidence.json"][0]["quote"] == sources[0]["text"]
    assert output["conclusions.json"][0]["supporting_facts"][0]["quote"] == sources[0]["text"]
    assert output["confidence.json"][0]["object_ref"]["quote"] == sources[0]["text"]
    assert snapshot == before
    sources[0]["text"] = "发生了其他事情"
    with pytest.raises(ValueError, match="differs"):
        competition_files(snapshot, source_paragraphs=sources)


def test_build_snapshot_validates_disjoint_views_against_supplied_whole_source():
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    peer = deepcopy(result["evidence"][0])
    text = "通报称：失联3人。另有说明"
    start = text.index("另有说明")
    peer.update(id="disjoint", start_cp=start, end_cp=len(text), quote="另有说明", fact_des="另有说明")
    result["evidence"].append(peer)
    sources = [{"id": "paragraph-internal", "raw_doc_id": "doc_001", "raw_para_id": "doc_001#p1", "text": text}]
    snapshot = build_snapshot(topic, questions, results, revision_id="test-ready", graph_revision=1,
                              run_id="test-run", source_paragraphs=sources)
    assert competition_files(snapshot, source_paragraphs=sources)["evidence.json"][0]["quote"] == text
    assert len(snapshot["evidence"]) == 2
    sources[0]["text"] = "通报称：失联3人。另有描述"
    with pytest.raises(ValueError, match="differs"):
        build_snapshot(topic, questions, results, revision_id="test-ready", graph_revision=1,
                       run_id="test-run", source_paragraphs=sources)


def test_duplicate_paragraph_quality_tamper_is_checked_before_discarding_it():
    topic, questions, results = fixture()
    peer = deepcopy(results[questions[0]["id"]]["evidence"][0])
    peer.update(id="other-view", fact_des="另一个描述")
    results[questions[0]["id"]]["evidence"].append(peer)
    snapshot = build(topic, questions, results)
    snapshot["confidence"][1]["score"] = .987654
    with pytest.raises(ValueError, match="quality projection"):
        competition_files(snapshot)


def test_known_quote_swap_in_quality_is_not_repaired_by_merge():
    topic, questions, results = fixture()
    peer = deepcopy(results[questions[0]["id"]]["evidence"][0])
    peer.update(id="other-view", start_cp=6, end_cp=8, quote="3人", fact_des="数值")
    results[questions[0]["id"]]["evidence"].append(peer)
    snapshot = build(topic, questions, results)
    snapshot["confidence"][0]["object_ref"]["quote"] = "3人"
    with pytest.raises(ValueError, match="quality projection"):
        competition_files(snapshot)


def test_full_label_does_not_cover_another_unselected_annotation_in_same_paragraph():
    topic, questions, results = fixture()
    result = results[questions[0]["id"]]
    peer = deepcopy(result["evidence"][0])
    peer.update(id="other-view", fact_des="另一个尚未采纳的断言")
    result["evidence"].append(peer)
    result["final"]["evidence_verdict_labels"] = {"evidence-internal": "full_support"}
    files = competition_files(build(topic, questions, results))
    ref = files["conclusions.json"][0]["supporting_facts"][0]
    assert ref["verdict_label"] == "partial_support"
    assert "尚未采纳" not in ref["fact_des"]
