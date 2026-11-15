"""UNAPPLIED v14 edge fixes, synthetic facts only, no external model calls.

These tests exercise real domain guards and the real LangGraph lifecycle. The
explicit provider decisions are control-flow fixtures, not semantic P/R scores.
"""
import copy
import hashlib
import json

import pytest

from app.domain.refine_input import create_candidate_artifact, digest, source_manifest
from app.domain.strict_verification import (
    build_fact_views, build_scope_request, build_narrowing_request,
    resolve_scope_verification, resolve_annotation_narrowing,
)
from app.workflows.analysis import _filter_operator_exclusions, run_analysis
from test_scope_refine_workflow import Q, annotation, decision, ScopeProvider


NEGATIVE_TEXT = "检测机构公告称设备未通过安全检测，复检结论尚未公布。"
POSITIVE_FACT = "设备通过安全检测。"
ATTRIBUTED_NEGATIVE = "检测机构公告称设备未通过安全检测。"
QUALIFIER = "复检结论尚未公布。"


def para(identifier, text):
    return {"id": identifier, "document_id": "doc-" + identifier,
            "raw_doc_id": "raw-doc-" + identifier, "raw_para_id": "raw-para-" + identifier,
            "start_cp": 0, "end_cp": len(text), "text": text}


def anchor(identifier="anchor", text=NEGATIVE_TEXT, *, start=0, paragraph_id="p", origin="model"):
    return {"id": identifier, "paragraph_id": paragraph_id,
            "raw_doc_id": "raw-doc-" + paragraph_id, "raw_para_id": "raw-para-" + paragraph_id,
            "quote": text, "start_cp": start, "end_cp": start + len(text),
            "interpretation_origin": origin, **annotation(POSITIVE_FACT)}


def verdict(entailment="contradicted", relation="direct_answer"):
    row = decision()
    row.update(fact_entailment=entailment, target_relation=relation,
               target_fact_present_but_annotation_incomplete=False,
               target_question_text=None if relation == "outside_target" else "设备是否通过安全检测",
               qualifier_dimension="certainty" if relation == "necessary_qualifier" else None)
    return row


@pytest.mark.parametrize("relation", ["direct_answer", "explicit_counterclaim", "necessary_qualifier"])
def test_in_target_contradiction_requests_refocus_without_accepting_old_annotation(relation):
    pool = [anchor()]
    original = copy.deepcopy(pool)
    payload, _, views = build_scope_request(Q, pool)
    ledger = resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"],
        "decisions": {"C1": verdict(relation=relation)}}, Q, pool)
    assert ledger["narrowing_view_ids"] == [views[0]["id"]]
    assert ledger["selected_fact_view_ids"] == [] and ledger["verified_evidence"] == []
    assert ledger["assessments"][0]["fact_entailment"] == "contradicted"
    assert ledger["assessments"][0]["target_fact_present_but_annotation_incomplete"] is False
    assert pool == original


@pytest.mark.parametrize("entailment,relation", [
    ("contradicted", "outside_target"), ("unsupported", "direct_answer"),
    ("unsupported", "outside_target"),
])
def test_new_trigger_does_not_broaden_outside_target_or_unsupported_facts(entailment, relation):
    pool = [anchor()]
    payload, _, _ = build_scope_request(Q, pool)
    result = resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"],
        "decisions": {"C1": verdict(entailment, relation)}}, Q, pool)
    assert result["narrowing_view_ids"] == result["selected_fact_view_ids"] == []


class ContradictedProvider(ScopeProvider):
    def __init__(self, *, reject_second=False, no_supported=False):
        super().__init__(facts=[POSITIVE_FACT])
        self.reject_second = reject_second
        self.no_supported = no_supported
        self.reviewed = []
        self.correction_inputs = []

    async def verify_evidence_scope(self, question, evidence):
        self.calls.append("scope")
        self.verify_count += 1
        self.reviewed.append(copy.deepcopy(evidence))
        payload, _, views = build_scope_request(question, evidence)
        decisions = {}
        for index, view in enumerate(views):
            first = self.verify_count == 1
            relation = "necessary_qualifier" if view["fact_des"] == QUALIFIER else "explicit_counterclaim"
            decisions[f"C{index + 1}"] = verdict(
                "contradicted" if first or self.reject_second else "entailed", relation)
        return resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"],
            "decisions": decisions}, question, evidence)

    async def narrow_evidence_annotations(self, question, evidence, fact_view_ids):
        self.calls.append("narrow")
        self.correction_inputs.append(copy.deepcopy(evidence))
        payload, _, views = build_narrowing_request(question, evidence, fact_view_ids)
        corrections = {f"C{index + 1}": {
            "status": "no_supported_annotation" if self.no_supported else "corrected",
            "annotations": [] if self.no_supported else [annotation(ATTRIBUTED_NEGATIVE), annotation(QUALIFIER)],
        } for index, _ in enumerate(views)}
        return resolve_annotation_narrowing({"candidate_catalog_id": payload["candidate_catalog_id"],
            "corrections": corrections}, question, evidence, fact_view_ids)


async def run_local(tmp_path, provider, paragraphs, *, thread="edge", **kwargs):
    return await run_analysis(Q, paragraphs, provider, checkpoint_path=tmp_path / "edge.sqlite3",
        thread_id=thread, cancel_check=lambda: False, emit=lambda *_: None, **kwargs)


async def test_false_incomplete_flag_still_refocuses_negative_quote_and_verifies_every_child(tmp_path):
    paragraphs = [para("negative", NEGATIVE_TEXT)]
    provider = ContradictedProvider()
    result = await run_local(tmp_path, provider, paragraphs)
    assert provider.calls.count("narrow") == 1 and provider.calls.count("scope") == 2
    assert len(build_fact_views(provider.reviewed[1])) == 2
    assert {row["fact_des"] for row in result["evidence"]} == {ATTRIBUTED_NEGATIVE, QUALIFIER}
    assert result["candidate_evidence"][0]["fact_des"] == POSITIVE_FACT
    assert len(result["strict_scope_review"]["annotation_versions"]) == 2
    assert result["strict_scope_review"]["unresolved_count"] == 0
    original = result["candidate_evidence"][0]
    for reviewed in [*provider.reviewed, *provider.correction_inputs]:
        for row in reviewed:
            assert (row["quote"], row["start_cp"], row["end_cp"], row["paragraph_id"]) == (
                NEGATIVE_TEXT, 0, len(NEGATIVE_TEXT), paragraphs[0]["id"])
    assert all(row["source_evidence_id"] == original["id"] and row["scope_verified"]
               for row in result["evidence"])


async def test_second_contradiction_is_audited_and_never_loops_or_selects(tmp_path):
    provider = ContradictedProvider(reject_second=True)
    result = await run_local(tmp_path, provider, [para("negative", NEGATIVE_TEXT)])
    assert provider.calls.count("narrow") == 1 and provider.calls.count("scope") == 2
    assert result["evidence"] == [] and result["final"]["decision"] == "undetermined"
    audit = result["strict_scope_review"]
    second = [row for row in audit["assessments"] if row["phase"] == "reverify"]
    assert len(second) == 2 and all(row["fact_entailment"] == "contradicted" for row in second)
    assert all(row["fact_view_id"] in audit["unresolved_view_ids"] for row in second)
    assert audit["unresolved_count"] == 3  # failed parent + both corrected views
    assert result["candidate_evidence"][0]["fact_des"] == POSITIVE_FACT


async def test_no_supported_annotation_remains_unresolved_without_inventing_a_fact(tmp_path):
    provider = ContradictedProvider(no_supported=True)
    result = await run_local(tmp_path, provider, [para("negative", NEGATIVE_TEXT)])
    assert provider.calls.count("narrow") == 1 and provider.calls.count("scope") == 1
    assert result["evidence"] == [] and result["strict_scope_review"]["unresolved_count"] == 1
    assert result["strict_scope_review"]["annotation_versions"] == []


async def test_contradicted_manual_annotation_is_not_automatically_rewritten(tmp_path):
    provider = ContradictedProvider()
    quote = "复检结论尚未公布"
    start = NEGATIVE_TEXT.index(quote)
    manual = {"paragraph_id": "negative", "quote": quote, "start_cp": start,
              "end_cp": start + len(quote), **annotation("人工待核实断言设备通过检测。")}
    result = await run_local(tmp_path, provider, [para("negative", NEGATIVE_TEXT)], manual_evidence=[manual])
    human = [row for row in result["evidence"] if row.get("interpretation_origin") == "human"]
    assert len(human) == 1 and human[0]["manual_review_required"] and not human[0]["scope_verified"]
    assert human[0]["fact_des"] == manual["fact_des"] and human[0]["quote"] == quote
    assert provider.calls.count("narrow") == 1
    assert all(row.get("interpretation_origin") != "human" for batch in provider.correction_inputs for row in batch)
    assert not result["strict_scope_review"]["formal_export_ready"]


@pytest.mark.parametrize("start,end", [(4, 6), (1, 8), (0, 12), (0, 5), (9, 12)])
def test_any_overlap_filters_complete_window_including_human_anchors(start, end):
    text = "abcdefghijkl"
    excluded = {("p", 4, 10, text[4:10])}
    rows = [anchor("auto", text[start:end], start=start),
            anchor("manual", text[start:end], start=start, origin="human")]
    original = copy.deepcopy(rows)
    kept, audit = _filter_operator_exclusions(rows, excluded)
    assert kept == [] and {row["evidence_id"] for row in audit} == {"auto", "manual"}
    assert rows == original
    assert all(row["reason"] == "operator_excluded_overlap" for row in audit)
    assert all(row["quote_sha256"] == hashlib.sha256(text[start:end].encode()).hexdigest() for row in audit)
    assert all(row["fact_view_ids"] for row in audit)
    assert text[start:end] not in json.dumps(audit) and text[4:10] not in json.dumps(audit)


def test_half_open_touching_spans_and_different_paragraphs_are_not_excluded():
    rows = [anchor("before", "abcd", start=0), anchor("after", "kl", start=10),
            anchor("other", "efghij", start=4, paragraph_id="other")]
    original = copy.deepcopy(rows)
    kept, audit = _filter_operator_exclusions(rows, {("p", 4, 10, "efghij")})
    assert kept == original and audit == []
    kept[0]["fact_des"] = "mutated copy"
    assert rows == original


TEXTS = ["甲设备未通过安全检测，报告已发布。", "乙设备通过安全检测，编号乙已公开。"]
PARAGRAPHS = [para("first", TEXTS[0]), para("second", TEXTS[1])]
BLOCKED_QUOTE = "未通过"
BLOCKED_START = TEXTS[0].index(BLOCKED_QUOTE)
EXCLUSIONS = {(PARAGRAPHS[0]["id"], BLOCKED_START, BLOCKED_START + len(BLOCKED_QUOTE), BLOCKED_QUOTE)}


class ExclusionProvider(ScopeProvider):
    def __init__(self):
        super().__init__()
        self.received_scope = []
        self.received_analyst = []

    async def extract(self, question, paragraphs):
        self.calls.append("extract")
        return [{"paragraph_id": row["id"], "quote": row["text"], **annotation(row["text"])}
                for row in paragraphs]

    async def verify_evidence_scope(self, question, evidence):
        self.received_scope.extend(copy.deepcopy(evidence))
        return await super().verify_evidence_scope(question, evidence)

    async def analyze(self, role, question, evidence, snapshot_id):
        self.received_analyst.extend(copy.deepcopy(evidence))
        return await super().analyze(role, question, evidence, snapshot_id)


async def test_small_explicit_exclusion_blocks_whole_model_window_before_paid_scope(tmp_path):
    provider = ExclusionProvider()
    manual = {"paragraph_id": PARAGRAPHS[0]["id"], "quote": BLOCKED_QUOTE,
              "start_cp": BLOCKED_START, "end_cp": BLOCKED_START + len(BLOCKED_QUOTE),
              **annotation("人工解释也服从同一显式排除。")}
    result = await run_local(tmp_path, provider, PARAGRAPHS, excluded_spans=EXCLUSIONS,
                             manual_evidence=[manual])
    assert {row["paragraph_id"] for row in result["evidence"]} == {PARAGRAPHS[1]["id"]}
    assert all(row["paragraph_id"] == PARAGRAPHS[1]["id"]
               for row in [*provider.received_scope, *provider.received_analyst])
    assert len(result["operator_exclusion_audit"]) == 2
    assert {row["interpretation_origin"] for row in result["operator_exclusion_audit"]} == {"model", "human"}
    audit = result["strict_scope_review"]
    assert audit["operator_excluded_candidate_count"] == 2 and audit["unresolved_count"] == 0
    assert len(audit["operator_excluded_fact_view_ids"]) == 2


async def test_imported_new_refine_job_keeps_original_artifact_but_never_reviews_excluded_window(tmp_path):
    seed = await run_local(tmp_path, ExclusionProvider(), PARAGRAPHS, thread="source")
    artifact = create_candidate_artifact(topic_id="topic-test", source_revision_id="revision-test",
        source_artifact_sha256="a" * 64, source_manifest_sha256=digest(source_manifest(PARAGRAPHS)),
        question=Q, paragraphs=PARAGRAPHS, result={**seed, "origin_run_id": "source-run-test"},
        source_config={"source_batch_codepoints": 20_000, "source_batch_spans": 128},
        overlays={"excluded_spans": sorted(EXCLUSIONS)}, created_at="2026-10-18T00:00:00Z")
    before = copy.deepcopy(artifact)
    provider = ExclusionProvider()
    result = await run_local(tmp_path, provider, PARAGRAPHS, thread="refined",
                             candidate_artifact=artifact, excluded_spans=EXCLUSIONS)
    assert artifact == before
    assert provider.calls.count("extract") == 0
    assert result["candidate_evidence"] == artifact["candidate_evidence"]
    assert len(result["candidate_evidence"]) == 2  # immutable imported ledger remains complete
    assert {row["paragraph_id"] for row in result["evidence"]} == {PARAGRAPHS[1]["id"]}
    assert all(row["paragraph_id"] == PARAGRAPHS[1]["id"]
               for row in [*provider.received_scope, *provider.received_analyst])
    assert len(result["operator_exclusion_audit"]) == 1
    assert result["strict_scope_review"]["operator_excluded_candidate_count"] == 1
    assert not result["candidate_import_provenance"]["new_source_scan_performed"]
