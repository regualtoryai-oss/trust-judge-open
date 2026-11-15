"""Synthetic lifecycle checks for real LangGraph scope/refocus integration.

Providers below are explicit test doubles, not semantic quality measurements.
"""
import copy
from uuid import UUID

import pytest

from app.domain.strict_verification import (
    ScopeVerificationInvalid,
    build_fact_views, build_scope_request, build_narrowing_request,
    resolve_scope_verification, resolve_annotation_narrowing,
)
from app.agents.provider import ModelOutputInvalid
from app.domain.scope_batches import plan_scope_batches, subset_evidence, merge_corrections
from app.workflows.analysis import AnalysisCancelled, run_analysis

Q = {"id": "internal-question", "raw_question_id": "question-test", "question": "设备是否通过安全检测？"}
TEXT = "甲公司声称设备通过安全检测，但报告尚未公开。"
PARAS = [{"id": "internal-para", "document_id": "internal-doc", "raw_doc_id": "doc-test", "raw_para_id": "para-test",
          "start_cp": 0, "end_cp": len(TEXT), "text": TEXT}]
KEYS = ("时间", "主体", "地点", "内容", "影响", "数值口径")


def annotation(text):
    return {"fact_des": text, "argument": {key: None for key in KEYS}, "entities": ["设备"]}


def decision(*, valid=True, needs=False, incomplete=False):
    return {"fact_entailment": "needs_narrowing" if needs else "entailed" if valid else "unsupported",
            "target_relation": "direct_answer", "target_question_text": "设备是否通过安全检测",
            "qualifier_dimension": None, "requires_unstated_bridge": False,
            "target_fact_present_but_annotation_incomplete": incomplete,
            "reason": "合成测试显式返回标签，不表示模型语义能力。"}


class ScopeProvider:
    def __init__(self, *, facts=None, mode="all", cancel_after_narrow=None, interrupt_narrow=False):
        self.facts = facts or ["有效事实"]
        self.mode = mode
        self.calls = []
        self.verify_count = 0
        self.cancel_after_narrow = cancel_after_narrow
        self.interrupt_narrow = interrupt_narrow
        self.narrow_inputs = []

    async def extract(self, question, paragraphs):
        self.calls.append("extract")
        return [{"paragraph_id": paragraphs[0]["id"], "quote": paragraphs[0]["text"], **annotation(text)} for text in self.facts]

    async def verify_evidence_scope(self, question, evidence):
        self.calls.append("scope")
        self.verify_count += 1
        payload, _, views = build_scope_request(question, evidence)
        rows = {}
        for index, view in enumerate(views):
            if self.mode == "first_bad":
                row = decision(valid=view["fact_des"] != "错误首解释")
            elif self.mode == "refocus":
                row = decision(needs=self.verify_count == 1)
            elif self.mode == "repeat_rejected":
                row = decision(needs=self.verify_count > 1, incomplete=True)
            elif self.mode == "manual_bad":
                row = decision(valid=False)
            elif self.mode == "mixed_needs":
                manual = next(source for source in evidence if source["id"] == view["evidence_id"]).get("interpretation_origin") == "human"
                row = decision(needs=manual or self.verify_count == 1)
            else:
                row = decision()
            rows[f"C{index + 1}"] = row
        return resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": rows}, question, evidence)

    async def narrow_evidence_annotations(self, question, evidence, fact_view_ids):
        self.calls.append("narrow")
        self.narrow_inputs.append({"evidence": copy.deepcopy(evidence), "fact_view_ids": list(fact_view_ids)})
        if self.interrupt_narrow:
            raise AnalysisCancelled("Synthetic interruption after reservation")
        try:
            payload, _, views = build_narrowing_request(question, evidence, fact_view_ids)
        except ScopeVerificationInvalid as error:
            raise ModelOutputInvalid("Synthetic adapter source guard", stage="verify", details=error.details) from None
        corrections = {}
        for index, view in enumerate(views):
            annotations = [annotation(view["fact_des"])] if self.mode == "repeat_rejected" else [annotation("重抽后的有归属事实"), annotation("重抽后的明确限定")]
            corrections[f"C{index + 1}"] = {"status": "corrected", "annotations": annotations}
        ledger = resolve_annotation_narrowing({"candidate_catalog_id": payload["candidate_catalog_id"], "corrections": corrections}, question, evidence, fact_view_ids)
        if self.cancel_after_narrow is not None:
            self.cancel_after_narrow[0] = True
        return ledger

    async def analyze(self, role, question, evidence, snapshot_id):
        self.calls.append(role)
        return self.answer()

    async def arbitrate(self, question, evidence, analyses):
        self.calls.append("arbitrate")
        return {**self.answer(), "conflicts": []}

    @staticmethod
    def answer():
        return {"decision": "undetermined", "conclusion": "公开材料尚有待核实的部分。", "judgments": [],
                "supporting_evidence_ids": [], "conflicting_evidence_ids": []}


class SecondLookProvider(ScopeProvider):
    """Explicit labels isolate scheduling from semantic model performance."""

    def __init__(self, *, initial="outside", reject_corrected=False,
                 bad_correction=False, no_supported=False, **kwargs):
        super().__init__(facts=["首次遗漏的来源解释"], **kwargs)
        self.initial = initial
        self.reject_corrected = reject_corrected
        self.bad_correction = bad_correction
        self.no_supported = no_supported
        self.scope_inputs = []

    async def verify_evidence_scope(self, question, evidence):
        self.calls.append("scope")
        self.verify_count += 1
        payload, _, views = build_scope_request(question, evidence)
        self.scope_inputs.append(copy.deepcopy(views))
        decisions = {}
        for index, view in enumerate(views):
            if self.verify_count == 1:
                row = decision(valid=self.initial in {"outside", "selected_incomplete"},
                               incomplete=self.initial == "selected_incomplete")
                if self.initial == "outside":
                    row.update(target_relation="outside_target", target_question_text=None)
            else:
                # A successful correction is still independently judged. The
                # synthetic extra annotation deliberately remains unsupported.
                row = decision(valid=not self.reject_corrected and view["fact_des"] == "重抽后的有归属事实")
                if self.reject_corrected:
                    row = decision(needs=True, incomplete=True)
            decisions[f"C{index + 1}"] = row
        return resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"],
            "decisions": decisions}, question, evidence)

    async def narrow_evidence_annotations(self, question, evidence, fact_view_ids):
        if self.no_supported:
            self.calls.append("narrow")
            self.narrow_inputs.append({"evidence": copy.deepcopy(evidence), "fact_view_ids": list(fact_view_ids)})
            payload, _, views = build_narrowing_request(question, evidence, fact_view_ids)
            return resolve_annotation_narrowing({"candidate_catalog_id": payload["candidate_catalog_id"],
                "corrections": {f"C{index + 1}": {"status": "no_supported_annotation", "annotations": []}
                                for index in range(len(views))}}, question, evidence, fact_view_ids)
        result = await super().narrow_evidence_annotations(question, evidence, fact_view_ids)
        if self.bad_correction:
            # Mutate the supposedly resolved projection, not the immutable
            # input. Workflow ledger replay must detect this before merge.
            result["corrected_evidence"][0]["quote"] = "伪造的引用"
        return result


async def run(tmp_path, provider, *, thread="scope-run", **kwargs):
    return await run_analysis(Q, PARAS, provider, checkpoint_path=tmp_path / "scope.sqlite3", thread_id=thread,
        cancel_check=kwargs.pop("cancel_check", lambda: False), emit=kwargs.pop("emit", lambda *_: None), **kwargs)


async def test_all_first_and_additional_views_are_independently_preserved_as_canonical_facts(tmp_path):
    provider = ScopeProvider(facts=["原子事实甲", "原子事实乙"])
    result = await run(tmp_path, provider)
    assert len(result["candidate_evidence"]) == 1
    assert len(result["candidate_evidence"][0]["additional_interpretations"]) == 1
    assert {row["fact_des"] for row in result["evidence"]} == {"原子事实甲", "原子事实乙"}
    assert len({UUID(row["id"]) for row in result["evidence"]}) == 2
    assert len({row["fact_view_id"] for row in result["evidence"]}) == 2
    assert all(row["scope_verified"] is True and row["quote"] == TEXT for row in result["evidence"])
    assert all(row["source_evidence_id"] == result["candidate_evidence"][0]["id"] for row in result["evidence"])


async def test_rejected_first_interpretation_does_not_hide_valid_additional_fact(tmp_path):
    provider = ScopeProvider(facts=["错误首解释", "有效附加解释"], mode="first_bad")
    result = await run(tmp_path, provider)
    # v17 inspects the rejected interpretation's fixed quote once more. This
    # explicit synthetic verifier accepts the new annotations, not the old one.
    assert {row["fact_des"] for row in result["evidence"]} == {
        "有效附加解释", "重抽后的有归属事实", "重抽后的明确限定"}
    assert provider.calls.count("scope") == 2 and provider.calls.count("narrow") == 1
    assert len(result["strict_scope_review"]["recall_recovery_plan"]["second_look_view_ids"]) == 1
    assert result["candidate_evidence"][0]["fact_des"] == "错误首解释"


async def test_one_content_refocus_then_every_new_fact_is_independently_verified(tmp_path):
    provider = ScopeProvider(mode="refocus")
    events = []
    result = await run(tmp_path, provider, emit=lambda kind, data: events.append((kind, copy.deepcopy(data))))
    assert provider.calls.count("narrow") == 1 and provider.calls.count("scope") == 2
    assert {row["fact_des"] for row in result["evidence"]} == {"重抽后的有归属事实", "重抽后的明确限定"}
    audit = result["strict_scope_review"]
    assert len(audit["narrowed_parent_ids"]) == 1
    assert len(audit["annotation_versions"]) == 2
    assert audit["unresolved_count"] == 0
    assert all(version["parent_fact_view_id"] in audit["narrowed_parent_ids"] for version in audit["annotation_versions"])
    assert any(kind == "tool_completed" and data["name"] == "reverify_evidence_scope" for kind, data in events)


async def test_unchanged_corrected_view_rejected_on_second_verify_cannot_keep_prior_pass(tmp_path):
    provider = ScopeProvider(mode="repeat_rejected")
    result = await run(tmp_path, provider)
    assert provider.calls.count("narrow") == 1 and provider.calls.count("scope") == 2
    assert result["evidence"] == []
    audit = result["strict_scope_review"]
    assert audit["unresolved_count"] >= 1
    assert result["final"]["decision"] == "undetermined"


async def test_failed_manual_opinion_retains_user_source_and_explicit_unverified_state(tmp_path):
    provider = ScopeProvider(mode="manual_bad")
    manual = {"paragraph_id": PARAS[0]["id"], "quote": TEXT, "start_cp": 0, "end_cp": len(TEXT), **annotation("人工待核验解释")}
    result = await run(tmp_path, provider, manual_evidence=[manual])
    humans = [row for row in result["evidence"] if row["interpretation_origin"] == "human"]
    assert len(humans) == 1 and humans[0]["quote"] == TEXT
    assert humans[0]["manual_review_required"] and not humans[0]["scope_verified"]
    # The separate automatic opinion may be retried, even at identical source
    # geometry. Human interpretation/source identity must never be rewritten.
    assert all(row.get("interpretation_origin") != "human"
               for call in provider.narrow_inputs for row in call["evidence"])
    assert humans[0]["fact_des"] == "人工待核验解释"
    plan = result["strict_scope_review"]["recall_recovery_plan"]
    assert not {humans[0]["fact_view_id"]} & set(plan["parent_view_ids"])


async def test_manual_needs_narrowing_does_not_block_automatic_same_batch_refocus(tmp_path):
    provider = ScopeProvider(mode="mixed_needs")
    quote = "报告尚未公开"
    start = TEXT.index(quote)
    manual = {"paragraph_id": PARAS[0]["id"], "quote": quote, "start_cp": start, "end_cp": start + len(quote), **annotation("人工待核验解释")}
    result = await run(tmp_path, provider, manual_evidence=[manual])
    auto = [row for row in result["evidence"] if row["interpretation_origin"] != "human"]
    humans = [row for row in result["evidence"] if row["interpretation_origin"] == "human"]
    assert {row["fact_des"] for row in auto} == {"重抽后的有归属事实", "重抽后的明确限定"}
    assert len(humans) == 1 and humans[0]["manual_review_required"]
    assert provider.calls.count("narrow") == 1


async def test_cancel_after_successful_narrow_checkpoint_resumes_only_second_verification(tmp_path):
    stop = [False]
    provider = ScopeProvider(mode="refocus", cancel_after_narrow=stop)
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, thread="original", cancel_check=lambda: stop[0])
    assert provider.calls.count("narrow") == 1
    stop[0] = False
    provider.cancel_after_narrow = None
    resumed = await run(tmp_path, provider, thread="resumed", resume_from_thread_id="original")
    assert provider.calls.count("extract") == 1 and provider.calls.count("narrow") == 1
    assert len(resumed["evidence"]) == 2


async def test_interrupted_reserved_narrow_is_not_paid_again_after_restart(tmp_path):
    provider = ScopeProvider(mode="refocus", interrupt_narrow=True)
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, thread="interrupted")
    assert provider.calls.count("narrow") == 1
    provider.interrupt_narrow = False
    resumed = await run(tmp_path, provider, thread="new-process", resume_from_thread_id="interrupted")
    assert provider.calls.count("narrow") == 1
    assert resumed["strict_scope_review"]["unresolved_count"] >= 1
    assert resumed["final"]["decision"] == "undetermined"


@pytest.mark.parametrize("initial", ["outside", "unsupported"])
async def test_unrequested_second_look_recovers_same_quote_then_independently_reverifies(tmp_path, initial):
    provider = SecondLookProvider(initial=initial)
    events = []
    result = await run(tmp_path, provider, emit=lambda kind, data: events.append((kind, copy.deepcopy(data))))
    assert provider.calls.count("narrow") == 1 and provider.calls.count("scope") == 2
    audit = result["strict_scope_review"]
    plan = audit["recall_recovery_plan"]
    assert plan["narrowing_view_ids"] == [] and len(plan["second_look_view_ids"]) == 1
    assert plan["parent_view_ids"] == audit["narrowed_parent_ids"]
    assert [row["fact_des"] for row in result["evidence"]] == ["重抽后的有归属事实"]
    assert result["candidate_evidence"][0]["fact_des"] == "首次遗漏的来源解释"
    assert len(audit["annotation_versions"]) == 2  # Both proposed; only one passed.
    assert len(audit["selected_fact_view_ids"]) == 1
    assert all(view["quote"] == TEXT and view["start_cp"] == 0 and view["end_cp"] == len(TEXT)
               for batch in provider.scope_inputs for view in batch)
    assert {view["fact_des"] for view in provider.scope_inputs[1]} == {
        "重抽后的有归属事实", "重抽后的明确限定"}
    assert any(kind == "tool_completed" and data["name"] == "reverify_evidence_scope" for kind, data in events)
    assert sum(kind == "recall_recovery_planned" for kind, _ in events) == 1
    assert result["final"]["decision"] == "undetermined"


async def test_second_look_invalid_quote_projection_is_not_accepted_or_reverified(tmp_path):
    provider = SecondLookProvider(bad_correction=True)
    events = []
    result = await run(tmp_path, provider, emit=lambda kind, data: events.append((kind, copy.deepcopy(data))))
    assert provider.calls.count("narrow") == provider.calls.count("scope") == 1
    assert result["candidate_evidence"][0]["quote"] == TEXT
    assert result["evidence"] == []
    audit = result["strict_scope_review"]
    assert audit["annotation_versions"] == [] and audit["unresolved_count"] == 1
    assert audit["narrowing_ledger"][0]["status"] == "invalid"
    assert result["final"]["decision"] == "undetermined"
    assert any(kind == "scope_narrowing_unresolved" and data["code"] == "narrow_annotation_invalid"
               for kind, data in events)


async def test_second_look_no_supported_annotation_keeps_audited_unresolved_without_fake_evidence(tmp_path):
    provider = SecondLookProvider(no_supported=True)
    result = await run(tmp_path, provider)
    assert provider.calls.count("narrow") == provider.calls.count("scope") == 1
    assert result["evidence"] == []
    audit = result["strict_scope_review"]
    assert audit["unresolved_count"] == 1 and audit["annotation_versions"] == []
    assert audit["narrowing_ledger"][0]["corrections"][0]["status"] == "no_supported_annotation"
    assert result["final"]["decision"] == "undetermined"


async def test_second_look_rejected_children_never_trigger_another_semantic_loop(tmp_path):
    provider = SecondLookProvider(reject_corrected=True)
    events = []
    result = await run(tmp_path, provider, emit=lambda kind, data: events.append((kind, copy.deepcopy(data))))
    assert provider.calls.count("narrow") == 1 and provider.calls.count("scope") == 2
    assert result["evidence"] == [] and result["final"]["decision"] == "undetermined"
    assert result["strict_scope_review"]["unresolved_count"] >= 1
    assert sum(kind == "recall_recovery_planned" for kind, _ in events) == 1
    assert len(result["strict_scope_review"]["narrowing_ledger"]) == 1


async def test_explicit_selected_incomplete_parent_is_preserved_when_only_new_children_are_rejected(tmp_path):
    provider = SecondLookProvider(initial="selected_incomplete", reject_corrected=True)
    result = await run(tmp_path, provider)
    assert provider.calls.count("narrow") == 1 and provider.calls.count("scope") == 2
    assert [row["fact_des"] for row in result["evidence"]] == ["首次遗漏的来源解释"]
    audit = result["strict_scope_review"]
    plan = audit["recall_recovery_plan"]
    assert len(plan["narrowing_view_ids"]) == 1 and plan["second_look_view_ids"] == []
    assert plan["narrowing_view_ids"] == plan["audit"]["retained_selected_narrowing_view_ids"]
    assert plan["narrowing_view_ids"][0] in audit["selected_fact_view_ids"]
    assert {view["id"] for view in provider.scope_inputs[0]}.isdisjoint(
        {view["id"] for view in provider.scope_inputs[1]})
    assert audit["unresolved_count"] >= 1  # The omitted facts remain unresolved.


async def test_cancel_after_second_look_reuses_checkpoint_and_only_pays_independent_reverify(tmp_path):
    stop = [False]
    provider = SecondLookProvider(cancel_after_narrow=stop)
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, thread="second-look-original", cancel_check=lambda: stop[0])
    assert provider.calls.count("extract") == provider.calls.count("narrow") == provider.calls.count("scope") == 1
    stop[0] = False
    provider.cancel_after_narrow = None
    resumed = await run(tmp_path, provider, thread="second-look-resumed", resume_from_thread_id="second-look-original")
    assert provider.calls.count("extract") == provider.calls.count("narrow") == 1
    assert provider.calls.count("scope") == 2
    assert [row["fact_des"] for row in resumed["evidence"]] == ["重抽后的有归属事实"]
    assert len(resumed["strict_scope_review"]["recall_recovery_plan"]["second_look_view_ids"]) == 1


async def test_interrupted_second_look_reservation_is_not_paid_again_on_resume(tmp_path):
    provider = SecondLookProvider(interrupt_narrow=True)
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, thread="second-look-interrupted")
    assert provider.calls.count("narrow") == 1
    provider.interrupt_narrow = False
    result = await run(tmp_path, provider, thread="second-look-restarted", resume_from_thread_id="second-look-interrupted")
    assert provider.calls.count("extract") == provider.calls.count("narrow") == provider.calls.count("scope") == 1
    assert result["evidence"] == [] and result["final"]["decision"] == "undetermined"
    audit = result["strict_scope_review"]
    assert audit["unresolved_count"] == 1 and audit["narrowing_ledger"][0]["status"] == "interrupted_unknown"


def test_scope_batch_split_covers_first_additional_views_once_without_cutting_source():
    row = {"id": "anchor", "paragraph_id": "para", "raw_doc_id": "doc", "raw_para_id": "rawpara",
           "quote": TEXT, "start_cp": 0, "end_cp": len(TEXT), **annotation("原子甲"),
           "additional_interpretations": [annotation("原子乙"), annotation("原子丙")]}
    original = copy.deepcopy(row)
    batches = plan_scope_batches([row], question=Q, max_views=1)
    assert len(batches) == 3 and len({value for batch in batches for value in batch}) == 3
    for batch in batches:
        subset = subset_evidence([row], batch)
        assert len(build_fact_views(subset)) == 1
        assert subset[0]["quote"] == TEXT and subset[0]["start_cp"] == 0 and subset[0]["end_cp"] == len(TEXT)
    assert row == original


def test_merge_corrections_cannot_change_original_geometry_or_internal_ledger():
    row = {"id": "anchor", "paragraph_id": "para", "raw_doc_id": "doc", "raw_para_id": "rawpara",
           "quote": TEXT, "start_cp": 0, "end_cp": len(TEXT), **annotation("原始甲")}
    original = copy.deepcopy(row)
    corrected = {**copy.deepcopy(row), **annotation("新增乙")}
    merged = merge_corrections([row], [corrected])
    assert row == original
    assert {view["fact_des"] for view in build_fact_views(merged)} == {"原始甲", "新增乙"}
    corrected["end_cp"] += 1
    with pytest.raises(ValueError, match="scope_correction_geometry"):
        merge_corrections([row], [corrected])
