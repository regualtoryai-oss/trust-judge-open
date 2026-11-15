"""Synthetic source/ledger authority tests; no gold or model-quality claims."""
from copy import deepcopy
import json

import pytest

from app.domain.premise_bank import (
    PremiseBankInvalid, build_quote_faithful_bank, build_fact_purpose_request,
    resolve_fact_purpose, validate_fact_purpose_ledger, compile_purpose_projection,
    validate_premise_projection, plan_fact_purpose_batches,
)
from app.domain.strict_verification import (
    build_scope_request, resolve_scope_verification, build_fact_views,
    build_narrowing_request, resolve_annotation_narrowing,
)
from app.domain.material_eligibility import validate_workflow_material_eligibility, Invalid


Q = {"id": "q-local", "raw_question_id": "q_raw", "question": "场馆应归为应急避险设施，还是日常服务设施？"}
KEYS = ("时间", "主体", "地点", "内容", "影响", "数值口径")


def source(identity="a", *, human=False, text="场馆可在预警时安置疏散人员。平时用于体育活动。", fact=None):
    return {"id": identity, "paragraph_id": "p-" + identity, "raw_doc_id": "doc-" + identity,
        "raw_para_id": "para-" + identity, "start_cp": 0, "end_cp": len(text), "quote": text,
        "fact_des": fact or text, "argument": {key: "场馆" if key == "主体" else None for key in KEYS},
        "entities": ["场馆"], "interpretation_origin": "human" if human else "model",
        "source_group_id": "g-" + identity, "source_group_basis": "same_document_only_not_complete_origin_detection",
        "source_level": "unknown", "grade_score": None}


def paragraphs(rows):
    return [{"id": row["paragraph_id"], "document_id": "d-" + row["id"], "raw_doc_id": row["raw_doc_id"],
             "raw_para_id": row["raw_para_id"], "text": row["quote"]} for row in rows]


def decision(view, *, target="outside_target", target_bridge=False, quote_bridge=False, field_failure=None):
    checks = {"fact_des": "entailed", "entities": "entailed" if view["entities"] else "not_present",
              "argument": {key: "entailed" if value else "not_present" for key, value in view["argument"].items()}}
    if field_failure == "fact_des": checks["fact_des"] = "unsupported"
    elif field_failure == "entities": checks["entities"] = "unsupported"
    elif field_failure: checks["argument"][field_failure] = "needs_narrowing"
    return {"fact_entailment": "entailed", "target_relation": target,
        "target_question_text": Q["question"] if target != "outside_target" else None,
        "qualifier_dimension": None, "requires_unstated_bridge": quote_bridge or target_bridge,
        "quote_requires_unstated_bridge": quote_bridge, "target_requires_unstated_bridge": target_bridge,
        "field_checks": checks, "target_fact_present_but_annotation_incomplete": False,
        "reason": "合成标签检验程序边界，不表示真实模型判断能力。"}


def scope(rows, *, q=Q, mutate=None, **options):
    payload, _, views = build_scope_request(q, rows, detailed=True)
    decisions = {f"C{i + 1}": decision(view, **options) for i, view in enumerate(views)}
    if mutate: mutate(decisions)
    ledger = resolve_scope_verification({"candidate_catalog_id": payload["candidate_catalog_id"], "decisions": decisions}, q, rows)
    return {"complete": True, "assessments": [{"phase": "original", **row} for row in ledger["assessments"]],
        "narrowing_ledger": [], "selected_fact_view_ids": sorted(ledger["selected_fact_view_ids"]),
        "operator_excluded_fact_view_ids": [], "exact_annotation_duplicate_map": []}


def bank_for(rows=None, **options):
    rows = rows or [source()]
    audit = scope(rows, **options)
    return rows, audit, build_quote_faithful_bank(Q, rows, audit, paragraphs(rows))


def purpose(bank, *, role="necessary_premise", ids=None, mutate=None):
    ids = [row["id"] for row in bank["rows"]] if ids is None else ids
    payload, _, _ = build_fact_purpose_request(Q, bank, ids)
    output = {"bank_id": bank["bank_id"], "purpose_catalog_id": payload["purpose_catalog_id"], "decisions": {
        row["id"]: {"purpose_binding_id": f"T0.{role}" if role not in {"background", "uncertain", "duplicate"} else role,
                    "introduces_unstated_source_fact": False, "equivalent_to": None,
                    "reason": "该对象的具体功能可用于分类比较，但原句没有独自给出完整分类结论。"}
        for row in payload["candidate_catalog"]}}
    if mutate: mutate(output)
    return resolve_fact_purpose(output, Q, bank, ids)


def result_for(rows, audit, bank, ledgers):
    formal, review, material = compile_purpose_projection(Q, rows, audit, bank, ledgers, paragraphs(rows))
    return {"candidate_evidence": rows, "strict_scope_review": audit, "quote_faithful_bank": bank,
        "purpose_review": review, "material_eligibility": material, "evidence": formal,
        "verification_protocol": bank["version"]}


def test_rehashed_stored_bank_cannot_smuggle_unknown_question_fields_into_purpose_payload():
    from app.domain.premise_bank import digest
    _rows, _scope, bank = bank_for()
    bank["question"].update(answer="PRIVATE_REFERENCE", conclusion={"body": "PRIVATE_GOLD"})
    bank["bank_id"] = "QB" + digest({key: value for key, value in bank.items() if key != "bank_id"})
    payload, _schema, _rows = build_fact_purpose_request(Q, bank, [row["id"] for row in bank["rows"]])
    assert "PRIVATE" not in json.dumps(payload)
    assert set(payload["question"]) == set(Q)


@pytest.mark.parametrize("limits", [{"max_rows": 0}, {"max_rows": True}, {"max_codepoints": 0}, {"max_codepoints": "1"}])
def test_invalid_batch_limits_do_not_drop_or_skip_fact_coverage(limits):
    _rows, _scope, bank = bank_for()
    with pytest.raises(PremiseBankInvalid): plan_fact_purpose_batches(Q, bank, **limits)


def test_target_bridge_does_not_reject_quote_faithful_premise_or_rewrite_original_selection():
    rows, audit, bank = bank_for(target="direct_answer", target_bridge=True)
    before = deepcopy((rows, audit))
    assert audit["selected_fact_view_ids"] == [] and len(bank["rows"]) == 1
    assert bank["formal_evidence"] is False and bank["rows"][0]["purpose_verified"] is False
    selected = purpose(bank)
    result = result_for(rows, audit, bank, [selected])
    assert len(result["evidence"]) == 1
    assert result["purpose_review"]["ledgers"][0]["assessments"][0]["role"] == "necessary_premise"
    assert result["evidence"][0]["quote"] == rows[0]["quote"]
    assert (rows, audit) == before and audit["selected_fact_view_ids"] == []
    validate_premise_projection(result, Q, paragraphs(rows))
    validate_workflow_material_eligibility(result, Q, paragraphs(rows))


@pytest.mark.parametrize("failure", ["fact_des", "entities", "主体", "时间", "数值口径"])
def test_no_target_permission_can_override_bad_quote_annotation_fields(failure):
    rows, audit, bank = bank_for(field_failure=failure)
    assert bank["rows"] == []
    assert bank["decisions"][0]["quote_faithful"] is False
    result = result_for(rows, audit, bank, [])
    assert result["evidence"] == []
    validate_premise_projection(result, Q, paragraphs(rows))


def test_quote_bridge_remains_hard_rejection_even_for_same_target_object():
    _rows, _audit, bank = bank_for(target="direct_answer", quote_bridge=True)
    assert bank["rows"] == []


def test_background_can_enter_internal_faithful_bank_but_not_formal_export_pool():
    rows, audit, bank = bank_for()
    assert len(bank["rows"]) == 1
    result = result_for(rows, audit, bank, [purpose(bank, role="background")])
    assert result["evidence"] == [] and result["purpose_review"]["complete"]


def test_introducing_unstated_source_fact_blocks_even_named_necessary_premise():
    rows, audit, bank = bank_for()
    ledger = purpose(bank, mutate=lambda wire: wire["decisions"]["C1"].update(introduces_unstated_source_fact=True))
    assert ledger["selected_evidence_ids"] == []
    assert result_for(rows, audit, bank, [ledger])["evidence"] == []


def test_purpose_prompt_projection_has_no_old_target_judgments_or_analyst_answers():
    _rows, _audit, bank = bank_for(target="direct_answer", target_bridge=True)
    q = {**Q, "conclusion": "PRIVATE_GOLD", "supporting_facts": ["PRIVATE"]}
    payload, _, _ = build_fact_purpose_request(q, bank, [bank["rows"][0]["id"]])
    encoded = json.dumps(payload)
    assert "PRIVATE" not in encoded
    for forbidden in ("selected", "original_scope", "target_requires_unstated_bridge", "field_checks", "reason", "analyses"):
        assert forbidden not in encoded
    assert payload["candidate_catalog"][0]["source_group_id"] == "g-a"


def test_original_success_followed_by_failed_independent_reverify_does_not_stay_in_bank():
    rows = [source()]
    audit = scope(rows, target="direct_answer")
    ids = [row["id"] for row in build_fact_views(rows)]
    request, _, _ = build_narrowing_request(Q, rows, ids)
    annotation = {key: deepcopy(rows[0][key]) for key in ("fact_des", "argument", "entities")}
    narrow = resolve_annotation_narrowing({"candidate_catalog_id": request["candidate_catalog_id"], "corrections": {
        "C1": {"status": "corrected", "annotations": [annotation]}}}, Q, rows, ids)
    again = scope(rows, target="direct_answer", field_failure="entities")
    audit["assessments"].extend({**row, "phase": "reverify"} for row in again["assessments"])
    audit["narrowing_ledger"] = [narrow]
    audit["selected_fact_view_ids"] = []
    bank = build_quote_faithful_bank(Q, rows, audit, paragraphs(rows))
    assert bank["rows"] == [] and bank["decisions"][0]["review_phase"] == "reverify"


@pytest.mark.parametrize("change", ["missing_row", "unknown_row", "no_fields", "no_quote_bridge", "fake_selected", "extra_assessment"])
def test_scope_proof_missing_or_forged_data_cannot_be_upgraded_to_bank(change):
    rows, audit, _bank = bank_for()
    if change == "missing_row": audit["assessments"] = []
    elif change == "unknown_row": audit["assessments"][0]["fact_view_id"] = "unknown"
    elif change == "no_fields": audit["assessments"][0]["field_checks"] = None
    elif change == "no_quote_bridge": audit["assessments"][0]["quote_requires_unstated_bridge"] = None
    elif change == "fake_selected": audit["selected_fact_view_ids"] = [build_fact_views(rows)[0]["id"]]
    else: audit["assessments"][0]["PRIVATE"] = "PRIVATE"
    with pytest.raises(PremiseBankInvalid):
        build_quote_faithful_bank(Q, rows, audit, paragraphs(rows))


@pytest.mark.parametrize("mutation", ["quote", "coordinate", "raw_id"])
def test_literal_source_validation_is_independent_of_quote_length_consistency(mutation):
    rows, audit, _bank = bank_for()
    if mutation == "quote": rows[0]["quote"] = "假" * len(rows[0]["quote"])
    elif mutation == "coordinate": rows[0]["start_cp"] = 1; rows[0]["end_cp"] += 1
    else: rows[0]["raw_para_id"] = "other"
    with pytest.raises(PremiseBankInvalid):
        build_quote_faithful_bank(Q, rows, audit, paragraphs([source()]))


@pytest.mark.parametrize("change", ["all_bank", "missing_ledger", "duplicate_ledger", "missing_authorization", "changed_fact", "changed_question", "changed_bank"])
def test_material_freeze_publish_and_export_replay_reject_selection_bypasses(change):
    rows, audit, bank = bank_for([source("a"), source("b", text="场馆外墙是蓝色。")])
    ids = [row["id"] for row in bank["rows"]]
    ledger = purpose(bank, mutate=lambda wire: wire["decisions"]["C2"].update(purpose_binding_id="background"))
    result = result_for(rows, audit, bank, [ledger])
    q = deepcopy(Q)
    if change == "all_bank": result["evidence"] = deepcopy(bank["rows"])
    elif change == "missing_ledger": result["purpose_review"]["ledgers"] = []
    elif change == "duplicate_ledger": result["purpose_review"]["ledgers"] *= 2
    elif change == "missing_authorization": result.pop("purpose_review"); result.pop("quote_faithful_bank")
    elif change == "changed_fact": result["evidence"][0]["fact_des"] = "一个伪造的新结论"
    elif change == "changed_question": q["question"] += "资金情况如何？"
    else: result["quote_faithful_bank"]["rows"][0]["quote"] += "伪造"
    with pytest.raises(Invalid):
        validate_workflow_material_eligibility(result, q, paragraphs(rows))


def test_manual_primary_is_not_silently_rewritten_removed_or_covered_by_another_view():
    row = source(human=True)
    row["additional_interpretations"] = [{"fact_des": "场馆平时用于体育活动。", "argument": {key: None for key in KEYS}, "entities": ["场馆"]}]
    rows, audit, bank = bank_for([row])
    assert len({r["id"] for r in bank["rows"]}) == 2
    ids = [r["id"] for r in bank["rows"]]
    def reject_primary(wire):
        primary = f"C{ids.index('a') + 1}"
        wire["decisions"][primary]["purpose_binding_id"] = "background"
    result = result_for(rows, audit, bank, [purpose(bank, mutate=reject_primary)])
    assert result["purpose_review"]["manual_pending_ids"] == ["a"]
    with pytest.raises(PremiseBankInvalid, match="premise_manual_purpose_unverified"):
        validate_premise_projection(result, Q, paragraphs(rows))
    assert rows[0]["fact_des"] == row["fact_des"]


def test_material_cutoff_cannot_be_bypassed_by_approved_purpose():
    q = {**Q, "question": "仅依据截至2040年1月2日12:00（北京时间）已发布的材料，场馆用途是什么？"}
    row = source()
    ps = paragraphs([row]); ps[0]["raw_publish_time"] = "2040年1月3日12:00（北京时间）"
    audit = scope([row], q=q)
    bank = build_quote_faithful_bank(q, [row], audit, ps)
    ids = [r["id"] for r in bank["rows"]]
    payload, _, _ = build_fact_purpose_request(q, bank, ids)
    wire = {"bank_id": bank["bank_id"], "purpose_catalog_id": payload["purpose_catalog_id"], "decisions": {
        "C1": {"purpose_binding_id": "T0.necessary_premise", "introduces_unstated_source_fact": False,
               "equivalent_to": None, "reason": "直接记载该对象用途。"}}}
    ledger = resolve_fact_purpose(wire, q, bank, ids)
    formal, _purpose, material = compile_purpose_projection(q, [row], audit, bank, [ledger], ps)
    assert formal == [] and material["counts"]["excluded"] == 1


def test_program_bindings_reject_old_question_and_duplicate_json_keys():
    _rows, _audit, bank = bank_for()
    ids = [r["id"] for r in bank["rows"]]
    with pytest.raises(PremiseBankInvalid, match="premise_question_binding"):
        build_fact_purpose_request({**Q, "question": "另一个属性是否成立？"}, bank, ids)
    payload, _, _ = build_fact_purpose_request(Q, bank, ids)
    raw = json.dumps({"bank_id": bank["bank_id"], "purpose_catalog_id": payload["purpose_catalog_id"], "decisions": {}})
    raw = raw.replace('"decisions":', '"decisions": {}, "decisions":')
    with pytest.raises(PremiseBankInvalid, match="premise_purpose_duplicate_json_key"):
        resolve_fact_purpose(raw, Q, bank, ids)


def test_every_bank_row_is_planned_once_without_output_or_candidate_quotas():
    _rows, _audit, bank = bank_for([source(str(i)) for i in range(7)])
    batches = plan_fact_purpose_batches(Q, bank, max_rows=2, max_codepoints=40000)
    assert [fid for batch in batches for fid in batch] == [r["id"] for r in bank["rows"]]
    assert len(batches) == 4


def test_purpose_ledger_cannot_change_selected_flag_without_changed_independent_wire():
    _rows, _audit, bank = bank_for()
    ledger = purpose(bank)
    ledger["assessments"][0]["selected"] = False
    with pytest.raises(PremiseBankInvalid):
        validate_fact_purpose_ledger(ledger, Q, bank)
