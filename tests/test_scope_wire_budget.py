"""Real wire projection size; synthetic inputs and intercepted SDK only.

No .env, live model, API, runtime database, or sample answers are read.
"""
from copy import deepcopy
import json

import pytest

from app.domain.scope_batches import plan_scope_batches, scope_payload_codepoints, subset_evidence
from app.domain.strict_verification import build_fact_views, build_narrowing_request, build_scope_request, canonical
from app.domain.compiled_scope import build_compiled_scope_request
from test_deepseek_provider import intercepted_adapter
from test_scope_refine_workflow import Q, ScopeProvider, annotation, run


def anchor(quote="设备记录。" * 200, views=48, *, identity="anchor", start=17):
    return {"id": identity, "paragraph_id": "paragraph-" + identity, "raw_doc_id": "doc-" + identity,
            "raw_para_id": "para-" + identity, "start_cp": start, "end_cp": start + len(quote),
            "quote": quote, **annotation("原子解释0"),
            "additional_interpretations": [annotation(f"原子解释{index}") for index in range(1, views)]}


def assert_lossless_payload_batches(evidence, question, batches, limit, wanted=None):
    original_views = build_fact_views(evidence)
    wanted = set(wanted) if wanted is not None else {view["id"] for view in original_views}
    flat = [identity for batch in batches for identity in batch]
    assert len(flat) == len(set(flat)) == len(wanted)
    assert set(flat) == wanted
    before = {view["id"]: view for view in original_views}
    for batch in batches:
        projection = subset_evidence(evidence, batch)
        payload, _, actual = build_compiled_scope_request(question, projection)
        assert {view["id"] for view in actual} == set(batch)
        assert all(view == before[view["id"]] for view in actual)
        measured = len(json.dumps(payload, ensure_ascii=False))
        assert scope_payload_codepoints(question, projection) == measured
        assert measured <= limit or len(batch) == 1


def test_many_interpretations_repeat_complete_quote_in_actual_budget():
    evidence = [anchor()]
    original = deepcopy(evidence)
    old_anchor_size = len(canonical(evidence))
    actual_size = scope_payload_codepoints(Q, evidence)
    assert old_anchor_size < 40_000 < actual_size
    batches = plan_scope_batches(evidence, question=Q)
    assert len(batches) > 1
    assert_lossless_payload_batches(evidence, Q, batches, 40_000)
    for batch in batches:
        payload, _, _ = build_scope_request(Q, subset_evidence(evidence, batch))
        assert all(row["quote"] == evidence[0]["quote"] for row in payload["candidate_catalog"])
    assert evidence == original


def test_question_projection_counts_toward_target_and_changes_partition():
    evidence = [anchor(views=3)]
    ids = [view["id"] for view in build_fact_views(evidence)]
    one = scope_payload_codepoints(Q, subset_evidence(evidence, ids[:1]))
    two = scope_payload_codepoints(Q, subset_evidence(evidence, ids[:2]))
    longer = {**Q, "question": Q["question"] + "额" * ((two - one) // 2)}
    short_batches = plan_scope_batches(evidence, question=Q, max_codepoints=two)
    long_batches = plan_scope_batches(evidence, question=longer, max_codepoints=two)
    assert [len(batch) for batch in short_batches] == [2, 1]
    assert [len(batch) for batch in long_batches] == [1, 1, 1]
    assert_lossless_payload_batches(evidence, longer, long_batches, two)


def test_unprojected_question_fields_do_not_inflate_actual_payload_measurement():
    evidence = [anchor(views=2)]
    question = {**Q, "not_sent_to_provider": "ignored" * 10_000}
    assert scope_payload_codepoints(question, evidence) == scope_payload_codepoints(Q, evidence)


def test_oversized_single_view_stays_whole_with_unicode_geometry():
    evidence = [anchor(quote='甲🙂\n"已核验"\\原始证据。' * 1000, views=3, start=31)]
    original = deepcopy(evidence)
    batches = plan_scope_batches(evidence, question=Q, max_codepoints=100)
    assert [len(batch) for batch in batches] == [1, 1, 1]
    assert_lossless_payload_batches(evidence, Q, batches, 100)
    assert evidence == original
    for batch in batches:
        row = subset_evidence(evidence, batch)[0]
        assert row["start_cp"] == 31
        assert row["end_cp"] - row["start_cp"] == len(row["quote"])
        assert row["quote"] == evidence[0]["quote"]


def test_selected_subset_and_max_views_cover_each_requested_f_once():
    evidence = [anchor(quote="甲🙂", views=6, identity="one"), anchor(quote="乙\n", views=7, identity="two")]
    views = build_fact_views(evidence)
    wanted = [view["id"] for view in views if view["fact_des"] in {"原子解释0", "原子解释2", "原子解释4", "原子解释6"}]
    batches = plan_scope_batches(evidence, question=Q, max_views=3, view_ids=list(reversed(wanted)))
    assert all(len(batch) <= 3 for batch in batches)
    assert_lossless_payload_batches(evidence, Q, batches, 40_000, wanted)
    with pytest.raises(ValueError, match="scope_batch_view_identity"):
        plan_scope_batches(evidence, question=Q, view_ids=[wanted[0], wanted[0]])
    with pytest.raises(ValueError, match="scope_batch_view_identity"):
        plan_scope_batches(evidence, question=Q, view_ids=["unknown-view"])


def test_empty_pool_and_question_contract_are_explicit():
    assert plan_scope_batches([], question=Q) == []
    with pytest.raises(ValueError, match="scope_question_invalid"):
        plan_scope_batches([], question={"question": ""})
    with pytest.raises(TypeError):
        plan_scope_batches([])


@pytest.mark.parametrize("operation", ["review", "narrow"])
async def test_size_equals_real_sdk_user_json_with_schema_overhead_separate(operation):
    evidence = [anchor(quote='设备🙂说："通过"\n未公开\\完整报告。', views=2)]
    view_ids = [view["id"] for view in build_fact_views(evidence)]

    def response(body):
        user = json.loads(body["messages"][1]["content"])
        if operation == "narrow":
            return json.dumps({"candidate_catalog_id": user["candidate_catalog_id"],
                "corrections": {row["id"]: {"status": "no_supported_annotation", "annotations": []}
                                for row in user["candidate_catalog"]}}, ensure_ascii=False)
        return json.dumps({"candidate_catalog_id": user["candidate_catalog_id"], "target_catalog_id": user["target_catalog_id"],
            "decisions": {row["id"]: {"fact_entailment": "entailed", "target_binding_id": "outside_target",
                "quote_requires_unstated_bridge": False, "target_requires_unstated_bridge": False,
                "field_checks": {"fact_des": "entailed", "entities": "entailed" if row["entities"] else "not_present",
                    "argument": {key: "entailed" if value else "not_present" for key, value in row["argument"].items()}},
                "target_fact_present_but_annotation_incomplete": False, "reason": "Synthetic transport label only."}
                for row in user["candidate_catalog"]}}, ensure_ascii=False)

    async with intercepted_adapter(content=response) as (adapter, requests):
        if operation == "narrow":
            await adapter.narrow_evidence_annotations(Q, evidence, view_ids)
            expected, schema, _ = build_narrowing_request(Q, evidence, view_ids)
        else:
            await adapter.verify_evidence_scope(Q, evidence)
            expected, schema, _ = build_compiled_scope_request(Q, evidence)
    assert len(requests) == 1
    wire = json.loads(requests[0].content)
    human, system = wire["messages"][1]["content"], wire["messages"][0]["content"]
    assert human == json.dumps(expected, ensure_ascii=False)
    if operation == "review":
        assert len(human) == scope_payload_codepoints(Q, evidence)
    else:
        assert len(human) < scope_payload_codepoints(Q, evidence)
    instruction, schema_json = system.split("\nJSON Schema:\n", 1)
    assert schema_json == json.dumps(schema.model_json_schema(), ensure_ascii=False)
    assert len(instruction) > 0 and len(schema_json) > 0
    assert len(system) > len(schema_json)  # Both are outside the candidate payload soft target.


async def test_workflow_passes_bound_question_to_initial_narrow_and_reverify_plans(tmp_path, monkeypatch):
    import app.domain.scope_batches as module
    original = module.plan_scope_batches
    seen = []

    def record(evidence, **kwargs):
        assert kwargs.get("question") == Q
        seen.append({"question": deepcopy(kwargs["question"]), "views": kwargs.get("view_ids")})
        return original(evidence, **kwargs)

    monkeypatch.setattr(module, "plan_scope_batches", record)
    provider = ScopeProvider(mode="refocus")
    result = await run(tmp_path, provider)
    assert len(seen) == 3
    assert seen[0]["views"] is None
    assert seen[1]["views"] and seen[2]["views"]
    assert provider.calls.count("scope") == 2 and provider.calls.count("narrow") == 1
    assert result["strict_scope_review"]["verified_fact_view_count"] == 2
