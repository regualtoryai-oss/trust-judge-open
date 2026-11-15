"""Explicit synthetic providers test v19 contracts/lifecycle, never semantic P/R."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.agents.provider import LangChainOpenAIProvider, ModelNotConfigured, ModelOutputInvalid, get_requested_output_budget
from app.config import Settings
from app.domain import paragraph_ranked as p
from app.workflows.analysis import AnalysisCancelled, run_analysis

Q = {"id": "q-internal", "raw_question_id": "q-test", "question": "甲设备已获批准还是尚未批准？仅判断本次检测的状态。"}
TEXT = "甲公司称设备已获批准，但主管部门表示检测尚未结束，批准尚未确认。"


def sources(n=2, *, mirrors=False):
    return [{"id": f"p{i}", "document_id": f"d{i}", "raw_doc_id": f"doc{i}", "raw_para_id": f"doc{i}#p1",
             "text": TEXT if mirrors or i == 0 else "甲设备检测尚未结束。", "title": "甲设备检测消息",
             "source_credibility_json": "0.95", "raw_publish_time": "刚刚"} for i in range(n)]


def queries():
    return {"groups": [{"group": key, "terms": terms} for key, terms in
                       (("X", ["甲设备", "批准"]), ("Y", ["尚未批准", "检测"]), ("official", ["主管部门", "确认"]))]}


def rank(catalog, score=3):
    return [{"paragraph_ref": row["paragraph_ref"], "score": score, "critical_qualifier": score >= 2,
             "stance": "qualifier", "reason": "合成fixture保留尚未确认的必要限定，不评估真实模型能力。"} for row in catalog]


def annotations(refs):
    return [{"paragraph_ref": ref, "fact_des": "甲公司称已批准，但主管部门表示尚未确认。",
             "argument": {key: None for key in p.ARGUMENTS}, "entities": ["甲公司", "主管部门"]} for ref in refs]


def fields(rows, *, valid=True):
    return [{"paragraph_ref": row["paragraph_ref"], "annotation_sha256": p.digest(row),
             "fact_des_supported": valid, "field_checks": {key: True for key in p.ARGUMENTS},
             "entities_supported": True, "attribution_preserved": True, "reason": "仅合成字段fixture标签。"} for row in rows]


def compiled(paras=None, *, cfg=None, excluded=()):
    paras = paras or sources(); cfg = p.config_values(cfg)
    qs = p.query_catalog(Q, queries()); recall = p.recall_paragraphs(Q, paras, qs, config=cfg, excluded_spans=excluded)
    ranks = rank(recall["candidate_catalog"]); selected = p.select_ranked(recall, ranks, config=cfg)
    anns = annotations(selected)
    result = p.compile_ranked_result(Q, paras, queries=qs, recall=recall, ranks=ranks, selected_refs=selected,
                                     annotations=anns, field_reviews=fields(anns), config=cfg, excluded_spans=excluded)
    return result, paras


def test_settings_default_v17_and_explicit_exclusive_v19():
    default = Settings(_env_file=None)
    assert default.model_paragraph_ranked_enabled is False and "v17" in default.model_prompt_version
    assert default.paragraph_mirror_enabled is False and p.config_values()["mirror_enabled"] is False
    assert "v19" in Settings(_env_file=None, model_paragraph_ranked_enabled=True).model_prompt_version
    with pytest.raises(ValueError): Settings(_env_file=None, model_premise_bank_enabled=True, model_paragraph_ranked_enabled=True)


def test_three_queries_keep_complete_original_question_and_all_source_qualifiers():
    catalog = p.query_catalog(Q, queries())
    assert [g["group"] for g in catalog["groups"]] == ["X", "Y", "official"]
    assert all(g["full_original_question"] == Q["question"] and g["object_time_scope_context"] == Q["question"] and
               Q["question"] in g["search_text"] for g in catalog["groups"])
    with pytest.raises(ValueError): p.query_catalog(Q, {"groups": [queries()["groups"][0]] * 3})


def test_recall_union_is_bounded_and_rank_cap_one_record_per_source_mirrors_retained():
    result, paras = compiled(sources(70, mirrors=True))
    proof = result["paragraph_ranked_proof"]
    assert len(proof["recall"]["candidate_catalog"]) == 60 and len(proof["selected_refs"]) == 10
    assert len(result["evidence"]) == 10 and len({(r["raw_doc_id"], r["raw_para_id"]) for r in result["evidence"]}) == 10
    assert len({r["source_group_id"] for r in result["evidence"]}) == 1
    assert all(r["quote"] == TEXT and r["start_cp"] == 0 and r["end_cp"] == len(TEXT) for r in result["evidence"])
    assert all(r["grade_score"] == .95 and "metadata" in r["grade_basis"] for r in result["evidence"])
    assert proof["recall"]["mirror_candidate_index"]["independent_source_votes_added"] == 0
    assert not any("scope_verified" in r for r in result["evidence"])
    p.validate_ranked_result(result, Q, paras)


@pytest.mark.parametrize("attack", ["source", "question", "quote", "fact", "config", "rank", "fields", "marker", "oldledger"])
def test_proof_replay_rejects_projection_or_binding_downgrades(attack):
    result, paras = compiled()
    question = deepcopy(Q); expected = None
    if attack == "source": paras[0]["text"] += "后加更正。"
    elif attack == "question": question["question"] = "另一道问题？"
    elif attack == "quote": result["evidence"][0]["quote"] = "模型替代原文"
    elif attack == "fact": result["evidence"][0]["fact_des"] = "主管部门确认已批准"
    elif attack == "config": expected = {"selection_cap": 1}
    elif attack == "rank": result["paragraph_ranked_proof"]["rank_decisions"][0]["score"] = 0
    elif attack == "fields": result["paragraph_ranked_proof"]["field_reviews"][0]["fact_des_supported"] = False
    elif attack == "marker": result.pop("verification_protocol")
    else: result["strict_scope_review"] = {"complete": True}
    with pytest.raises(ValueError): p.validate_ranked_result(result, question, paras, expected_config=expected)


def test_invalid_critical_low_score_duplicate_unknown_rank_and_atomic_annotations_rejected():
    _, paras = compiled(); recall = p.recall_paragraphs(Q, paras, p.query_catalog(Q, queries()))
    ranks = rank(recall["candidate_catalog"])
    bad = deepcopy(ranks); bad[0]["score"] = 1
    with pytest.raises(ValueError): p.select_ranked(recall, bad)
    with pytest.raises(ValueError): p.select_ranked(recall, ranks + [ranks[0]])
    with pytest.raises(ValueError): p.select_ranked(recall, [{**r, "paragraph_ref": "UNKNOWN"} for r in ranks])
    with pytest.raises(ValueError): p.validate_annotations({"annotations": annotations(["P1", "P1"])}, ["P1"])
    with pytest.raises(ValueError): p.validate_fields({"decisions": fields(annotations(["P1"]))}, annotations(["P2"]))


def test_manual_exclusion_drops_whole_paragraph_without_cropping_and_binding_is_replayed():
    paras = sources(); span = (paras[0]["id"], 0, 3, paras[0]["text"][:3])
    result, paras = compiled(paras, excluded=[span])
    assert all(row["paragraph_id"] != "p0" for row in result["evidence"])
    p.validate_ranked_result(result, Q, paras, excluded_spans=[span])
    with pytest.raises(ValueError): p.validate_ranked_result(result, Q, paras, excluded_spans=[])


def test_publication_cutoff_precedes_recall_and_cap_late_unknown_cannot_crowd_qualified_source():
    q = {**Q, "question": "仅依据截至2038年5月9日18:00（北京时间）已发布的材料，甲设备是否已获批准？"}
    paras = sources(3)
    paras[0]["raw_publish_time"] = "2038-05-10T08:00:00+08:00"
    paras[1]["raw_publish_time"] = "2038-05-08T08:00:00+08:00"
    paras[2]["raw_publish_time"] = "刚刚"
    cfg = p.config_values({"selection_cap": 1})
    qs = p.query_catalog(q, queries()); recall = p.recall_paragraphs(q, paras, qs, config=cfg)
    assert recall["source_material_policy"]["counts"] == {"qualified": 1, "excluded": 1, "unknown": 1}
    assert [row["paragraph_id"] for row in recall["candidate_catalog"]] == ["p1"]
    ranks = rank(recall["candidate_catalog"]); refs = p.select_ranked(recall, ranks, config=cfg); anns = annotations(refs)
    result = p.compile_ranked_result(q, paras, queries=qs, recall=recall, ranks=ranks, selected_refs=refs,
        annotations=anns, field_reviews=fields(anns), config=cfg)
    assert len(result["evidence"]) == 1 and result["evidence"][0]["paragraph_id"] == "p1"


def test_canonical_question_transport_aliases_and_annotation_ids_do_not_cross_questions():
    result, paras = compiled()
    p.validate_ranked_result(result, {**Q, "topic_id": "incidental-transport"}, paras)
    proof = result["paragraph_ranked_proof"]
    other = {**Q, "id": "other-question"}
    qs = p.query_catalog(other, queries()); recall = p.recall_paragraphs(other, paras, qs)
    otherresult = p.compile_ranked_result(other, paras, queries=qs, recall=recall, ranks=proof["rank_decisions"], selected_refs=proof["selected_refs"],
        annotations=proof["annotations"], field_reviews=proof["field_reviews"])
    assert {r["id"] for r in result["evidence"]}.isdisjoint(r["id"] for r in otherresult["evidence"])


class RankedProvider:
    paragraph_ranked_enabled = True
    paragraph_ranked_config = p.config_values()
    prompt_version = p.PROMPT_VERSION
    premise_bank_enabled = False
    model_name = "synthetic-no-network"

    def __init__(self, *, reject_fields=False, bad_rank_once=False, zero_selected=False, truncate_once=False):
        self.calls = []; self.inputs = []; self.reject_fields = reject_fields
        self.bad_rank_once = bad_rank_once; self.zero_selected = zero_selected; self.truncate_once = truncate_once
        self.analysts_started = set(); self.both = asyncio.Event()

    def record(self, name):
        self.calls.append(name)
        LangChainOpenAIProvider._record_usage(self, name, {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})

    async def paragraph_queries(self, question, *, diagnostics=None):
        self.record("paragraph_queries"); return queries()

    async def paragraph_rank(self, question, catalog, *, diagnostics=None):
        self.record("paragraph_rank"); rows = rank(catalog, 0 if self.zero_selected else 3)
        if self.truncate_once and self.calls.count("paragraph_rank") == 1:
            raise ModelOutputInvalid("Synthetic length", stage="paragraph_rank", code="model_output_truncated", details={"finish_reason": "length"})
        if self.bad_rank_once and self.calls.count("paragraph_rank") == 1:
            rows.append(deepcopy(rows[0]))
        return {"decisions": rows}

    async def paragraph_annotate(self, question, catalog, *, diagnostics=None):
        self.record("paragraph_annotate")
        assert all(set(row) == {"paragraph_ref", "text"} for row in catalog)
        return {"annotations": annotations([r["paragraph_ref"] for r in catalog])}

    async def paragraph_fields(self, question, catalog, rows, *, diagnostics=None):
        self.record("paragraph_fields")
        assert all(set(row) == {"paragraph_ref", "text"} for row in catalog)
        return {"decisions": fields(rows, valid=not self.reject_fields)}

    async def analyze(self, role, question, evidence, snapshot_id):
        self.record(role); self.inputs.append((role, deepcopy(question), deepcopy(evidence), snapshot_id))
        self.analysts_started.add(role)
        if len(self.analysts_started) == 2: self.both.set()
        await asyncio.wait_for(self.both.wait(), 3)
        if evidence: evidence[0]["quote"] = "PRIVATE_MUTATION"
        return self.answer()

    async def arbitrate(self, question, evidence, analyses):
        self.record("arbitrate"); return {**self.answer(), "conflicts": []}

    @staticmethod
    def answer():
        return {"decision": "undetermined", "conclusion": "尚有待核实部分，保留归属与限定。", "judgments": [],
                "supporting_evidence_ids": [], "conflicting_evidence_ids": []}

    def capacity_policy(self, stage):
        return {"initial_tokens": 8, "max_tokens": 32, "retries": 2}


async def run(tmp_path, provider, **kwargs):
    return await run_analysis(Q, sources(), provider, checkpoint_path=tmp_path / "ranked.sqlite3",
        thread_id=kwargs.pop("thread_id", "v19-run"), cancel_check=kwargs.pop("cancel_check", lambda: False),
        emit=kwargs.pop("emit", lambda *_: None), **kwargs)


async def test_real_langgraph_v19_parallel_tail_proof_events_and_usage(tmp_path):
    provider = RankedProvider(); events = []; usage = []
    result = await run(tmp_path, provider, emit=lambda k, d: events.append((k, d)), usage_callback=lambda r: usage.append(r))
    assert provider.calls == ["paragraph_queries", "paragraph_rank", "paragraph_annotate", "paragraph_fields", "authority_analyst", "cross_validation_analyst", "arbitrate"]
    assert provider.inputs[0][1:] == provider.inputs[1][1:]
    assert result["verification_protocol"] == p.VERSION and "strict_scope_review" not in result
    assert all(row["quote"] != "PRIVATE_MUTATION" for row in result["evidence"])
    p.validate_ranked_result(result, Q, sources())
    assert {d["node"] for k, d in events if k == "node_completed"} >= {"paragraph_queries", "paragraph_recall", "paragraph_rank", "paragraph_annotate", "paragraph_fields", "freeze", "arbitrate"}
    assert all(not d.get("complete_source_scan") for k, d in events if k == "paragraph_recall_planned")
    assert len(result["usage"]) == 7 and len(usage[-1]) == 7 and sum(r["total_tokens"] for r in usage[-1]) == 84


async def test_negative_field_verification_fails_once_no_analysts_and_usage_survives(tmp_path):
    provider = RankedProvider(reject_fields=True); usage = []
    with pytest.raises(ModelOutputInvalid) as error:
        await run(tmp_path, provider, usage_callback=lambda value: usage.append(value))
    assert error.value.stage == "paragraph_fields" and provider.calls.count("paragraph_fields") == 1
    assert not provider.analysts_started and len(usage[-1]) == 4


async def test_mirror_additions_outside_base_are_actually_ranked_and_background_is_not_formal(tmp_path):
    source = sources(65, mirrors=True)
    class MirrorProvider(RankedProvider):
        def __init__(self):
            super().__init__(); self.ranked_ids = []; self.paragraph_ranked_config = p.config_values({"mirror_enabled": True})
            # Explicit fixture bookkeeping stays outside the model catalog.
            fixture_recall = p.recall_paragraphs(Q, source, p.query_catalog(Q, queries()), config=self.paragraph_ranked_config)
            self.fixture_ids = {r["paragraph_ref"]: r["paragraph_id"] for r in fixture_recall["candidate_catalog"]}
        async def paragraph_rank(self, question, catalog, *, diagnostics=None):
            self.record("paragraph_rank"); self.ranked_ids.extend(self.fixture_ids[row["paragraph_ref"]] for row in catalog)
            assert all("paragraph_id" not in row and "retrieval_hits" not in row for row in catalog)
            return {"decisions": [dict(row, score=(3 if self.fixture_ids[item["paragraph_ref"]] == "p60" else 0),
                critical_qualifier=self.fixture_ids[item["paragraph_ref"]] == "p60", stance="qualifier" if self.fixture_ids[item["paragraph_ref"]] == "p60" else "neutral")
                for row, item in zip(rank(catalog), catalog)]}
    provider = MirrorProvider(); events = []
    result = await run_analysis(Q, source, provider, checkpoint_path=tmp_path / "mirrors.sqlite3", thread_id="mirrors",
        cancel_check=lambda: False, emit=lambda kind, detail: events.append((kind, detail)))
    recall = result["paragraph_ranked_proof"]["recall"]
    assert (recall["base_candidate_count"], recall["mirror_candidate_count"], len(recall["candidate_catalog"])) == (60, 5, 65)
    assert provider.ranked_ids == [row["paragraph_id"] for row in recall["candidate_catalog"]]
    assert len(provider.ranked_ids) == len(set(provider.ranked_ids)) == 65
    assert recall["candidate_catalog"][60]["candidate_origin"] == "mirror_expansion"
    assert recall["candidate_catalog"][60]["mirror_origins"][0]["match"] == "exact"
    assert [row["paragraph_id"] for row in result["evidence"]] == ["p60"]
    planned = next(detail for kind, detail in events if kind == "paragraph_recall_planned")
    assert planned["base_candidate_count"] == 60 and planned["mirror_candidate_count"] == 5 and planned["mirror_enabled"] is True
    p.validate_ranked_result(result, Q, source, expected_config=provider.paragraph_ranked_config)


def test_mirror_budget_additions_keep_source_votes_conservative_and_disabled_base_identical():
    source = sources(65, mirrors=True); qs = p.query_catalog(Q, queries())
    base = p.recall_paragraphs(Q, source, qs)
    expanded = p.recall_paragraphs(Q, source, qs, config={"mirror_enabled": True, "mirror_candidate_limit": 2})
    assert expanded["candidate_catalog"][:60] == base["candidate_catalog"]
    assert expanded["mirror_expansion"]["budget_truncated"] and expanded["mirror_candidate_count"] == 2
    cfg = p.config_values({"mirror_enabled": True, "selection_cap": 30})
    recall = p.recall_paragraphs(Q, source, qs, config=cfg)
    ranks = rank(recall["candidate_catalog"])
    # Request at least one appended mirror by giving base background 0.
    for row in ranks:
        if int(row["paragraph_ref"][1:]) <= 59: row.update(score=0, critical_qualifier=False)
    selected = p.select_ranked(recall, ranks, config=cfg); anns = annotations(selected)
    result = p.compile_ranked_result(Q, source, queries=qs, recall=recall, ranks=ranks, selected_refs=selected,
        annotations=anns, field_reviews=fields(anns), config=cfg)
    assert len(result["evidence"]) == 6 and len({row["source_group_id"] for row in result["evidence"]}) == 1


def test_near_mirror_retains_two_source_records_without_adding_an_independent_confidence_vote():
    source = sources(41, mirrors=True)
    source[-1]["text"] = TEXT + "。"
    # Freeze a base-bound helper result through the actual recall/compiler;
    # the near source can fall on either side of BM25's deterministic cutoff.
    cfg = p.config_values({"candidate_limit": 40, "mirror_enabled": True})
    qs = p.query_catalog(Q, queries()); recall = p.recall_paragraphs(Q, source, qs, config=cfg)
    assert len(recall["candidate_catalog"]) == 41 and recall["mirror_candidate_count"] == 1
    ranks = rank(recall["candidate_catalog"])
    keep = {"p0", "p40"}
    for decision, catalog in zip(ranks, recall["candidate_catalog"]):
        if catalog["paragraph_id"] not in keep: decision.update(score=0, critical_qualifier=False)
    refs = p.select_ranked(recall, ranks, config=cfg); anns = annotations(refs)
    result = p.compile_ranked_result(Q, source, queries=qs, recall=recall, ranks=ranks, selected_refs=refs,
        annotations=anns, field_reviews=fields(anns), config=cfg)
    assert {row["paragraph_id"] for row in result["evidence"]} == keep
    assert len({row["source_group_id"] for row in result["evidence"]}) == 1
    assert all("not_proven_independence" in row["source_group_basis"] for row in result["evidence"])
    assert len({(row["raw_doc_id"], row["raw_para_id"]) for row in result["evidence"]}) == 2


async def test_rank_structure_repair_once_and_length_capacity_do_not_restart_prior_nodes(tmp_path):
    provider = RankedProvider(bad_rank_once=True)
    result = await run(tmp_path, provider)
    assert provider.calls.count("paragraph_queries") == 1 and provider.calls.count("paragraph_rank") == 2
    assert len(result["usage"]) == 8
    provider = RankedProvider(truncate_once=True); events = []
    await run(tmp_path, provider, thread_id="capacity", emit=lambda k, d: events.append((k, d)))
    assert provider.calls.count("paragraph_queries") == 1 and provider.calls.count("paragraph_rank") == 2
    assert any(k == "model_capacity_retry_completed" and d["node"] == "paragraph_rank" for k, d in events)


async def test_zero_selection_is_unresolved_and_does_not_invent_annotation_model_calls(tmp_path):
    provider = RankedProvider(zero_selected=True); events = []
    result = await run(tmp_path, provider, emit=lambda k, d: events.append((k, d)))
    assert result["evidence"] == [] and result["final"]["decision"] == "undetermined"
    assert "paragraph_annotate" not in provider.calls and "paragraph_fields" not in provider.calls
    assert not any(k == "tool_started" and d["node"] in {"paragraph_annotate", "paragraph_fields"} for k, d in events)


async def test_pause_after_ranking_resume_reuses_queries_and_rank_without_new_input(tmp_path):
    provider = RankedProvider(); cancelled = [False]
    def emit(kind, detail):
        if kind == "node_completed" and detail["node"] == "paragraph_rank": cancelled[0] = True
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, emit=emit, cancel_check=lambda: cancelled[0])
    assert provider.calls == ["paragraph_queries", "paragraph_rank"]
    cancelled[0] = False; events = []
    result = await run(tmp_path, provider, thread_id="resumed", resume_from_thread_id="v19-run", emit=lambda k, d: events.append((k, d)))
    assert provider.calls.count("paragraph_queries") == provider.calls.count("paragraph_rank") == 1
    reused = next(d for k, d in events if k == "checkpoint_reused")["reused_nodes"]
    assert {"paragraph_queries", "paragraph_recall", "paragraph_rank"} <= set(reused)
    assert not {"retrieve", "extract", "verify", "scope_review"} & set(reused)
    p.validate_ranked_result(result, Q, sources())


async def test_partial_paid_rank_batch_resume_reuses_only_exact_completed_prefix(tmp_path):
    provider = RankedProvider(); provider.paragraph_ranked_config = p.config_values({"rank_batch_size": 1})
    cancelled = [False]
    def emit(kind, detail):
        if kind == "paragraph_rank_progress" and detail["completed_batches"] == 1: cancelled[0] = True
    with pytest.raises(AnalysisCancelled):
        await run(tmp_path, provider, emit=emit, cancel_check=lambda: cancelled[0])
    assert provider.calls == ["paragraph_queries", "paragraph_rank"]
    cancelled[0] = False; events = []
    result = await run(tmp_path, provider, thread_id="partial-resumed", resume_from_thread_id="v19-run",
                       emit=lambda k, d: events.append((k, d)))
    assert provider.calls.count("paragraph_queries") == 1 and provider.calls.count("paragraph_rank") == 2
    reused = next(d for k, d in events if k == "checkpoint_reused")["reused_nodes"]
    assert "paragraph_recall" in reused and "paragraph_rank" not in reused
    p.validate_ranked_result(result, Q, sources(), expected_config=provider.paragraph_ranked_config)


@pytest.mark.parametrize("damage", ["query", "recall", "selected", "annotations", "proof", "null_proof", "partial_prefix", "field_channel", "pending_without_proof"])
async def test_checkpoint_downstream_requires_complete_bound_prerequisites_before_fork(damage):
    from types import SimpleNamespace
    from app.workflows.checkpoint_resume import fork_checkpoint
    result, paras = compiled(); proof = result["paragraph_ranked_proof"]; cfg = p.config_values()
    initial = {"question": Q, "paragraphs": paras, "manual_candidates": [], "interpretation_overrides": {},
               "paragraph_ranked_enabled": True, "paragraph_ranked_config": cfg, "paragraph_ranked_excluded_spans": []}
    refs = [row["paragraph_ref"] for row in proof["recall"]["candidate_catalog"]]
    channels = {**deepcopy(initial), **deepcopy(result), "workflow_input_sha256": "explicit-synthetic-binding",
        "ranked_query_catalog": deepcopy(proof["query_catalog"]), "ranked_recall": deepcopy(proof["recall"]),
        "ranked_rank_batches": [refs], "ranked_rank_index": 1, "ranked_ranks": deepcopy(proof["rank_decisions"]),
        "ranked_selected_refs": deepcopy(proof["selected_refs"]), "ranked_annotations": deepcopy(proof["annotations"]),
        "ranked_field_reviews": deepcopy(proof["field_reviews"]), "evidence_snapshot_id": "synthetic-frozen",
        "authority_analysis": {"evidence_snapshot_id": "synthetic-frozen"}}
    pending = []
    keys = {"query": "ranked_query_catalog", "recall": "ranked_recall", "selected": "ranked_selected_refs",
            "annotations": "ranked_annotations", "proof": "paragraph_ranked_proof"}
    if damage in keys: channels.pop(keys[damage])
    elif damage == "null_proof": channels["paragraph_ranked_proof"] = None
    elif damage == "partial_prefix": channels["ranked_rank_index"] = 0
    elif damage == "field_channel": channels["ranked_field_reviews"][0]["attribution_preserved"] = False
    else:
        channels.pop("paragraph_ranked_proof")
        pending = [("paid-sibling", "authority_analysis", channels.pop("authority_analysis"))]
    class MemorySaver:
        async def aget_tuple(self, config):
            if config["configurable"]["thread_id"] == "target": return None
            return SimpleNamespace(checkpoint={"channel_values": channels, "channel_versions": {}}, pending_writes=pending,
                                   config={"configurable": {"checkpoint_id": "synthetic"}}, metadata={})
        async def aput(self, *_args): pytest.fail("Malformed downstream checkpoint must fail before any target write")
        async def aput_writes(self, *_args): pytest.fail("Malformed downstream checkpoint must not copy paid writes")
    with pytest.raises(ModelOutputInvalid) as error:
        await fork_checkpoint(MemorySaver(), source_thread_id="source", target_thread_id="target", initial_state=initial,
                              input_sha256="explicit-synthetic-binding", snapshot_id=lambda _rows: "target-snapshot")
    assert error.value.stage == "resume" and error.value.code == "resume_checkpoint_invalid"


async def test_strategy_or_manual_overlay_mismatch_is_rejected_before_paid_calls(tmp_path):
    provider = RankedProvider(); provider.prompt_version = "v17"
    with pytest.raises(ModelNotConfigured): await run(tmp_path, provider)
    assert provider.calls == []
    provider = RankedProvider()
    with pytest.raises(ModelNotConfigured): await run(tmp_path, provider, interpretation_overrides={"x": {"fact_des": "人工改写"}})
    assert provider.calls == []


async def test_rank_repair_exhaustion_retains_exact_stage_and_never_runs_tail(tmp_path):
    class AlwaysBad(RankedProvider):
        async def paragraph_rank(self, question, catalog, *, diagnostics=None):
            self.record("paragraph_rank")
            rows = rank(catalog); return {"decisions": [*rows, rows[0]]}
    provider = AlwaysBad(); usage = []
    with pytest.raises(ModelOutputInvalid) as error:
        await run(tmp_path, provider, usage_callback=lambda rows: usage.append(rows))
    assert error.value.stage == "paragraph_rank" and error.value.details["repair_exhausted"] is True
    assert provider.calls == ["paragraph_queries", "paragraph_rank", "paragraph_rank"] and len(usage[-1]) == 3


async def test_real_provider_v19_contracts_metadata_and_duplicate_json_guard_without_network():
    from langchain_core.messages import AIMessage
    class LocalModel:
        def __init__(self, content): self.content = content; self.messages = []
        def bind(self, **_options): return self
        async def ainvoke(self, messages):
            self.messages = messages
            return AIMessage(content=self.content, response_metadata={"finish_reason": "stop"},
                             usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})
    settings = Settings(_env_file=None, model_name="synthetic-v19-wire", model_api_key="explicit-test-only",
                        model_paragraph_ranked_enabled=True)
    provider = LangChainOpenAIProvider(settings)
    local = LocalModel(json.dumps(queries(), ensure_ascii=False)); provider._model = provider._extract_model = local
    value = await provider.paragraph_queries(Q)
    assert value == queries() and provider.paragraph_ranked_enabled is True and provider.prompt_version == p.PROMPT_VERSION
    assert json.loads(local.messages[1].content)["question"] == Q
    local.content = '{"decisions":[],"decisions":[]}'
    with pytest.raises(ModelOutputInvalid) as error:
        await provider.paragraph_rank(Q, [{"paragraph_ref": "P1", "text": TEXT, "raw_publish_time": "昨天"}])
    assert error.value.details["check"] == "paragraph_ranked_duplicate_json_key"
    wire = json.loads(local.messages[1].content)
    assert wire["catalog"] == [{"paragraph_ref": "P1", "text": TEXT}]
