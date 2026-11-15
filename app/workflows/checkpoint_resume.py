"""Fork a durable LangGraph checkpoint without mutating its original audit trail."""

import copy
from collections import defaultdict

from app.agents.provider import ModelOutputInvalid


_REUSE_CHANNELS = {
    "candidate_import_provenance": "candidate_import", "strict_scope_review": "scope_review",
    "retrieved": "retrieve", "candidates": "extract", "extraction_reviews": "extract",
    "evidence": "verify",
    "quote_faithful_bank": "scope_review", "purpose_review": "purpose_review",
    "evidence_snapshot_id": "freeze", "authority_analysis": "authority_analyst",
    "cross_analysis": "cross_validation_analyst", "baseline_analysis": "baseline", "final": "arbitrate",
    "strict_selection": "strict_selection", "strict_artifact": "strict_annotation",
    "source_bundle_response": "source_bundle_selection",
}
_DISCARDED_WRITES = {"__error__", "__error_source_node__", "__interrupt__", "__resume__"}


async def fork_checkpoint(saver, *, source_thread_id, target_thread_id, initial_state, input_sha256, snapshot_id):
    """Copy successful parallel writes as well as completed supersteps.

    Keeping checkpoint/task IDs stable inside the new thread lets LangGraph
    recognize a completed parallel sibling. A new input must never be supplied
    to ainvoke afterward: that would restart START and repeat paid requests.
    """
    def refuse(check):
        raise ModelOutputInvalid(
            "续接检查点与当前任务输入不一致，未调用模型。", stage="resume",
            code="resume_checkpoint_invalid", details={"check": check},
        )

    if not source_thread_id or source_thread_id == target_thread_id:
        refuse("thread_identity")
    target = {"configurable": {"thread_id": target_thread_id, "checkpoint_ns": ""}}
    if await saver.aget_tuple(target) is not None:
        refuse("target_thread_exists")
    stored = await saver.aget_tuple({"configurable": {"thread_id": source_thread_id}})
    if stored is None:
        refuse("source_checkpoint_missing")
    checkpoint = copy.deepcopy(stored.checkpoint)
    channels = checkpoint.get("channel_values", {})
    # A pause can land after LangGraph saves its initial input but before
    # START projects it into state channels. Resuming that checkpoint is still
    # ainvoke(None): no workflow node has executed and no paid work is lost.
    source_input = channels.get("__start__", {}) if "question" not in channels else channels
    if not isinstance(source_input, dict):
        refuse("source_input_missing")
    for key in ("question", "paragraphs", "manual_candidates", "interpretation_overrides"):
        if source_input.get(key) != initial_state[key]:
            refuse("source_input_mismatch")
    if initial_state.get("source_bundle_enabled") is True:
        for key in ("source_bundle_enabled", "source_bundle_protocol", "source_documents", "source_bundle_excluded_spans"):
            if source_input.get(key) != initial_state.get(key):
                refuse("source_bundle_input_mismatch")
    previous_hash = source_input.get("workflow_input_sha256")
    if previous_hash != input_sha256:
        # Old completed publication checkpoints predate input fingerprints.
        # They are intentionally recovered by the explicit recovery pathway;
        # an incomplete legacy graph cannot prove all runtime options identical.
        refuse("source_input_fingerprint")

    writes = defaultdict(list)
    for task_id, channel, value in stored.pending_writes or []:
        if channel not in _DISCARDED_WRITES:
            writes[task_id].append((channel, copy.deepcopy(value)))
    reused = set()
    for channel, node in _REUSE_CHANNELS.items():
        if channel in channels or any(key == channel for items in writes.values() for key, _ in items):
            reused.add(node)
    # A durable batch is paid work worth reusing, but it is not the completed
    # extraction stage. Likewise exact verification precedes the independently
    # checkpointed relevance review in complete-source-scan workflows. A recall
    # recovery plan is also partial scope work; only its final ledger or bank
    # proves completion of the scope stage.
    def restored_value(name):
        value = channels.get(name)
        for items in writes.values():
            for channel, pending in items:
                if channel == name:
                    value = pending
        return value

    coverage = restored_value("source_coverage")
    if isinstance(coverage, dict):
        if not coverage.get("complete"):
            reused.discard("extract")
        if restored_value("evidence_selection") is None:
            reused.discard("verify")
    if initial_state.get("candidate_artifact") is not None:
        # Imported candidates populate evidence/source coverage without running
        # the initial graph's retrieve/extract/verify nodes. Their stored rows
        # are reusable data, not proof that those stages executed in this job.
        reused.difference_update({"retrieve", "extract", "verify"})
    if initial_state.get("paragraph_ranked_enabled") is True:
        from app.domain import paragraph_ranked as ranked
        reused.difference_update({"retrieve", "extract", "verify", "scope_review", "purpose_review"})
        try:
            def present(name):
                return name in channels or any(name == key for items in writes.values() for key, _ in items)

            downstream = ("paragraph_ranked_proof", "verification_protocol", "evidence", "material_eligibility",
                "candidate_evidence", "evidence_selection", "source_coverage", "ranked_field_reviews",
                "evidence_snapshot_id", "authority_analysis", "cross_analysis", "baseline_analysis", "final")
            rank_state = ("ranked_rank_batches", "ranked_rank_index", "ranked_ranks")
            needs_fields = any(present(name) for name in downstream)
            needs_annotations = needs_fields or present("ranked_annotations")
            needs_selection = needs_annotations or present("ranked_selected_refs")
            needs_recall = needs_selection or present("ranked_recall") or any(present(name) for name in rank_state)
            if ((needs_recall and not present("ranked_query_catalog"))
                    or (needs_recall and not present("ranked_recall"))
                    or (needs_selection and not present("ranked_selected_refs"))
                    or (needs_annotations and not present("ranked_annotations"))
                    or (needs_fields and not isinstance(restored_value("paragraph_ranked_proof"), dict))):
                refuse("ranked_checkpoint_missing_prerequisite")
            query = restored_value("ranked_query_catalog")
            if present("ranked_query_catalog"):
                query_output = {"groups": [{k: g[k] for k in ("group", "terms")} for g in query["groups"]]}
                if query != ranked.query_catalog(initial_state["question"], query_output):
                    refuse("ranked_query_binding")
                reused.add("paragraph_queries")
            recall = restored_value("ranked_recall")
            if needs_recall:
                expected = ranked.recall_paragraphs(initial_state["question"], initial_state["paragraphs"], query,
                    config=initial_state["paragraph_ranked_config"], excluded_spans=initial_state.get("paragraph_ranked_excluded_spans", []))
                if recall != expected:
                    refuse("ranked_recall_binding")
                refs = [r["paragraph_ref"] for r in recall["candidate_catalog"]]
                size = initial_state["paragraph_ranked_config"]["rank_batch_size"]
                batches = [refs[i:i + size] for i in range(0, len(refs), size)]
                index = restored_value("ranked_rank_index")
                if (not all(present(name) for name in rank_state) or restored_value("ranked_rank_batches") != batches
                        or type(index) is not int or not 0 <= index <= len(batches)):
                    refuse("ranked_partial_batch_binding")
                # A paid partial batch is reusable, but only for its exact
                # already completed source prefix. It is not a complete rank stage.
                completed_refs = [ref for batch in batches[:index] for ref in batch]
                ranked.validate_rank_batch({"decisions": restored_value("ranked_ranks")}, completed_refs)
                reused.add("paragraph_recall")
            selected = restored_value("ranked_selected_refs")
            if needs_selection:
                if (index != len(batches) or selected != ranked.select_ranked(recall, restored_value("ranked_ranks"),
                                                                          config=initial_state["paragraph_ranked_config"])):
                    refuse("ranked_selection_binding")
                reused.add("paragraph_rank")
            annotations = restored_value("ranked_annotations")
            if needs_annotations:
                ranked.validate_annotations({"annotations": annotations}, selected)
                reused.add("paragraph_annotate")
            proof = restored_value("paragraph_ranked_proof")
            if needs_fields:
                ranked.validate_ranked_result({"verification_protocol": ranked.VERSION, "paragraph_ranked_proof": proof,
                    "evidence": restored_value("evidence"), "material_eligibility": restored_value("material_eligibility")},
                    initial_state["question"], initial_state["paragraphs"], expected_config=initial_state["paragraph_ranked_config"],
                    excluded_spans=initial_state.get("paragraph_ranked_excluded_spans", []))
                if (restored_value("verification_protocol") != ranked.VERSION
                        or proof["query_catalog"] != query or proof["recall"] != recall
                        or proof["rank_decisions"] != restored_value("ranked_ranks")
                        or proof["selected_refs"] != selected or proof["annotations"] != annotations
                        or proof["field_reviews"] != restored_value("ranked_field_reviews")):
                    refuse("ranked_checkpoint_stage_binding")
                reused.add("paragraph_fields")
        except (KeyError, TypeError, ValueError):
            refuse("ranked_checkpoint_projection")

    if initial_state.get("source_bundle_enabled") is True:
        from app.workflows.bundle_workbench import validate_result
        reused.difference_update({"retrieve", "extract", "verify", "scope_review", "purpose_review"})
        try:
            def present_bundle(name):
                return name in channels or any(name == key for items in writes.values() for key, _ in items)
            fields = ("evidence", "candidate_evidence", "material_eligibility", "source_bundle_protocol",
                "verification_protocol", "source_bundle_proof", "evidence_selection", "source_coverage")
            downstream = ("source_bundle_response", "source_bundle_proof", "evidence", "evidence_snapshot_id",
                "authority_analysis", "cross_analysis", "final")
            if any(present_bundle(name) for name in downstream):
                if not all(present_bundle(name) for name in fields) or not present_bundle("source_bundle_response"):
                    refuse("source_bundle_checkpoint_missing_prerequisite")
                projection = {key: restored_value(key) for key in fields}
                validate_result(projection, initial_state["question"], initial_state["paragraphs"], initial_state["source_documents"],
                    excluded_spans=initial_state.get("source_bundle_excluded_spans", []))
                if projection["source_bundle_proof"]["response"] != restored_value("source_bundle_response"):
                    refuse("source_bundle_checkpoint_stage_binding")
                reused.add("source_bundle_selection")
        except (KeyError, TypeError, ValueError, OSError):
            refuse("source_bundle_checkpoint_projection")

    if initial_state.get("strict_evidence_enabled") is True:
        from app.workflows.strict_workbench import prepare_case, validate_result
        from app.agents.strict_evidence import validate_selection
        reused.difference_update({"retrieve", "extract", "verify", "scope_review", "purpose_review"})
        try:
            def present_strict(name):
                return name in channels or any(name == key for items in writes.values() for key, _ in items)
            needs_case = any(present_strict(name) for name in ("strict_selection", "strict_artifact", "strict_evidence_proof"))
            expected_case = prepare_case(initial_state["question"], initial_state["paragraphs"],
                candidate_limit=initial_state["strict_candidate_limit"],
                excluded_spans=initial_state.get("strict_excluded_spans", []))
            if needs_case and restored_value("strict_case") != expected_case:
                refuse("strict_checkpoint_source_binding")
            if present_strict("strict_selection"):
                validate_selection(expected_case, restored_value("strict_selection"))
                reused.add("strict_selection")
            if present_strict("strict_artifact") or present_strict("strict_evidence_proof"):
                projection = {key: restored_value(key) for key in ("evidence", "candidate_evidence", "material_eligibility",
                    "verification_protocol", "strict_evidence_proof", "evidence_selection", "source_coverage")}
                validate_result(projection, initial_state["question"], initial_state["paragraphs"],
                    candidate_limit=initial_state["strict_candidate_limit"],
                    excluded_spans=initial_state.get("strict_excluded_spans", []))
                if (projection["strict_evidence_proof"]["selection"] != restored_value("strict_selection")
                        or projection["strict_evidence_proof"]["annotation"] != restored_value("strict_artifact")):
                    refuse("strict_checkpoint_stage_binding")
                reused.add("strict_annotation")
        except (KeyError, TypeError, ValueError):
            refuse("strict_checkpoint_projection")

    purpose_coverage = None
    purpose = restored_value("purpose_review")
    if "purpose_review" in reused and isinstance(purpose, dict) and purpose.get("complete") is True:
        from app.domain.premise_bank import validate_premise_projection, PremiseBankInvalid
        projection = {key: restored_value(key) for key in (
            "candidate_evidence", "strict_scope_review", "quote_faithful_bank",
            "purpose_review", "material_eligibility", "evidence",
        )}
        try:
            # Display only completed counts that replay against the same bound
            # source, question and formal projection. Partial or unprovable
            # checkpoints keep this optional metadata absent.
            validate_premise_projection(projection, initial_state["question"], initial_state["paragraphs"])
        except PremiseBankInvalid:
            pass
        else:
            bank = projection["quote_faithful_bank"]
            purpose_coverage = {"bank_id": bank["bank_id"], "bank_fact_count": len(bank["rows"]),
                "formal_fact_count": len(purpose["formal_evidence_ids"]), "complete": True, "reused": True}
    original_snapshot_id = channels.get("evidence_snapshot_id")
    if original_snapshot_id:
        # Evidence snapshot rows are execution-scoped. Preserve content while
        # assigning a fresh ID to this audited continuation and both analysts.
        channels["evidence_snapshot_id"] = snapshot_id(channels.get("evidence", []))
        for channel in ("authority_analysis", "cross_analysis", "baseline_analysis"):
            if channel in channels:
                channels[channel]["evidence_snapshot_id"] = channels["evidence_snapshot_id"]
        for task_writes in writes.values():
            for channel, value in task_writes:
                if channel in {"authority_analysis", "cross_analysis", "baseline_analysis"}:
                    value["evidence_snapshot_id"] = channels["evidence_snapshot_id"]

    origin = {
        "kind": "checkpoint_continuation", "source_thread_id": source_thread_id,
        "checkpoint_id": stored.config["configurable"]["checkpoint_id"],
        "reused_nodes": sorted(reused), "source_evidence_snapshot_id": original_snapshot_id,
        **({"reused_source_batches": coverage.get("completed_batches", 0)} if isinstance(coverage, dict) else {}),
        **({"reused_purpose_coverage": purpose_coverage} if purpose_coverage is not None else {}),
    }
    saved = await saver.aput(target, checkpoint, {**stored.metadata, "continuation": origin}, checkpoint["channel_versions"])
    for task_id, task_writes in writes.items():
        if task_writes:
            await saver.aput_writes(saved, task_writes, task_id)
    return origin
