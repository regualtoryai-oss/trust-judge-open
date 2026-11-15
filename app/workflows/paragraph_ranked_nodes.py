"""v19 front-half LangGraph nodes; shared tail/capacity/usage remain in analysis."""
from copy import deepcopy
from pydantic import ValidationError

from app.agents.provider import ModelOutputInvalid
from app.domain import paragraph_ranked as protocol


def build_nodes(*, question, paragraphs, provider, config, excluded, begin, complete, emit,
                request_with_capacity, repair_once, instrument):
    byid = {p["id"]: p for p in paragraphs}

    def invalid(stage, check):
        return ModelOutputInvalid("段落策略输出未通过来源绑定、完整性或字段验真，未采纳结果。", stage=stage,
            code="validation_failed", details={"check": check, "question_id": question["id"]})

    async def model(stage, request, validate):
        def checked(value):
            try:
                return validate(value)
            except (ValidationError, TypeError, ValueError):
                raise invalid(stage, "paragraph_contract") from None
        try:
            raw = await request_with_capacity(stage, lambda: request(None))
            return checked(raw)
        except ModelOutputInvalid as error:
            if error.code not in {"validation_failed", "model_output_invalid"}:
                raise
            return await repair_once(stage, error, lambda: request({"check": "paragraph_contract"}), checked)

    async def queries(state):
        begin("paragraph_queries")
        async def call(_state):
            return {"ranked_query_catalog": await model("paragraph_queries",
                lambda diagnostics: provider.paragraph_queries(deepcopy(question), diagnostics=diagnostics),
                lambda value: protocol.query_catalog(question, value))}
        value = await instrument("paragraph_queries", "LangChain", "model", call)(state)
        complete("paragraph_queries", {"query_group_count": 3})
        return value

    async def recall(state):
        begin("paragraph_recall")
        value = protocol.recall_paragraphs(question, paragraphs, state["ranked_query_catalog"], config=config, excluded_spans=excluded)
        refs = [r["paragraph_ref"] for r in value["candidate_catalog"]]
        size = config["rank_batch_size"]
        batches = [refs[i:i + size] for i in range(0, len(refs), size)]
        emit("paragraph_recall_planned", {"question_id": question["id"], "source_paragraph_count": len(paragraphs),
            "candidate_count": len(refs), "batch_count": len(batches), "candidate_limit": config["candidate_limit"],
            "base_candidate_count": value["base_candidate_count"], "mirror_candidate_count": value["mirror_candidate_count"],
            "mirror_enabled": config["mirror_enabled"], "mirror_candidate_limit": config["mirror_candidate_limit"],
            "complete_source_scan": False})
        complete("paragraph_recall", {"candidate_count": len(refs)})
        return {"ranked_recall": value, "ranked_rank_batches": batches, "ranked_rank_index": 0, "ranked_ranks": [],
                "retrieved": [byid[r["paragraph_id"]] for r in value["candidate_catalog"]]}

    async def rank(state):
        begin("paragraph_rank")
        index, batches = state["ranked_rank_index"], state["ranked_rank_batches"]
        ranks = list(state["ranked_ranks"])
        if index < len(batches):
            refs = batches[index]
            catalog = protocol.rank_batch_catalog(state["ranked_recall"], byid, refs)
            async def call(_state):
                return {"ranked_batch_decisions": await model("paragraph_rank",
                    lambda diagnostics: provider.paragraph_rank(deepcopy(question), deepcopy(catalog), diagnostics=diagnostics),
                    lambda value: protocol.validate_rank_batch(value, refs))}
            value = await instrument("paragraph_rank", "LangChain", "model", call)(state)
            ranks.extend(value["ranked_batch_decisions"]); index += 1
            emit("paragraph_rank_progress", {"question_id": question["id"], "completed_batches": index,
                "batch_count": len(batches), "ranked_paragraph_count": len(ranks),
                "candidate_count": len(state["ranked_recall"]["candidate_catalog"])})
        output = {"ranked_ranks": ranks, "ranked_rank_index": index}
        if index == len(batches):
            try:
                output["ranked_selected_refs"] = protocol.select_ranked(state["ranked_recall"], ranks, config=config)
            except (TypeError, ValueError):
                raise invalid("paragraph_rank", "rank_selection_or_critical_qualifier") from None
            complete("paragraph_rank", {"candidate_count": len(ranks), "selected_paragraph_count": len(output["ranked_selected_refs"]),
                                        "selection_cap": config["selection_cap"], "minimum_score": config["minimum_score"]})
        return output

    async def annotate(state):
        begin("paragraph_annotate")
        refs = state["ranked_selected_refs"]
        annotations = []
        if refs:
            catalog = protocol.rank_batch_catalog(state["ranked_recall"], byid, refs, stage="annotate")
            async def call(_state):
                return {"ranked_annotations": await model("paragraph_annotate",
                    lambda diagnostics: provider.paragraph_annotate(deepcopy(question), deepcopy(catalog), diagnostics=diagnostics),
                    lambda value: protocol.validate_annotations(value, refs))}
            value = await instrument("paragraph_annotate", "LangChain", "model", call)(state)
            annotations = value["ranked_annotations"]
        complete("paragraph_annotate", {"selected_paragraph_count": len(refs), "annotation_count": len(annotations)})
        return {"ranked_annotations": annotations}

    async def fields(state):
        begin("paragraph_fields")
        refs, annotations = state["ranked_selected_refs"], state["ranked_annotations"]
        reviewed = []
        if refs:
            catalog = protocol.rank_batch_catalog(state["ranked_recall"], byid, refs, stage="fields")
            # One independent semantic field review. A negative determination
            # is never "repaired" by asking it to change false to true.
            async def call(_state):
                raw = await request_with_capacity("paragraph_fields", lambda: provider.paragraph_fields(
                    deepcopy(question), deepcopy(catalog), deepcopy(annotations)))
                try:
                    return {"ranked_field_reviews": protocol.validate_fields(raw, annotations)}
                except (ValidationError, TypeError, ValueError):
                    raise invalid("paragraph_fields", "paragraph_fields_not_verified") from None
            value = await instrument("paragraph_fields", "LangChain", "model", call)(state)
            reviewed = value["ranked_field_reviews"]
        try:
            result = protocol.compile_ranked_result(question, paragraphs, queries=state["ranked_query_catalog"],
                recall=state["ranked_recall"], ranks=state["ranked_ranks"], selected_refs=refs, annotations=annotations,
                field_reviews=reviewed, config=config, excluded_spans=excluded)
        except (ValidationError, TypeError, ValueError):
            raise invalid("paragraph_fields", "paragraph_source_proof_replay") from None
        complete("paragraph_fields", {"field_verified_count": len(reviewed), "formal_paragraph_count": len(result["evidence"])})
        return {**result, "ranked_field_reviews": reviewed,
            "candidate_evidence": deepcopy(result["material_eligibility"]["input_evidence"]),
            "evidence_selection": {"method": protocol.VERSION, "selected_evidence_ids": [r["id"] for r in result["evidence"]]},
            "source_coverage": {"strategy": "ranked_paragraph_subset", "complete": False,
                "source_paragraph_count": len(paragraphs), "recalled_paragraph_count": len(state["ranked_recall"]["candidate_catalog"]),
                "selected_paragraph_count": len(refs), "formal_paragraph_count": len(result["evidence"])}}

    return {"paragraph_queries": queries,
            "paragraph_recall": instrument("paragraph_recall", "BM25/jieba", "local", recall),
            "paragraph_rank": rank, "paragraph_annotate": annotate, "paragraph_fields": fields}
