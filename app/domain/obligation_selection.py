"""Experimental dependency-bundle reading plans, never evidence or truth checks.

The injected cost is for the COMPLETE proposed payload, not additive paragraph
estimates. It must account for the caller's real tokenizer/message construction.
Final messages still require their own tokenizer preflight before any request.
"""
from copy import deepcopy
from fractions import Fraction

from app.domain.evidence_obligations import validate_obligation_plan
from app.domain.paragraph_contributions import source_blocks


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"selection_{label}_nonnegative_integer_required")
    return value


def plan_obligation_selection(question, catalog, raw_output, *, input_token_cost,
                              token_limit, reserved_tokens=0, read_refs=()):
    """Revalidate raw model schema and plan whole dependency closures atomically.

    No rank/credibility/person/source-name heuristics or fixed paragraph quotas.
    Catalog completeness and read_refs are caller declarations, not verified raw
    parent bindings or evidence of understanding. Unlinked sources stay unknown.
    """
    limit = _integer(token_limit, "token_limit")
    reserved = _integer(reserved_tokens, "reserved_tokens")
    if not callable(input_token_cost):
        raise ValueError("selection_cost_callback_required")
    catalog = list(catalog)
    validated = validate_obligation_plan(question, catalog, raw_output)
    sources = {row["paragraph_ref"]: row for row in catalog}
    identities = set()
    for row in catalog:
        pair = tuple(row.get(key) for key in ("doc_id", "para_id"))
        if any(not isinstance(value, str) or not value.strip() for value in pair) or pair in identities:
            raise ValueError("selection_unique_original_doc_para_required")
        identities.add(pair)
    read_refs = list(read_refs)
    if (any(not isinstance(ref, str) or ref not in sources for ref in read_refs)
            or len(set(read_refs)) != len(read_refs)):
        raise ValueError("selection_known_unique_read_refs_required")
    read = set(read_refs)
    graph = {key: deepcopy(validated[key]) for key in ("obligations", "judgments", "answer_depends_on")}
    judgments = {row["judgment_id"]: row for row in graph["judgments"]}
    order = validated["dependency_order"]
    active = set(validated["proposed_answer_reachable_judgment_ids"])
    closures = {}
    for ref in order:
        closures[ref] = {ref}.union(*(closures[dep] for dep in judgments[ref]["depends_on"]))
    # A structural anchor can be supportive, contrary or limiting. Empty
    # unresolved nodes are still pending, but cannot buy utility with IDs alone.
    source_anchored = {ref: any(judgments[j]["links"] for j in closures[ref]) for ref in order}
    # Sharing is EXACT provided-text equality only. Identity and local S scope remain.
    body_ids = {}
    for row in catalog:
        body_ids.setdefault(row["text"], f"R{len(body_ids) + 1}")

    def payload(chosen):
        included = [judgments[ref] for ref in order if ref in chosen]
        refs = {link["paragraph_ref"] for row in included for link in row["links"]}
        obligations = {oid for row in included for oid in row["obligation_ids"]}
        units = {}
        for ref, row in sources.items():
            if ref not in refs:
                continue
            unit = units.setdefault(row["text"], {"reading_unit_ref": body_ids[row["text"]],
                "blocks": source_blocks(row["text"]) if row["text"] else [], "source_aliases": []})
            alias = {key: row[key] for key in ("paragraph_ref", "doc_id", "para_id")}
            alias.update(title=row.get("title", ""), paragraph_complete=row.get("paragraph_complete", False))
            unit["source_aliases"].append(alias)
        return {"question": {"question": question["question"]},
            "proposed_answer_targets": deepcopy(graph["answer_depends_on"]),
            "unprovided_judgment_ids": [ref for ref in order if ref not in chosen],
            "untested_obligations": [deepcopy(row) for row in graph["obligations"]
                                      if row["obligation_id"] in obligations],
            "proposed_judgments": deepcopy(included), "reading_units": list(units.values()),
            "span_scope": "span_ref_is_local_to_each_original_paragraph_alias",
            "semantics_verified": False, "publishable": False}

    cost_trace = []

    def measure(chosen):
        proposed = payload(chosen)
        try:
            value = input_token_cost(deepcopy(proposed))
        except Exception:
            value = None  # A callback failure never admits a free request.
        if value is not None:
            _integer(value, "callback_cost")
        cost_trace.append({"judgment_ids": [ref for ref in order if ref in chosen],
                           "input_tokens": value, "cost_known": value is not None})
        return proposed, value

    def coverage(chosen):
        return {(ref, link["obligation_id"], link["relation"])
                for ref in chosen for link in judgments[ref]["links"]}

    chosen = set()
    reading_payload, current_cost = measure(chosen)
    budget_base_valid = current_cost is not None and current_cost + reserved <= limit
    while budget_base_valid:
        best = None
        for index, ref in enumerate(order):
            if ref not in active or ref in chosen:
                continue
            union = chosen | closures[ref]
            gain = len(coverage(union) - coverage(chosen))
            gain += sum(source_anchored[j] for j in union - chosen)
            if not gain:
                continue
            proposed, cost = measure(union)
            if cost is None or cost + reserved > limit:
                continue
            # Coverage counts concrete proposed uses, never truth or role quotas.
            key = (Fraction(gain, max(1, cost - current_cost)), gain, -index)
            if best is None or key > best[0]:
                best = (key, union, proposed, cost)
        if best is None:
            break
        _, chosen, reading_payload, current_cost = best

    # Re-evaluate every remaining COMPLETE bundle against the final union. No
    # stale standalone costs, partial admission, or assumed-monotone tokenizer.
    deferred = []
    for ref in order:
        if ref in active and ref not in chosen:
            _, cost = measure(chosen | closures[ref])
            deferred.append({"judgment_id": ref, "required_judgment_ids": [j for j in order if j in closures[ref]],
                "union_input_tokens": cost, "reason": "unknown_cost" if cost is None else
                "unknown_base_cost" if current_cost is None else
                "source_unanchored_unknown" if not source_anchored[ref] else
                "base_payload_out_of_budget" if not budget_base_valid else "token_budget"})
    selected = {link["paragraph_ref"] for ref in chosen for link in judgments[ref]["links"]}
    linked = {link["paragraph_ref"] for row in graph["judgments"] for link in row["links"]}
    active_refs = {link["paragraph_ref"] for ref in active for link in judgments[ref]["links"]}
    plans = []
    for ref in order:
        closure = closures[ref]
        # Pure derivations can use their dependency sources. An unlinked O is
        # never filled merely because it was declared or exists elsewhere.
        linked_o = {link["obligation_id"] for j in closure for link in judgments[j]["links"]}
        provided_o = {link["obligation_id"] for j in closure & chosen for link in judgments[j]["links"]}
        plans.append({"judgment_id": ref, "selected_in_reading_graph": ref in chosen,
            "required_dependency_ids": [j for j in order if j in closure - {ref}],
            "missing_dependency_ids": [j for j in order if j in closure - chosen - {ref}],
            "unlinked_obligation_ids": [oid for oid in judgments[ref]["obligation_ids"] if oid not in linked_o],
            "unprovided_obligation_ids": [oid for oid in judgments[ref]["obligation_ids"] if oid not in provided_o],
            "state": "selected_unverified" if ref in chosen else "pending_unknown"})
    decisions = []
    for ref, row in sources.items():
        reason = ("required_by_selected_dependency_closure" if ref in selected else
                  "unlinked_unknown" if ref not in linked else
                  "outside_proposed_answer_paths_unknown" if ref not in active_refs else "deferred_bundle_unknown")
        decisions.append({"paragraph_ref": ref, "doc_id": row["doc_id"], "para_id": row["para_id"],
            "selected": ref in selected, "reason": reason, "declared_read": ref in read,
            "provided_context": "full_declared_unverified" if row.get("paragraph_complete", False) else "partial_unknown",
            "new_reading_required": ref in selected and ref not in read,
            "unseen_context_unknown": not row.get("paragraph_complete", False)})
    missing_o = sorted({oid for row in plans if row["judgment_id"] in active for oid in row["unprovided_obligation_ids"]})
    provided_o = {link["obligation_id"] for ref in chosen for link in judgments[ref]["links"]}
    structural_selected = bool(active) and active <= chosen and not missing_o and budget_base_valid
    partial = any(not sources[ref].get("paragraph_complete", False) for ref in selected)
    return {"status": ("partial" if partial else "selected_unverified") if structural_selected else "incomplete",
        "reading_payload": reading_payload if budget_base_valid else None,
        "proposed_selection_refs": [ref for ref in sources if ref in selected],
        "selected_judgment_ids": [ref for ref in order if ref in chosen],
        "judgment_plans": plans, "missing_obligation_ids": missing_o,
        "pending_obligation_ids": [row["obligation_id"] for row in graph["obligations"] if row["obligation_id"] not in provided_o],
        "deferred_bundles": deferred, "paragraph_decisions": decisions,
        "pending_paragraph_refs": [ref for ref in sources if ref not in selected],
        "unread_paragraph_refs": [ref for ref in sources if ref not in read],
        "orphan_judgment_ids": validated["orphan_judgment_ids"], "original_proposed_plan": graph,
        "input_tokens": current_cost, "reserved_tokens": reserved, "token_limit": limit,
        "within_planning_budget": budget_base_valid,
        "planning_budget_status": "known_within_limit" if budget_base_valid else
            "unknown_cost" if current_cost is None else "exceeds_limit",
        "cost_basis": "injected_integer_planning_ledger_actual_final_message_tokenizer_preflight_required",
        "cost_trace": cost_trace, "all_original_source_identities_preserved": True,
        "sharing_implies_fact_or_source_equivalence": False, "independent_source_votes_created": 0,
        "answer_path_selection_proves_counterevidence_complete": False,
        "dependency_selection_proves_necessity_or_inference_truth": False,
        "parent_source_mapping_verified": False, "judgement_states_verified": False,
        "semantic_entailment_verified": False, "rule_applicability_verified": False,
        "formal_evidence_generated": False, "publishable": False}
