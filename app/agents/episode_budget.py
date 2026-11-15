"""Pre-paid whole-chain forecasts using the production native request builder.

These forecasts are conservative planning envelopes, not billed usage or a
mathematical upper bound on provider tokenization or arbitrary future prose.
Actual request guards and complete accounting remain authoritative. There is
no model invocation, response repair, source shortening or agent loop here.
"""
from copy import deepcopy
from hashlib import sha256
import json
import subprocess

from app.agents.provider import ROLE_INSTRUCTIONS, ARBITRATION_INSTRUCTIONS, GROUNDING_INSTRUCTIONS
from app.agents.schemas import AnalysisOutput, ArbitrationOutput
from app.agents.strict_transport import (
    ROOT, TOKENIZER, TOKENIZER_SHA, canonical, prepare_strict_request, build_strict_request_payload,
)

VERSION = "native-whole-chain-planning-v1"
POLICY = {"version": VERSION, "selected_anchor_cap": 10,
          "future_analysis_wire_allowance": 10240, "safety_per_request": 512,
          "source_size_basis": "largest_complete_range_per_anchor_then_largest_10_serialized_rows",
          "future_analysis_allowance_is_not_a_proven_global_bound": True,
          "output_limits_unchanged": True, "paid_retries": 0}
PARAGRAPH_QUALITY_POLICY = {**deepcopy(POLICY),
    "version": "native-whole-chain-paragraph-quality-planning-v1", "selected_anchor_cap": None,
    "source_size_basis": "all_offered_original_anchors_largest_complete_range_per_anchor_no_top_k",
    "source_quote_envelope_is_not_an_annotation_or_future_analysis_bound": True,
    "unavailable_future_model_fields_use_explicit_size_only_placeholders": True}
PARAGRAPH_QUALITY_COMPACT_V4_POLICY = {**deepcopy(PARAGRAPH_QUALITY_POLICY),
    "version": "native-whole-chain-paragraph-quality-compact-planning-v4",
    "front_token_budget_semantics": "new_front_tokens_excluding_paid_ancestors",
    "paid_ancestors_still_count_toward_whole_episode": True}
PARAGRAPH_QUALITY_BUDGET_V5_POLICY = {**deepcopy(PARAGRAPH_QUALITY_COMPACT_V4_POLICY),
    "version": "native-whole-chain-paragraph-quality-budget-planning-v5",
    "source_size_basis": "all_offered_token_packed_complete_original_parent_anchors"}


def count_texts(texts):
    """Pinned local tokenizer, a single subprocess; no network or credentials."""
    if sha256(TOKENIZER.read_bytes()).hexdigest() != TOKENIZER_SHA:
        raise ValueError("episode_budget_tokenizer_changed")
    run = subprocess.run([str(ROOT / ".runtime/v20-embedding-env/Scripts/python.exe"), "-c",
        "import json,sys;from tokenizers import Tokenizer;"
        "t=Tokenizer.from_file(sys.argv[1]);"
        "a=json.load(sys.stdin);print(json.dumps([len(t.encode(x).ids) for x in a]))", str(TOKENIZER)],
        input=json.dumps(texts, ensure_ascii=False).encode("utf8"), capture_output=True, timeout=45)
    if run.returncode:
        raise ValueError("episode_budget_tokenizer_failed")
    counts = json.loads(run.stdout)
    if len(counts) != len(texts) or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError("episode_budget_tokenizer_invalid")
    return counts


def native_request(operation, question, evidence, *, snapshot_id, model_name, output_limits,
                   compact_analyses, analyses=None, issue_frame_enabled=False):
    payload = {"question": deepcopy(question), "evidence": deepcopy(evidence)}
    if operation == "arbitrate":
        payload["analyses"] = deepcopy(analyses)
        instructions, schema = ARBITRATION_INSTRUCTIONS, ArbitrationOutput
    else:
        payload["evidence_snapshot_id"] = snapshot_id
        instructions, schema = ROLE_INSTRUCTIONS[operation], AnalysisOutput
    prepared = prepare_strict_request(operation, instructions + GROUNDING_INSTRUCTIONS,
        payload, schema, balanced_scoped_enabled=True, compact_analyses_enabled=compact_analyses,
        issue_frame_enabled=issue_frame_enabled)
    return build_strict_request_payload(prepared, model_name=model_name,
                                       max_output_tokens=output_limits[operation])


def _empty_analyses(snapshot_id):
    # Explicit size-only placeholders; never passed to inference or published.
    return [{"role": role, "evidence_snapshot_id": snapshot_id, "decision": "undetermined",
             "conclusion": "尺寸占位；未来模型判断尚不存在。", "judgments": [],
             "supporting_evidence_ids": [], "conflicting_evidence_ids": []}
            for role in ("authority_analyst", "cross_validation_analyst")]


def plan_tail(question, evidence, *, known_tokens, known_calls, threshold, max_calls,
              model_name, output_limits, compact_analyses, snapshot_id="budget-only-snapshot",
              analyses=None, counter=count_texts, issue_frame_enabled=False, planning_policy=None):
    policy = deepcopy(POLICY if planning_policy is None else planning_policy)
    if policy not in (POLICY, PARAGRAPH_QUALITY_POLICY, PARAGRAPH_QUALITY_COMPACT_V4_POLICY, PARAGRAPH_QUALITY_BUDGET_V5_POLICY):
        raise ValueError("whole_chain_forecast_unknown_policy")
    if analyses is None and not compact_analyses:
        raise ValueError("whole_chain_forecast_requires_compact_analysis_projection")
    operations = (["authority_analyst", "cross_validation_analyst", "arbitrate"]
                  if analyses is None else ["arbitrate"])
    wires = [native_request(op, question, evidence, snapshot_id=snapshot_id, model_name=model_name,
        output_limits=output_limits, compact_analyses=compact_analyses,
        analyses=_empty_analyses(snapshot_id) if analyses is None else analyses,
        issue_frame_enabled=issue_frame_enabled) for op in operations]
    counts = counter([wire["canonical_wire_json"] for wire in wires])
    growth = policy["future_analysis_wire_allowance"] if analyses is None else 0
    rows = [{"operation": op, "serialized_json_input_estimate": n,
             "reserved_output_tokens": output_limits[op], "safety_tokens": policy["safety_per_request"],
             "native_wire_sha256": wire["canonical_wire_sha256"]}
            for op, n, wire in zip(operations, counts, wires)]
    required = sum(r["serialized_json_input_estimate"] + r["reserved_output_tokens"] + r["safety_tokens"] for r in rows) + growth
    return {"format": policy["version"], "phase": "selected_sources" if analyses is None else "actual_analyses",
            "policy": policy, "known_tokens": known_tokens, "known_calls": known_calls,
            "threshold": threshold, "max_calls": max_calls, "remaining_requests": len(rows),
            "requests": rows, "future_analysis_wire_allowance": growth,
            "required_remaining_planning_tokens": required,
            "planned_whole_tokens": known_tokens + required,
            "admitted": known_tokens + required <= threshold and known_calls + len(rows) <= max_calls,
            "is_actual_usage": False, "is_guaranteed_global_upper_bound": False,
            **({"issue_frame_enabled": True} if issue_frame_enabled else {}),
            "selected_quote_codepoints": sum(len(row["quote"]) for row in evidence),
            "source_count": len(evidence), "wire_bodies": [wire["body"] for wire in wires]}


def planning_sources(prepared, *, counter=count_texts):
    """A size envelope only: it does not select or hide any candidate source."""
    from app.domain.lc_bundle_case import _literal_lead_sentence
    rows = []
    for item in prepared["catalog"]:
        source = item["source_bundle"]
        rows.append({"id": item["source_window_id"], "paragraph_id": source["paragraph_id"],
                     "quote": source["quote"], "fact_des": _literal_lead_sentence(source["quote"]),
                     "raw_doc_id": source["raw_doc_id"], "raw_para_id": source["raw_para_id"],
                     "grade_score": 0.95, "source_group_id": source["document_id"]})
    counts = counter([canonical(row) + "\n" for row in rows]) if rows else []
    by_anchor = {}
    for row, cost in zip(rows, counts):
        key = row["paragraph_id"]
        if key not in by_anchor or cost > by_anchor[key][1]:
            by_anchor[key] = (row, cost)
    envelope = sorted(by_anchor.values(), key=lambda pair: pair[1], reverse=True)
    if prepared.get("paragraph_quality") is not True:
        envelope = envelope[:POLICY["selected_anchor_cap"]]
    return [row for row, _ in envelope]


def plan_source_episode(question, prepared, *, known_tokens, known_calls, threshold, max_calls,
                        model_name, output_limits, compact_analyses, counter=count_texts, issue_frame_enabled=False):
    paragraph_quality = prepared.get("paragraph_quality") is True
    if (prepared.get("graded_selection") is not True and not paragraph_quality) or not compact_analyses:
        raise ValueError("whole_chain_forecast_requires_graded_compact_protocol")
    policy = PARAGRAPH_QUALITY_POLICY if paragraph_quality else POLICY
    if prepared.get("version") == "lc-paragraph-quality-v4":
        policy = PARAGRAPH_QUALITY_COMPACT_V4_POLICY
    if prepared.get("version") == "lc-paragraph-quality-v5":
        policy = PARAGRAPH_QUALITY_BUDGET_V5_POLICY
    # The production source schema and messages are used without summarizing.
    if paragraph_quality:
        from app.domain.paragraph_quality_case import paragraph_quality_schema as selection_schema
        if prepared.get("paragraph_quality_compact") is True:
            from app.domain.paragraph_quality_compact_case import paragraph_quality_schema as selection_schema
        if prepared.get("parent_token_pack") is True:
            from app.domain.paragraph_quality_budget_case import paragraph_quality_schema as selection_schema
    elif prepared.get("complete_grading") is True:
        from app.domain.complete_context_grading import complete_schema as selection_schema
    elif prepared.get("source_geometry") == "complete-sentence-context-v1":
        from app.domain.lc_context_case import selection_schema
    else:
        from app.domain.lc_bundle_case import selection_schema
    front = {"schema": selection_schema(prepared), "messages": prepared["messages"],
             "tool": deepcopy(prepared["wire_tool"])}
    front_wire = build_strict_request_payload(front, model_name=model_name,
                                             max_output_tokens=output_limits["strict_selection"])
    front_input = counter([front_wire["canonical_wire_json"]])[0]
    tail = plan_tail(question, planning_sources(prepared, counter=counter), known_tokens=known_tokens,
        known_calls=known_calls, threshold=threshold, max_calls=max_calls, model_name=model_name,
        output_limits=output_limits, compact_analyses=compact_analyses, counter=counter,
        issue_frame_enabled=issue_frame_enabled, planning_policy=policy)
    front_required = front_input + output_limits["strict_selection"] + policy["safety_per_request"]
    tail.update(phase="before_first_paid_request", remaining_requests=4,
                front_request={"operation": "strict_selection", "serialized_json_input_estimate": front_input,
                    "reserved_output_tokens": output_limits["strict_selection"], "safety_tokens": policy["safety_per_request"],
                    "native_wire_sha256": front_wire["canonical_wire_sha256"]},
                required_remaining_planning_tokens=front_required + tail["required_remaining_planning_tokens"],
                planned_whole_tokens=front_required + tail["planned_whole_tokens"])
    tail["admitted"] = tail["planned_whole_tokens"] <= threshold and known_calls + 4 <= max_calls
    tail["wire_bodies"].insert(0, front_wire["body"])
    return tail
