"""Offline source-ID diagnostic for a root-hash-bound CLOSED reading probe.

References are opened only after all closure/source/message checks. Stage recall
is a candidate-coverage proxy; only proposed_selection receives ID-only P/R/F1.
No semantic judgment or final supporting/conflicting classification is created.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.domain.frozen_selection_inputs import build_frozen_selection_input, canonical
from app.domain.paragraph_contributions import source_blocks
from app.domain.evidence_obligations import validate_obligation_plan
from app.domain.obligation_selection import plan_obligation_selection

STAGES = ("fixed_pool", "exposed", "locally_fetched", "presented_full_context",
          "completed_response_full_context", "final_graph_linked", "proposed_selection")


def digest(value): return sha256(canonical(value)).hexdigest()


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative_known_integer_required")
    return value


class Bindings:
    def __init__(self): self.guards = {}

    def read(self, descriptor, within=None):
        if not isinstance(descriptor, dict) or set(descriptor) != {"file", "sha256"}:
            raise ValueError("exact_file_sha_descriptor_required")
        path = Path(descriptor["file"]).resolve(); checksum = descriptor["sha256"]
        if (path.name.lower() in {".env", "evidence_samples.json", "conclusions.json", "expected.json"}
                or within is not None and not path.is_relative_to(within)
                or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum)):
            raise ValueError("source_binding_scope_forbidden")
        data = path.read_bytes()
        if sha256(data).hexdigest() != checksum:
            raise ValueError("closed_probe_bound_file_changed")
        self.guards[str(path)] = checksum
        return data

    def json(self, descriptor, within=None):
        return json.loads(self.read(descriptor, within).decode("utf-8-sig"))

    def unchanged(self):
        if any(sha256(Path(p).read_bytes()).hexdigest() != h for p, h in self.guards.items()):
            raise ValueError("closed_probe_changed_during_score")


def known_ledger(guard):
    if guard.get("unknown_usage_requests") != 0 or guard.get("pending_request") is not None:
        raise ValueError("closed_probe_usage_unknown_or_pending")
    invoked = [r for r in guard["records"] if r.get("actual_request_started") is True]
    for row in invoked:
        if row.get("usage_known") is not True:
            raise ValueError("closed_probe_actual_usage_not_known")
        a, b, total = (integer(row[k]) for k in ("input_tokens", "output_tokens", "total_tokens"))
        if a + b != total:
            raise ValueError("closed_probe_usage_inconsistent")
    if (len(invoked) != integer(guard["actual_requests_started"])
            or sum(r["total_tokens"] for r in invoked) != integer(guard["known_total_tokens"])):
        raise ValueError("closed_probe_usage_ledger_mismatch")
    return invoked


def refs(values, sources):
    if (not isinstance(values, list) or any(not isinstance(r, str) or r not in sources for r in values)
            or len(values) != len(set(values))):
        raise ValueError("known_unique_program_paragraph_refs_required")
    return set(values)


def tool_scopes(events, sources):
    exposed, fetched = set(), set()
    for index, event in enumerate(events, 1):
        if event.get("call_index") != index or event.get("local_only") is not True or event.get("model_calls") != 0:
            raise ValueError("actual_local_tool_trace_required")
        returned = refs(event["returned_paragraph_refs"], sources)
        operation, status, args = event["operation"], event["status"], event["arguments"]
        if operation not in {"search", "read"} or status not in {"completed", "empty_query_no_search", "failed", "rejected"}:
            raise ValueError("unknown_local_tool_operation_or_state")
        if status != "completed":
            if returned: raise ValueError("failed_tool_cannot_expose_or_fetch")
            continue
        if operation == "search": exposed |= returned
        else:
            if refs(args["paragraph_refs"], sources) != returned or not returned <= exposed or returned & fetched:
                raise ValueError("tool_read_outside_previous_exposure_or_repeated")
            fetched |= returned
    return exposed, fetched


def message_scope(messages, question, sources):
    if (not isinstance(messages, list) or len(messages) != 2 or messages[0].get("role") != "system"
            or messages[1].get("role") != "human" or not isinstance(messages[0].get("content"), str)):
        raise ValueError("exact_native_final_messages_required")
    payload = json.loads(messages[1]["content"])
    if payload.get("question") != {"question": question["question"]}:
        raise ValueError("message_original_question_changed")
    rows = payload["full_read_paragraphs"]; present = refs([r["paragraph_ref"] for r in rows], sources)
    if [r["paragraph_ref"] for r in rows] != [r for r in sources if r in present]:
        raise ValueError("full_context_original_source_order_required")
    for row in rows:
        source = sources[row["paragraph_ref"]]
        expected = {k: source[k] for k in ("paragraph_ref", "title", "doc_id", "para_id")}
        expected.update(blocks=source_blocks(source["text"]) if source["text"] else [],
                        paragraph_complete=True, semantic_verified=False, parent_codepoints=len(source["text"]))
        if row != expected: raise ValueError("full_context_original_blocks_changed")
    excerpts = payload.get("latest_search_results", [])
    excerpt_refs = refs([r["paragraph_ref"] for r in excerpts], sources)
    for row in excerpts:
        source = sources[row["paragraph_ref"]]; start, end = row["excerpt_start_cp"], row["excerpt_end_cp"]
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(source["text"])
                or row["excerpt_text"] != source["text"][start:end]
                or any(row[k] != source[k] for k in ("doc_id", "para_id", "title"))
                or row["paragraph_complete"] != (start == 0 and end == len(source["text"]))
                or row.get("semantic_verified") is not False):
            raise ValueError("search_excerpt_original_span_changed")
    return present, excerpt_refs, {r["paragraph_ref"] for r in excerpts if r["paragraph_complete"]}


def validate_closed_probe(final_path, expected_sha256, *, _bindings=None, _seen=()):
    """No references, provider, credentials, tokenizer execution or network here."""
    bindings = _bindings or Bindings(); final_path = Path(final_path).resolve(); run = final_path.parent
    if final_path in _seen: raise ValueError("closed_receipt_cycle")
    final = bindings.json({"file": str(final_path), "sha256": expected_sha256})
    if (final.get("closed") is not True or final.get("source_unchanged") is not True
            or final.get("reference_reads") != [] or final.get("publishable") is not False
            or final.get("formal_evidence_generated") is not False
            or final.get("status") not in {"stopped", "partial_unverified", "completed_unverified"}):
        raise ValueError("closed_unverified_probe_receipt_required")
    invoked = known_ledger(final["guard"])
    protocol = bindings.json({"file": str(run / "probe-protocol.json"), "sha256": final["protocol_sha256"]}, run)
    if protocol.get("format") != "one-uniform-tool-reading-probe-v1" or protocol.get("reference_reads") != []:
        raise ValueError("reviewed_tool_probe_protocol_required")
    config = protocol["config"]; files = protocol["bound_input_files"]; bundle = protocol["source_bundle"]
    if (digest(bundle["files"]) != bundle["sha256"]
            or any(p in files and files[p] != h for p,h in bundle["files"].items())):
        raise ValueError("source_bundle_digest_changed")
    for path, checksum in {**files, **bundle["files"]}.items(): bindings.read({"file": path, "sha256": checksum})
    def source(key): return bindings.json({"file": config[key], "sha256": files[config[key]]})
    case, index, freeze = source("case_input"), source("input_index"), source("input_freeze")
    if (config["case_sha256"] != final["case_sha256"] or files[config["case_input"]] != final["case_sha256"]
            or freeze.get("status") != "CLOSED_RAW_INPUT_MAPPING_ONLY" or index.get("reference_reads") != []
            or freeze.get("reference_reads") != [] or freeze["index"]["sha256"] != files[config["input_index"]]
            or not any(r["file"] == config["case_input"] and r["sha256"] == final["case_sha256"] for r in index["cases"])):
        raise ValueError("closed_raw_case_index_binding_required")
    raw_bytes = bindings.read({"file": config["raw_input"], "sha256": files[config["raw_input"]]})
    pool_bytes = bindings.read({"file": config["pool_input"], "sha256": files[config["pool_input"]]})
    rebuilt = build_frozen_selection_input(raw_input_bytes=raw_bytes, pool_input_bytes=pool_bytes,
        topic_id=case["topic_id"], question_id=case["question_id"], variant=case["binding"]["variant"],
        expected_raw_sha256=case["binding"]["raw_input_file_sha256"], expected_pool_sha256=case["binding"]["pool_input_file_sha256"])
    if case != rebuilt or protocol["case_identity"] != {k: case[k] for k in ("topic_id", "question_id")}:
        raise ValueError("raw_pool_case_rebuild_changed")
    prior = 0
    for descriptor in config["prior_receipts"]:
        old = validate_closed_probe(descriptor["file"], descriptor["sha256"], _bindings=bindings, _seen=(*_seen, final_path))
        if (old["case"]["binding"]["catalog_sha256"] != case["binding"]["catalog_sha256"]
                or old["case"]["question"] != case["question"]
                or any(old["case"][k] != case[k] for k in ("topic_id", "question_id"))):
            raise ValueError("prior_probe_wrong_case")
        prior += old["receipt"]["guard"]["known_total_tokens"]
    if (prior != protocol["prior"]["tokens"] or prior != final["prior_known_tokens"]
            or final["whole_episode_total_tokens"] != prior + final["guard"]["known_total_tokens"]):
        raise ValueError("whole_episode_usage_not_closed")
    sources = {r["paragraph_ref"]: r for r in case["catalog"]}
    exposed, fetched = tool_scopes(final["tool_trace"], sources)
    if (refs(final["tool_scope"]["exposed_paragraph_refs"], sources) != exposed
            or refs(final["tool_scope"]["read_paragraph_refs"], sources) != fetched):
        raise ValueError("tool_scope_not_actual_trace")
    preflights = {}
    for wrapper in final["preflights"]:
        messages = bindings.json(wrapper["messages"], run); pre = bindings.json(wrapper["preflight"], run)
        if (pre["admission"] != wrapper["admission"] or pre["protocol_sha256"] != final["protocol_sha256"]
                or pre["message_file"] != wrapper["messages"] or digest(messages) != pre["admission"]["messages_sha256"]):
            raise ValueError("preflight_exact_messages_changed")
        for counter in pre["counter_receipts"]:
            cm = bindings.json(counter["messages"], run); cr = bindings.json(counter["counter_receipt"], run)
            if digest(cm) != cr["message_content_sha256"]:
                raise ValueError("counter_message_binding_changed")
        if "final_lookahead_messages" in wrapper: bindings.json(wrapper["final_lookahead_messages"], run)
        preflights[wrapper["preflight"]["sha256"]] = (pre, messages, wrapper["preflight"])
    if len(preflights) != len(final["preflights"]): raise ValueError("duplicate_preflight_receipt")
    presented, completed, short_complete = set(), set(), set(); settled_index = 0
    for step in final["steps"]:
        stored = bindings.json(step["receipt"], run)
        if stored != {k: v for k, v in step.items() if k != "receipt"}:
            raise ValueError("step_receipt_changed")
        pre, messages, pre_descriptor = preflights[step["preflight"]["sha256"]]
        if (step["preflight"] != pre_descriptor or step["messages"] != pre["message_file"]
                or step["messages_sha256"] != digest(messages) or pre["admission"].get("admitted") is not True):
            raise ValueError("step_messages_not_admitted_preflight")
        full, excerpt, complete_excerpt = message_scope(messages, case["question"], sources)
        if (refs(step["full_context_refs"], sources) != full or refs(step["search_excerpt_refs"], sources) != excerpt
                or refs(step["complete_search_excerpt_refs"], sources) != complete_excerpt or not full <= fetched or not excerpt <= exposed):
            raise ValueError("step_context_not_bound_tool_scope")
        if step.get("actual_request_started") is True:
            ledger = invoked[settled_index]; settled_index += 1
            if (step.get("usage_settled") is not True or step.get("unknown_usage_requests") != 0
                    or ledger["message_content_sha256"] != step["messages_sha256"]
                    or any(step["settlement"].get(k) != ledger.get(k) for k in ("usage_known", "input_tokens", "output_tokens", "total_tokens", "outcome", "stop_reason"))):
                raise ValueError("step_actual_usage_not_matching_guard")
            presented |= full
            if step.get("response_completed") is True: completed |= full; short_complete |= complete_excerpt
        elif step.get("response_completed") is True: raise ValueError("uninvoked_response_cannot_complete")
    if settled_index != len(invoked): raise ValueError("actual_calls_missing_step_receipts")
    result = bindings.json(final["output_artifact"], run) if final["output_artifact"] else None
    if final["status"] != "stopped" and result is None: raise ValueError("finished_probe_missing_output_artifact")
    linked, selected, deferred = set(), set(), set()
    if result is not None:
        if (result.get("usage_after_boundary_unknown") is not False or result["known_episode_tokens"] != final["whole_episode_total_tokens"]
                or result["tool_trace"] != final["tool_trace"] or result["scope"] != final["tool_scope"]
                or refs(result["fetched_full_paragraph_refs"], sources) != fetched
                or refs(result["presented_full_paragraph_refs"], sources) != presented):
            raise ValueError("flow_result_scope_or_usage_changed")
        graph, selection = result.get("final_proposed_plan"), result.get("selection_plan")
        if graph is not None:
            if not final["steps"] or final["steps"][-1].get("stage") != "final" or final["steps"][-1].get("response_completed") is not True:
                raise ValueError("final_graph_without_completed_final_request")
            validated = validate_obligation_plan(case["question"], [r for r in case["catalog"] if r["paragraph_ref"] in completed], graph)
            linked = {r["paragraph_ref"] for r in validated["resolved_links"]}
        if selection is not None:
            if graph is None: raise ValueError("selection_without_final_graph")
            costs = iter(selection["cost_trace"])
            def recorded_cost(payload):
                entry = next(costs)
                if entry["judgment_ids"] != [j["judgment_id"] for j in payload["proposed_judgments"]]:
                    raise ValueError("selection_cost_trace_changed")
                return entry["input_tokens"]
            replay = plan_obligation_selection(case["question"], case["catalog"], graph,
                input_token_cost=recorded_cost, token_limit=selection["token_limit"], reserved_tokens=selection["reserved_tokens"],
                read_refs=result["bound_full_context_paragraph_refs"])
            if replay != selection or next(costs, None) is not None:
                raise ValueError("selection_program_replay_changed")
            selected = refs(selection["proposed_selection_refs"], sources)
            deferred = {r["paragraph_ref"] for r in selection["paragraph_decisions"] if r["reason"] == "deferred_bundle_unknown"}
    if not selected <= linked <= completed <= presented <= fetched <= exposed <= set(sources):
        raise ValueError("probe_source_stage_scope_not_nested")
    stages = dict(zip(STAGES, (set(sources), exposed, fetched, presented, completed, linked, selected)))
    bindings.unchanged()
    return {"case": case, "receipt": final, "result": result, "stages": stages, "budget_deferred": deferred,
            "completed_complete_search_excerpt_refs": short_complete, "bindings": bindings}


def reference_tools():
    """Reuse pinned functions without importing the historical global hook."""
    from scripts.offline_reference_tools import load_reference_tools
    mapper, engine, guards = load_reference_tools()
    return mapper, engine, [*guards, {"file": str(Path(__file__).resolve()),
                                    "sha256": sha256(Path(__file__).read_bytes()).hexdigest()}]


def score_closed_probe(final_path, expected_sha256, *, reference_loader):
    checked = validate_closed_probe(final_path, expected_sha256)
    mapper, engine, implementation = reference_tools()
    # FIRST reference access: all actual usage/artifact/source checks are above.
    case = checked["case"]; evidence, conclusion, reference_guards = reference_loader(deepcopy(case))
    if conclusion.get("question_id") != case["question_id"] or conclusion.get("question") != case["question"]["question"]:
        raise ValueError("original_reference_question_changed")
    reference, issues = mapper.question_union_reference(evidence, conclusion)
    catalog = {r["paragraph_ref"]: r for r in case["catalog"]}
    def source_rows(rs): return [{k: catalog[r][k] for k in ("doc_id", "para_id")} for r in catalog if r in rs]
    comparisons = {name: engine.compare_rows(case["topic_id"], source_rows(rs), reference)["unique_paragraphs"]["A_id_only"]["metrics"]
                   for name, rs in checked["stages"].items()}
    coverage = {name: {k: row[k] for k in ("matches", "reference_records", "recall")} | {"source_positions": row["submitted_records"]}
                for name, row in comparisons.items()}
    selected_locations = {mapper.key(r) for r in source_rows(checked["stages"]["proposed_selection"])}
    reference_locations = {mapper.key(r) for r in reference if mapper.key(r) is not None}
    stage_locations = {name: {mapper.key(r) for r in source_rows(rs)} for name, rs in checked["stages"].items()}
    causes = ("pool_missing", "not_exposed", "not_fetched", "not_presented", "no_completed_response_context", "not_linked", "not_selected")
    misses = []
    for location in sorted(reference_locations - selected_locations):
        reason = next(cause for name, cause in zip(STAGES, causes) if location not in stage_locations[name])
        if reason == "not_selected" and location in {mapper.key(r) for r in source_rows(checked["budget_deferred"])}:
            reason = "budget_deferred"
        misses.append({"doc_id": location[0], "para_id": location[1], "reason": reason})
    checked["bindings"].unchanged()
    for guard in [*implementation, *reference_guards]:
        if sha256(Path(guard["file"]).read_bytes()).hexdigest() != guard["sha256"]: raise ValueError("offline_score_input_changed")
    return {"format": "closed-obligation-probe-source-ID-diagnostic-v1", "topic_id": case["topic_id"], "question_id": case["question_id"],
        "probe_status": checked["receipt"]["status"], "stage_source_coverage": coverage,
        "selected_source_A_proxy": comparisons["proposed_selection"], "reference_issues": issues,
        "reference_denominator": "per-question supporting union conflicting, unique original doc_id/para_id; unresolved identities retained",
        "reference_raw_union_records": len(reference), "misses": misses,
        "selected_nonreference_IDs": [{"doc_id": d, "para_id": p, "semantic_false_fact_asserted": False} for d, p in sorted(selected_locations-reference_locations)],
        "completed_complete_search_excerpt_count_separate": len(checked["completed_complete_search_excerpt_refs"]),
        "official_T1_score": False, "semantic_score": False, "final_evidence_or_relation_score": False,
        "graph_link_sides_are_final_conclusion_relations": False, "model_comprehension_verified": False,
        "new_model_calls": 0, "input_guards": [{"file": p, "sha256": h} for p,h in checked["bindings"].guards.items()]+implementation+reference_guards}


def load_reference(case, sample_root):
    """Call only from score_closed_probe's post-validation loader branch."""
    sample_root = Path(sample_root).resolve(); folder = (sample_root / case["topic_title"]).resolve()
    if folder.parent != sample_root: raise ValueError("reference_topic_directory_escape")
    paths = [folder/"evidence_samples.json", folder/"conclusions.json"]
    guards = [{"file": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in paths]
    evidence, conclusions = [json.loads(p.read_text("utf-8-sig")) for p in paths]
    rows = [r for r in conclusions if r.get("question_id") == case["question_id"]]
    if len(rows) != 1: raise ValueError("unique_original_reference_question_required")
    return evidence, rows[0], guards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-receipt", type=Path, required=True); parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--sample-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); sample_root = args.sample_root.resolve(); output = args.output.resolve()
    if output.exists() or output.is_relative_to(sample_root) or output.is_relative_to(args.closed_receipt.resolve().parent):
        raise ValueError("new_separate_offline_score_path_required")
    result = score_closed_probe(args.closed_receipt, args.expected_sha256, reference_loader=lambda case: load_reference(case, sample_root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+"\n")
    print(json.dumps({"output": str(output), "new_model_calls": 0}))


if __name__ == "__main__":
    def audit(event, args):
        if event.startswith("socket."): raise PermissionError("offline_probe_score_network_forbidden")
        if event == "open" and args and not isinstance(args[0], int) and Path(os.fsdecode(args[0])).name.lower() == ".env":
            raise PermissionError("offline_probe_score_credentials_forbidden")
    sys.addaudithook(audit); main()
