"""Replay a single LangChain source-book selection before the graph tail.

This verifier has no model call or retrieval fallback. Frozen application
bytes prepare the source book and compile the model's original structured
output through one fixed, isolated CLI. The actual request, complete source
delivery and provider accounting are independent of annotation semantics.
"""
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from app.workflows.bundle_workbench import _bound, _json, _question, _usage_total, digest


PROOF_FORMAT = "langchain-source-bundle-workbench-proof-v1"
_ENTRY = "scripts/lc_bundle_case.py"
_RANK_ENTRY = "scripts/local_parent_rank.py"
_APP_SOURCE = re.compile(r"app/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py\Z")
_REQUIRED = {_ENTRY, "app/__init__.py", "app/domain/lc_bundle_case.py",
             "app/domain/source_bundles.py", "app/domain/source_bundle_tools.py",
             "app/domain/numeric_literals.py"}


def _sources(archives):
    if not isinstance(archives, dict):
        raise ValueError("lc_bundle_codec_archives_required")
    sources = {}
    for relative, descriptor in archives.items():
        if not isinstance(relative, str):
            raise ValueError("lc_bundle_archive_relative_path_required")
        name = relative.replace("\\", "/")
        path = PurePosixPath(name)
        if (path.is_absolute() or any(part in {"", ".", ".."} or ":" in part for part in name.split("/"))
                or name in sources or (name.startswith("app/") and not _APP_SOURCE.fullmatch(name))):
            raise ValueError("lc_bundle_archive_relative_path_required")
        sources[name] = _bound(descriptor, raw=True)
    if not _REQUIRED <= set(sources):
        raise ValueError("lc_bundle_codec_archives_required")
    return {name: data for name, data in sources.items() if name in {_ENTRY, _RANK_ENTRY} or _APP_SOURCE.fullmatch(name)}


@contextmanager
def _archived_codec(sources, raw_bytes, raw_sha256):
    """Execute only the SHA-bound source compiler, with no live app fallback."""
    with TemporaryDirectory(prefix="trust-judge-lc-source-replay-") as directory:
        root = Path(directory).resolve()
        for relative, data in sources.items():
            path = root / PurePosixPath(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        case_path = root / "raw-case.json"
        case_path.write_bytes(raw_bytes)
        # Isolated Python still loads installed site packages. Pin the regular
        # archived app package ahead of an editable workspace installation,
        # even if an old CLI did not set its own sys.path explicitly.
        bootstrap = root / "_lc_source_replay.py"
        bootstrap.write_text("import runpy,sys\nfrom pathlib import Path\n"
            "sys.path.insert(0,str(Path(__file__).resolve().parent))\n"
            "entry=sys.argv[1]\nsys.argv=sys.argv[1:]\nrunpy.run_path(entry,run_name='__main__')\n", encoding="utf-8")
        env = {key: os.environ[key] for key in ("SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP") if key in os.environ}

        def execute(request):
            try:
                completed = subprocess.run([sys.executable, "-I", "-B", str(bootstrap), str(root / _ENTRY), str(case_path), raw_sha256],
                    input=_json(request).encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=root, env=env, timeout=15, check=False)
            except (OSError, subprocess.TimeoutExpired):
                raise ValueError("lc_bundle_archived_replay_unavailable") from None
            if completed.returncode != 0 or len(completed.stdout) > 10_000_000:
                raise ValueError("lc_bundle_archived_replay_failed")
            try:
                output = json.loads(completed.stdout.decode("utf-8"))
            except (ValueError, UnicodeError):
                raise ValueError("lc_bundle_archived_replay_invalid_output") from None
            if not isinstance(output, dict):
                raise ValueError("lc_bundle_archived_replay_invalid_output")
            return output

        yield execute


def _provider_json(raw):
    def pairs(items):
        row = {}
        for key, value in items:
            if key in row:
                raise ValueError("lc_bundle_duplicate_json_key")
            row[key] = value
        return row
    try:
        return json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("lc_bundle_nonfinite_json")))
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("lc_bundle_invalid_original_json") from None


def _front_usage(proof, body):
    requests = proof.get("front_requests")
    if (not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], dict)
            or requests[0].get("operation") != "strict_selection"):
        raise ValueError("lc_bundle_exactly_one_front_request_required")
    previous = _bound(proof["previous_usage"])
    if not isinstance(previous, dict) or not isinstance(previous.get("records"), list):
        raise ValueError("lc_bundle_parent_usage_required")
    records = previous["records"]
    parent = 0
    for row in records:
        if not isinstance(row, dict) or row.get("usage_source") != "provider":
            raise ValueError("lc_bundle_unknown_parent_usage")
        parent += _usage_total(row)
    entry = requests[0]
    preflight, settlement, finished, visible = (_bound(entry[key]) for key in ("preflight", "settlement", "finished", "response"))
    episode, launch, closed = (_bound(proof[key]) for key in ("front_episode", "launch_manifest", "closed_receipt"))
    report = proof["usage_report"]
    if (not isinstance(episode, dict) or episode.get("method") != "function_calling" or episode.get("strict") is not True
            or episode.get("automatic_retries") != 0 or episode.get("thinking") != "disabled"
            or not isinstance(launch, dict) or launch.get("format") != "langchain-source-bundle-launch-v1"
            or launch.get("model") != body.get("model") or episode.get("model") != body.get("model")
            or type(launch.get("parent_request_count")) is not int or launch["parent_request_count"] != len(records)
            or type(launch.get("parent_tokens")) is not int or launch["parent_tokens"] != parent):
        raise ValueError("lc_bundle_front_episode_or_parent_binding")
    episode_archives, episode_sources = episode.get("source_archive"), episode.get("sources")
    proof_archives = {key.replace("\\", "/"): value for key, value in proof["source_archives"].items()}
    if not isinstance(episode_archives, dict) or not isinstance(episode_sources, dict) or not episode_archives:
        raise ValueError("lc_bundle_front_source_archive_required")
    if set(episode_archives) != set(episode_sources) or "app/agents/strict_transport.py" not in {key.replace("\\", "/") for key in episode_archives}:
        raise ValueError("lc_bundle_front_source_archive_required")
    for key, descriptor in episode_archives.items():
        frozen = proof_archives.get(key.replace("\\", "/"))
        if (not isinstance(descriptor, dict) or frozen is None or descriptor.get("sha256") != episode_sources[key]
                or descriptor.get("sha256") != frozen.get("sha256")):
            raise ValueError("lc_bundle_front_source_archive_changed")
        _bound(descriptor, raw=True)
    front, whole_limit, cap = (launch.get(key) for key in ("front_token_stop", "whole_token_stop", "max_requests"))
    incremental = proof.get("whole_chain_planning", {}).get("front_token_budget_semantics") == "new_front_tokens_excluding_paid_ancestors"
    incremental_keys = ("front_token_budget_semantics", "configured_new_front_token_budget", "cumulative_front_guard_threshold")
    if incremental:
        allowance, tail = launch.get("configured_new_front_token_budget"), launch.get("tail_reserved_tokens")
        expected_planning = ("native-whole-chain-paragraph-quality-budget-planning-v5" if proof.get("parent_token_pack") is True
            else "native-whole-chain-paragraph-quality-compact-planning-v4")
        if (proof.get("whole_chain_planning", {}).get("version") != expected_planning
                or proof.get("paragraph_quality_compact") is not True
                or type(allowance) is not int or allowance <= 0 or type(tail) is not int or tail <= 0
                or type(whole_limit) is not int or front != min(parent + allowance, whole_limit - tail)
                or launch.get("cumulative_front_guard_threshold") != front
                or any(launch.get(key) != report.get(key) or launch.get(key) != closed.get(key) for key in incremental_keys)):
            raise ValueError("lc_bundle_incremental_front_budget_changed")
    elif any(key in value for value in (launch, report, closed) for key in incremental_keys):
        raise ValueError("lc_bundle_legacy_front_budget_reinterpreted")
    if (any(type(value) is not int or value <= 0 for value in (front, whole_limit, cap))
            or front > whole_limit or episode.get("experimental_stop_threshold") != front
            or launch.get("front_max_requests") != 1 or launch.get("required_tail_requests") != 3
            or launch.get("front_guard_max_requests") != cap - 3
            or episode.get("max_requests") != cap - 3 or len(records) + 4 > cap):
        raise ValueError("lc_bundle_front_budget_policy")
    capacity, estimate, safety = body.get("max_tokens"), preflight.get("serialized_json_token_estimate"), preflight.get("safety_tokens")
    if (type(capacity) is not int or capacity <= 0 or type(estimate) is not int or estimate < 0
            or safety != 512 or preflight.get("operation") != "strict_selection"
            or preflight.get("wire") != entry["wire"] or preflight.get("known_tokens_before") != parent
            or preflight.get("reserved_output_tokens") != capacity or preflight.get("admitted") is not True
            or preflight.get("experimental_stop_threshold") != front or preflight.get("is_actual_usage") is not False
            or parent + estimate + capacity + safety > front):
        raise ValueError("lc_bundle_front_preflight_not_admitted")
    usage = visible.get("usage")
    total = _usage_total(usage)
    if (not isinstance(settlement, dict) or settlement.get("operation") != "strict_selection"
            or settlement.get("ticket_id") != 1 or settlement.get("actual_request_started") is not True
            or settlement.get("usage_known") is not True or settlement.get("outcome") != "success"
            or settlement.get("provider_http_status") != 200 or settlement.get("structure_repair") is not False
            or settlement.get("length_recovery") is not False or settlement.get("is_usage") is not True
            or settlement.get("known_tokens_before") != 0 or settlement.get("known_total_tokens_after") != total
            or settlement.get("credited_total_tokens") != total
            or settlement.get("stop_reason") != ("cost_guard_known_token_soft_threshold_reached" if parent + total == front else None)
            or settlement.get("requested_max_output_tokens") != capacity or settlement.get("reserved_max_output_tokens") != capacity
            or settlement.get("soft_total_token_threshold") != front - parent
            or any(settlement.get(key) != usage[key] for key in ("input_tokens", "output_tokens", "total_tokens"))
            or usage["output_tokens"] > capacity or parent + total > front or parent + total >= whole_limit):
        raise ValueError("lc_bundle_front_provider_usage_not_complete")
    guard_messages = [*body["messages"], {"role": "system", "content": _json({"tools": body["tools"], "tool_choice": body["tool_choice"]}, sort=True)}]
    guard_bytes = _json(guard_messages, sort=True).encode("utf-8")
    if (settlement.get("message_content_sha256") != sha256(guard_bytes).hexdigest()
            or settlement.get("context_sha256") != sha256(guard_bytes).hexdigest()
            or settlement.get("message_content_utf8_bytes") != len(guard_bytes)):
        raise ValueError("lc_bundle_front_guard_actual_message_binding")
    if (not isinstance(finished, dict) or finished.get("operation") != "strict_selection"
            or finished.get("request_started") is not True or finished.get("outcome") != "success"
            or finished.get("cache_hit") is not False or finished.get("safe_error_code") is not None
            or finished.get("known_total_tokens") != total or finished.get("known_episode_tokens_with_ancestors") != parent + total
            or finished.get("unknown_usage_requests") != 0):
        raise ValueError("lc_bundle_front_finish_not_successful")
    expected = {"new_request_count": 1, "new_tokens": total, "parent_tokens": parent, "whole_episode_tokens": parent + total}
    if (not isinstance(report, dict) or report.get("known_usage_complete") is not True
            or not isinstance(closed, dict) or closed.get("format") != "langchain-source-bundle-closed-receipt-v1"
            or closed.get("status") != "succeeded" or closed.get("episode_closed") is not True
            or closed.get("known_usage_complete") is not True or closed.get("front_requests") != requests
            or closed.get("failure_code") is not None
            or any(type(report.get(key)) is not int or report[key] != value
                   or type(closed.get(key)) is not int or closed[key] != value for key, value in expected.items())):
        raise ValueError("lc_bundle_front_closed_usage_mismatch")
    paid = closed.get("usage_records")
    if (not isinstance(paid, list) or len(paid) != 1 or paid[0].get("usage_source") != "provider"
            or paid[0].get("operation") != "strict_selection" or paid[0].get("model") != body["model"]
            or any(paid[0].get(key) != usage[key] for key in ("input_tokens", "output_tokens", "total_tokens"))):
        raise ValueError("lc_bundle_closed_original_usage_binding")
    if (launch.get("prepared_case") != proof["prepared_case"] or launch.get("front_episode") != proof["front_episode"]
            or launch.get("reserved_output_tokens") != capacity or launch.get("safety_tokens") != safety
            or type(launch.get("tail_reserved_tokens")) is not int or launch["tail_reserved_tokens"] <= 0
            or front > whole_limit - launch["tail_reserved_tokens"]):
        raise ValueError("lc_bundle_launch_source_and_tail_reservation")
    return entry, visible, launch, closed


def _codec_result(execute, raw_sha256, request):
    value = execute(request)
    if (value.get("format") != "lc-source-bundle-tool-response-v1" or value.get("status") != "ok"
            or value.get("case_sha256") != raw_sha256 or value.get("model_calls") != 0
            or value.get("publishable") is not False or not isinstance(value.get("result"), dict)):
        raise ValueError("lc_bundle_archived_business_replay_failed")
    return value["result"]


def replay(question, paragraphs, documents, response, excluded):
    if not isinstance(response, dict) or set(response) != {"artifact", "proof", "usage_report"}:
        raise ValueError("lc_bundle_response_contract")
    proof = response["proof"]
    if (not isinstance(proof, dict) or proof.get("format") != PROOF_FORMAT
            or proof.get("usage_report") != response["usage_report"] or proof.get("excluded_spans") != excluded
            or {"native_requests", "native_projections", "presence", "submit_request", "tool_submission"} & set(proof)):
        raise ValueError("lc_bundle_independent_proof_required")
    sources = _sources(proof.get("source_archives"))
    raw_bytes = _bound(proof["raw_case"], raw=True)
    case = _provider_json(raw_bytes)
    if (not isinstance(case, dict) or case.get("format") != "frozen-source-bundle-case-v1"
            or case.get("question") != _question(question) or case.get("documents") != documents or case.get("paragraphs") != paragraphs
            or _bound(proof["documents"]) != documents or _bound(proof["paragraphs"]) != paragraphs):
        raise ValueError("lc_bundle_complete_original_case_binding")
    with _archived_codec(sources, raw_bytes, proof["raw_case"]["sha256"]) as execute:
        rank_options = {}
        selection_only = proof.get("selection_only")
        if selection_only is not None:
            if selection_only is not True:
                raise ValueError("lc_bundle_selection_mode_changed")
            rank_options["selection_only"] = True
        graded_selection = proof.get("graded_selection")
        if graded_selection is not None:
            if graded_selection is not True or selection_only is not True:
                raise ValueError("lc_bundle_graded_selection_mode_changed")
            rank_options["graded_selection"] = True
        compact_analyses = proof.get("compact_analyses")
        if compact_analyses is not None and compact_analyses is not True:
            raise ValueError("lc_bundle_compact_analyses_mode_changed")
        context_passages = proof.get("context_passages")
        if context_passages is not None and (context_passages is not True
                or graded_selection is not True or compact_analyses is not True
                or proof.get("source_bundle_protocol") != "document-source-context-v1"):
            raise ValueError("lc_bundle_context_mode_changed")
        if context_passages is None and any(key in proof for key in (
                "source_bundle_protocol", "context_plan", "context_window_scores", "context_window_scores_sha256")):
            raise ValueError("lc_bundle_context_mode_required")
        for feature in ("complete_grading", "hybrid_retrieval"):
            mode = proof.get(feature)
            if mode is not None:
                if mode is not True or context_passages is not True or not isinstance(proof.get(feature + "_policy"), dict):
                    raise ValueError("lc_bundle_context_feature_mode_changed")
                rank_options[feature] = True
            elif feature + "_policy" in proof:
                raise ValueError("lc_bundle_context_feature_mode_required")
        paragraph_quality = proof.get("paragraph_quality")
        if paragraph_quality is not None:
            if (paragraph_quality is not True or any(value is True for value in
                    (selection_only, graded_selection, context_passages, proof.get("complete_grading"), proof.get("hybrid_retrieval")))
                    or not isinstance(proof.get("paragraph_quality_policy"), dict) or compact_analyses is not True):
                raise ValueError("lc_bundle_paragraph_quality_distinct_protocol_required")
            rank_options["paragraph_quality"] = True
        elif "paragraph_quality_policy" in proof:
            raise ValueError("lc_bundle_paragraph_quality_mode_required")
        paragraph_quality_compact = proof.get("paragraph_quality_compact")
        if paragraph_quality_compact is not None:
            if (paragraph_quality_compact is not True or paragraph_quality is not True
                    or not isinstance(proof.get("paragraph_quality_compact_policy"), dict)):
                raise ValueError("lc_bundle_paragraph_quality_compact_mode_required")
            rank_options["paragraph_quality_compact"] = True
        elif "paragraph_quality_compact_policy" in proof:
            raise ValueError("lc_bundle_paragraph_quality_compact_mode_required")
        parent_token_pack = proof.get("parent_token_pack")
        if parent_token_pack is not None:
            if (parent_token_pack is not True or paragraph_quality_compact is not True
                    or not isinstance(proof.get("parent_token_pack_policy"), dict)
                    or proof.get("local_parent_rank") is not None):
                raise ValueError("lc_bundle_parent_token_pack_mode_required")
            rank_options["parent_token_pack"] = True
        elif "parent_token_pack_policy" in proof:
            raise ValueError("lc_bundle_parent_token_pack_mode_required")
        rank = proof.get("local_parent_rank")
        if rank is not None:
            if (not isinstance(rank, dict) or set(rank) != {"plan", "plan_sha256", "proof", "config"}
                    or _RANK_ENTRY not in sources or "app/domain/local_parent_rank.py" not in sources):
                raise ValueError("lc_bundle_local_rank_proof_required")
            plan = _bound(rank["plan"])
            config = rank["config"]
            if (not isinstance(config, dict) or config.get("enabled") is not True
                    or config.get("policy") != "full-question-literal-facets-body-rrf60-v1"
                    or config.get("candidate_limit") != 32 or config.get("per_document_soft_cap") != 3
                    or rank["plan_sha256"] != digest(plan)):
                raise ValueError("lc_bundle_local_rank_policy_changed")
            plan_options = {"rank_plan": plan, "rank_plan_sha256": rank["plan_sha256"]}
            rank_options.update(plan_options)
            _codec_result(execute, proof["raw_case"]["sha256"], {"operation": "verify_rank", **plan_options,
                "local_rank_proof": rank["proof"], "expected_model_manifest_sha256": config.get("model_manifest_sha256")})
        if context_passages is True:
            if rank is None:
                raise ValueError("lc_bundle_context_verified_rank_required")
            rank_options.update(context_passages=True, local_rank_proof=rank["proof"],
                expected_model_manifest_sha256=config["model_manifest_sha256"])
        prepared = _codec_result(execute, proof["raw_case"]["sha256"],
            {"operation": "prepare", "excluded_spans": excluded, **rank_options})
        if _bound(proof["prepared_case"]) != prepared:
            raise ValueError("lc_bundle_prepared_source_case_changed")
        if (proof.get("paragraph_quality", False) is not prepared.get("paragraph_quality", False)
                or paragraph_quality is True and proof["paragraph_quality_policy"] != prepared.get("paragraph_quality_policy")):
            raise ValueError("lc_bundle_paragraph_quality_policy_changed")
        if (proof.get("paragraph_quality_compact", False) is not prepared.get("paragraph_quality_compact", False)
                or paragraph_quality_compact is True and proof["paragraph_quality_compact_policy"] != prepared.get("paragraph_quality_compact_policy")):
            raise ValueError("lc_bundle_paragraph_quality_compact_policy_changed")
        if (proof.get("parent_token_pack", False) is not prepared.get("parent_token_pack", False)
                or parent_token_pack is True and proof["parent_token_pack_policy"] != prepared.get("parent_token_pack_policy")):
            raise ValueError("lc_bundle_parent_token_pack_policy_changed")
        for feature in ("complete_grading", "hybrid_retrieval"):
            if proof.get(feature, False) is not prepared.get(feature, False):
                raise ValueError("lc_bundle_prepared_context_feature_changed")
        if proof.get("complete_grading") is True and proof["complete_grading_policy"] != prepared.get("complete_grading_policy"):
            raise ValueError("lc_bundle_complete_grading_policy_changed")
        if context_passages is True:
            context_plan = _bound(proof["context_plan"])
            window_scores = _bound(proof["context_window_scores"])
            binding = prepared["source_binding"]
            if (digest(context_plan) != binding.get("context_plan_sha256")
                    or digest(window_scores) != proof.get("context_window_scores_sha256")
                    or digest(window_scores) != binding.get("full_window_scores_digest")):
                raise ValueError("lc_bundle_context_plan_or_windows_changed")
        requests = proof.get("front_requests")
        if not isinstance(requests, list) or len(requests) != 1:
            raise ValueError("lc_bundle_exactly_one_front_request_required")
        entry = requests[0]
        body = _provider_json(_bound(entry["actual_wire"], raw=True))
        if not isinstance(body, dict) or _bound(entry["wire"]) != body:
            raise ValueError("lc_bundle_actual_wire_binding")
        if (body.get("messages") != prepared.get("messages") or not isinstance(body.get("messages"), list)
                or len(body["messages"]) != 2 or [r.get("role") for r in body["messages"]] != ["system", "user"]
                or any(not isinstance(r.get("content"), str) for r in body["messages"])
                or body["messages"][1]["content"] != _json(prepared["model_input"], sort=True)
                or body.get("tools") != [prepared.get("wire_tool")]
                or body.get("tool_choice") != {"type": "function", "function": {"name": prepared["function_name"]}}
                or prepared["wire_tool"].get("type") != "function" or prepared["wire_tool"]["function"].get("strict") is not True
                or prepared["wire_tool"]["function"].get("name") != prepared["function_name"]
                or body.get("parallel_tool_calls") is not False or body.get("thinking") != {"type": "disabled"}
                or body.get("stream") is True or body.get("n", 1) != 1 or body.get("temperature") != 0
                or "reasoning_effort" in body or "effort" in body.get("output_config", {})):
            raise ValueError("lc_bundle_actual_full_source_or_strict_tool_missing")
        entry, visible, launch, closed = _front_usage(proof, body)
        if (launch.get("selection_only") != selection_only or launch.get("graded_selection") != graded_selection
                or launch.get("compact_analyses") != compact_analyses
                or launch.get("issue_frame") != proof.get("issue_frame")
                or "issue_frame" in proof and proof["issue_frame"] is not True
                or launch.get("context_passages") != context_passages
                or selection_only is True and body.get("max_tokens") != (2048 if graded_selection is True else 1024)
                or paragraph_quality is True and body.get("max_tokens") != 8192):
            raise ValueError("lc_bundle_selection_mode_changed")
        if context_passages is True and any(launch.get(key) != proof.get(key) for key in (
                "source_bundle_protocol", "context_plan", "context_window_scores", "context_window_scores_sha256")):
            raise ValueError("lc_bundle_context_launch_changed")
        if any(launch.get(key) != proof.get(key) for feature in ("complete_grading", "hybrid_retrieval", "paragraph_quality", "paragraph_quality_compact", "parent_token_pack")
               for key in (feature, feature + "_policy")):
            raise ValueError("lc_bundle_context_feature_launch_changed")
        planning = proof.get("whole_chain_planning")
        if (launch.get("whole_chain_planning") != planning
                or (planning is not None) != ((graded_selection is True or paragraph_quality is True) and compact_analyses is True)
                or planning is not None and (not isinstance(planning, dict)
                    or planning.get("future_analysis_allowance_is_not_a_proven_global_bound") is not True)):
            raise ValueError("lc_bundle_whole_chain_planning_mode_changed")
        if planning is not None:
            initial_binding = proof.get("initial_budget_plan")
            if not isinstance(initial_binding, dict) or launch.get("initial_budget_plan") != initial_binding:
                raise ValueError("lc_bundle_initial_budget_plan_missing")
            initial = _bound(initial_binding)
            if (not isinstance(initial, dict) or initial.get("format") != planning.get("version")
                    or initial.get("policy") != planning or initial.get("phase") != "before_first_paid_request"
                    or initial.get("admitted") is not True or initial.get("is_actual_usage") is not False
                    or initial.get("is_guaranteed_global_upper_bound") is not False
                    or initial.get("known_tokens") != launch.get("parent_tokens")
                    or initial.get("known_calls") != launch.get("parent_request_count")
                    or initial.get("threshold") != launch.get("whole_token_stop")
                    or initial.get("max_calls") != launch.get("max_requests")
                    or initial.get("front_request", {}).get("native_wire_sha256") != entry["wire"]["sha256"]
                    or initial.get("front_request", {}).get("reserved_output_tokens") != body.get("max_tokens")):
                raise ValueError("lc_bundle_initial_budget_plan_binding_changed")
        if graded_selection is True:
            if (proof.get("graded_selection_policy") != prepared.get("graded_selection_policy")
                    or launch.get("graded_selection_policy") != prepared.get("graded_selection_policy")
                    or proof.get("selection_output_tokens") != 2048 or launch.get("selection_output_tokens") != 2048):
                raise ValueError("lc_bundle_graded_selection_policy_changed")
        elif paragraph_quality is True:
            if ("graded_selection_policy" in proof or "graded_selection_policy" in launch
                    or proof.get("selection_output_tokens") != 8192 or launch.get("selection_output_tokens") != 8192):
                raise ValueError("lc_bundle_paragraph_quality_output_capacity_changed")
        elif any(key in proof or key in launch for key in ("graded_selection_policy", "selection_output_tokens")):
            raise ValueError("lc_bundle_graded_selection_mode_changed")
        if "syntax-recovery" in entry:
            raise ValueError("lc_bundle_original_arguments_not_strict_json")
        calls = visible.get("tool_calls_wire")
        if (visible.get("finish_reason") not in {"tool_calls", "stop"} or not isinstance(calls, list) or len(calls) != 1
                or calls[0].get("type") != "function" or calls[0].get("function", {}).get("name") != prepared["function_name"]
                or not isinstance(calls[0]["function"].get("arguments"), str)):
            raise ValueError("lc_bundle_unique_original_function_required")
        output = _provider_json(calls[0]["function"]["arguments"])
        parsed = visible.get("tool_calls")
        if (not isinstance(output, dict) or _bound(proof["model_output"]) != output or not isinstance(parsed, list) or len(parsed) != 1
                or parsed[0].get("id") != calls[0].get("id") or parsed[0].get("name") != prepared["function_name"]
                or parsed[0].get("args") != output):
            raise ValueError("lc_bundle_original_model_output_changed")
        if launch.get("local_parent_rank") != ({"plan_sha256": rank["plan_sha256"], "proof": rank["proof"],
                "config": rank["config"]} if rank is not None else None):
            raise ValueError("lc_bundle_local_rank_launch_changed")
        rebuilt = _codec_result(execute, proof["raw_case"]["sha256"],
            {"operation": "compile", "output": output, "excluded_spans": excluded, **rank_options})
    artifact, scope = rebuilt.get("artifact"), rebuilt.get("scope")
    if (not isinstance(artifact, dict) or not isinstance(scope, dict) or _bound(proof["artifact"]) != artifact
            or artifact != response["artifact"] or _bound(proof["scope"]) != scope
            or scope.get("binding") != prepared.get("source_binding") or scope.get("native_model_delivery_verified") is not False
            or closed.get("raw_case_sha256") != proof["raw_case"]["sha256"]
            or closed.get("artifact_sha256") != proof["artifact"]["sha256"] or closed.get("source_binding") != scope["binding"]
            or launch.get("raw_case_sha256") != proof["raw_case"]["sha256"] or launch.get("source_binding") != scope["binding"]):
        raise ValueError("lc_bundle_closed_source_or_artifact_binding")
    return artifact, scope
