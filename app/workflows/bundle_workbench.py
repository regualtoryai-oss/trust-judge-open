"""Replay source-only bundle selection before the independent graph tail.

No model, reference labels, provider fallback, or framework loop lives here.
Frozen tool/source files and the actual native Messages requests are checked
independently of the provider's summaries. Source identity does not establish
annotation semantics or convert proposed S/C/L roles into final relations.

Matching codecs use the local business entry. Changed codecs replay through
the same SHA-bound CLI entry in an isolated, temporary archived application
tree. Missing archived application dependencies fail closed; they cannot fall
back to the current application or change its imported modules.
"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

from app.domain.material_eligibility import apply_workflow_material_eligibility
from app.domain.numeric_literals import unsupported_numeric_claims
from app.domain.snapshot import SOURCE_BUNDLE_PROTOCOL, SOURCE_CONTEXT_PROTOCOL, SOURCE_DOCUMENT_PROTOCOLS, source_core_fields
from app.domain.source_bundles import SOURCE_BUNDLE_CORE_FIELDS, validate_source_bundle
from app.domain.source_metadata import source_credibility_value


PROOF_FORMAT = "dsh-source-bundle-workbench-proof-v1"
PROMPT_VERSION = "trusted-analysis-dsh-source-bundles-v1"
ARGUMENT_KEYS = {"时间", "主体", "地点", "内容", "影响", "数值口径"}


def _time_literal_grounded(row):
    """Validate compiler-joined time spans without inventing a continuous quote.

    Paragraph field extraction explicitly serializes multiple individually
    located literals with a semicolon. Its replayed CP audit is authoritative;
    older protocols without that audit keep their single-literal requirement.
    """
    value, quote = row["argument"]["时间"], row["quote"]
    if not value:
        return True
    fields = row.get("field_spans")
    if not isinstance(fields, dict) or not isinstance(fields.get("argument"), dict):
        return value in quote
    spans = fields["argument"].get("时间")
    if not isinstance(spans, list) or not spans:
        return False
    texts = []
    for span in spans:
        if not isinstance(span, dict):
            return False
        text, start, end = span.get("text"), span.get("start_cp"), span.get("end_cp")
        if (not isinstance(text, str) or not text or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(quote) or quote[start:end] != text):
            return False
        texts.append(text)
    return value == "；".join(texts)
REPLAY_SOURCES = ("scripts/source_bundle_tool.py", "app/domain/source_bundles.py",
                  "app/domain/source_bundle_tools.py", "app/domain/numeric_literals.py")
_ARCHIVED_ENTRY = "scripts/source_bundle_tool.py"
_APP_SOURCE = re.compile(r"app/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py\Z")


def _json(value, *, sort=False):
    return json.dumps(value, ensure_ascii=False, sort_keys=sort, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(_json(value, sort=True).encode("utf-8")).hexdigest()


def _bound(descriptor, *, lines=False, raw=False):
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), str) or not Path(descriptor["file"]).is_absolute():
        raise ValueError("source_bundle_proof_bound_file_required")
    data = Path(descriptor["file"]).read_bytes()
    if sha256(data).hexdigest() != descriptor.get("sha256"):
        raise ValueError("source_bundle_proof_file_sha256_changed")
    if raw:
        return data
    if lines:
        return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return json.loads(data.decode("utf-8"))


def _archive_sources(archives):
    """Validate every byte binding; copy only application Python and one entry."""
    if not isinstance(archives, dict):
        raise ValueError("source_bundle_codec_archives_required")
    sources = {}
    for relative, descriptor in archives.items():
        if not isinstance(relative, str):
            raise ValueError("source_bundle_archive_relative_path_required")
        name = relative.replace("\\", "/")
        path = PurePosixPath(name)
        if (path.is_absolute() or any(part in {"", ".", ".."} or ":" in part for part in name.split("/"))
                or name in sources):
            raise ValueError("source_bundle_archive_relative_path_required")
        if name.startswith("app/") and not _APP_SOURCE.fullmatch(name):
            raise ValueError("source_bundle_archive_python_source_required")
        sources[name] = _bound(descriptor, raw=True)
    if not set(REPLAY_SOURCES) <= set(sources):
        raise ValueError("source_bundle_codec_archives_required")
    return sources


@contextmanager
def _replay_executor(sources, raw_bytes, raw_sha256):
    """Run only the frozen business CLI, never a generic Python expression.

    A regular archived ``app`` package shadows the workspace package under
    Python isolated mode. Missing transitive modules therefore fail import
    instead of being filled from the live checkout. Other archived scripts,
    integrations, sessions, credentials and original runtime paths are never
    copied into this execution root. No parent-process imports are replaced.
    """
    root = Path(__file__).resolve().parents[2]
    replay_sources = {name: data for name, data in sources.items()
                      if name == _ARCHIVED_ENTRY or _APP_SOURCE.fullmatch(name)}
    changed = any(not (root / PurePosixPath(name)).is_file()
                  or sha256((root / PurePosixPath(name)).read_bytes()).digest() != sha256(data).digest()
                  for name, data in replay_sources.items())
    if not changed:
        from scripts.source_bundle_tool import execute
        yield lambda request: execute(raw_bytes, raw_sha256, request)
        return
    if "app/__init__.py" not in replay_sources:
        raise ValueError("source_bundle_archived_application_required")
    with TemporaryDirectory(prefix="trust-judge-source-replay-") as directory:
        archived_root = Path(directory).resolve()
        for relative, data in replay_sources.items():
            target = archived_root / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        case_path = archived_root / "raw-case.json"
        case_path.write_bytes(raw_bytes)
        entry = archived_root / _ARCHIVED_ENTRY
        env = {key: os.environ[key] for key in ("SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP") if key in os.environ}

        def execute_archived(request):
            try:
                completed = subprocess.run([sys.executable, "-I", "-B", str(entry), str(case_path), raw_sha256],
                    input=_json(request).encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=archived_root, env=env, timeout=15, check=False)
            except (OSError, subprocess.TimeoutExpired):
                raise ValueError("source_bundle_archived_replay_unavailable") from None
            if completed.returncode != 0 or len(completed.stdout) > 10_000_000:
                raise ValueError("source_bundle_archived_replay_failed")
            try:
                output = json.loads(completed.stdout.decode("utf-8"))
            except (ValueError, UnicodeError):
                raise ValueError("source_bundle_archived_replay_invalid_output") from None
            if not isinstance(output, dict):
                raise ValueError("source_bundle_archived_replay_invalid_output")
            return output

        yield execute_archived


def _question(question):
    return {"question_id": question.get("raw_question_id", question.get("question_id")), "question": question["question"]}


def _exclusions(paragraphs, excluded):
    by_id, projected = {p["id"]: p for p in paragraphs}, []
    for item in excluded:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise ValueError("source_bundle_exclusion_shape")
        pid, start, end, quote = item
        p = by_id.get(pid)
        if (p is None or type(start) is not int or type(end) is not int or not isinstance(quote, str)
                or not 0 <= start < end <= len(p["text"]) or p["text"][start:end] != quote):
            raise ValueError("source_bundle_exclusion_original_span")
        projected.append({"document_id": p["document_id"], "start_cp": p["document_start_cp"] + start,
                          "end_cp": p["document_start_cp"] + end})
    return sorted(map(list, excluded)), projected


def _usage(proof):
    report, receipt = proof["usage_report"], _bound(proof["closed_receipt"])
    if (not isinstance(report, dict) or not isinstance(receipt, dict)
            or receipt.get("format") != "dsh-source-bundle-closed-receipt-v1" or receipt.get("status") != "succeeded"
            or receipt.get("episode_closed") is not True or report.get("known_usage_complete") is not True):
        raise ValueError("source_bundle_episode_not_closed")
    parent = 0
    if proof.get("previous_usage") is not None:
        previous = _bound(proof["previous_usage"])
        if not isinstance(previous, dict) or not isinstance(previous.get("records"), list):
            raise ValueError("source_bundle_parent_usage_required")
        for record in previous["records"]:
            if record.get("usage_source") != "provider":
                raise ValueError("source_bundle_unknown_parent_usage")
            parent += _usage_total(record)
    events = _bound(proof["usage_ledger"], lines=True)
    starts, settlements = {}, {}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("source_bundle_invalid_usage_event")
        status, sequence = event.get("status"), event.get("request_sequence")
        if status == "usage_unknown":
            raise ValueError("source_bundle_unknown_actual_usage")
        if status not in ("request_started", "usage_settled"):
            continue
        if type(sequence) is not int or sequence < 1:
            raise ValueError("source_bundle_usage_sequence_required")
        target = starts if status == "request_started" else settlements
        if sequence in target or (status == "usage_settled" and sequence not in starts):
            raise ValueError("source_bundle_duplicate_or_unstarted_usage")
        target[sequence] = event
    native = proof["native_requests"]
    if not isinstance(native, list) or not native or set(starts) != set(range(1, len(native) + 1)) or set(starts) != set(settlements):
        raise ValueError("source_bundle_incomplete_native_usage")
    running = parent
    for sequence, descriptor in enumerate(native, 1):
        begun, settled = starts[sequence], settlements[sequence]
        body = _bound(descriptor)
        if (not isinstance(body, dict) or body.get("max_tokens") != begun.get("reserved_output_tokens") or body.get("thinking", {}).get("type") != "disabled"
                or "reasoning_effort" in body or "effort" in body.get("output_config", {})):
            raise ValueError("source_bundle_native_budget_or_thinking_policy")
        if begun.get("wire_sha256") != descriptor.get("sha256") or begun.get("known_episode_tokens_before") != running:
            raise ValueError("source_bundle_native_usage_binding")
        running += _usage_total(settled.get("provider_usage"))
        if (settled.get("known_episode_tokens_after") != running or settled.get("response_completed") is not True
                or settled.get("halted") is not False or settled.get("unknown_usage_requests") != 0
                or type(begun.get("reserved_output_tokens")) is not int or begun["reserved_output_tokens"] < 1
                or settled["provider_usage"]["output_tokens"] > begun["reserved_output_tokens"]):
            raise ValueError("source_bundle_episode_usage_accumulator")
    expected = {"new_request_count": len(native), "new_tokens": running - parent, "parent_tokens": parent, "whole_episode_tokens": running}
    if any(type(report.get(k)) is not int or report[k] != value or receipt.get(k) != value for k, value in expected.items()):
        raise ValueError("source_bundle_usage_summary_mismatch")
    if receipt.get("native_requests") != native:
        raise ValueError("source_bundle_closed_native_files_mismatch")
    return receipt


def _usage_total(record):
    if not isinstance(record, dict) or any(type(record.get(k)) is not int or record[k] < 0 for k in ("input_tokens", "output_tokens", "total_tokens")):
        raise ValueError("source_bundle_complete_provider_usage_required")
    if record["input_tokens"] + record["output_tokens"] != record["total_tokens"]:
        raise ValueError("source_bundle_provider_usage_arithmetic")
    return record["total_tokens"]


def _text_blocks(content):
    if isinstance(content, str):
        return [content]
    return [row["text"] for row in content if isinstance(row, dict) and row.get("type") == "text" and isinstance(row.get("text"), str)] if isinstance(content, list) else []


def _native_arguments(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("source_bundle_projection_duplicate_native_key")
            result[key] = value
        return result
    if not isinstance(raw, str) or len(raw) > 1_000_000:
        raise ValueError("source_bundle_projection_raw_arguments_required")
    try:
        result = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("source_bundle_projection_nonfinite_arguments")))
    except (ValueError, TypeError) as error:
        raise ValueError("source_bundle_projection_invalid_native_arguments") from error
    if not isinstance(result, dict):
        raise ValueError("source_bundle_projection_object_arguments_required")
    return result


def _handle_manifest(case, navigation, plan, raw_sha256):
    by_id = {p["id"]: p for p in case["paragraphs"]}
    paragraphs, refs = [], {}
    for doc in case["documents"]:
        for item in doc["paragraph_manifest"]:
            row = by_id[item["id"]]
            if (row["document_id"] != doc["id"] or row["document_start_cp"] != item["start_cp"]
                    or row["document_end_cp"] != item["end_cp"] or item["id"] in refs):
                raise ValueError("source_bundle_projection_original_manifest_mismatch")
            ref = "P" + str(len(paragraphs) + 1)
            refs[item["id"]] = ref
            paragraphs.append({"ref": ref, "paragraph_id": item["id"], "document_id": doc["id"],
                "start_cp": item["start_cp"], "end_cp": item["end_cp"]})
    if len(refs) != len(by_id) or navigation.get("tool") != "search_source_bundles" or plan.get("tool") != "plan_source_bundles" or plan.get("status") != "planned":
        raise ValueError("source_bundle_projection_complete_plan_required")
    nav_ids = [r["paragraph_id"] for r in navigation["results"]]
    if len(set(nav_ids)) != len(nav_ids) or len(nav_ids) > 40:
        raise ValueError("source_bundle_projection_navigation_identity")
    bundles, covered, seen = [], set(), set()
    for bundle in plan["bundles"]:
        anchors, ids = bundle["anchor_paragraph_ids"], bundle["source_paragraph_ids"]
        if (not anchors or not set(anchors) <= set(nav_ids) or bundle["bundle_id"] in seen or not 1 <= len(ids) <= 5):
            raise ValueError("source_bundle_projection_illegal_plan_range")
        seen.add(bundle["bundle_id"]); covered.update(anchors)
        paragraph_refs = [refs[pid] for pid in ids]
        bundles.append({"bundle_ref": "B" + str(len(bundles) + 1), "anchor_paragraph_id": anchors[0],
            "start_ref": paragraph_refs[0], "end_ref": paragraph_refs[-1], "paragraph_refs": paragraph_refs,
            "canonical_bundle": deepcopy(bundle)})
    if covered != set(nav_ids) or len(bundles) > 40:
        raise ValueError("source_bundle_projection_incomplete_navigation_plan")
    return {"format": "source-bundle-native-handles-v1", "case_sha256": raw_sha256,
        "source_binding": deepcopy(plan["source_binding"]), "backend_navigation_sha256": digest(navigation),
        "window_policy": {"before": 1, "after": 3, "max_paragraphs": 5}, "paragraphs": paragraphs, "bundles": bundles}


def _handle_read_arguments(handles, args):
    if not isinstance(args, dict) or set(args) != {"bundle_refs"} or not isinstance(args["bundle_refs"], dict):
        raise ValueError("source_bundle_projection_read_shape")
    selected, by_ref = args["bundle_refs"], {r["bundle_ref"]: r for r in handles["bundles"]}
    if not 1 <= len(selected) <= 12 or any(ref not in by_ref or value is not True for ref, value in selected.items()):
        raise ValueError("source_bundle_projection_unknown_or_oversize_read")
    return {"anchor_paragraph_ids": [by_ref[ref]["anchor_paragraph_id"] for ref in selected], "before": 1, "after": 3}


def _handle_read_view(handles, result, args):
    _handle_read_arguments(handles, args)
    plans = {r["bundle_ref"]: r for r in handles["bundles"]}
    by_id = {r["paragraph_id"]: r for r in handles["paragraphs"]}
    selected = [plans[ref] for ref in args["bundle_refs"]]
    if (result.get("tool") != "read_source_bundles" or result.get("status") != "completed"
            or result.get("complete_original_slices") is not True or not isinstance(result.get("delivery"), dict)
            or len(result.get("bundles", [])) != len(selected)):
        raise ValueError("source_bundle_projection_complete_read_required")
    views = []
    for bundle, planned in zip(result["bundles"], selected):
        if any(bundle.get(k) != planned["canonical_bundle"].get(k) for k in SOURCE_BUNDLE_CORE_FIELDS):
            raise ValueError("source_bundle_projection_read_source_changed")
        views.append({"bundle_ref": planned["bundle_ref"], "start_ref": planned["start_ref"], "end_ref": planned["end_ref"],
            "paragraph_refs": deepcopy(planned["paragraph_refs"]),
            "paragraph_spans": [[by_id[pid]["ref"], by_id[pid]["start_cp"] - bundle["start_cp"],
                by_id[pid]["end_cp"] - bundle["start_cp"]] for pid in bundle["source_paragraph_ids"]], "quote": bundle["quote"]})
    delivery = result["delivery"]
    return {"format": "source-bundle-native-read-v1", "tool": "read_source_bundles", "status": "completed",
        "complete_original_slices": True, "bundles": views,
        "delivery": {"read_call_index": delivery["read_call_index"], "complete_delivery_index": delivery["complete_delivery_index"],
            "new_paragraph_refs": [by_id[pid]["ref"] for pid in delivery["new_original_paragraph_ids"]],
            "already_read_paragraph_refs": [by_id[pid]["ref"] for pid in delivery["already_read_original_paragraph_ids"]],
            "full_original_source_bytes_returned": True}, "semantic_verified": False, "native_model_delivery_verified": False}


def _handle_submit_arguments(handles, reads, args):
    by_id, by_ref = ({r["paragraph_id"]: r for r in handles["paragraphs"]}, {r["ref"]: r for r in handles["paragraphs"]})
    allowed = {}
    for read in reads:
        for bundle in read["result"]["bundles"]:
            refs = [by_id[pid]["ref"] for pid in bundle["source_paragraph_ids"]]
            for i, start in enumerate(refs):
                allowed.setdefault(start, set()).update(refs[i:])
    if not isinstance(args, dict) or set(args) != {"selected"} or not isinstance(args["selected"], dict) or len(args["selected"]) > 30:
        raise ValueError("source_bundle_projection_selected_object_required")
    selected = []
    for start, row in args["selected"].items():
        if (not isinstance(row, dict) or set(row) != {"end_ref", "fact_des", "argument", "entities", "proposed_roles"}
                or row["end_ref"] not in allowed.get(start, set())):
            raise ValueError("source_bundle_projection_unread_or_cross_bundle_range")
        selected.append({"start_paragraph_id": by_ref[start]["paragraph_id"],
            "end_paragraph_id": by_ref[row["end_ref"]]["paragraph_id"],
            **deepcopy({key: row[key] for key in ("fact_des", "argument", "entities", "proposed_roles")})})
    return {"selected": selected}


def _native_presence_projected(body, presence, reads, question_data, *, final, handles, handles_sha256, projections):
    if (not isinstance(body, dict) or not isinstance(presence, dict) or not isinstance(body.get("system"), str)
            or not isinstance(body.get("messages"), list) or not isinstance(body.get("tools"), list)
            or body["system"].count(question_data) != 1 or presence.get("format") != "dsh-source-bundle-native-presence-v2"
            or presence.get("handles_sha256") != handles_sha256 or presence.get("semantic_verified") is not False
            or presence.get("publishable") is not False or presence.get("source_bytes_present_in_current_request") is not True
            or presence.get("native_messages_sha256") != sha256(_json(body["messages"]).encode()).hexdigest()):
        raise ValueError("source_bundle_native_projection_question_or_messages_missing")
    names = [t.get("name") for t in body["tools"]]
    if len(set(names)) != len(names) or not set(names) <= {"read_source_bundles", "submit_bundle_evidence"} or (final and "submit_bundle_evidence" not in names):
        raise ValueError("source_bundle_native_projection_tools_changed")
    calls = [b for m in body["messages"] if m.get("role") == "assistant" for b in m.get("content", [])
             if isinstance(m.get("content"), list) and isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "read_source_bundles"]
    results = [b for m in body["messages"] if m.get("role") == "user" for b in m.get("content", [])
               if isinstance(m.get("content"), list) and isinstance(b, dict) and b.get("type") == "tool_result"]
    all_text = [body["system"], *[s for m in body["messages"] for s in _text_blocks(m.get("content"))],
                *[s for r in results for s in _text_blocks(r.get("content"))]]
    receipts, consumed, use_ids = presence.get("read_receipts"), set(), set()
    if not isinstance(receipts, list):
        raise ValueError("source_bundle_native_projection_receipts_required")
    for receipt in receipts:
        use_id = receipt.get("tool_use_id")
        candidates = [p for p in projections if p.get("operation") == "read" and p.get("tool_use_id") == use_id]
        native_calls, native_results = [c for c in calls if c.get("id") == use_id], [r for r in results if r.get("tool_use_id") == use_id]
        if len(candidates) != 1 or len(native_calls) != 1 or len(native_results) != 1 or native_results[0].get("is_error") is True:
            raise ValueError("source_bundle_native_projection_call_or_result_missing")
        projection = candidates[0]
        args = _native_arguments(projection.get("raw_native_arguments"))
        if args != projection.get("native_arguments") or args != native_calls[0].get("input") or projection.get("handles_sha256") != handles_sha256:
            raise ValueError("source_bundle_native_projection_arguments_changed")
        business = _handle_read_arguments(handles, args)
        matches = [(i, r) for i, r in enumerate(reads) if r["arguments"] == business and r["result"] == projection.get("canonical_result")]
        if len(matches) != 1 or matches[0][0] in consumed or use_id in use_ids or business != projection.get("business_arguments"):
            raise ValueError("source_bundle_native_projection_unknown_or_repeated_read")
        index, read = matches[0]
        expected_view = _handle_read_view(handles, read["result"], args)
        rendered = _json(expected_view, sort=True)
        canonical_sha, native_sha = digest(read["result"]), sha256(rendered.encode()).hexdigest()
        if (projection.get("canonical_result_sha256") != canonical_sha or projection.get("native_result") != expected_view
                or projection.get("native_read_view_sha256") != native_sha
                or _text_blocks(native_results[0].get("content")) != [rendered] or sum(t.count(rendered) for t in all_text) != 1):
            raise ValueError("source_bundle_native_projection_complete_source_missing")
        expected_receipt = {"tool_use_id": use_id, "canonical_read_result_sha256": canonical_sha,
            "native_read_view_sha256": native_sha, "handles_sha256": handles_sha256,
            "bundle_ids": [b["bundle_id"] for b in read["result"]["bundles"]],
            "source_bundle_set_sha256": read["result"]["source_bundle_set_sha256"], "complete_original_slices_present": True}
        if receipt != expected_receipt:
            raise ValueError("source_bundle_native_projection_receipt_changed")
        consumed.add(index); use_ids.add(use_id)
    successful = {c.get("id") for c in calls if any(r.get("tool_use_id") == c.get("id") and r.get("is_error") is not True for r in results)}
    if successful != use_ids or (final and consumed != set(range(len(reads)))):
        raise ValueError("source_bundle_native_projection_untracked_or_missing_read")


def _native_presence(body, presence, reads, question_data, *, final, handles=None, handles_sha256=None, projections=None):
    if handles is not None:
        return _native_presence_projected(body, presence, reads, question_data, final=final, handles=handles,
            handles_sha256=handles_sha256, projections=projections)
    if (not isinstance(body, dict) or not isinstance(presence, dict) or not isinstance(body.get("system"), str) or not isinstance(body.get("messages"), list)
            or not isinstance(body.get("tools"), list) or body["system"].count(question_data) != 1
            or presence.get("format") != "dsh-source-bundle-native-presence-v1"
            or presence.get("semantic_verified") is not False or presence.get("publishable") is not False
            or presence.get("source_bytes_present_in_current_request") is not True
            or presence.get("native_messages_sha256") != sha256(_json(body["messages"]).encode()).hexdigest()):
        raise ValueError("source_bundle_native_question_or_messages_missing")
    tool_names = [t.get("name") for t in body["tools"]]
    if len(set(tool_names)) != len(tool_names) or not set(tool_names) <= {"search_source_bundles", "read_source_bundles", "submit_bundle_evidence"} or (final and "submit_bundle_evidence" not in tool_names):
        raise ValueError("source_bundle_native_tools_changed")
    calls = [block for m in body["messages"] if m.get("role") == "assistant" for block in (m.get("content") if isinstance(m.get("content"), list) else [])
             if block.get("type") == "tool_use" and block.get("name") == "read_source_bundles"]
    results = [block for m in body["messages"] if m.get("role") == "user" for block in (m.get("content") if isinstance(m.get("content"), list) else []) if block.get("type") == "tool_result"]
    all_text = [body["system"], *[s for m in body["messages"] for s in _text_blocks(m.get("content"))], *[s for r in results for s in _text_blocks(r.get("content"))]]
    receipts, consumed = presence.get("read_receipts"), set()
    if not isinstance(receipts, list):
        raise ValueError("source_bundle_native_read_receipts_required")
    for receipt in receipts:
        use_id = receipt.get("tool_use_id")
        matching_calls = [c for c in calls if c.get("id") == use_id]
        matching_results = [r for r in results if r.get("tool_use_id") == use_id]
        if len(matching_calls) != 1 or len(matching_results) != 1 or matching_results[0].get("is_error") is True:
            raise ValueError("source_bundle_native_read_call_or_result_missing")
        # The same range may be delivered twice within the bounded read budget.
        # Bind each occurrence to its entire canonical result, including the
        # program's unique delivery metadata; argument equality alone cannot
        # distinguish those real deliveries. Never merge partial results.
        native_text = _text_blocks(matching_results[0].get("content"))
        matches = [(i, row) for i, row in enumerate(reads)
                   if row["arguments"] == matching_calls[0].get("input")
                   and native_text == [_json(row["result"], sort=True)]]
        if len(matches) != 1 or matches[0][0] in consumed:
            raise ValueError("source_bundle_native_unknown_or_repeated_read")
        index, read = matches[0]
        rendered = _json(read["result"], sort=True)
        if _text_blocks(matching_results[0].get("content")) != [rendered] or sum(s.count(rendered) for s in all_text) != 1:
            raise ValueError("source_bundle_native_complete_source_text_missing")
        expected = {"tool_use_id": use_id, "native_read_result_sha256": sha256(rendered.encode()).hexdigest(),
            "bundle_ids": [b["bundle_id"] for b in read["result"]["bundles"]],
            "source_bundle_set_sha256": read["result"]["source_bundle_set_sha256"], "complete_original_slices_present": True}
        if receipt != expected:
            raise ValueError("source_bundle_native_read_receipt_changed")
        consumed.add(index)
    successful_read_ids = {c.get("id") for c in calls if any(r.get("tool_use_id") == c.get("id") and r.get("is_error") is not True for r in results)}
    if successful_read_ids != {r["tool_use_id"] for r in receipts} or (final and consumed != set(range(len(reads)))):
        raise ValueError("source_bundle_native_untracked_or_missing_reads")


def _replay(question, paragraphs, documents, response, excluded):
    if (isinstance(response, dict) and isinstance(response.get("proof"), dict)
            and response["proof"].get("format") == "langchain-source-bundle-workbench-proof-v1"):
        from app.workflows.lc_bundle_workbench import replay
        return replay(question, paragraphs, documents, response, excluded)
    if not isinstance(response, dict) or set(response) != {"artifact", "proof", "usage_report"}:
        raise ValueError("source_bundle_response_contract")
    proof = response["proof"]
    if not isinstance(proof, dict) or proof.get("format") != PROOF_FORMAT or proof.get("usage_report") != response["usage_report"] or proof.get("excluded_spans") != excluded:
        raise ValueError("source_bundle_native_proof_required")
    sources = _archive_sources(proof.get("source_archives"))
    raw_bytes = _bound(proof["raw_case"], raw=True)
    case = json.loads(raw_bytes)
    if case["question"] != _question(question) or case["documents"] != documents or case["paragraphs"] != paragraphs:
        raise ValueError("source_bundle_full_source_case_binding")
    request, submitted = _bound(proof["submit_request"]), _bound(proof["tool_submission"])
    if request.get("operation") != "submit":
        raise ValueError("source_bundle_terminal_submit_required")
    compact = "program_projection" in proof or "native_projections" in proof
    if compact and not {"program_projection", "native_projections"} <= set(proof):
        raise ValueError("source_bundle_projection_descriptors_required")
    reads, planned = [], {}
    with _replay_executor(sources, raw_bytes, proof["raw_case"]["sha256"]) as execute:
        replayed = execute(request)
        if replayed != submitted or replayed.get("status") != "ok":
            raise ValueError("source_bundle_tool_transcript_replay_failed")
        for i, entry in enumerate(request["history"]):
            if compact and entry.get("operation") in {"search", "plan"} and entry.get("status") == "ok":
                local = execute({"operation": entry["operation"], "arguments": entry["arguments"], "history": request["history"][:i]})
                if entry["operation"] in planned or local.get("status") != "ok":
                    raise ValueError("source_bundle_projection_unique_program_plan_required")
                planned[entry["operation"]] = (i, entry["arguments"], local["result"])
            if entry.get("operation") == "read" and entry.get("status") == "ok":
                local = execute({"operation": "read", "arguments": entry["arguments"], "history": request["history"][:i]})
                if local.get("status") != "ok":
                    raise ValueError("source_bundle_tool_transcript_replay_failed")
                reads.append({"arguments": entry["arguments"], "result": local["result"]})
    receipt = _usage(proof)
    if (receipt.get("source_binding") != submitted["scope"]["binding"] or receipt.get("raw_case_sha256") != proof["raw_case"]["sha256"]
            or receipt.get("artifact_sha256") != proof["artifact"]["sha256"]):
        raise ValueError("source_bundle_closed_source_binding")
    if not reads:
        raise ValueError("source_bundle_no_complete_reads")
    handles, projections, handles_sha256 = None, None, None
    if compact:
        if set(planned) != {"search", "plan"}:
            raise ValueError("source_bundle_projection_program_plan_required")
        si, search_args, navigation = planned["search"]
        pi, plan_args, plan = planned["plan"]
        if (si != 0 or pi != 1 or search_args != {"query": case["question"]["question"], "limit": 40, "exclude_paragraph_ids": []}
                or plan_args != {"anchor_paragraph_ids": [r["paragraph_id"] for r in navigation["results"]], "before": 1, "after": 3}
                or plan.get("source_binding") != submitted["scope"]["binding"]):
            raise ValueError("source_bundle_projection_original_navigation_plan_changed")
        handles = _bound(proof["program_projection"])
        handles_sha256 = proof["program_projection"]["sha256"]
        if handles != _handle_manifest(case, navigation, plan, proof["raw_case"]["sha256"]):
            raise ValueError("source_bundle_projection_handle_manifest_changed")
        projections = _bound(proof["native_projections"], lines=True)
        if (not projections or any(not isinstance(p, dict) or p.get("operation") not in {"read", "submit"} for p in projections)
                or len([p for p in projections if p["operation"] == "read"]) != len(reads)
                or len([p for p in projections if p["operation"] == "submit"]) != 1
                or projections[-1]["operation"] != "submit"):
            raise ValueError("source_bundle_projection_complete_native_transcript_required")
        use_ids = [p.get("tool_use_id") for p in projections]
        if any(not isinstance(value, str) or not value for value in use_ids) or len(set(use_ids)) != len(use_ids):
            raise ValueError("source_bundle_projection_unique_native_tool_ids_required")
        terminal = projections[-1]
        native_args = _native_arguments(terminal.get("raw_native_arguments"))
        if (native_args != terminal.get("native_arguments") or terminal.get("handles_sha256") != handles_sha256
                or _handle_submit_arguments(handles, reads, native_args) != request["arguments"]
                or terminal.get("business_arguments") != request["arguments"]
                or terminal.get("canonical_result") != submitted["result"]
                or terminal.get("canonical_result_sha256") != digest(submitted["result"])):
            raise ValueError("source_bundle_projection_terminal_submission_changed")
    presence = _bound(proof["presence"], lines=True)
    if len(presence) != len(proof["native_requests"]):
        raise ValueError("source_bundle_presence_request_count")
    scope = submitted["scope"]
    # JS receives canonical JSON from the Python bridge; nested key insertion
    # order is the canonical order, while this outer question section is fixed.
    question_data = _json({"original_question": json.loads(_json(case["question"], sort=True)), "corpus_document_count": len(documents),
        "corpus_paragraph_count": len(paragraphs), "source_binding": json.loads(_json(scope["binding"], sort=True)),
        "unseen_context": "unknown_not_irrelevant", "publishable": False})
    for i, descriptor in enumerate(proof["native_requests"]):
        _native_presence(_bound(descriptor), presence[i], reads, question_data, final=i == len(presence) - 1,
            handles=handles, handles_sha256=handles_sha256, projections=projections)
    artifact = _bound(proof["artifact"])
    expected = {**submitted["result"], "native_model_delivery_proof": presence[-1],
        "source_bytes_present_in_submission_request": True, "semantic_verified": False, "publishable": False}
    if artifact != expected or artifact != response["artifact"]:
        raise ValueError("source_bundle_native_artifact_mismatch")
    return artifact, scope


def _context_delivery(scope, documents, paragraphs):
    """Validate individual delivered ranges; their union never certifies a read."""
    from app.domain.source_passages import validate_source_passage
    delivered = scope.get("delivered_source_passages")
    if not isinstance(delivered, list):
        raise ValueError("source_context_delivered_passages_required")
    by_id = {row["id"]: row for row in paragraphs}
    complete, touched, identities, checked = set(), set(), set(), []
    for raw in delivered:
        locators = {"source_window_id", "source_context_id"}
        if (not isinstance(raw, dict) or not set(source_core_fields(raw)) <= set(raw)
                or set(raw) - set(source_core_fields(raw)) - locators
                or any(not isinstance(raw[key], str) or not raw[key].strip() for key in locators & set(raw))):
            raise ValueError("source_context_delivery_core_required")
        row = (validate_source_passage if "source_geometry" in raw else validate_source_bundle)(raw, documents, paragraphs)
        if row["bundle_id"] in identities:
            raise ValueError("source_context_duplicate_delivered_passage")
        identities.add(row["bundle_id"]); checked.append(row)
        touched.update(row["source_paragraph_ids"])
        # Every member must be fully covered by this one original payload.
        # Two complementary partial passages are not a full-parent delivery.
        for pid in row["source_paragraph_ids"]:
            parent = by_id[pid]
            if row["start_cp"] <= parent["document_start_cp"] and row["end_cp"] >= parent["document_end_cp"]:
                complete.add(pid)
    for key, expected in (("returned_complete_paragraph_ids", complete), ("exposed_paragraph_ids", touched)):
        values = scope.get(key)
        if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values)) or set(values) != expected):
            raise ValueError("source_context_full_parent_read_claim_changed")
    return checked, complete, touched


def compile_result(question, paragraphs, source_documents, response, *, excluded_spans=()):
    excluded, document_exclusions = _exclusions(paragraphs, excluded_spans)
    artifact, scope = _replay(question, paragraphs, source_documents, response, excluded)
    lc_source = response["proof"]["format"] == "langchain-source-bundle-workbench-proof-v1"
    context = response["proof"].get("context_passages", False)
    protocol = SOURCE_CONTEXT_PROTOCOL if context else SOURCE_BUNDLE_PROTOCOL
    if (type(context) is not bool or context and (not lc_source or response["proof"].get("source_bundle_protocol") != protocol)
            or "source_bundle_protocol" in response["proof"] and response["proof"]["source_bundle_protocol"] != protocol):
        raise ValueError("source_context_frozen_protocol_required")
    delivered, complete, touched = _context_delivery(scope, source_documents, paragraphs) if context else ([], set(), set())
    from app.workflows.strict_workbench import _groups
    canonical, anchors = [], set()
    selected_ids = {pid for row in artifact["evidence"] for pid in row["source_paragraph_ids"]}
    groups = _groups(paragraphs, selected_ids)
    by_id = {p["id"]: p for p in paragraphs}
    for raw in artifact["evidence"]:
        from app.domain.snapshot import validate_source_bundle_rows
        validate_source_bundle_rows([raw], protocol=protocol, source_documents=source_documents, source_paragraphs=paragraphs)
        row = deepcopy(raw)
        if context and not any(read["document_id"] == row["document_id"]
                and read["start_cp"] <= row["start_cp"] <= row["end_cp"] <= read["end_cp"]
                and set(row["source_paragraph_ids"]) <= set(read["source_paragraph_ids"])
                and read["quote"][row["start_cp"] - read["start_cp"]:row["end_cp"] - read["start_cp"]] == row["quote"]
                for read in delivered):
            raise ValueError("source_context_final_range_not_in_one_actual_delivery")
        if row.get("bundle_purpose") != "citation_source" or row["paragraph_id"] in anchors or not row["quote"]:
            raise ValueError("source_bundle_final_citation_scope")
        anchors.add(row["paragraph_id"])
        if any(x["document_id"] == row["document_id"] and row["start_cp"] < x["end_cp"] and x["start_cp"] < row["end_cp"] for x in document_exclusions):
            raise ValueError("source_bundle_selected_operator_excluded_span")
        argument, entities, roles = row.get("argument"), row.get("entities"), row.get("proposed_roles")
        if (not isinstance(row.get("fact_des"), str) or not row["fact_des"].strip() or not isinstance(argument, dict)
                or set(argument) != ARGUMENT_KEYS or any(not isinstance(x, str) for x in argument.values()) or not argument["内容"].strip()
                or not _time_literal_grounded(row)
                or not isinstance(entities, list) or any(not isinstance(x, str) or not x.strip() for x in entities) or len(set(entities)) != len(entities)
                or ("proposed_roles" in row if lc_source else
                    not isinstance(roles, list) or not roles or not set(roles) <= {"S", "C", "L"} or len(set(roles)) != len(roles))
                # Inspect actual field text independently. JSON escaping can
                # turn a real newline before 9 into the spurious identifier
                # n9; joining fields can likewise invent cross-field tokens.
                or any(unsupported_numeric_claims(text, row["quote"])
                    for text in [row["fact_des"], *argument.values(), *entities])):
            raise ValueError("source_bundle_annotation_literal_grounding")
        p = by_id[row["paragraph_id"]]
        grade, grade_basis = source_credibility_value(p)
        row.update(id=str(uuid5(NAMESPACE_URL, "trust-judge:source-bundle-evidence:" + digest({"question": question["id"], "source": raw}))),
            source_group_id=groups[p["document_id"]], source_group_basis="known_doc_identity_or_exact_paragraph_correlation_not_proven_independent",
            grade_score=grade, grade_basis=grade_basis, source_level="unknown", quote_verified=True, interpretation_origin="model",
            annotation_semantics_verified=False, model_span_corrected=False,
            span_location_method=("program_complete_original_sentence_context" if "source_geometry" in row
                                  else "program_complete_original_document_paragraph_bundle"),
            native_model_delivery_verified=True, proposed_roles_semantics_verified=False)
        canonical.append(row)
    canonical.sort(key=lambda r: r["id"])
    evidence, material = apply_workflow_material_eligibility(question, paragraphs, canonical,
        source_documents=source_documents, source_bundle_protocol=protocol)
    proof = {"version": protocol, "question_sha256": digest(question), "source_documents_sha256": digest(source_documents),
        "source_paragraphs_sha256": digest(paragraphs), "response": deepcopy(response), "operator_exclusions": excluded,
        "qualified_evidence_sha256": digest(evidence), "source_quote_span_verified": True, "native_source_delivery_replayed": True,
        "semantic_verified": False, "proposed_roles_used_for_final_relations": False}
    proof["proof_sha256"] = digest(proof)
    return {"evidence": evidence, "candidate_evidence": deepcopy(canonical), "material_eligibility": material,
        "source_bundle_protocol": protocol, "verification_protocol": protocol, "source_bundle_proof": proof,
        "evidence_selection": {"strategy": protocol, "selected_evidence_ids": [r["id"] for r in evidence], "semantic_verified": False},
        "source_coverage": {"strategy": protocol, "complete": False, "total_paragraphs": len(paragraphs),
            "provided_full_paragraphs": len(scope["returned_complete_paragraph_ids"]), "selected_paragraphs": len(selected_ids),
            **({"provided_full_paragraph_ids": sorted(complete), "partially_delivered_parent_ids": sorted(touched - complete),
                "delivered_source_passage_count": len(delivered), "delivered_source_passages": [
                    {key: deepcopy(value) for key, value in row.items() if key != "quote"} for row in delivered],
                "delivery_basis": "individually_replayed_native_contexts_not_union_or_whole_corpus_read"} if context else {}),
            "native_delivery_verified": True, "unread_context_state": "unknown_not_irrelevant"}}


def validate_result(result, question, paragraphs, source_documents, *, excluded_spans=None):
    proof = result.get("source_bundle_proof")
    protocol = result.get("source_bundle_protocol")
    if (protocol not in SOURCE_DOCUMENT_PROTOCOLS or result.get("verification_protocol") != protocol
            or not isinstance(proof, dict) or proof.get("version") != protocol):
        raise ValueError("source_bundle_workbench_proof_required")
    excluded = proof["operator_exclusions"] if excluded_spans is None else excluded_spans
    rebuilt = compile_result(question, paragraphs, source_documents, proof["response"], excluded_spans=excluded)
    if any(result.get(key) != value for key, value in rebuilt.items()):
        raise ValueError("source_bundle_workbench_projection_replay_failed")
    return rebuilt
