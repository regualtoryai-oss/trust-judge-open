"""SHA-bound pure LC source-book entry; isolated archival replay, no model or I/O tools."""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.lc_bundle_case import prepare_lc_case, compile_business, validate_rank_plan  # noqa: E402


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("lc_bundle_duplicate_json_key")
        result[key] = value
    return result


def decode(data):
    return json.loads(data.decode("utf8"), object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("lc_bundle_nonfinite_json")))


def execute(case_bytes, case_sha256, request):
    if (not isinstance(case_bytes, bytes) or len(case_bytes) > 10000000
            or not isinstance(case_sha256, str) or sha256(case_bytes).hexdigest() != case_sha256):
        raise ValueError("lc_bundle_bound_raw_case_changed")
    case = decode(case_bytes)
    if not isinstance(request, dict) or request.get("operation") not in {"prepare", "compile", "verify_rank"}:
        raise ValueError("lc_bundle_cli_request_shape")
    if request["operation"] == "verify_rank":
        if set(request) != {"operation", "rank_plan", "rank_plan_sha256", "local_rank_proof", "expected_model_manifest_sha256"}:
            raise ValueError("lc_bundle_cli_request_shape")
        validate_rank_plan(case, request["rank_plan"], request["rank_plan_sha256"])
        try:
            from app.domain.local_parent_rank import verify_local_rank
        except ImportError:
            raise ValueError("lc_bundle_rank_verifier_unavailable") from None
        result = verify_local_rank(case, request["rank_plan"], request["local_rank_proof"],
            expected_model_manifest_sha256=request["expected_model_manifest_sha256"], raw_case_sha256=case_sha256)
    else:
        if (set(request) - {"operation", "output", "excluded_spans", "rank_plan", "rank_plan_sha256", "selection_only", "graded_selection",
                "context_passages", "complete_grading", "hybrid_retrieval", "paragraph_quality", "paragraph_quality_compact", "parent_token_pack", "local_rank_proof", "expected_model_manifest_sha256"}
                or (request["operation"] == "prepare" and "output" in request)
                or (request["operation"] == "compile" and "output" not in request)
                or ("rank_plan" in request) != ("rank_plan_sha256" in request)
                or ("selection_only" in request and type(request["selection_only"]) is not bool)
                or ("graded_selection" in request and type(request["graded_selection"]) is not bool)
                or ("rank_plan" in request and (request["rank_plan"] is None or request["rank_plan_sha256"] is None))):
            raise ValueError("lc_bundle_cli_request_shape")
        excluded = request.get("excluded_spans", [])
        rank = {key: request[key] for key in ("rank_plan", "rank_plan_sha256", "selection_only", "graded_selection") if key in request}
        if "paragraph_quality_compact" in request and (request["paragraph_quality_compact"] is not True or request.get("paragraph_quality") is not True):
            raise ValueError("lc_bundle_paragraph_quality_compact_mode_required")
        if "parent_token_pack" in request and (request["parent_token_pack"] is not True
                or request.get("paragraph_quality_compact") is not True or "rank_plan" in request):
            raise ValueError("lc_bundle_parent_token_pack_mode_required")
        if request.get("paragraph_quality") is True:
            if any(request.get(key, False) for key in ("selection_only", "graded_selection", "context_passages", "complete_grading", "hybrid_retrieval")):
                raise ValueError("lc_bundle_paragraph_quality_distinct_protocol_required")
            # Legacy flags may be explicitly false in a pure request. They
            # do not become keyword options of the distinct paragraph codec.
            rank = {key: request[key] for key in ("rank_plan", "rank_plan_sha256") if key in request}
            from app.domain.paragraph_quality_case import prepare_paragraph_quality, compile_paragraph_quality
            if request.get("paragraph_quality_compact") is True:
                from app.domain.paragraph_quality_compact_case import prepare_paragraph_quality, compile_paragraph_quality
            if request.get("parent_token_pack") is True:
                from app.domain.paragraph_quality_budget_case import prepare_paragraph_quality, compile_paragraph_quality
            prepared = prepare_paragraph_quality(case, excluded_spans=excluded, **rank)
            result = prepared if request["operation"] == "prepare" else compile_paragraph_quality(
                case, prepared, request["output"], excluded_spans=excluded, **rank)
        elif request.get("context_passages") is True:
            if (rank.get("selection_only") is not True or rank.get("graded_selection") is not True
                    or not {"rank_plan", "rank_plan_sha256", "local_rank_proof", "expected_model_manifest_sha256"} <= set(request)):
                raise ValueError("lc_bundle_context_verified_rank_required")
            from app.domain.context_source_inputs import prepare_inputs, compile_inputs
            options = {key: request[key] for key in ("rank_plan", "rank_plan_sha256", "local_rank_proof", "expected_model_manifest_sha256")}
            options.update(raw_case_sha256=case_sha256, excluded_spans=excluded)
            for feature in ("complete_grading", "hybrid_retrieval", "paragraph_quality"):
                if feature in request:
                    if request[feature] is not True:
                        raise ValueError("lc_bundle_context_feature_mode_invalid")
                    options[feature] = True
            prepared = prepare_inputs(case, **options)["prepared"]
            result = prepared if request["operation"] == "prepare" else compile_inputs(case, prepared, request["output"], **options)
        else:
            if any(key in request for key in ("context_passages", "complete_grading", "hybrid_retrieval", "paragraph_quality", "paragraph_quality_compact", "parent_token_pack", "local_rank_proof", "expected_model_manifest_sha256")):
                raise ValueError("lc_bundle_context_mode_required")
            prepared = prepare_lc_case(case, excluded_spans=excluded, **rank)
            result = prepared if request["operation"] == "prepare" else compile_business(case, prepared, request["output"], excluded_spans=excluded, **rank)
    return {"format": "lc-source-bundle-tool-response-v1", "status": "ok", "result": result,
        "case_sha256": case_sha256, "model_calls": 0, "publishable": False}


def main():
    try:
        if len(sys.argv) != 3 or not Path(sys.argv[1]).is_absolute():
            raise ValueError("lc_bundle_cli_bound_case_arguments_required")
        path, expected = Path(sys.argv[1]), sys.argv[2]
        raw = sys.stdin.buffer.read(4000001)
        if len(raw) > 4000000:
            raise ValueError("lc_bundle_cli_request_too_large")
        result = execute(path.read_bytes(), expected, decode(raw))
        code = 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        reason = str(error)
        safe = reason if re.fullmatch(r"(?:lc_bundle|local_parent_rank|source_bundle(?:_tools)?)_[a-z0-9_]+", reason) else "lc_bundle_cli_rejected"
        result = {"format": "lc-source-bundle-tool-response-v1", "status": "rejected", "error_code": safe,
            "model_calls": 0, "publishable": False}
        code = 1
    sys.stdout.buffer.write((json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
