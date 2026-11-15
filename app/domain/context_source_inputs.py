"""Rebuild the context book from closed, actually computed local scores.

The same entry is used by runtime and the isolated, archived source compiler.
There is no provider call, inference, repair or alternate orchestration loop.
"""
from app.domain.local_parent_rank import fingerprint
from app.domain.lc_bundle_case import validate_rank_plan
from app.domain.local_passage_rank import load_bound_window_scores


def prepare_inputs(raw, *, raw_case_sha256, rank_plan, rank_plan_sha256,
                   local_rank_proof, expected_model_manifest_sha256, excluded_spans=(), complete_grading=False,
                   hybrid_retrieval=False):
    from app.domain.context_candidates import build_context_plan
    from app.domain.lc_context_case import prepare_context_case
    validate_rank_plan(raw, rank_plan, rank_plan_sha256)
    windows = load_bound_window_scores(raw, rank_plan, local_rank_proof,
        expected_model_manifest_sha256, raw_case_sha256)
    windows_sha = fingerprint(windows)
    plan_options = {"hybrid_retrieval": True} if hybrid_retrieval is True else {}
    if type(hybrid_retrieval) is not bool:
        raise ValueError("lc_context_hybrid_retrieval_mode_invalid")
    plan = build_context_plan(raw, windows, full_window_scores_digest=windows_sha, **plan_options)
    plan_sha = fingerprint(plan)
    prepared = prepare_context_case(raw, plan, plan_sha, window_scores=windows,
        full_window_scores_digest=windows_sha, excluded_spans=excluded_spans, **plan_options)
    base_prepared = prepared
    if type(complete_grading) is not bool:
        raise ValueError("lc_context_complete_grading_mode_invalid")
    if complete_grading:
        from app.domain.complete_context_grading import prepare_complete
        prepared = prepare_complete(base_prepared)
    return {"prepared": prepared, "context_plan": plan, "context_plan_sha256": plan_sha,
            "window_scores": windows, "window_scores_sha256": windows_sha, "base_prepared": base_prepared}


def compile_inputs(raw, prepared, output, **options):
    from app.domain.lc_context_case import compile_context_business
    rebuilt = prepare_inputs(raw, **options)
    if rebuilt["prepared"] != prepared:
        raise ValueError("lc_context_prepared_contract_changed")
    compiler = compile_context_business
    if options.get("complete_grading") is True:
        from app.domain.complete_context_grading import compile_complete_business
        compiler = compile_complete_business
    return compiler(raw, rebuilt["base_prepared"], output,
        context_plan=rebuilt["context_plan"], context_plan_sha256=rebuilt["context_plan_sha256"],
        window_scores=rebuilt["window_scores"], full_window_scores_digest=rebuilt["window_scores_sha256"],
        excluded_spans=options.get("excluded_spans", ()),
        **({"hybrid_retrieval": True} if options.get("hybrid_retrieval") is True else {}))
