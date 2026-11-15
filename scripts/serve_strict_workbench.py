"""Run the real API + durable worker with the bounded strict workflow enabled.

Opening the workbench does not start paid work. Runs still require an explicit
UI/API task action and use normal repository, checkpoints, publication/export.
"""
import argparse
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.main import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--cost-parent-run-id", action="append", default=[])
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--stop-tokens", type=int)
    parser.add_argument("--reuse-paid-closure", action="append", default=[])
    parser.add_argument("--reuse-paid-stage", action="append", default=[])
    parser.add_argument("--balanced", action="store_true", help="Experimental balanced paragraphs and explicit claim relations")
    parser.add_argument("--source-bundles", action="store_true", help="Experimental complete-source selection and normal A/B/arbitration")
    parser.add_argument("--bundle-framework", choices=("dsh", "langchain"), default="dsh")
    parser.add_argument("--bundle-compact-handles", action="store_true", help="Use source-owned legal range handles in the explicit source-bundle workflow")
    parser.add_argument("--bundle-local-rank", action="store_true", help="Rank every complete source parent locally before the LangChain source book")
    parser.add_argument("--bundle-selection-only", action="store_true", help="Select source IDs; derive literal metadata in code")
    parser.add_argument("--bundle-graded-selection", action="store_true", help="Grade source IDs 0-3, keep at most 10 with audited counterclaim input coverage; not final conflict labels")
    parser.add_argument("--bundle-compact-analyses", action="store_true", help="Use reversible source-ID aliases in the two independent analyses")
    parser.add_argument("--bundle-issue-frame", action="store_true", help="Anchor final relations to an explicit question proposition and directional qualifications")
    parser.add_argument("--bundle-context-passages", action="store_true", help="Read complete sentence contexts ranked from all local source windows")
    parser.add_argument("--bundle-complete-grading", action="store_true", help="Require an explicit grade for every actually offered context ID")
    parser.add_argument("--bundle-hybrid-retrieval", action="store_true", help="Fuse source-only lexical windows with the verified local BGE ranking")
    parser.add_argument("--bundle-paragraph-quality", action="store_true", help="Select complete original paragraphs and extract grounded fields in the single front call; no score threshold or evidence count cap")
    parser.add_argument("--bundle-paragraph-quality-compact", action="store_true", help="Use versioned compact literal fields for paragraph quality; requires --bundle-paragraph-quality")
    parser.add_argument("--bundle-parent-token-pack", action="store_true", help="Pack complete source parents within a source-text token budget; requires compact paragraph quality, excludes local rank")
    parser.add_argument("--bundle-recover-front-run-id", help="Explicitly reuse a succeeded sealed front from this failed run; no new source-model request")
    parser.add_argument("--bundle-recover-front-response", type=Path, help="Original immutable provider-response.json for explicit sealed-front recovery")
    parser.add_argument("--bundle-local-model-manifest", type=Path,
        default=ROOT / ".runtime/models/bge-reranker-v2-m3-6f5ff6529851-int8/local-model-manifest.json")
    parser.add_argument("--bundle-front-tokens", type=int, default=12000)
    parser.add_argument("--bundle-tail-reserve", type=int, default=18000)
    args = parser.parse_args()
    recovery_sha = None
    if args.bundle_recover_front_run_id or args.bundle_recover_front_response:
        if not (args.bundle_recover_front_run_id and args.bundle_recover_front_response and args.bundle_paragraph_quality_compact):
            parser.error("sealed front recovery requires run-id, original response path and compact paragraph quality")
        args.bundle_recover_front_response = args.bundle_recover_front_response.resolve()
        try:
            recovery_sha = sha256(args.bundle_recover_front_response.read_bytes()).hexdigest()
        except OSError:
            parser.error("sealed front recovery response is unavailable")
        args.cost_parent_run_id = list(dict.fromkeys([*args.cost_parent_run_id, args.bundle_recover_front_run_id]))
    if args.bundle_compact_handles and not args.source_bundles:
        parser.error("bundle-compact-handles requires source-bundles")
    if args.bundle_framework != "dsh" and not args.source_bundles:
        parser.error("bundle-framework requires source-bundles")
    if args.bundle_local_rank and (not args.source_bundles or args.bundle_framework != "langchain"):
        parser.error("bundle-local-rank requires LangChain source-bundles")
    if args.bundle_selection_only and (not args.source_bundles or args.bundle_framework != "langchain"):
        parser.error("bundle-selection-only requires LangChain source-bundles")
    if args.bundle_graded_selection and not args.bundle_selection_only:
        parser.error("bundle-graded-selection requires explicit bundle-selection-only")
    if args.bundle_compact_analyses and (not args.source_bundles or args.bundle_framework != "langchain"):
        parser.error("bundle-compact-analyses requires LangChain source-bundles")
    if args.bundle_issue_frame and (not args.source_bundles or args.bundle_framework != "langchain"):
        parser.error("bundle-issue-frame requires LangChain source-bundles")
    local_manifest_sha = None
    if args.bundle_local_rank:
        try:
            local_manifest_sha = sha256(args.bundle_local_model_manifest.read_bytes()).hexdigest()
        except OSError:
            parser.error("the explicit local model manifest is unavailable")
    bundle_paid_tail = (args.source_bundles and args.bundle_framework == "langchain"
        and args.bundle_recover_front_run_id and args.reuse_paid_closure
        and set(args.reuse_paid_stage) in ({"authority_analyst", "cross_validation_analyst"},
            {"authority_analyst", "cross_validation_analyst", "arbitrate"}))
    if args.source_bundles and (args.balanced or (args.reuse_paid_closure or args.reuse_paid_stage) and not bundle_paid_tail):
        parser.error("source-bundles permits exact A/B or full-tail cache only with explicit sealed-front recovery")
    settings = Settings(model_strict_evidence_enabled=not args.source_bundles,
        model_source_bundle_enabled=args.source_bundles,
        source_bundle_front_framework=args.bundle_framework,
        source_bundle_compact_handles=args.bundle_compact_handles,
        source_bundle_local_rank_enabled=args.bundle_local_rank,
        source_bundle_selection_only=args.bundle_selection_only,
        source_bundle_graded_selection=args.bundle_graded_selection,
        source_bundle_compact_analyses_enabled=args.bundle_compact_analyses,
        source_bundle_issue_frame_enabled=args.bundle_issue_frame,
        source_bundle_context_passages_enabled=args.bundle_context_passages,
        source_bundle_complete_grading_enabled=args.bundle_complete_grading,
        source_bundle_hybrid_retrieval_enabled=args.bundle_hybrid_retrieval,
        source_bundle_paragraph_quality_enabled=args.bundle_paragraph_quality,
        source_bundle_paragraph_quality_compact_enabled=args.bundle_paragraph_quality_compact,
        source_bundle_parent_token_pack_enabled=args.bundle_parent_token_pack,
        source_bundle_recover_front_run_id=args.bundle_recover_front_run_id,
        source_bundle_recover_front_response=args.bundle_recover_front_response,
        source_bundle_recover_front_response_sha256=recovery_sha,
        source_bundle_local_rank_manifest=args.bundle_local_model_manifest if args.bundle_local_rank else None,
        source_bundle_local_rank_manifest_sha256=local_manifest_sha,
        source_bundle_front_token_stop=args.bundle_front_tokens,
        source_bundle_tail_reserved_tokens=args.bundle_tail_reserve,
        strict_evidence_balanced_enabled=args.balanced,
        model_paragraph_ranked_enabled=False, model_premise_bank_enabled=False,
        model_thinking="disabled", model_max_retries=0, model_capacity_retries=0,
        max_questions_per_run=1, strict_evidence_max_requests=args.max_calls or (10 if args.source_bundles else 5),
        strict_evidence_cost_parent_run_ids=args.cost_parent_run_id,
        strict_evidence_token_stop=args.stop_tokens or (30000 if args.source_bundles else 40000),
        strict_evidence_response_reuse_closures=args.reuse_paid_closure,
        strict_evidence_response_reuse_operations=args.reuse_paid_stage or ["strict_selection", "strict_annotation"])
    import uvicorn
    uvicorn.run(create_app(settings, embedded_worker=True), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
