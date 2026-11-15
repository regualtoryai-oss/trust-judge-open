"""Freeze two raw-only UI inputs and existing baselines; never opens gold or calls AI.

Run from the repository root. This is pre-registration, not a launch freeze: the
normal UI import must be followed by an actual-UUID v6 budget/config freeze.
"""
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_bundle_workbench_real import prepare_closed
from scripts.score_real_quality_episode import original_sources
from scripts.score_workbench_real_episode import known_usage, verify_legacy_sources


def bind(path):
    path = Path(path).resolve()
    return {"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return bind(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        parser.error("pre-registration output must be inside .runtime")
    old = ROOT / ".runtime/sample-benchmark-v15-20261018-r1"
    cases = []
    for tid, qid in (("02", "q_002"), ("04", "q_001")):
        source = old / f"inputs/topic_{tid}/input.json"
        raw = load(source)
        if set(raw) != {"title", "documents", "questions"}:
            raise ValueError("raw_only_original_input_required")
        questions = [q for q in raw["questions"] if q["question_id"] == qid]
        if len(questions) != 1:
            raise ValueError("one_unchanged_original_question_required")
        contexts, _ = original_sources(raw)
        projection = {**raw, "questions": questions}
        # Nothing except the other questions is removed. All complete original
        # source documents, metadata, text, paragraph IDs and offsets survive.
        if projection["documents"] != raw["documents"]:
            raise ValueError("source_projection_changed_original_documents")
        case_dir = out / f"{tid}-{qid.replace('_', '')}"
        input_binding = save(case_dir / "input.json", projection)
        if tid == "02":
            v5dir = ROOT / ".runtime/paragraph-quality-token-pack-real-20261021/02-q002"
            closed = bind(v5dir / "closed/closed-inputs.json")
            browser = bind(v5dir / "browser-download-binding.json")
            prepared = prepare_closed(closed["file"], closed["sha256"], browser["file"], browser["sha256"])
            if prepared["failed"] or prepared["raw"] != raw or prepared["question"] != questions[0]:
                raise ValueError("latest_closed_workbench_baseline_source_or_question_changed")
            baseline = {"kind": "closed_bundle_workbench_v5", "closed_inputs": closed,
                "browser_binding": browser, "question_result": prepared["closure"]["artifacts"]["question_result"],
                "actual_download": prepared["browser_download"]["download_file"],
                "origin_run_id": prepared["run"]["id"], "status": "succeeded",
                "usage": {"total_tokens": prepared["tail"]["whole_episode_known_tokens"]},
                "actual_provider_calls": prepared["tail"]["whole_episode_actual_requests"],
                "new_experiment_cost": False, "native_and_browser_audit_passed": True,
                "source_bound_result_validated": True}
        else:
            history = old / "topics/topic_04/questions" / sha256(qid.encode()).hexdigest()
            terminal_path, result_path = history / "terminal.json", history / "question_result.json"
            terminal, result = load(terminal_path), load(result_path)
            if (terminal["status"] != "succeeded" or result["origin_run_id"] != terminal["id"]
                    or list(terminal["usage"].values()) != [result["usage"]]):
                raise ValueError("historical_closed_terminal_result_and_usage_must_agree")
            verify_legacy_sources(result, contexts)
            baseline = {"kind": "legacy_v15", "closed_receipt": bind(terminal_path),
                "result": bind(result_path), "raw_input": bind(source),
                "origin_run_id": result["origin_run_id"], "status": "succeeded",
                "usage": known_usage(result["usage"]), "actual_provider_calls": len(result["usage"]),
                "new_experiment_cost": False, "native_and_browser_audit_passed": False,
                "source_bound_result_validated": True, "literal_evidence_rows_verified": len(result["evidence"]),
                "historical_baseline_is_same_framework_causal_pair": False}
        cases.append({"raw_topic_id": tid, "question": questions[0], "original_raw_sources": bind(source),
            "UI_import_input": input_binding, "counts": {"documents": len(raw["documents"]),
                "paragraphs": len(contexts), "source_codepoints": sum(len(d["content"]) for d in raw["documents"]),
                "original_questions": len(raw["questions"]), "imported_questions": 1},
            "baseline": baseline, "selection_uses_reference_answers_or_scores": False,
            "selection_reason": "Existing closed same-question baseline, different topic, no paid baseline regeneration",
            "after": {"required": "new normal worker run from actual UI; actual JSON and ZIP downloads",
                "baseline_response_cache_allowed": False, "recovery_allowed_for_primary_success": False,
                "recompute_actual_UUID_budget_after_UI_import": True, "automatic_model_retries": 0,
                "requested_max_provider_calls": 4, "failure_retained_in_same_case_cost": True}})
    # Record excluded choices without parsing their answers or reading gold.
    excluded = []
    for tid, qid in (("03", "q_001"), ("06", "q_003")):
        history = old / f"topics/topic_{tid}/questions" / sha256(qid.encode()).hexdigest()
        terminal = load(history / "terminal.json")
        excluded.append({"topic_id": tid, "question_id": qid, "terminal": bind(history / "terminal.json"),
            "terminal_status": terminal["status"], "result_exists": (history / "question_result.json").is_file(),
            "reason": "No successful closed same-question v15 result; do not pay to recreate before"})
    result = {"format": "raw-only-paired-quality-preregistration-v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(), "cases": cases, "excluded": excluded,
        "new_model_calls": 0, "new_tokens": 0, "reference_files_read": False,
        "runtime_contains_baseline_answers": False, "is_actual_UUID_launch_freeze": False,
        "baseline_is_offline_only": True, "success_criteria_preregistered": {
            "per_question": ["E unique-source-ID P/R/F1", "supporting P/R", "conflicting P/R",
                "pool/provided/final reference coverage separately", "field nonempty counts distinct from accuracy",
                "literal quote validation", "actual provider usage including failures", "wall time"],
            "same_scorer_on_before_after": True, "do_not_claim_official_score": True,
            "semantic_B": "unknown unless separately authorized and measured",
            "gold_IO_gate": "Only after new native published result + actual browser JSON binding + ZIP verification",
            "no_post_gold_runtime_or_candidate_changes_in_this_round": True,
            "two_topics_are_not_all_topic_generalization": True}}
    print(json.dumps(save(out / "pair-input-freeze.json", result), ensure_ascii=False))


if __name__ == "__main__":
    main()
