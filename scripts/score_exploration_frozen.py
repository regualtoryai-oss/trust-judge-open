"""Bind the post-freeze reference-loader implementation without changing plans."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.evaluate_exploration_schedule import descriptor, bound, save, validate_freeze, unchanged
from scripts.offline_reference_tools import load_reference_tools


def score_views(freeze_path, expected_sha, sample_root, *, reference_loader=None):
    fixed, cases, guards = validate_freeze(freeze_path, expected_sha)
    mapper, engine, implementation = load_reference_tools(); guards += implementation
    if reference_loader is None:
        from scripts.score_obligation_selection_probe import load_reference
        reference_loader = lambda case: load_reference(case, sample_root)
    reports = []
    for (_, case, full), row in zip(cases, fixed["cases"]):
        # No reference access occurs until the complete 36-plan validation.
        evidence, conclusion, reference_guards = reference_loader(case); guards += reference_guards
        if conclusion["question"] != case["question"]["question"] or conclusion["question_id"] != case["question_id"]:
            raise ValueError("original_reference_question_changed")
        reference, issues = mapper.question_union_reference(evidence, conclusion)
        reference_units = engine._units(reference, True)
        domains, matched = {}, {}
        for domain, source in (("frozen_widepool", case), ("raw_all_paragraphs", full)):
            catalog = {p["paragraph_ref"]: p for p in source["catalog"]}; coverage, matches = {}, {}
            for name in ("domain_all_sources", "bootstrap5", "original20", "original40", "multi_facet20", "multi_facet40"):
                aliases = list(catalog) if name == "domain_all_sources" else row["schedules"][domain][name]
                submitted = [{k: catalog[p][k] for k in ("doc_id", "para_id")} for p in aliases]
                result = engine.compare_rows(case["topic_id"], submitted, reference)["unique_paragraphs"]["A_id_only"]
                m = result["metrics"]; coverage[name] = {k: m[k] for k in ("matches", "reference_records", "recall")}
                coverage[name]["exposed_source_positions"] = m["submitted_records"]
                matches[name] = {reference_units[match["reference_index"]]["key"] for match in result["matching"]}
            domains[domain] = {"source_ID_coverage": coverage,
                "delta_from_bootstrap5": {name: {"gained_IDs": sorted(matches[name]-matches["bootstrap5"]),
                    "lost_IDs": sorted(matches["bootstrap5"]-matches[name])} for name in ("multi_facet20", "multi_facet40")},
                "same_budget_scheduler_vs_original": {str(n): {"gained_IDs": sorted(matches[f"multi_facet{n}"]-matches[f"original{n}"]),
                    "lost_IDs": sorted(matches[f"original{n}"]-matches[f"multi_facet{n}"])} for n in (20, 40)}}
            matched[domain] = matches
        reports.append({"topic_id": case["topic_id"], "question_id": case["question_id"], "topic_title": case["topic_title"],
            "domains": domains, "reference_issues": issues, "same_budget_fullraw_vs_widepool": {name: {
                "gained_IDs": sorted(matched["raw_all_paragraphs"][name]-matched["frozen_widepool"][name]),
                "lost_IDs": sorted(matched["frozen_widepool"][name]-matched["raw_all_paragraphs"][name])}
                for name in ("bootstrap5", "original20", "original40", "multi_facet20", "multi_facet40")}})
    totals = {}
    for label, selected in [("all", reports), *[(t, [r for r in reports if r["topic_id"] == t]) for t in sorted({r["topic_id"] for r in reports})]]:
        totals[label] = {}
        for domain in ("frozen_widepool", "raw_all_paragraphs"):
            totals[label][domain] = {}
            for name in ("domain_all_sources", "bootstrap5", "original20", "original40", "multi_facet20", "multi_facet40"):
                rows = [r["domains"][domain]["source_ID_coverage"][name] for r in selected]
                hit, den = sum(r["matches"] for r in rows), sum(r["reference_records"] for r in rows)
                totals[label][domain][name] = {"matches": hit, "reference_records": den, "coverage": hit/den if den else None,
                    "exposed_source_positions": sum(r["exposed_source_positions"] for r in rows)}
    unchanged(guards)
    return {"format": "fixed-exploration-ID-coverage-retrospective-v1", "cases": reports, "topic_and_total": totals,
        "reference_denominator": "per-question supporting union conflicting, unique source positions, established v25 mapper/evaluator",
        "reference_used_after_all_candidates_frozen": True, "candidate_coverage_is_final_precision_or_recall": False,
        "semantic_score_created": False, "model_reading_executed": False, "official_score": False, "model_calls": 0,
        "retrospective_public_sample_regression_not_unseen_test": True, "input_guards": guards}


def run(freeze_path, expected_sha, sample_root, output_directory):
    output = Path(output_directory).resolve(); sample_root = Path(sample_root).resolve()
    if output.exists() or output.is_relative_to(sample_root) or output.is_relative_to(Path(freeze_path).resolve().parent):
        raise ValueError("new_separate_implementation_bound_score_directory_required")
    # This dependency is used ONLY in offline scoring, not candidate generation.
    implementations = [descriptor(p) for p in (Path(__file__), ROOT/"scripts/evaluate_exploration_schedule.py",
        ROOT/"scripts/score_obligation_selection_probe.py")]
    output.mkdir(parents=True)
    report = score_views(freeze_path, expected_sha, sample_root)
    intermediate = save(output/"base-score.json", report)
    for item in implementations: bound(item)
    report["input_guards"] += implementations
    report["offline_implementation_extension"] = {"generation_source_or_candidates_changed": False,
        "implementation_bound_before_reference": True, "base_report": intermediate,
        "reason": "explicit_reference_loader_source_provenance_after_candidate_freeze"}
    final = save(output/"score.json", report)
    save(output/"score-implementation-receipt.json", {"status": "CLOSED_OFFLINE_ID_COVERAGE_ONLY",
        "freeze": {"file": str(Path(freeze_path).resolve()), "sha256": expected_sha},
        "implementations_bound_before_and_after_reference": implementations,
        "base_report": intermediate, "final_report": final, "model_calls": 0})
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--freeze", required=True)
    p.add_argument("--expected-sha256", required=True); p.add_argument("--sample-root", required=True); p.add_argument("--output", required=True)
    args = p.parse_args()
    def audit(event, values):
        if event.startswith("socket."): raise PermissionError("offline_exploration_score_network_forbidden")
        if event == "open" and values and not isinstance(values[0], int) and Path(os.fsdecode(values[0])).name.lower() == ".env":
            raise PermissionError("offline_exploration_score_credentials_forbidden")
    sys.addaudithook(audit)
    print(json.dumps(run(args.freeze, args.expected_sha256, args.sample_root, args.output), ensure_ascii=False))
