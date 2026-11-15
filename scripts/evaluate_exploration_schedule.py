"""Two separate zero-API phases: freeze all local exposures, then ID coverage.

prepare cannot read references. score requires an expected CLOSED freeze SHA and
rebuilds raw/pool source cases before its first offline reference loader call.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from app.domain.exploration_schedule import (build_exploration_schedule, build_full_raw_exploration_schedule,
    build_full_raw_exploration_input, POLICY)
from app.domain.frozen_selection_inputs import build_frozen_selection_input, canonical


def descriptor(path):
    path = Path(path).resolve()
    return {"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def bound(item):
    if descriptor(item["file"]) != item: raise ValueError("exploration_bound_file_changed")
    return Path(item["file"])


def load(item): return json.loads(bound(item).read_text("utf-8-sig"))


def save(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()+b"\n")
    return descriptor(path)


def code_binding():
    return [descriptor(ROOT/path) for path in ("scripts/evaluate_exploration_schedule.py", "app/domain/exploration_schedule.py",
        "app/domain/question_recall.py", "app/domain/retrieval.py", "app/domain/query_context.py", "app/domain/selection_read_tools.py",
        "app/domain/frozen_selection_inputs.py", "app/domain/paragraph_contributions.py", "scripts/offline_reference_tools.py")]


def source_cases(index_descriptor):
    index = load(index_descriptor)
    if (index.get("format") != "frozen-unified-selection-input-index-v1" or index.get("reference_reads") != []
            or index.get("model_calls") != 0 or index.get("network_attempts") != [] or index.get("new_candidates_created") is not False
            or len(index["cases"]) != index["case_count"]):
        raise ValueError("whole_closed_raw_only_input_index_required")
    pool_descriptor = index["source_pool"]; pool_bytes = bound(pool_descriptor).read_bytes()
    raw = {}; guards = [index_descriptor, pool_descriptor]
    for entry in index["source_inputs"]:
        item = {"file": entry["path"], "sha256": entry["sha256"]}; guards.append(item)
        if entry["topic_id"] in raw: raise ValueError("duplicate_raw_topic")
        raw[entry["topic_id"]] = bound(item).read_bytes()
    cases, seen = [], set()
    for entry in index["cases"]:
        item = {k: entry[k] for k in ("file", "sha256")}; guards.append(item); case = load(item)
        identity = case["topic_id"], case["question_id"]
        if identity in seen or identity != (entry["topic_id"], entry["question_id"]): raise ValueError("duplicate_or_changed_case")
        seen.add(identity)
        rebuilt = build_frozen_selection_input(raw_input_bytes=raw[identity[0]], pool_input_bytes=pool_bytes,
            topic_id=identity[0], question_id=identity[1], expected_raw_sha256=case["binding"]["raw_input_file_sha256"],
            expected_pool_sha256=pool_descriptor["sha256"], variant=case["binding"]["variant"])
        if rebuilt != case or entry["binding"] != case["binding"]: raise ValueError("raw_pool_original_case_changed")
        cases.append((item, case))
    if len({k[0] for k in seen}) != index["topic_count"] or len(cases) != index["case_count"]:
        raise ValueError("all_frozen_input_topics_questions_required")
    return cases, guards, raw


def unchanged(guards):
    for item in guards: bound(item)


def prepare(index_path, expected_index_sha, output):
    output = Path(output).resolve(); index_item = {"file": str(Path(index_path).resolve()), "sha256": expected_index_sha}
    if output.exists(): raise ValueError("new_local_exploration_directory_required")
    output.mkdir(parents=True); code = code_binding()
    policy = save(output/"policy-before-data.json", {"policy": POLICY, "source_index": index_item,
        "code": code, "scope": "all_indexed_questions_unchanged_pools", "extra_queries": 0,
        "model_calls": 0, "reference_reads": [], "parameter_search": False})
    start = time.perf_counter(); cases, guards, raw = source_cases(index_item); plans = []
    for item, case in cases:
        schedule = build_exploration_schedule(case)
        full = build_full_raw_exploration_schedule(raw[case["topic_id"]], case["binding"]["raw_input_file_sha256"], case["topic_id"], case["question_id"])
        plans.append({"case_input": item, "topic_id": case["topic_id"], "question_id": case["question_id"],
            "topic_title": case["topic_title"], "schedules": {"frozen_widepool": schedule, "raw_all_paragraphs": full}})
    unchanged([*code, *guards, policy])
    fixed = save(output/"fixed-exposures.json", {"format": "closed-all-question-local-exposures-v1", "policy": policy,
        "source_index": index_item, "case_count": len(plans), "cases": plans, "model_calls": 0,
        "reference_reads": [], "full_text_sent_to_model": False, "original_widepool_changed": False,
        "separate_raw_all_source_domain": True,
        "input_tokens": None, "token_scope": "no_model_message_created", "elapsed_seconds": time.perf_counter()-start})
    closed = save(output/"exploration-freeze.json", {"status": "CLOSED_LOCAL_EXPOSURE_ONLY", "fixed_exposures": fixed,
        "policy": policy, "source_index": index_item, "source_guards": guards, "code": code,
        "all_indexed_cases_included": True, "model_calls": 0, "reference_reads": [], "network_calls": 0})
    return closed


def validate_freeze(freeze_path, expected_sha):
    item = {"file": str(Path(freeze_path).resolve()), "sha256": expected_sha}; receipt = load(item)
    if (receipt.get("status") != "CLOSED_LOCAL_EXPOSURE_ONLY" or receipt.get("reference_reads") != []
            or receipt.get("model_calls") != 0 or receipt.get("all_indexed_cases_included") is not True):
        raise ValueError("closed_exposure_freeze_required")
    unchanged([*receipt["source_guards"], *receipt["code"], receipt["policy"]])
    fixed = load(receipt["fixed_exposures"]); policy = load(receipt["policy"])
    if (policy["policy"] != POLICY or fixed["policy"] != receipt["policy"] or fixed.get("reference_reads") != []
            or fixed["source_index"] != receipt["source_index"] or fixed.get("original_widepool_changed") is not False
            or fixed.get("separate_raw_all_source_domain") is not True):
        raise ValueError("frozen_policy_or_scope_changed")
    cases, guards, raw = source_cases(receipt["source_index"])
    if len(cases) != fixed["case_count"] or len(cases) != len(fixed["cases"]): raise ValueError("frozen_case_scope_changed")
    paired = []
    for (case_item, case), row in zip(cases, fixed["cases"]):
        full = build_full_raw_exploration_input(raw[case["topic_id"]], case["binding"]["raw_input_file_sha256"], case["topic_id"], case["question_id"])
        paired.append((case_item, case, full))
        if row["case_input"] != case_item or (row["topic_id"], row["question_id"]) != (case["topic_id"], case["question_id"]):
            raise ValueError("exposure_source_case_binding_changed")
        if set(row["schedules"]) != {"frozen_widepool", "raw_all_paragraphs"}: raise ValueError("both_preregistered_search_domains_required")
        for domain, source in (("frozen_widepool", case), ("raw_all_paragraphs", full)):
            schedule = row["schedules"][domain]; aliases = {p["paragraph_ref"] for p in source["catalog"]}
            if (schedule["search_domain"] != domain or schedule["input_format"] != source["format"]
                    or schedule["question"] != case["question"] or schedule["pool_catalog_sha256"] != source["binding"]["catalog_sha256"]
                    or schedule["policy_sha256"] != sha256(canonical(POLICY)).hexdigest()):
                raise ValueError("search_domain_catalog_binding_changed")
            for name, limit in (("bootstrap5", 5), ("original20", 20), ("original40", 40), ("multi_facet20", 20), ("multi_facet40", 40)):
                refs = schedule[name]
                if len(refs) > limit or len(refs) != len(set(refs)) or not set(refs) <= aliases:
                    raise ValueError("exposure_not_unique_fixed_source_subset")
            if schedule["multi_facet20"] != schedule["multi_facet40"][:20]: raise ValueError("exposure_prefix_changed")
    return fixed, paired, [item, receipt["fixed_exposures"], receipt["policy"], *receipt["code"], *guards]


def score(freeze_path, expected_sha, sample_root, output):
    output = Path(output).resolve(); sample_root = Path(sample_root).resolve()
    if output.exists() or output.is_relative_to(sample_root) or output.is_relative_to(Path(freeze_path).resolve().parent):
        raise ValueError("new_separate_coverage_report_required")
    fixed, cases, guards = validate_freeze(freeze_path, expected_sha)
    from scripts.offline_reference_tools import load_reference_tools
    from scripts.score_obligation_selection_probe import load_reference
    mapper, engine, implementation = load_reference_tools(); guards += implementation
    reports = []
    for (_, case, full), row in zip(cases, fixed["cases"]):
        # FIRST reference reads happen after the COMPLETE all-question freeze
        # and raw/pool/index/source-code validation above.
        evidence, conclusion, refs = load_reference(case, sample_root); guards += refs
        if conclusion["question"] != case["question"]["question"]: raise ValueError("original_reference_question_changed")
        reference, issues = mapper.question_union_reference(evidence, conclusion)
        domains, matched = {}, {}
        for domain, source in (("frozen_widepool", case), ("raw_all_paragraphs", full)):
            catalog = {p["paragraph_ref"]: p for p in source["catalog"]}; coverage = {}; matches = {}
            for name in ("domain_all_sources", "bootstrap5", "original20", "original40", "multi_facet20", "multi_facet40"):
                aliases = list(catalog) if name == "domain_all_sources" else row["schedules"][domain][name]
                submitted = [{k: catalog[p][k] for k in ("doc_id", "para_id")} for p in aliases]
                result = engine.compare_rows(case["topic_id"], submitted, reference)["unique_paragraphs"]["A_id_only"]
                m = result["metrics"]; coverage[name] = {k: m[k] for k in ("matches", "reference_records", "recall")}
                coverage[name]["exposed_source_positions"] = m["submitted_records"]
                reference_units = engine._units(reference, True)
                matches[name] = {reference_units[i]["key"] for _, i in result["matching"]}
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
                rows = [r["domains"][domain]["source_ID_coverage"][name] for r in selected]; hit = sum(r["matches"] for r in rows); den = sum(r["reference_records"] for r in rows)
                totals[label][domain][name] = {"matches": hit, "reference_records": den, "coverage": hit/den if den else None,
                    "exposed_source_positions": sum(r["exposed_source_positions"] for r in rows)}
    unchanged(guards)
    result = {"format": "fixed-exploration-ID-coverage-retrospective-v1", "cases": reports, "topic_and_total": totals,
        "reference_denominator": "per-question supporting union conflicting, unique source positions, established v25 mapper/evaluator",
        "reference_used_after_all_candidates_frozen": True, "candidate_coverage_is_final_precision_or_recall": False,
        "semantic_score_created": False, "model_reading_executed": False, "official_score": False, "model_calls": 0,
        "retrospective_public_sample_regression_not_unseen_test": True, "input_guards": guards}
    output.parent.mkdir(parents=True, exist_ok=True); save(output, result)
    return descriptor(output)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("mode", choices=("prepare", "score")); p.add_argument("--input", required=True)
    p.add_argument("--expected-sha256", required=True); p.add_argument("--output", required=True); p.add_argument("--sample-root")
    args = p.parse_args()
    def audit(event, values):
        if event.startswith("socket."): raise PermissionError("local_exploration_network_forbidden")
        if event == "open" and values and not isinstance(values[0], int):
            name = Path(os.fsdecode(values[0])).name.lower()
            if name == ".env" or args.mode == "prepare" and name in {"evidence_samples.json", "conclusions.json", "expected.json"}:
                raise PermissionError("local_exploration_credentials_or_generation_reference_forbidden")
    sys.addaudithook(audit)
    result = prepare(args.input, args.expected_sha256, args.output) if args.mode == "prepare" else score(args.input, args.expected_sha256, args.sample_root, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
