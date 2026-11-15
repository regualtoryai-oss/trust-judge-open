"""CLOSED-only, zero-model source-ID diagnostics for the fixed title experiment.

References are opened only after all 18 candidates, source bytes and closure
bindings pass. This measures candidate locations, never final facts or truth.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.frozen_selection_inputs import build_frozen_selection_input
from app.domain.title_aware_rerank import prepare_title_pairs, fuse_title_hybrid
from scripts.audit_dsh_closed_episode import bound
from scripts.offline_reference_tools import load_reference_tools
from scripts.two_stage_evidence_matching import location

CASE_IDS = [(f"{topic:02d}", f"q_{question:03d}") for topic in range(1, 7) for question in range(1, 4)]
BASELINE = "baseline_hybrid40"
MAIN = "title_bge_hybrid_rrf40"
DIAGNOSTIC = "title_bge_only40"
VARIANTS = (BASELINE, MAIN, DIAGNOSTIC)
POOL_SHA = "6544144737d01c0bed15206f3925ec7a548e3540dcbbeebf7ab92774e229945a"
OLD_SHA = "709052e31f73272948b40c3afcb1c93b70e429208c869d984a4ad42e051bc436"
FORBIDDEN = {".env", "evidence_samples.json", "conclusions.json", "raw_documents.json"}


def _read(path, guards, expected=None, *, references=False):
    path = Path(path).resolve()
    if not references and path.name.lower() in FORBIDDEN:
        raise ValueError("reference_or_credentials_before_source_gate_forbidden")
    data, descriptor = bound(path, expected)
    guards.append(descriptor)
    return json.loads(data.decode("utf-8-sig")), data


def _one(rows, predicate):
    selected = [row for row in rows if predicate(row)]
    if len(selected) != 1:
        raise ValueError("one_frozen_identity_required")
    return selected[0]


def _order(rows):
    if [(row["topic_id"], row["question_id"]) for row in rows] != CASE_IDS:
        raise ValueError("all_eighteen_original_cases_in_order_required")


def _clean(value):
    if type(value.get("paid_api_calls")) is not int or value["paid_api_calls"] != 0:
        raise ValueError("zero_paid_generation_required")
    if value.get("reference_reads") != []:
        raise ValueError("reference_free_generation_required")


def _source_rows(rows, case):
    return [{**row, "id": case["original_ids"][row["paragraph_ref"]],
             "raw_doc_id": row["doc_id"], "raw_para_id": row["para_id"],
             "source_text_sha256": row["source_text_sha256_utf8"]} for row in rows]


def closed_sources(freeze_path, reviewed_sha256, *, root=ROOT, pool_path=None, pool_sha256=POOL_SHA,
                   baseline_path=None, baseline_sha256=OLD_SHA):
    """Return a source-only score plan; this function never opens references."""
    root = Path(root).resolve(); folder = Path(freeze_path).resolve().parent; guards = []
    freeze, _ = _read(freeze_path, guards, reviewed_sha256)
    if freeze.get("generation_complete") is not True or type(freeze.get("cases")) is not int or freeze["cases"] != 18:
        raise ValueError("closed_complete_eighteen_required")
    _clean(freeze)
    script = root / "scripts/run_title_aware_full.py"
    _, descriptor = bound(script, freeze["script_sha256"]); guards.append(descriptor)
    prep, _ = _read(folder / "preparation.json", guards, freeze["preparation_sha256"])
    if prep.get("format") != "title-aware-bge-widepool-preparation-v1" or prep.get("overflow") != [] or prep.get("queries_truncated") != 0:
        raise ValueError("unchanged_complete_question_no_overflow_preparation_required")
    _clean(prep); _order(prep["cases"])
    for name, key in (("scripts/probe_title_aware_rerank.py", "script_sha256"),
                      ("app/domain/title_aware_rerank.py", "helper_sha256")):
        _, descriptor = bound(root / name, prep[key]); guards.append(descriptor)
    for guard in prep["guards"]:
        if Path(guard["file"]).name.lower() in FORBIDDEN:
            raise ValueError("reference_or_credentials_in_generation_guards")
        _, descriptor = bound(guard["file"], guard["sha256"]); guards.append(descriptor)
    fixed, _ = _read(folder / "fixed-candidates.json", guards, freeze["fixed_candidates_sha256"])
    ranked, _ = _read(folder / "all-parent-ranking.json", guards, freeze["all_parent_ranking_sha256"])
    fit, _ = _read(folder / "native-fit-index.json", guards, freeze["native_fit_index_sha256"])
    if (fixed.get("format") != "title-aware-full-window-local-rerank-candidates-v1"
            or fixed.get("preparation_sha256") != freeze["preparation_sha256"]
            or fixed.get("all_parent_windows_scored") is not True
            or fit.get("format") != "frozen-local-rerank-native-fit-inputs-v1"):
        raise ValueError("expected_complete_frozen_artifact_formats_required")
    _clean(fixed)
    for artifact in (fixed, ranked, fit):
        _order(artifact["cases"])
    receipts = []
    total_batches = (len(prep["schedule"]) + 3) // 4
    if len(freeze["shard_receipts"]) != 2 or len(prep["schedule"]) != len(set(prep["schedule"])) or set(prep["schedule"]) != set(prep["pairs"]):
        raise ValueError("two_disjoint_complete_shards_required")
    for shard, binding in enumerate(freeze["shard_receipts"]):
        receipt, _ = _read(binding["file"], guards, binding["sha256"])
        start, end = (0, total_batches // 2) if shard == 0 else (total_batches // 2, total_batches)
        if (receipt.get("closed") is not True or receipt.get("shard") != shard
                or receipt.get("preparation_sha256") != freeze["preparation_sha256"]
                or receipt.get("script_sha256") != freeze["script_sha256"]
                or receipt.get("global_batch_start") != start or receipt.get("global_batch_end_exclusive") != end
                or receipt.get("pairs") != len(prep["schedule"][start*4:end*4])
                or type(receipt.get("paid_api_calls")) is not int or receipt["paid_api_calls"] != 0
                or receipt.get("network_attempts") != [] or receipt.get("reference_attempts") != []
                or receipt.get("probe_scores_reused") is not False):
            raise ValueError("closed_clean_nonreused_shard_required")
        receipts.append(receipt)
    pool_path = pool_path or root / ".runtime/v22-channel-pool-r2/fixed-candidates.json"
    baseline_path = baseline_path or root / "docs/reports/v20-semantic/fixed-candidates.json"
    pool, pool_bytes = _read(pool_path, guards, pool_sha256)
    old, _ = _read(baseline_path, guards, baseline_sha256)
    _order(pool["cases"]); _order(old["cases"])
    raw_sources = {}
    for source in pool["inputs"]:
        _, raw_bytes = _read(source["path"], guards, source["sha256"])
        if source["topic_id"] in raw_sources:
            raise ValueError("duplicate_raw_topic_binding")
        raw_sources[source["topic_id"]] = (source, raw_bytes)
    cases, pair_messages = [], {}
    for p, current, ranking, old_case, fit_case in zip(prep["cases"], fixed["cases"], ranked["cases"], old["cases"], fit["cases"]):
        topic, qid = p["topic_id"], p["question_id"]
        source, raw_bytes = raw_sources[topic]
        rebuilt = build_frozen_selection_input(raw_input_bytes=raw_bytes, pool_input_bytes=pool_bytes,
            topic_id=topic, question_id=qid, expected_raw_sha256=source["sha256"], expected_pool_sha256=pool_sha256)
        original, _ = _read(p["case_input"], guards, p["case_sha256"])
        if original != rebuilt:
            raise ValueError("complete_original_source_case_changed")
        common = {"topic_id": topic, "question_id": qid, "title": rebuilt["topic_title"], "question": rebuilt["question"]["question"]}
        if any(any(row.get(key) != value for key, value in common.items()) for row in (p | {"question": p["plan"]["question"]}, current, ranking, old_case)):
            raise ValueError("original_question_title_or_case_changed")
        full_plan = prepare_title_pairs(rebuilt["question"], rebuilt["catalog"], model_fingerprint=prep["model_manifest_sha256"])
        for record in full_plan["records"]:
            key = record["pair_sha256"]; wire = prep["pairs"].get(key)
            if (not isinstance(wire, dict) or set(wire) != {"query", "document", "tokens"}
                    or wire["query"] != full_plan["question"] or wire["document"] != record["model_document"]
                    or type(wire["tokens"]) is not int or not 0 < wire["tokens"] <= 512):
                raise ValueError("exact_original_question_title_window_pair_required")
            pair_messages[key] = wire
        full_plan["records"] = [{key: value for key, value in r.items() if key not in {"window_text", "model_document"}} for r in full_plan["records"]]
        if p["plan"] != full_plan or p["original_ids"] != {ref: meta["paragraph_id"] for ref, meta in rebuilt["source_metadata"].items()}:
            raise ValueError("frozen_title_full_window_plan_or_original_ids_changed")
        by_ref = {row["paragraph_ref"]: row for row in full_plan["parents"]}
        by_source = {location(row): ref for ref, row in by_ref.items()}
        wide = [{**row, "id": rebuilt["source_metadata"][row["paragraph_ref"]]["paragraph_id"]} for row in full_plan["parents"]]
        old_rows = _one(old_case["outputs"], lambda row: row["variant"] == "hybrid40")["candidates"]
        old_refs = []
        for row in old_rows:
            key = row["raw_doc_id"], row["raw_para_id"]
            ref = by_source.get(key)
            if ref is None or row["id"] != p["original_ids"][ref] or row["source_text_sha256"] != by_ref[ref]["source_text_sha256_utf8"]:
                raise ValueError("baseline_original_source_outside_pool_or_changed")
            old_refs.append(ref)
        if len(old_refs) != 40 or len(set(old_refs)) != 40 or p["old_hybrid_refs"] != old_refs:
            raise ValueError("same_unique_original_forty_baseline_required")
        rows = ranking["candidates"]; refs = [row["paragraph_ref"] for row in rows]
        if len(refs) != len(by_ref) or set(refs) != set(by_ref):
            raise ValueError("all_parent_ranking_incomplete_or_duplicated")
        windows = {r["window_id"]: r for r in full_plan["records"]}
        for row in rows:
            ref = row["paragraph_ref"]; record = windows.get(row.get("best_window_id")); score = row.get("local_logit")
            if type(score) not in (int, float) or not math.isfinite(score) or record is None or record["paragraph_ref"] != ref:
                raise ValueError("finite_original_parent_window_score_required")
            expected = {**by_ref[ref], "local_logit": score, "best_window_id": record["window_id"],
                        "best_window_start_cp": record["start_cp"], "best_window_end_cp": record["end_cp"]}
            if row != _source_rows([expected], p)[0]:
                raise ValueError("ranked_parent_source_metadata_or_window_changed")
        order = {ref: i for i, ref in enumerate(by_ref)}
        if refs != [r["paragraph_ref"] for r in sorted(rows, key=lambda r: (-r["local_logit"], order[r["paragraph_ref"]]))]:
            raise ValueError("frozen_descending_rank_or_original_tie_order_changed")
        clean_ranking = [{key: row[key] for key in (*by_ref[row["paragraph_ref"]], "local_logit", "best_window_id", "best_window_start_cp", "best_window_end_cp")} for row in rows]
        expected_variants = {DIAGNOSTIC: _source_rows(clean_ranking[:40], p),
            MAIN: _source_rows(fuse_title_hybrid(full_plan, clean_ranking, old_refs), p)}
        if len(current["outputs"]) != 2:
            raise ValueError("only_fixed_main_and_diagnostic_variants_required")
        variants = {BASELINE: _source_rows([by_ref[ref] for ref in old_refs], p)}
        for variant, expected in expected_variants.items():
            actual = _one(current["outputs"], lambda row: row["variant"] == variant)["candidates"]
            if actual != expected or len(actual) != 40:
                raise ValueError("fixed_forty_or_rrf60_projection_changed")
            variants[variant] = actual
        if len(fit_case["variants"]) != 2:
            raise ValueError("both_native_fit_inputs_required")
        for label in (BASELINE, MAIN):
            binding = _one(fit_case["variants"], lambda row: row["label"] == label)
            fit_source, _ = _read(binding["file"], guards, binding["sha256"])
            selected = {row["paragraph_ref"] for row in variants[label]}
            expected = {"format": "dsh-local-rerank-raw-subset-v1", "question": rebuilt["question"], "topic_title": rebuilt["topic_title"],
                "catalog": [row for row in rebuilt["catalog"] if row["paragraph_ref"] in selected],
                "source_metadata": {ref: meta for ref, meta in rebuilt["source_metadata"].items() if ref in selected},
                "binding": {"parent_case_file": p["case_input"], "parent_case_sha256": p["case_sha256"], "variant": label},
                "unprovided_context_state": "unknown_not_irrelevant", "publishable": False}
            if fit_source != expected:
                raise ValueError("complete_native_fit_source_projection_changed")
        cases.append({**common, "wide": wide, "variants": variants})
    if (freeze["parents"] != sum(len(c["wide"]) for c in cases) or freeze["parents"] != prep["parents"]
            or freeze["windows"] != prep["windows"] or freeze["unique_pairs"] != len(prep["pairs"])
            or freeze["windows"] != sum(len(c["plan"]["records"]) for c in prep["cases"])
            or pair_messages != prep["pairs"]
            or prep["schedule"] != sorted(pair_messages, key=lambda key: (pair_messages[key]["tokens"], key))):
        raise ValueError("complete_parent_window_totals_changed")
    _recheck(guards)
    return cases, guards, {"generation_complete": True, "shard_receipts": receipts,
                           "native_fit_input_sources_verified": True, "native_fit_execution_assessed": False}


def _recheck(guards):
    for descriptor in guards:
        bound(descriptor["file"], descriptor["sha256"])


def _load_references(sample_root, cases, guards):
    """The caller must complete closed_sources and freeze its score plan first."""
    mapper, engine, mapper_guards = load_reference_tools(); guards.extend(mapper_guards)
    references, topics = {}, {}; sample_root = Path(sample_root).resolve()
    for case in cases:
        topic = case["topic_id"]
        if topic not in topics:
            folder = (sample_root / case["title"]).resolve()
            if folder.parent != sample_root:
                raise ValueError("reference_topic_directory_escape")
            evidence, _ = _read(folder / "evidence_samples.json", guards, references=True)
            conclusions, _ = _read(folder / "conclusions.json", guards, references=True)
            topics[topic] = (evidence, conclusions)
        evidence, conclusions = topics[topic]
        conclusion = _one(conclusions, lambda row: row["question_id"] == case["question_id"])
        if conclusion["question"] != case["question"]:
            raise ValueError("complete_original_reference_question_changed")
        union, issues = mapper.question_union_reference(evidence, conclusion)
        sides = [engine._relation_rows(conclusion[side], evidence, submitted=False, issues=issues,
                 context=[topic, case["question_id"], side]) for side in ("supporting_facts", "conflicting_facts")]
        references[(topic, case["question_id"])] = {"question_union": union,
            "supporting_sources": sides[0], "conflicting_sources": sides[1], "topic_evidence": evidence, "issues": issues}
    return references, engine


def _coverage(engine, topic, candidates, reference):
    measured = engine.compare_rows(topic, candidates, reference)["unique_paragraphs"]["A_id_only"]["metrics"]
    return {"hits": measured["matches"], "candidate_sources": measured["submitted_records"],
        "reference_sources": measured["reference_records"], "source_id_coverage": measured["recall"],
        "reference_id_density": measured["precision"],
        "unlocated_reference_records_retained": sum(location(row) is None for row in reference)}


def _aggregate(parts):
    hits = sum(p["hits"] for p in parts); candidates = sum(p["candidate_sources"] for p in parts); reference = sum(p["reference_sources"] for p in parts)
    return {"hits": hits, "candidate_sources": candidates, "reference_sources": reference,
            "source_id_coverage": hits / reference if reference else None,
            "reference_id_density": hits / candidates if candidates else None,
            "unlocated_reference_records_retained": sum(p["unlocated_reference_records_retained"] for p in parts)}


def measure_cases(cases, references, engine):
    """Pure candidate measurement using the existing deterministic ID matcher."""
    reports, topic_reports = [], []
    categories = ("question_union", "supporting_sources", "conflicting_sources")
    for case in cases:
        topic, qid = case["topic_id"], case["question_id"]; reference = references[(topic, qid)]
        gold = {location(row) for row in reference["question_union"] if location(row)}
        wide_keys = {location(row) for row in case["wide"]}
        runs = {}
        for variant in VARIANTS:
            rows = case["variants"][variant]; keys = {location(row) for row in rows}
            nonreference = [{"doc_id": r["doc_id"], "para_id": r["para_id"], "id": r["id"],
                             "reason": "source_id_not_in_question_reference_semantics_unknown"} for r in rows if location(r) not in gold]
            missed = [{"doc_id": unit["key"][0] if unit["key"] else None, "para_id": unit["key"][1] if unit["key"] else None,
                "reason": "unlocated_reference_retained" if unit["key"] is None else "outside_wide_pool_entry" if unit["key"] not in wide_keys else "inside_wide_pool_not_in_fixed40"}
                for unit in engine._units(reference["question_union"], True) if unit["key"] is None or unit["key"] not in keys]
            runs[variant] = {category: _coverage(engine, topic, rows, reference[category]) for category in categories}
            runs[variant].update(candidate_sources=rows, nonreference_source_count=len(nonreference),
                nonreference_source_examples=nonreference[:5], missed_reference_sources=missed)
        old_keys = {location(r) for r in case["variants"][BASELINE]}; new_keys = {location(r) for r in case["variants"][MAIN]}
        diff = {label: [{"doc_id": key[0], "para_id": key[1], "in_question_reference_by_id": key in gold}
                for key in sorted(keys)] for label, keys in (("added_sources", new_keys-old_keys), ("dropped_sources", old_keys-new_keys))}
        reports.append({key: case[key] for key in ("topic_id", "question_id", "title", "question")} | {
            "wide_pool": {category: _coverage(engine, topic, case["wide"], reference[category]) for category in categories},
            "variants": runs, "main_vs_baseline": diff, "reference_issues": reference["issues"]})
    for topic in dict.fromkeys(c["topic_id"] for c in cases):
        local = [r for r in reports if r["topic_id"] == topic]
        totals = {variant: {cat: _aggregate([r["variants"][variant][cat] for r in local]) for cat in categories} for variant in VARIANTS}
        selected_cases = [c for c in cases if c["topic_id"] == topic]
        topic_evidence = references[(topic, selected_cases[0]["question_id"])]["topic_evidence"]
        topic_union = {}
        for variant in VARIANTS:
            unique = {location(row): row for c in selected_cases for row in c["variants"][variant]}
            topic_union[variant] = _coverage(engine, topic, list(unique.values()), topic_evidence)
        topic_reports.append({"topic_id": topic, "question_micro": totals,
            "wide_pool_question_micro": {cat: _aggregate([r["wide_pool"][cat] for r in local]) for cat in categories},
            "main_net_source_hits": totals[MAIN]["question_union"]["hits"]-totals[BASELINE]["question_union"]["hits"],
            "topic_three_question_candidate_union_vs_complete_topic_evidence": topic_union})
    totals = {variant: {category: _aggregate([r["variants"][variant][category] for r in reports]) for category in categories} for variant in VARIANTS}
    denominator_ok = [totals[MAIN][cat]["reference_sources"] for cat in categories] == [181, 105, 83]
    baseline_ok = totals[BASELINE]["question_union"]["hits"] == 82
    gates = {"reference_denominators_181_105_83": denominator_ok, "baseline_82_reproduced": baseline_ok,
        "main_at_least_88_hits": totals[MAIN]["question_union"]["hits"] >= 88,
        "at_least_three_topics_net_gain": sum(r["main_net_source_hits"] > 0 for r in topic_reports) >= 3,
        "no_topic_loses_more_than_one": all(r["main_net_source_hits"] >= -1 for r in topic_reports)}
    return {"cases": reports, "per_topic": topic_reports, "question_micro": totals,
        "wide_pool_question_micro": {cat: _aggregate([r["wide_pool"][cat] for r in reports]) for cat in categories},
        "fixed_candidate_gate": {**gates, "candidate_gate_passed": all(gates.values()),
            "native_fit_gate": "not_assessed_requires_separate_closed_native_message_counts", "promotion_authorized": False},
        "examples_are_source_id_proxies_not_semantic_false_positives": True}


def score_closed(freeze_path, reviewed_sha256, sample_root, output, **source_options):
    output = Path(output).resolve(); sample_root = Path(sample_root).resolve(); generation = Path(freeze_path).resolve().parent
    source_plan_path = output.with_suffix(".source-plan.json")
    if output.exists() or source_plan_path.exists() or output.is_relative_to(sample_root) or output.is_relative_to(generation):
        raise ValueError("new_separate_offline_score_output_required")
    cases, guards, closure = closed_sources(freeze_path, reviewed_sha256, **source_options)
    # Freeze the complete source projection before the FIRST answer-reference IO.
    output.parent.mkdir(parents=True, exist_ok=True)
    with source_plan_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"format": "title-aware-source-only-score-plan-v1", "cases": cases,
                   "contains_reference_material": False, "generation_freeze_sha256": reviewed_sha256}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    _, descriptor = bound(source_plan_path); guards.append(descriptor)
    references, engine = _load_references(sample_root, cases, guards)
    result = measure_cases(cases, references, engine)
    for path in (Path(__file__), ROOT / "app/domain/frozen_selection_inputs.py"):
        _, descriptor = bound(path); guards.append(descriptor)
    _recheck(guards)
    result.update(format="closed-title-aware-candidate-source-diagnostics-v1", closure=closure,
        primary_variant=MAIN, diagnostic_variant=DIAGNOSTIC, variant_selected_after_reference_access=False,
        reference_scope="per-question S union C; topic evidence candidate-union is separate",
        official_score=False, final_evidence_or_relation_score=False, semantic_score_created=False,
        candidate_presence_proves_model_reading=False, relation_sides_inferred_from_candidate_rank=False,
        source_guards=guards, source_guards_unchanged=True, contains_reference_material=True,
        reference_written_to_runtime=False, new_model_calls=0, new_network_calls=0)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("freeze", "reviewed-sha256", "sample-root", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    report = score_closed(args.freeze, args.reviewed_sha256, args.sample_root, args.output)
    print(json.dumps({"main": report["question_micro"][MAIN], "candidate_gate": report["fixed_candidate_gate"], "new_model_calls": 0}))
