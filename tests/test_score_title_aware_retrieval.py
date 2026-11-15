"""Synthetic closure/source and denominator regressions; no competition answers."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from app.domain.frozen_selection_inputs import build_frozen_selection_input, canonical, POOL_FORMAT
from app.domain.title_aware_rerank import prepare_title_pairs, rank_title_parents, fuse_title_hybrid
from scripts import score_title_aware_retrieval as scoring
from scripts.offline_reference_tools import load_reference_tools


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value); path.write_bytes(data)
    return sha256(data).hexdigest()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def closed(tmp_path):
    folder = tmp_path / "generated"; folder.mkdir()
    pool = {"format": POOL_FORMAT, "reference_reads": [], "inputs": [], "cases": []}
    old = {"format": "local-e5-dense-hybrid-fixed40-v1", "cases": []}
    raw_inputs = {}
    for topic in dict.fromkeys(t for t, _ in scoring.CASE_IDS):
        paragraphs, content = [], ""
        for i in range(1, 46):
            # Two identical literal parents keep independent source identities.
            text = "相同原文😀。\n" if i in (1, 2) else f"原始段落 {i}，不改空白。\r\n"
            paragraphs.append({"para_id": f"p{i}", "text": text, "char_start": len(content), "char_end": len(content)+len(text)})
            content += text
        raw = {"title": f"合成专题{topic}", "documents": [{"doc_id": "d", "title": "原始标题", "content": content, "paragraphs": paragraphs}],
               "questions": [{"question_id": qid, "question": f"合成{topic} {qid}完整比较条件？"} for t, qid in scoring.CASE_IDS if t == topic]}
        path = tmp_path / "raw" / f"{topic}.json"; raw_sha = save(path, raw)
        pool["inputs"].append({"topic_id": topic, "title": raw["title"], "path": str(path), "sha256": raw_sha})
        raw_inputs[topic] = (path, raw_sha)
        for question in raw["questions"]:
            rows = [{"id": f"{topic}/d/p{i}", "raw_doc_id": "d", "raw_para_id": f"p{i}",
                     "source_text_sha256": sha256(p["text"].encode()).hexdigest()} for i, p in enumerate(paragraphs, 1)]
            common = {"topic_id": topic, "question_id": question["question_id"], "title": raw["title"], "question": question["question"]}
            pool["cases"].append({**common, "outputs": [{"variant": "four_channel_quota400", "limit": 400, "candidates": rows,
                                                          "candidate_sha256": sha256(canonical(rows)).hexdigest()}]})
            old["cases"].append({**common, "outputs": [{"variant": "hybrid40", "candidates": rows[:40]}]})
    pool_path = tmp_path / "pool.json"; pool_sha = save(pool_path, pool)
    old_path = tmp_path / "old.json"; old_sha = save(old_path, old)
    prep = {"format": "title-aware-bge-widepool-preparation-v1", "overflow": [], "queries_truncated": 0,
        "paid_api_calls": 0, "reference_reads": [], "guards": [], "cases": [], "pairs": {}, "schedule": [],
        "model_manifest_sha256": "f"*64, "parents": 0, "windows": 0,
        "script_sha256": digest(scoring.ROOT / "scripts/probe_title_aware_rerank.py"),
        "helper_sha256": digest(scoring.ROOT / "app/domain/title_aware_rerank.py")}
    fixed = {"format": "title-aware-full-window-local-rerank-candidates-v1", "all_parent_windows_scored": True,
             "paid_api_calls": 0, "reference_reads": [], "cases": []}
    ranked = {"cases": []}; fit = {"format": "frozen-local-rerank-native-fit-inputs-v1", "cases": []}
    for ordinal, (topic, qid) in enumerate(scoring.CASE_IDS, 1):
        raw_path, raw_sha = raw_inputs[topic]
        case = build_frozen_selection_input(raw_input_bytes=raw_path.read_bytes(), pool_input_bytes=pool_path.read_bytes(),
            topic_id=topic, question_id=qid, expected_raw_sha256=raw_sha, expected_pool_sha256=pool_sha)
        case_path = tmp_path / "cases" / f"{ordinal}.json"; case_sha = save(case_path, case)
        plan = prepare_title_pairs(case["question"], case["catalog"], model_fingerprint=prep["model_manifest_sha256"])
        scores = {row["pair_sha256"]: -float(i) for i, row in enumerate(plan["records"])}
        all_rows = rank_title_parents(plan, scores); old_refs = [r["paragraph_ref"] for r in case["catalog"][:40]]
        fused = fuse_title_hybrid(plan, all_rows, old_refs)
        p = {"topic_id": topic, "question_id": qid, "title": case["topic_title"], "case_input": str(case_path), "case_sha256": case_sha,
             "old_hybrid_refs": old_refs, "original_ids": {r: m["paragraph_id"] for r, m in case["source_metadata"].items()}}
        common = {"topic_id": topic, "question_id": qid, "title": case["topic_title"], "question": case["question"]["question"]}
        fixed["cases"].append({**common, "outputs": [{"variant": scoring.DIAGNOSTIC, "candidates": scoring._source_rows(all_rows[:40], p)},
                                                        {"variant": scoring.MAIN, "candidates": scoring._source_rows(fused, p)}]})
        ranked["cases"].append({**common, "candidates": scoring._source_rows(all_rows, p)})
        fits = []
        for label in (scoring.BASELINE, scoring.MAIN):
            fit_source = {"format": "dsh-local-rerank-raw-subset-v1", "question": case["question"], "topic_title": case["topic_title"],
                "catalog": case["catalog"][:40], "source_metadata": {r["paragraph_ref"]: case["source_metadata"][r["paragraph_ref"]] for r in case["catalog"][:40]},
                "binding": {"parent_case_file": str(case_path), "parent_case_sha256": case_sha, "variant": label},
                "unprovided_context_state": "unknown_not_irrelevant", "publishable": False}
            path = folder / "fit" / f"{ordinal}-{label}.json"; hashed = save(path, fit_source)
            fits.append({"label": label, "file": str(path), "sha256": hashed})
        fit["cases"].append({"topic_id": topic, "question_id": qid, "variants": fits})
        for row in plan["records"]:
            prep["pairs"][row["pair_sha256"]] = {"query": plan["question"], "document": row["model_document"], "tokens": 1}
        plan["records"] = [{k: v for k, v in row.items() if k not in {"window_text", "model_document"}} for row in plan["records"]]
        p["plan"] = plan; prep["cases"].append(p)
        prep["parents"] += len(plan["parents"]); prep["windows"] += len(plan["records"])
    prep["schedule"] = sorted(prep["pairs"])
    prep_sha = save(folder / "preparation.json", prep); fixed["preparation_sha256"] = prep_sha
    script_sha = digest(scoring.ROOT / "scripts/run_title_aware_full.py")
    receipts = []; batches = (len(prep["schedule"])+3)//4
    for shard in (0, 1):
        start, end = (0, batches//2) if shard == 0 else (batches//2, batches)
        receipt = {"closed": True, "shard": shard, "preparation_sha256": prep_sha, "script_sha256": script_sha,
            "global_batch_start": start, "global_batch_end_exclusive": end, "pairs": len(prep["schedule"][start*4:end*4]),
            "paid_api_calls": 0, "network_attempts": [], "reference_attempts": [], "probe_scores_reused": False}
        path = folder / f"shard-{shard}" / "final-receipt.json"; hashed = save(path, receipt)
        receipts.append({"file": str(path), "sha256": hashed})
    freeze = {"generation_complete": True, "cases": 18, "parents": prep["parents"], "windows": prep["windows"], "unique_pairs": len(prep["pairs"]),
        "fixed_candidates_sha256": save(folder / "fixed-candidates.json", fixed), "all_parent_ranking_sha256": save(folder / "all-parent-ranking.json", ranked),
        "native_fit_index_sha256": save(folder / "native-fit-index.json", fit), "preparation_sha256": prep_sha,
        "shard_receipts": receipts, "reference_reads": [], "paid_api_calls": 0, "script_sha256": script_sha}
    freeze_path = folder / "candidate-freeze.json"; freeze_sha = save(freeze_path, freeze)
    return {"folder": folder, "freeze_path": freeze_path, "freeze": freeze, "sha": freeze_sha,
            "options": {"pool_path": pool_path, "pool_sha256": pool_sha, "baseline_path": old_path, "baseline_sha256": old_sha}}


def test_closed_all18_source_reconstruction_and_independent_same_body_ids(closed):
    cases, guards, status = scoring.closed_sources(closed["freeze_path"], closed["sha"], **closed["options"])
    assert len(cases) == 18 and len(guards) > 18
    assert len(cases[0]["wide"]) == 45
    assert cases[0]["wide"][0]["source_text_sha256_utf8"] == cases[0]["wide"][1]["source_text_sha256_utf8"]
    assert cases[0]["wide"][0]["id"] != cases[0]["wide"][1]["id"]
    assert all(len(c["variants"][v]) == 40 for c in cases for v in scoring.VARIANTS)
    assert status["native_fit_input_sources_verified"] and not status["native_fit_execution_assessed"]


def test_resealed_pair_cannot_replace_title_or_question_behind_original_pair_key(closed):
    folder, freeze = closed["folder"], closed["freeze"]
    path = folder / "preparation.json"; prep = json.loads(path.read_bytes())
    prep["pairs"][prep["schedule"][0]]["document"] = "改写的标题与窗口"
    prep_sha = save(path, prep); freeze["preparation_sha256"] = prep_sha
    fixed_path = folder / "fixed-candidates.json"; fixed = json.loads(fixed_path.read_bytes()); fixed["preparation_sha256"] = prep_sha
    freeze["fixed_candidates_sha256"] = save(fixed_path, fixed)
    for binding in freeze["shard_receipts"]:
        receipt = json.loads(Path(binding["file"]).read_bytes()); receipt["preparation_sha256"] = prep_sha
        binding["sha256"] = save(Path(binding["file"]), receipt)
    hashed = save(closed["freeze_path"], freeze)
    with pytest.raises(ValueError, match="exact_original_question_title_window_pair"):
        scoring.closed_sources(closed["freeze_path"], hashed, **closed["options"])


@pytest.mark.parametrize("damage", ["not_closed", "unbound_bytes", "question", "duplicate", "source_sha", "source_id", "shard_network", "fit_truncated"])
def test_invalid_generation_never_reads_references(closed, tmp_path, monkeypatch, damage):
    folder, freeze = closed["folder"], closed["freeze"]
    if damage == "not_closed":
        freeze["generation_complete"] = False
    elif damage == "shard_network":
        path = Path(freeze["shard_receipts"][0]["file"]); receipt = json.loads(path.read_bytes()); receipt["network_attempts"] = ["socket.connect"]
        freeze["shard_receipts"][0]["sha256"] = save(path, receipt)
    elif damage == "fit_truncated":
        path = folder / "native-fit-index.json"; fit = json.loads(path.read_bytes()); binding = fit["cases"][0]["variants"][0]
        source = json.loads(Path(binding["file"]).read_bytes()); source["catalog"][0]["text"] = "截断"
        binding["sha256"] = save(Path(binding["file"]), source); freeze["native_fit_index_sha256"] = save(path, fit)
    else:
        path = folder / "fixed-candidates.json"; value = json.loads(path.read_bytes()); case = value["cases"][0]
        row = case["outputs"][0]["candidates"][0]
        if damage == "question": case["question"] += "另加条件"
        elif damage == "duplicate": case["outputs"][0]["candidates"][1] = deepcopy(row)
        elif damage == "source_sha": row["source_text_sha256_utf8"] = "0"*64
        elif damage == "source_id": row["id"] = "01/another/p1"
        save(path, value)
        if damage != "unbound_bytes": freeze["fixed_candidates_sha256"] = digest(path)
        else: path.write_bytes(path.read_bytes()+b" ")
    hashed = save(closed["freeze_path"], freeze)
    def forbidden(*args):
        pytest.fail("reference loader reached before closure/source validation")
    monkeypatch.setattr(scoring, "_load_references", forbidden)
    with pytest.raises(ValueError):
        scoring.score_closed(closed["freeze_path"], hashed, tmp_path / "answers", tmp_path / "report.json", **closed["options"])
    assert not (tmp_path / "report.source-plan.json").exists()


def row(p, doc="d"):
    return {"id": doc+"/"+p, "doc_id": doc, "para_id": p}


def test_question_union_and_topic_union_keep_distinct_denominators_and_diagnostic_cannot_replace_main():
    mapper, engine, _ = load_reference_tools()
    facts = [{**row("p1"), "fact_id": "f1"}, {**row("p2"), "fact_id": "f2"}, {**row("p3"), "fact_id": "f3"}]
    union, issues = mapper.question_union_reference(facts, {"supporting_facts": [{"fact_id": "f1"}, {"fact_id": "f2"}],
        "conflicting_facts": [{"fact_id": "f2"}]})
    assert issues == [] and len(union) == 2
    cases, refs = [], {}
    for topic, qid in (("a", "q1"), ("a", "q2"), ("b", "q1")):
        cases.append({"topic_id": topic, "question_id": qid, "title": "synthetic", "question": "完整题面？", "wide": facts,
            "variants": {scoring.BASELINE: [row("p1")], scoring.MAIN: [row("p1")], scoring.DIAGNOSTIC: [row("p1"), row("p2")]}})
        refs[(topic, qid)] = {"question_union": union, "supporting_sources": facts[:2], "conflicting_sources": facts[1:2], "topic_evidence": facts, "issues": []}
    report = scoring.measure_cases(cases, refs, engine)
    main = report["question_micro"][scoring.MAIN]
    assert main["question_union"]["hits"] == 3 and main["question_union"]["reference_sources"] == 6
    assert main["supporting_sources"]["reference_sources"] == 6 and main["conflicting_sources"]["reference_sources"] == 3
    assert report["question_micro"][scoring.DIAGNOSTIC]["question_union"]["hits"] == 6
    topic_union = report["per_topic"][0]["topic_three_question_candidate_union_vs_complete_topic_evidence"][scoring.MAIN]
    assert topic_union["candidate_sources"] == 1 and topic_union["hits"] == 1 and topic_union["reference_sources"] == 3
    assert report["per_topic"][0]["question_micro"][scoring.MAIN]["question_union"]["hits"] == 2
    assert not report["fixed_candidate_gate"]["promotion_authorized"]
    assert "not_assessed" in report["fixed_candidate_gate"]["native_fit_gate"]


def test_unlocated_reference_and_entry_vs_fixed40_misses_are_not_silently_dropped():
    _, engine, _ = load_reference_tools()
    case = {"topic_id": "a", "question_id": "q", "title": "synthetic", "question": "原题？", "wide": [row("p1"), row("p2")],
        "variants": {v: [row("p1"), row("nonreference")] for v in scoring.VARIANTS}}
    reference = [row("p2"), row("outside"), {"_unresolved_reference": "missing"}]
    refs = {("a", "q"): {"question_union": reference, "supporting_sources": reference, "conflicting_sources": [], "topic_evidence": reference, "issues": []}}
    report = scoring.measure_cases([case], refs, engine)
    run = report["cases"][0]["variants"][scoring.MAIN]
    assert run["question_union"]["reference_sources"] == 3 and run["question_union"]["unlocated_reference_records_retained"] == 1
    assert {r["reason"] for r in run["missed_reference_sources"]} == {"inside_wide_pool_not_in_fixed40", "outside_wide_pool_entry", "unlocated_reference_retained"}
    assert run["nonreference_source_count"] == 2
    assert report["examples_are_source_id_proxies_not_semantic_false_positives"]


@pytest.mark.parametrize("new_hits,expected", [([8, 12, 21, 21, 6, 20], True),
    ([9, 13, 19, 21, 6, 20], False), ([8, 12, 22, 19, 7, 20], False), ([7, 11, 20, 21, 6, 20], False)])
def test_fixed_gate_requires_six_hit_gain_three_topics_and_no_two_hit_topic_loss(new_hits, expected):
    _, engine, _ = load_reference_tools(); cases, references = [], {}
    for i, (n, s, c, before, after) in enumerate(zip([31,30,30,30,30,30], [18,18,18,17,17,17],
            [14,14,14,14,14,13], [6,10,19,21,6,20], new_hits), 1):
        gold = [row(f"p{x}") for x in range(n)]; overlap = s+c-n
        chosen = lambda hits: gold[:hits]+[row(f"b{x}") for x in range(40-hits)]
        cases.append({"topic_id": str(i), "question_id": "q", "title": "合成", "question": "完整原题？", "wide": gold,
            "variants": {scoring.BASELINE: chosen(before), scoring.MAIN: chosen(after), scoring.DIAGNOSTIC: chosen(n)}})
        references[(str(i), "q")] = {"question_union": gold, "supporting_sources": gold[:s],
            "conflicting_sources": gold[s-overlap:s-overlap+c], "topic_evidence": gold, "issues": []}
    report = scoring.measure_cases(cases, references, engine); gate = report["fixed_candidate_gate"]
    assert gate["reference_denominators_181_105_83"] and gate["baseline_82_reproduced"]
    assert gate["candidate_gate_passed"] is expected and gate["promotion_authorized"] is False
    # Perfect diagnostic ranking cannot rescue a failed preregistered main gate.
    assert report["question_micro"][scoring.DIAGNOSTIC]["question_union"]["hits"] == 181


def test_source_plan_is_persisted_before_reference_loader_and_report_never_certifies_final_metrics(closed, tmp_path, monkeypatch):
    output = tmp_path / "offline" / "report.json"
    def synthetic_references(sample_root, cases, guards):
        plan = output.with_suffix(".source-plan.json")
        assert plan.exists() and json.loads(plan.read_bytes())["contains_reference_material"] is False
        _, engine, _ = load_reference_tools()
        refs = {(c["topic_id"], c["question_id"]): {"question_union": [row("p1"), row("p2")],
            "supporting_sources": [row("p1")], "conflicting_sources": [row("p2")],
            "topic_evidence": [row("p1"), row("p2"), row("p45")], "issues": []} for c in cases}
        return refs, engine
    monkeypatch.setattr(scoring, "_load_references", synthetic_references)
    report = scoring.score_closed(closed["freeze_path"], closed["sha"], tmp_path / "answers", output, **closed["options"])
    assert output.exists() and report["question_micro"][scoring.MAIN]["question_union"]["hits"] == 36
    assert report["primary_variant"] == scoring.MAIN and report["diagnostic_variant"] == scoring.DIAGNOSTIC
    assert report["new_model_calls"] == 0 and report["new_network_calls"] == 0
    for key in ("official_score", "final_evidence_or_relation_score", "semantic_score_created", "candidate_presence_proves_model_reading", "reference_written_to_runtime"):
        assert report[key] is False
    with pytest.raises(ValueError, match="new_separate"):
        scoring.score_closed(closed["freeze_path"], closed["sha"], tmp_path / "answers", output, **closed["options"])
