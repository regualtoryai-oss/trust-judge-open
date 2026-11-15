"""Offline, read-only diagnostics against development samples; never a model score.

Gold answers are consumed only by this explicitly invoked script. No runtime
repository, provider, index, HTTP client or model is created. Both SQLite inputs
are opened mode=ro; outputs live in docs, outside runtime ingestion.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain.snapshot import competition_files
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


CASES = (
    ("2026·8·26 吉隆口岸泥石流灾害", "87eeea08-4062-4df2-8d00-89bee4e0d5b5"),
    ("刘翔编制“二选一”事件", "f350a502-58e2-4202-91a2-9f8ba389d15e"),
    ("印度航空 AI171 空难（2025.6–2026.5）", "a3077d34-8fa9-435b-8858-e1348b4ed706"),
    ("王虹“北大discouraged”事件", "d66c3ba9-a45a-48ce-a129-7ed88b80268b"),
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ro_connect(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def source_fingerprint(connection):
    return fingerprint({table: [dict(r) for r in connection.execute("SELECT * FROM " + table + " ORDER BY id")]
                        for table in ("documents", "paragraphs")})


def paragraph_key(row):
    return row["doc_id"], row["para_id"]


def quote_key(row):
    return *paragraph_key(row), row["quote"]


def rows_for_keys(values):
    return [{"doc_id": v[0], "para_id": v[1]} for v in sorted(values)]


def set_metrics(predicted, reference):
    """Deduplicate exact keys; empty precision is undefined, never a free 100%."""
    predicted, reference = set(predicted), set(reference)
    hit = len(predicted & reference)
    precision = hit / len(predicted) if predicted else None
    recall = hit / len(reference) if reference else None
    f1 = (2 * hit / (len(predicted) + len(reference))) if predicted or reference else None
    return {"matched": hit, "predicted": len(predicted), "reference": len(reference),
            "precision": precision, "recall": recall, "f1": f1}


def evidence_metrics(predicted, reference):
    """Paragraph filtering is necessary but explicitly not semantic matching."""
    p_unique = {quote_key(x): x for x in predicted}
    g_unique = {quote_key(x): x for x in reference}
    pc, gc = Counter(x[:2] for x in p_unique), Counter(x[:2] for x in g_unique)
    potential = sum(min(pc[key], gc[key]) for key in pc.keys() & gc.keys())
    return {
        "raw_counts": {"predicted": len(predicted), "reference": len(reference)},
        "paragraph_set": set_metrics(pc, gc),
        "strict_quote_set": set_metrics(p_unique, g_unique),
        "one_to_one_paragraph_filter_upper_bound": {
            "potential_matches": potential, "predicted_unique_quotes": len(p_unique),
            "reference_unique_quotes": len(g_unique),
            "precision_upper_bound": potential / len(p_unique) if p_unique else None,
            "recall_upper_bound": potential / len(g_unique) if g_unique else None,
            "f1_upper_bound": 2 * potential / (len(p_unique) + len(g_unique)) if p_unique or g_unique else None,
            "meaning": "Maximum one-to-one same-document/paragraph candidates before semantic checks. Not official M; quote duplicates removed. No cross-source semantic rescue is assumed.",
        },
        "missed_reference_paragraphs": rows_for_keys(set(gc) - set(pc)),
        "predicted_paragraphs_outside_reference": rows_for_keys(set(pc) - set(gc)),
    }


def raw_lookup(raw):
    docs = {d["doc_id"]: d for d in raw}
    paras = {(d["doc_id"], p["para_id"]): d["content"][p["char_start"]:p["char_end"]]
             for d in raw for p in d["paragraphs"]}
    return docs, paras


def quote_audit(evidence, raw):
    docs, paras = raw_lookup(raw)
    counts = Counter()
    issues = []
    for index, item in enumerate(evidence):
        document = docs.get(item["doc_id"])
        para = paras.get(paragraph_key(item))
        quote = item["quote"]
        valid_doc = document is not None and quote in document["content"]
        valid_para = para is not None and quote in para
        counts["document_literal_hits"] += bool(valid_doc)
        counts["declared_paragraph_literal_hits"] += bool(valid_para)
        if not valid_para:
            issues.append({"index": index, "fact_id": item.get("fact_id"),
                           **dict(zip(("doc_id", "para_id"), paragraph_key(item))),
                           "quote_in_document": valid_doc, "quote_sha256": hashlib.sha256(quote.encode()).hexdigest()})
        start, end = item.get("char_start"), item.get("char_end")
        if type(start) is int and type(end) is int:
            counts["declared_span_rows"] += 1
            absolute = document is not None and 0 <= start < end <= len(document["content"]) and document["content"][start:end] == quote
            relative = para is not None and 0 <= start < end <= len(para) and para[start:end] == quote
            counts["declared_document_spans_exact"] += bool(absolute)
            counts["declared_paragraph_relative_spans_exact"] += bool(relative)
    return {"records": len(evidence), **{k: counts[k] for k in (
        "document_literal_hits", "declared_paragraph_literal_hits", "declared_span_rows",
        "declared_document_spans_exact", "declared_paragraph_relative_spans_exact")},
        "unlocatable_declared_paragraph_quotes": issues,
        "note": "Span coordinate conventions in supplied examples are diagnosed separately; no repaired gold or normalized quote is used."}


def stored_span_audit(evidence, raw):
    """Internal Unicode codepoint positions, independent of sample quote matching."""
    _, paragraphs = raw_lookup(raw)
    exact = 0
    for item in evidence:
        paragraph = paragraphs.get(paragraph_key(item))
        start, end = item.get("start_cp"), item.get("end_cp")
        exact += bool(paragraph is not None and type(start) is int and type(end) is int
                      and 0 <= start < end <= len(paragraph) and paragraph[start:end] == item["quote"])
    return {"records": len(evidence), "stored_codepoint_spans_exact": exact,
            "original_raw_literal_audit": quote_audit(evidence, raw)}


def field_exact_diagnostic(predicted, reference):
    p, g = {quote_key(x): x for x in predicted}, {quote_key(x): x for x in reference}
    pairs = []
    for key in sorted(p.keys() & g.keys()):
        left, right = p[key], g[key]
        le, re = set(left["entities"]), set(right["entities"])
        entity = set_metrics(le, re)
        pairs.append({"doc_id": key[0], "para_id": key[1], "reference_fact_id": right.get("fact_id"),
                      "quote_sha256": hashlib.sha256(key[2].encode()).hexdigest(),
                      "argument_literal_equal": {name: left["argument"].get(name) == right["argument"].get(name)
                                                 for name in ("时间", "主体", "地点", "数值口径")},
                      "entity_set": entity,
                      "entities_p_and_r_at_least_point8": entity["precision"] is not None and entity["recall"] is not None and min(entity["precision"], entity["recall"]) >= .8})
    return {"strict_quote_pair_count": len(pairs), "pairs": pairs,
            "note": "Literal equality on strictly paired quotes only. Different wording/null versus omitted is not semantically graded; this is NOT official Field_acc."}


def load_retrieved(checkpoints, run_id, question_id, *, include_channels=False):
    thread_id = run_id + ":" + question_id
    row = checkpoints.execute("SELECT checkpoint_id,type,checkpoint FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1", (thread_id,)).fetchone()
    if row is None:
        result = ({"available": False, "thread_id": thread_id}, [])
        return (*result, {}) if include_channels else result
    cp = JsonPlusSerializer().loads_typed((row["type"], row["checkpoint"]))
    retrieved = cp["channel_values"].get("retrieved")
    if not isinstance(retrieved, list):
        result = ({"available": False, "thread_id": thread_id, "checkpoint_id": row["checkpoint_id"]}, [])
        return (*result, cp["channel_values"]) if include_channels else result
    result = ({"available": True, "thread_id": thread_id, "checkpoint_id": row["checkpoint_id"],
            "checkpoint_blob_sha256": hashlib.sha256(row["checkpoint"]).hexdigest(),
            "retrieved_fingerprint_sha256": fingerprint(retrieved), "paragraph_count": len(retrieved),
            "meaning": "Checkpoint source selection or scan plan only; listing all paragraphs does not prove completed model scanning.",
            "recorded_source_coverage": cp["channel_values"].get("source_coverage")}, retrieved)
    return (*result, cp["channel_values"]) if include_channels else result


def selection_audit(selection, candidates, selected_evidence, relevant_gold):
    """Visible model dispositions; paragraph overlap is not a false-reject label."""
    if not isinstance(selection, dict) or not isinstance(candidates, list):
        return {"available": False, "note": "No recorded precision-selection candidate ledger; old outputs are not reconstructed."}
    converted = [{"doc_id": item["raw_doc_id"], "para_id": item["raw_para_id"], **item} for item in candidates]
    selected = {item["id"] for item in selected_evidence}
    rejected = [item for item in converted if item["id"] not in selected]
    assessments = selection.get("assessments")
    by_id = {row["evidence_id"]: row for row in assessments} if isinstance(assessments, list) else {}
    return {"available": True, "applied": selection.get("applied"),
            "candidate_records": len(candidates), "selected_records": len(selected_evidence), "rejected_records": len(rejected),
            "candidate_reference_paragraph_comparison": evidence_metrics(converted, relevant_gold),
            "rejected_reference_paragraph_filter": evidence_metrics(rejected, relevant_gold),
            "assessment_relation_counts": dict(Counter(row.get("relation") for row in by_id.values())),
            "candidate_dispositions": [{"evidence_id": item["id"], "doc_id": item["doc_id"], "para_id": item["para_id"],
                                         "quote_sha256": hashlib.sha256(item["quote"].encode()).hexdigest(),
                                         "retained": item["id"] in selected,
                                         **{key: by_id.get(item["id"], {}).get(key) for key in ("relation", "reason", "equivalent_to")}}
                                        for item in converted],
            "note": "A rejected candidate sharing a sample paragraph is only a semantic-review candidate, not proof of erroneous deletion. Manual preserved evidence remains visible. Raw candidate source quotes are hashed; no hidden model reasoning is read."}


def source_scan_audit(connection, run_id, question_id, retrieved, checkpoint_coverage=None):
    """Audit actual persistent scan progress separately from retrieval planning.

    Old runs lack these events; unknown is not zero and a full retrieved corpus
    alone is never treated as completed scanning or semantic evidence recall.
    """
    recorded = []
    plan, progress = None, None
    for row in connection.execute("SELECT seq,type,data_json FROM run_events WHERE run_id=? AND type IN ('extract_coverage_planned','extract_coverage_progress','source_coverage_planned','source_coverage_progress') ORDER BY seq", (run_id,)):
        data = json.loads(row["data_json"])
        if data.get("question_id") != question_id:
            continue
        totals = [data.get(name) for name in ("batch_count", "total_paragraphs")]
        if any(type(value) is not int or value < 0 for value in totals):
            continue
        if row["type"].endswith("_planned"):
            if data.get("strategy") != "full_source_scan":
                continue
            plan, progress = {"seq": row["seq"], **data}, None
        else:
            counts = [data.get(name) for name in ("completed_batches", "scanned_paragraphs")]
            if any(type(value) is not int or value < 0 for value in counts) or counts[0] > totals[0] or counts[1] > totals[1]:
                continue
            progress = {"seq": row["seq"], **data}
        recorded.append(row["seq"])
    progress_origin = "persisted_progress_events" if progress else None
    if isinstance(checkpoint_coverage, dict) and checkpoint_coverage.get("strategy") == "full_source_scan":
        keys = ("batch_count", "total_paragraphs", "completed_batches", "scanned_paragraphs")
        counts = [checkpoint_coverage.get(name) for name in keys]
        if (all(type(value) is int and value >= 0 for value in counts)
                and counts[2] <= counts[0] and counts[3] <= counts[1]
                and type(checkpoint_coverage.get("complete")) is bool):
            progress = checkpoint_coverage
            progress_origin = "persisted_checkpoint_source_coverage"
    total = progress or plan
    complete = (progress["completed_batches"] == progress["batch_count"]
                and progress["scanned_paragraphs"] == progress["total_paragraphs"]
                and progress.get("complete", True) is True) if progress else None
    # Exact per-paragraph identities may be inferred only when recorded full scan
    # completion and the retrieved plan agree on the whole declared corpus.
    retrieved_keys = {(r["raw_doc_id"], r["raw_para_id"]) for r in retrieved}
    fully_scanned_keys = retrieved_keys if complete is True and len(retrieved_keys) == progress["total_paragraphs"] else None
    return {"strategy": (progress or plan).get("strategy") if progress or plan else None,
            "planned_source_paragraphs": len(retrieved_keys),
            "recorded_planned_total_paragraphs": total.get("total_paragraphs") if total else None,
            "recorded_batch_count": total.get("batch_count") if total else None,
            "recorded_completed_batches": progress.get("completed_batches") if progress else None,
            "recorded_scanned_paragraphs": progress.get("scanned_paragraphs") if progress else None,
            "recorded_scan_complete": complete,
            "source_paragraph_scan_coverage": (progress["scanned_paragraphs"] / progress["total_paragraphs"]
                                               if progress and progress["total_paragraphs"] else None),
            "actual_scanned_paragraph_identity_available": fully_scanned_keys is not None,
            "actual_progress_source": progress_origin,
            "progress_event_sequences": recorded,
            "note": "Recorded source scanning coverage is not evidence recall. Missing historical scan events mean unknown, not no scanning. Partial counts cannot identify which gold paragraphs were scanned."}, fully_scanned_keys


def side_diagnostic(predicted_refs, gold_refs, facts, retrieved, extracted, scanned=None):
    predicted = {paragraph_key(r) for r in predicted_refs}
    gold = {paragraph_key(facts[r["fact_id"]]) for r in gold_refs}
    retrieval_set = {(r["raw_doc_id"], r["raw_para_id"]) for r in retrieved}
    extracted_set = {(r["raw_doc_id"], r["raw_para_id"]) for r in extracted}
    return {"paragraph_set": set_metrics(predicted, gold),
            "gold_reference_rows": len(gold_refs),
            "gold_paragraphs_retrieved": len(gold & retrieval_set),
            "gold_paragraphs_planned": len(gold & retrieval_set),
            "gold_paragraphs_actual_scanned": len(gold & scanned) if scanned is not None else None,
            "gold_paragraphs_extracted": len(gold & extracted_set),
            "gold_paragraphs_final_linked": len(gold & predicted),
            "not_retrieved": rows_for_keys(gold - retrieval_set),
            "retrieved_not_extracted": rows_for_keys((gold & retrieval_set) - extracted_set),
            "extracted_not_final_linked": rows_for_keys((gold & extracted_set) - predicted),
            "extra_final_paragraphs": rows_for_keys(predicted - gold)}


def inspect_topic(connection, checkpoints, sample_root, title, topic_id, revision_id=None):
    topic = dict(connection.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone())
    revision_id = revision_id or topic["published_revision_id"]
    revision = connection.execute("SELECT topic_id,state FROM graph_revisions WHERE id=?", (revision_id,)).fetchone()
    if revision is None or revision["topic_id"] != topic_id or revision["state"] != "ready":
        raise ValueError("Evaluation requires a ready revision belonging to the selected topic.")
    row = connection.execute("SELECT payload_json FROM revision_artifacts WHERE revision_id=?", (revision_id,)).fetchone()
    snapshot = json.loads(row[0])
    files = competition_files(snapshot)  # validates schemas, refs and reproducible confidence; no runtime writes
    path = sample_root / title
    raw, gold, conclusions = (load(path / name) for name in ("raw_documents.json", "evidence_samples.json", "conclusions.json"))
    facts = {item["fact_id"]: item for item in gold}
    gold_q = {item["question_id"]: item for item in conclusions}
    questions = {r["raw_question_id"]: dict(r) for r in connection.execute("SELECT * FROM questions WHERE topic_id=?", (topic_id,))}
    question_reports = []
    completed_reference_ids = set()
    for conclusion in files["conclusions.json"]:
        qid = conclusion["question_id"]
        internal = questions[qid]["id"]
        result = snapshot["question_results"][internal]
        run_id = result["origin_run_id"]
        ck, retrieved, channels = load_retrieved(checkpoints, run_id, internal, include_channels=True)
        scan, scanned = source_scan_audit(connection, run_id, internal, retrieved, ck.get("recorded_source_coverage"))
        run = dict(connection.execute("SELECT model_config_json FROM runs WHERE id=?", (run_id,)).fetchone())
        cfg = json.loads(run["model_config_json"])
        ref = gold_q[qid]
        relevant_ids = {r["fact_id"] for side in ("supporting_facts", "conflicting_facts") for r in ref[side]}
        completed_reference_ids.update(relevant_ids)
        relevant_gold = [facts[fid] for fid in relevant_ids]
        extracted = [{"doc_id": e["raw_doc_id"], "para_id": e["raw_para_id"], **e} for e in result["evidence"]]
        conflict_records = [r for r in files["conflicts.json"] if r["question_id"] == qid]
        question_reports.append({
            "question_id": qid, "question": conclusion["question"], "origin_run_id": run_id,
            "prompt_version": cfg.get("prompt_version"), "retrieval_limit": cfg.get("retrieval_limit"),
            "input_question_exact": conclusion["question"] == ref["question"],
            "retrieval_checkpoint": ck,
            "source_scan": scan,
            "retrieval_paragraphs": [{"doc_id": r["raw_doc_id"], "para_id": r["raw_para_id"]} for r in retrieved],
            "question_referenced_gold_evidence": evidence_metrics(extracted, relevant_gold),
            "predicted_source_validation": stored_span_audit(extracted, raw),
            "evidence_selection_audit": selection_audit(result.get("evidence_selection", channels.get("evidence_selection")),
                                                         channels.get("candidate_evidence"), result["evidence"], relevant_gold),
            "support": side_diagnostic(conclusion["supporting_facts"], ref["supporting_facts"], facts, retrieved, result["evidence"], scanned),
            "conflict": side_diagnostic(conclusion["conflicting_facts"], ref["conflicting_facts"], facts, retrieved, result["evidence"], scanned),
            "predicted_decision": result["final"]["decision"],
            "predicted_conclusion": conclusion["conclusion"], "reference_conclusion": ref["conclusion"],
            "reference_verdict_label_counts": dict(Counter(r["verdict_label"] for side in ("supporting_facts", "conflicting_facts") for r in ref[side])),
            "reference_resolution_type_counts": dict(Counter(r["resolution_type"] for r in ref["conflicting_facts"])),
            "predicted_resolution_type_counts": dict(Counter(r["resolution_type"] for r in conflict_records)),
            "predicted_resolution_result_counts": dict(Counter(r["result"] for r in conflict_records)),
            "independent_conflict_record_count": len(conflict_records),
            "note": "Reference verdict_label belongs to each evidence relation, not a top-level decision label. Resolution records and sample conflicting-fact rows have different granularity; no false accuracy denominator is constructed.",
        })
    db_para = {(r["raw_doc_id"], r["raw_para_id"]): dict(r) for r in connection.execute(
        "SELECT p.*,d.raw_doc_id FROM paragraphs p JOIN documents d ON d.id=p.document_id WHERE d.topic_id=?", (topic_id,))}
    exact = sum(db_para[(e["raw_doc_id"], e["raw_para_id"])]["text"][e["start_cp"]:e["end_cp"]] == e["quote"] for e in snapshot["evidence"])
    return {"topic": title, "topic_id": topic_id, "revision_id": revision_id,
            "snapshot_payload_sha256": hashlib.sha256(row[0].encode()).hexdigest(),
            "scope": snapshot["scope"], "completed_questions": len(question_reports), "reference_questions": len(conclusions),
            "four_export_schemas_and_reference_integrity_valid": True,
            "formal_complete_topic_export_eligible": snapshot["scope"]["is_complete_topic"],
            "predicted_source_validation": {"records": len(snapshot["evidence"]), "stored_codepoint_spans_exact": exact,
                                            "original_raw_literal_audit": quote_audit(files["evidence.json"], raw)},
            "topic_wide_evidence_comparison": evidence_metrics(files["evidence.json"], gold),
            "completed_question_reference_evidence_comparison": evidence_metrics(files["evidence.json"], [facts[fid] for fid in completed_reference_ids]),
            "completed_question_reference_definition": "Union of evidence fact_ids referenced by gold answers only for completed questions; deduplicated across those questions. This is an explicitly scoped diagnostic target, not the official whole-topic evidence denominator.",
            "field_literal_diagnostic": field_exact_diagnostic(files["evidence.json"], gold),
            "unknown_source_level_evidence": sum(e.get("source_level") == "unknown" for e in snapshot["evidence"]),
            "questions": question_reports,
            "scope_warning": None if snapshot["scope"]["is_complete_topic"] else "Whole-topic gold contains unexecuted questions. Whole-topic evidence recall is coverage of available output, NOT completed-question accuracy or a full submission."}


def evaluate(database, checkpoints, sample_root, revision_map=None):
    input_files = [p for topic in sorted(sample_root.iterdir()) if topic.is_dir()
                   for name in ("raw_documents.json", "evidence_samples.json", "conclusions.json")
                   if (p := topic / name).exists()]
    initial = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    with ro_connect(database) as connection, ro_connect(checkpoints) as checkpoint_connection:
        before = source_fingerprint(connection)
        topics = [inspect_topic(connection, checkpoint_connection, sample_root, title, tid, (revision_map or {}).get(tid)) for title, tid in CASES]
        samples = [{"topic": p.name, "gold_quote_audit": quote_audit(load(p / "evidence_samples.json"), load(p / "raw_documents.json"))}
                   for p in sorted(sample_root.iterdir()) if p.is_dir() and (p / "evidence_samples.json").exists()]
        after = source_fingerprint(connection)
    assert before == after, "Source fingerprint changed during audit; reject report."
    assert initial == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}, "Sample inputs changed during audit."
    evaluated_titles = {t["topic"]: t for t in topics}
    coverage = [{"topic": p.name, "completed_question_count": evaluated_titles[p.name]["completed_questions"] if p.name in evaluated_titles else 0,
                 "reference_question_count": len(load(p / "conclusions.json")),
                 "evaluated": p.name in evaluated_titles,
                 "precision_or_recall_for_unexecuted_topic": None,
                 "note": "Published real model output evaluated." if p.name in evaluated_titles else "No published real output; excluded from precision/recall, not scored as zero accuracy."}
                for p in sorted(sample_root.iterdir()) if p.is_dir() and (p / "conclusions.json").exists()]
    return {"evaluated_at": datetime.now(timezone(timedelta(hours=8), "Asia/Shanghai")).isoformat(),
            "purpose": "Offline diagnostics against already-visible development examples; no official score and no held-out/generalization claim.",
            "model_calls": 0, "network_calls": 0, "database_access": "Both connections mode=ro and query_only=ON; no runtime repository/provider instantiated.",
            "source_fingerprint_before": before, "source_fingerprint_after": after, "source_unchanged": True,
            "sample_input_sha256": initial, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "export_contract_sha256": hashlib.sha256((ROOT / "contracts/competition-export.schema.json").read_bytes()).hexdigest(),
            "scoring_text_sha256": hashlib.sha256((ROOT.parent / "_analysis/design_input_20261017/S4-多智能体可信研判与证据链溯源系统_评分标准.txt").read_bytes()).hexdigest(),
            "method": {"matching": "Exact doc_id+para_id filtering only; official second-stage subject/time/number/scope semantic matching is NOT performed.",
                       "strict_quote": "Exact Unicode doc_id+para_id+quote after deduplication; diagnostic only, not the official semantic criterion.",
                       "supports_conflicts": "Each completed question uses deduplicated document/paragraph refs; missing questions excluded from per-question tables.",
                       "gold_quality": "All examples checked against original raw text without normalizing/fixing gold or treating supplied offsets as authoritative.",
                       "limitations": ["Mixed origin prompt versions in current published snapshots.", "Only published development-example question subsets; no unseen contest data.", "No semantic conclusion score, official Field_acc, confidence calibration or official total score is computed.", "Raw source validity is not fact correctness or complete evidence recall.", "Sample anomalies remain in official-denominator diagnostics; no convenient rows silently dropped."]},
            "sample_topic_coverage": coverage, "topics": topics, "all_sample_quote_audits": samples}


def _question_index(report):
    indexed = {}
    for topic in report["topics"]:
        for question in topic["questions"]:
            key = topic["topic_id"], question["question_id"]
            if key in indexed:
                raise ValueError("Duplicate topic/question identity in offline report.")
            indexed[key] = (topic, question)
    return indexed


def _metric_comparison(before, after):
    return {"before": before, "after": after,
            "delta": {name: after.get(name) - before.get(name)
                      if isinstance(after.get(name), (int, float)) and isinstance(before.get(name), (int, float)) else None
                      for name in ("precision", "recall", "f1")}}


def _micro_metrics(rows):
    """Question-local exact-key counts; shared evidence can count once per question."""
    values = {name: sum(row[name] for row in rows) for name in ("matched", "predicted", "reference")}
    hit, predicted, reference = (values[name] for name in ("matched", "predicted", "reference"))
    return {**values, "precision": hit / predicted if predicted else None,
            "recall": hit / reference if reference else None,
            "f1": 2 * hit / (predicted + reference) if predicted or reference else None}


def compare_reports(baseline, current):
    """Pair immutable reports on exact topic/question IDs and identical questions.

    No gold is placed in runtime state. New/unexecuted questions are reported
    outside the matched denominator, and no claim of held-out quality is made.
    """
    before, after = _question_index(baseline), _question_index(current)
    shared, excluded = [], []
    metrics = ("paragraph_set", "strict_quote_set", "support", "conflict")
    for key in sorted(before.keys() & after.keys()):
        bt, bq = before[key]
        at, aq = after[key]
        if bq.get("question") != aq.get("question"):
            excluded.append({"topic_id": key[0], "question_id": key[1], "reason": "Question text changed; not a matched-question comparison."})
            continue
        evidence_before = bq["question_referenced_gold_evidence"]
        evidence_after = aq["question_referenced_gold_evidence"]
        row = {"topic_id": key[0], "topic": at["topic"], "question_id": key[1],
               "baseline_revision_id": bt["revision_id"], "current_revision_id": at["revision_id"],
               "baseline_origin_run_id": bq["origin_run_id"], "current_origin_run_id": aq["origin_run_id"],
               "baseline_prompt_version": bq.get("prompt_version"), "current_prompt_version": aq.get("prompt_version"),
               "same_origin_run": bq["origin_run_id"] == aq["origin_run_id"],
               "paragraph_set": _metric_comparison(evidence_before["paragraph_set"], evidence_after["paragraph_set"]),
               "strict_quote_set": _metric_comparison(evidence_before["strict_quote_set"], evidence_after["strict_quote_set"]),
               "support": _metric_comparison(bq["support"]["paragraph_set"], aq["support"]["paragraph_set"]),
               "conflict": _metric_comparison(bq["conflict"]["paragraph_set"], aq["conflict"]["paragraph_set"]),
               "source_validation": {"before": bq.get("predicted_source_validation"), "after": aq.get("predicted_source_validation")},
               "source_scan": {"before": bq.get("source_scan"), "after": aq.get("source_scan")},
               "evidence_selection_audit": {"before": bq.get("evidence_selection_audit"), "after": aq.get("evidence_selection_audit")}}
        # Exact-set reference cardinalities should agree for identical samples.
        # A mismatch cannot be folded into an apparent algorithm improvement.
        if any(row[name]["before"]["reference"] != row[name]["after"]["reference"] for name in metrics):
            excluded.append({"topic_id": key[0], "question_id": key[1], "reason": "Reference cardinality changed; not a matched-denominator comparison."})
            continue
        shared.append(row)
    aggregate = {name: _metric_comparison(_micro_metrics([row[name]["before"] for row in shared]),
                                         _micro_metrics([row[name]["after"] for row in shared])) for name in metrics}
    sample_hashes_equal = bool(baseline.get("sample_input_sha256")) and baseline.get("sample_input_sha256") == current.get("sample_input_sha256")
    return {"purpose": "Offline matched-question development-example comparison; no official score or unseen-data prediction.",
            "baseline_report_fingerprint_sha256": fingerprint(baseline),
            "current_report_fingerprint_sha256": fingerprint(current),
            "source_database_fingerprint_matches": bool(baseline.get("source_fingerprint_before")) and baseline.get("source_fingerprint_before") == current.get("source_fingerprint_before"),
            "sample_input_sha256_matches": sample_hashes_equal,
            "comparison_input_warning": None if sample_hashes_equal else "Sample file fingerprints differ; do not interpret metric deltas as a controlled algorithm-only improvement.",
            "matched_question_count": len(shared),
            "changed_origin_question_count": sum(not row["same_origin_run"] for row in shared),
            "before_only_questions": [{"topic_id": key[0], "question_id": key[1]} for key in sorted(before.keys() - after.keys())],
            "after_only_questions": [{"topic_id": key[0], "question_id": key[1]} for key in sorted(after.keys() - before.keys())],
            "excluded_shared_questions": excluded,
            "matched_question_micro": aggregate, "questions": shared,
            "notes": ["Exact document/paragraph matching is not semantic fact correctness or official M.",
                      "Micro counts sum question-local deduplicated sets; one source paragraph can count for different questions.",
                      "Raw quote/source-span validity, planned retrieval, completed source scanning and evidence recall are separate measurements.",
                      "Missing old scan/source-validation measurements remain null rather than being reconstructed as perfect or zero.",
                      "Reused same-origin outputs are explicitly marked and included only to preserve matched-question scope."]}


def comparison_markdown(comparison):
    def percent(value):
        return "—" if value is None else f"{100 * value:.1f}%"

    def change(row, metric, field):
        return f"{percent(row[metric]['before'][field])} → {percent(row[metric]['after'][field])}"

    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = ["# 样例同题前后对比", "", "只比较同专题、同问题编号且问题原文与参照分母一致的已发布结果。这里是已知样例的机械匹配诊断，不是官方分数或隐藏赛题精度。", "",
             f"匹配问题：{comparison['matched_question_count']}；其中新运行替换：{comparison['changed_origin_question_count']}。原文扫描覆盖与证据召回率分开记录。", "",
             "| 专题 / 问题 | 段落精确率 | 段落召回率 | 段落 F1 | 支撑引用召回 | 冲突引用召回 | 实际扫描 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in comparison["questions"]:
        scan = row["source_scan"]["after"]
        scanned, total = (scan.get("recorded_scanned_paragraphs"), scan.get("recorded_planned_total_paragraphs")) if scan else (None, None)
        scan_label = f"{scanned}/{total} 段" if scanned is not None and total is not None else "未采集"
        suffix = "（复用原结果）" if row["same_origin_run"] else ""
        lines.append(f"| {safe(row['topic'])} / {safe(row['question_id'])}{suffix} | {change(row, 'paragraph_set', 'precision')} | {change(row, 'paragraph_set', 'recall')} | {change(row, 'paragraph_set', 'f1')} | {change(row, 'support', 'recall')} | {change(row, 'conflict', 'recall')} | {scan_label} |")
    aggregate = comparison["matched_question_micro"]
    lines.extend(["", f"同题微平均段落 P / R / F1：{change(aggregate, 'paragraph_set', 'precision')} / {change(aggregate, 'paragraph_set', 'recall')} / {change(aggregate, 'paragraph_set', 'f1')}。按问题内去重后求和；不同问题可分别计入同一段落。", "",
                  "严格引文匹配、原文逐字有效性、运行来源、版本和未匹配问题详见配套 JSON。旧运行没有扫描事件时记为未知，不能记成 0% 或 100%。"])
    if comparison["comparison_input_warning"]:
        lines.extend(["", "输入指纹发生变化，前后差异不能直接归因为算法调整。"])
    if comparison["before_only_questions"] or comparison["after_only_questions"] or comparison["excluded_shared_questions"]:
        lines.extend(["", f"未纳入同题分母：仅基线 {len(comparison['before_only_questions'])} 题、仅新版 {len(comparison['after_only_questions'])} 题、问题或参照分母变化 {len(comparison['excluded_shared_questions'])} 题；没有把未运行题目按零准确率计入。"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / ".runtime/business.sqlite3")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / ".runtime/checkpoints.sqlite3")
    parser.add_argument("--sample-root", type=Path, default=ROOT.parent / "多智能体可信研判与证据链溯源/样例集-拓尔思-20261012")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/样例离线差距评估-20261018.json")
    parser.add_argument("--revision", action="append", default=[], metavar="TOPIC_ID=REVISION_ID", help="Pin immutable ready revisions; repeat for each topic.")
    parser.add_argument("--baseline", type=Path, help="Compare a frozen prior report on the same completed questions; baseline remains read-only.")
    parser.add_argument("--comparison-output", type=Path, help="Write matched-question JSON separately from both reports; requires --baseline.")
    parser.add_argument("--comparison-markdown", type=Path, help="Write a concise before/after table; requires --baseline.")
    args = parser.parse_args()
    if not args.baseline and (args.comparison_output or args.comparison_markdown):
        parser.error("Comparison outputs require --baseline.")
    output_paths = [args.output, *[path for path in (args.comparison_output, args.comparison_markdown) if path]]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        parser.error("Every output needs a distinct path.")
    if args.baseline and args.baseline.resolve() in {path.resolve() for path in output_paths}:
        parser.error("A frozen --baseline cannot be an output path; choose new report paths.")
    baseline = load(args.baseline) if args.baseline else None
    baseline_bytes_sha256 = hashlib.sha256(args.baseline.read_bytes()).hexdigest() if args.baseline else None
    revision_map = {}
    for item in args.revision:
        topic_id, separator, revision_id = item.partition("=")
        if not separator or topic_id not in {tid for _, tid in CASES} or not revision_id or topic_id in revision_map:
            parser.error("Each --revision must be a unique known TOPIC_ID=REVISION_ID pair.")
        revision_map[topic_id] = revision_id
    report = evaluate(args.database, args.checkpoints, args.sample_root, revision_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if baseline is not None:
        comparison = compare_reports(baseline, report)
        comparison["baseline_file"] = {"path": str(args.baseline.resolve()), "sha256": baseline_bytes_sha256}
        comparison["current_file"] = {"path": str(args.output.resolve()), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
        for path, text in ((args.comparison_output, json.dumps(comparison, ensure_ascii=False, indent=2) + "\n"),
                           (args.comparison_markdown, comparison_markdown(comparison))):
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
        assert hashlib.sha256(args.baseline.read_bytes()).hexdigest() == baseline_bytes_sha256, "Frozen baseline changed; reject comparison."
    print(json.dumps({"report": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "source_unchanged": report["source_unchanged"],
                      "topics": [{"topic": t["topic"], "scope": f"{t['completed_questions']}/{t['reference_questions']}",
                                  "evidence": t["topic_wide_evidence_comparison"]} for t in report["topics"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
