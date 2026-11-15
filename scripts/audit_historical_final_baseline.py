"""Verify archived complete legacy final outputs; no reference/model access.

This metadata exporter reuses historical returned usage; it does not re-run the
old route or present its unbounded old cost as a current budget recommendation.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from scripts.final_evidence_scoring import normalize_legacy


def descriptor(path):
    data = path.read_bytes()
    return {"file": str(path.resolve()), "sha256": sha256(data).hexdigest(), "bytes": len(data)}


def audit(report_root, case_ids):
    root = Path(report_root).resolve()
    receipt_path = root / "generation/final-receipt.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    plan_path = root / "frozen-inputs/plan.json"
    plan = json.loads(plan_path.read_text("utf-8"))
    if receipt["plan_sha256"] != descriptor(plan_path)["sha256"] or not receipt["all_cases_succeeded"]:
        raise ValueError("closed_plan_and_complete_generation_required")
    cases = []
    for cid in case_ids:
        p = next(row for row in plan["cases"] if row["case_id"] == cid)
        r = next(row for row in receipt["cases"] if row["case_id"] == cid)
        source = root / "frozen-inputs" / p["input_file"]
        terminal_path = root / "generation" / cid / "terminal.json"
        if descriptor(source)["sha256"] != p["input_sha256"] or descriptor(terminal_path)["sha256"] != r["terminal"]["sha256"]:
            raise ValueError("closed_input_or_terminal_sha_changed")
        inp = json.loads(source.read_text("utf-8"))
        terminal = json.loads(terminal_path.read_text("utf-8"))
        if r["status"] != "succeeded" or terminal["status"] != "succeeded" or terminal["input_sha256"] != p["input_sha256"]:
            raise ValueError("complete_same_input_final_required")
        artifacts = {}
        for name, bound in terminal["artifacts"].items():
            artifact = root / "generation" / cid / Path(bound["file"]).name
            artifacts[name] = descriptor(artifact)
            if artifacts[name]["sha256"] != bound["sha256"] or artifacts[name]["bytes"] != bound["bytes"]:
                raise ValueError("closed_final_artifact_changed")
        result = json.loads(Path(artifacts["result"]["file"]).read_text("utf-8"))
        normalized = normalize_legacy(result, inp["question"])
        standard = json.loads(Path(artifacts["standard_evidence"]["file"]).read_text("utf-8"))
        fields = ("doc_id", "para_id", "quote", "fact_des", "argument", "entities")
        projection = lambda rows: Counter(json.dumps({k: row[k] for k in fields}, ensure_ascii=False, sort_keys=True) for row in rows)
        if projection(normalized["evidence"]) != projection(standard):
            raise ValueError("standard_export_differs_from_complete_result")
        usage = terminal["usage"]
        if any(row.get("usage_source") != "provider" or row.get("model") != receipt["model_name"]
               or any(type(row.get(k)) is not int or row[k] < 0 for k in ("input_tokens", "output_tokens", "total_tokens"))
               or row["input_tokens"] + row["output_tokens"] != row["total_tokens"] for row in usage):
            raise ValueError("actual_provider_usage_required")
        totals = {key: sum(row[key] for row in usage) for key in ("input_tokens", "output_tokens", "total_tokens")}
        if (any(terminal["usage_summary"][k] != value or r["usage_summary"][k] != value for k, value in totals.items())
                or terminal["usage_summary"]["unknown_usage_records"] != 0 or len(usage) != terminal["usage_summary"]["records"]):
            raise ValueError("closed_usage_summary_differs")
        cases.append({"case_id": cid, "topic_title": p["title"], "question": inp["question"],
                      "frozen_input": descriptor(source), "original_raw_sha256": p["source_input_sha256"],
                      "source_paragraphs": len(inp["paragraphs"]), "full_topic_paragraphs": p["source_paragraphs"],
                      "terminal": descriptor(terminal_path), "artifacts": artifacts,
                      "final_counts": {k: len(normalized[k]) for k in ("evidence", "supporting_facts", "conflicting_facts")},
                      "actual_usage_by_operation": usage, "actual_usage_totals": totals,
                      "unknown_usage_records": 0, "elapsed_seconds": terminal["elapsed_seconds"],
                      "input_has_source_credibility": all("source_credibility" in row for row in inp["paragraphs"]),
                      "source_credibility_must_not_be_guessed": True})
    return {"format": "archived-complete-final-baseline-index-v1", "receipt": descriptor(receipt_path),
            "plan": descriptor(plan_path), "model_name": receipt["model_name"], "provider": receipt["provider"],
            "prompt_version": receipt["prompt_version"], "cases": cases,
            "comparison_kind": "historical_architecture_route_not_pure_framework_causal_effect",
            "references_read": False, "new_model_calls": 0, "invoice_cost_known": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.report_root, args.case)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"cases": len(result["cases"]), "output_sha256": descriptor(path)["sha256"], "new_model_calls": 0}))


if __name__ == "__main__":
    main()
