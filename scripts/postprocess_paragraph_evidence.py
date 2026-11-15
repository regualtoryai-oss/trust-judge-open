"""T2: immutable offline paragraph consolidation and T1 before/after scoring."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain.paragraph_outputs import consolidate_submission
from scripts.evaluate_paragraphs import (comparison_markdown, evaluate_bundle, file_descriptor,
                                         make_closed_pilot_bundle)


def execute(comparison_dir, pilot_plan, sample_root, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("new_output_directory_required")
    for source in (Path(comparison_dir), Path(pilot_plan).parent, Path(sample_root)):
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output_overlaps_source")
    bundle, guards = make_closed_pilot_bundle(comparison_dir, pilot_plan, sample_root, "v17")
    baseline = evaluate_bundle(bundle)
    updated = deepcopy(bundle)
    topics = {topic["topic_id"]: topic for topic in bundle["topics"]}
    audits = []
    for index, submitted in enumerate(bundle["submissions"]):
        merged, audit = consolidate_submission(submitted, topics[submitted["topic_id"]]["source_paragraphs"])
        updated["submissions"][index] = merged
        audits.append({"topic_id": submitted["topic_id"], **audit})
    after = evaluate_bundle(updated)
    for source in (Path(__file__), ROOT/"app/domain/paragraph_outputs.py", ROOT/"scripts/evaluate_paragraphs.py"):
        guards.append(file_descriptor(source))
    if any(file_descriptor(row["file"])["sha256"] != row["sha256"] for row in guards):
        raise ValueError("source_changed_during_postprocessing")
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output/name).write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode())
    report = {"format": "t2-paragraph-postprocessing-comparison-v1", "new_model_calls": 0,
              "new_tokens": 0, "source_guards_unchanged": True, "source_guards": guards,
              "baseline_generation_not_repeated": True, "reports": {"v17_closed": baseline, "v17_T2": after},
              "limitations": ["Existing fact_des and field text are joined without new claims; no new paragraph summary model was called.",
                              "Changed quote or merged annotation has no renewed semantic label; B retains unknown bounds.",
                              "Conclusion membership is preserved and deduplicated; retrieved candidates are not promoted to support.",
                              "This remains the closed 6-question/144-paragraph subset, not a full18 generation run."]}
    write("report.json", report)
    write("postprocessed-submissions.json", {"submissions": updated["submissions"], "consolidation_audit": audits,
                                            "runtime_input": False, "contains_offline_reference_comparison": False})
    text = comparison_markdown(report["reports"])
    text = text.replace("# T1：闭合样例的确定性段落对照", "# T2：段落输出纯后处理对照", 1)
    text += ("\n本项0新调用、0 tokens。相同doc/para只保留一条，quote由程序扩展至完整原段，镜像保留各自出处。"
             "fact_des仅合并旧描述，尚非重新生成的整段摘要；生产新摘要由后续T3完成。"
             "修改后的quote/描述不得复用不再完全相同的旧语义标签，B明确保留未判范围。原子事实切分计划取消。\n")
    (output/"report.md").write_bytes(text.encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.comparison_dir, args.pilot_plan, args.sample_root, args.output)
    print(json.dumps({"output": str(args.output), "new_model_calls": 0, "new_tokens": 0,
                      "labels": list(result["reports"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
