"""Offline F1-F9 public-sample audit. Gold never enters application runtime.

This script reads reference answers solely for deterministic diagnostics. It
constructs no Settings/provider/repository/database and makes no network call.
Source files are immutable and their byte SHA256 values are checked afterwards.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.domain.paragraph_mirrors import (build_paragraph_mirror_index,
    find_literal_fragment_mirror_candidates, character_similarity, SIMILARITY)

FORMAT = "public-sample-paragraph-rules-audit-v1"


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_input_json_key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def _distribution(values):
    return dict(sorted(Counter(str(value) for value in values).items()))


def _fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
        "rate": numerator / denominator if denominator else None}


def raw_paragraphs(documents):
    output, seen = [], set()
    for doc in documents:
        content = doc["content"]
        for para in doc["paragraphs"]:
            start, end = para["char_start"], para["char_end"]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(content)):
                raise ValueError("invalid_raw_paragraph_span")
            text = content[start:end]
            if para.get("text", text) != text:
                raise ValueError("raw_paragraph_text_mismatch")
            key = (doc["doc_id"], para["para_id"])
            if key in seen:
                raise ValueError("duplicate_raw_paragraph_location")
            seen.add(key)
            output.append({"doc_id": key[0], "para_id": key[1], "text": text,
                "char_start": start, "char_end": end})
    return output


def cross_paragraph_matches(row, document, start_paragraph, *, max_paragraphs=5):
    """Literal matches starting in the declared paragraph, reaching a later one.

    Three separately named representations are reported. No normalization or
    replacement of the declared para_id is permitted. Results keep the minimum
    covering window for each representation; extending a successful 2-window
    to 3/4/5 does not multiply one fact into four successful records.
    """
    quote = row.get("quote")
    if not isinstance(quote, str) or not quote:
        return []
    ordered = sorted(document["paragraphs"], key=lambda p: (p["char_start"], p["char_end"], p["para_id"]))
    index = next((i for i, p in enumerate(ordered) if p["para_id"] == start_paragraph), None)
    if index is None:
        return []
    found = {}
    for count in range(2, min(max_paragraphs, len(ordered) - index) + 1):
        selected = ordered[index:index + count]
        texts = [document["content"][p["char_start"]:p["char_end"]] for p in selected]
        variants = {"concatenated_no_separator": "".join(texts),
            "joined_single_newline": "\n".join(texts),
            "original_document_slice_preserving_gaps": document["content"][selected[0]["char_start"]:selected[-1]["char_end"]]}
        for method, text in variants.items():
            if method in found:
                continue
            position = text.find(quote)
            if 0 <= position < len(texts[0]) and position + len(quote) > len(texts[0]):
                found[method] = {"method": method, "paragraph_count": count,
                    "para_ids": [p["para_id"] for p in selected],
                    "window_start_cp": position, "window_end_cp": position + len(quote),
                    "declared_doc_id": document["doc_id"], "declared_para_id": start_paragraph,
                    "reference_identity_changed": False}
    return list(found.values())


def publish_time_category(value):
    if value is None:
        return "null"
    if not isinstance(value, str):
        return "non_string"
    if not value.strip():
        return "empty_string"
    if re.search(r"(?i)xx|未知|不详|不明|待定", value):
        return "partial_or_unknown_placeholder"
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            datetime.fromisoformat(value)
            return "iso_date"
        if "T" in value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return "iso_datetime_with_timezone" if parsed.tzinfo else "iso_datetime_without_timezone"
    except ValueError:
        return "invalid_iso_calendar_value"
    if re.fullmatch(r"\d{4}(-\d{2})?", value):
        return "year_or_month_only"
    return "other_text_not_parsed"


def _doc_groups(documents):
    parents = {doc["doc_id"]: doc["doc_id"] for doc in documents}
    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    seen = {}
    for doc in documents:
        signals = ["content:" + sha256(doc["content"].encode()).hexdigest()]
        if doc.get("content_url"):
            signals.append("explicit_url:" + doc["content_url"])
        for signal in signals:
            if signal in seen:
                left, right = sorted((root(doc["doc_id"]), root(seen[signal])))
                parents[right] = left
            else:
                seen[signal] = doc["doc_id"]
    groups = defaultdict(list)
    for key in parents:
        groups[root(key)].append(key)
    return [sorted(members) for _, members in sorted(groups.items()) if len(members) > 1]


def mirror_reference_coverage(index, reference_locations):
    """Location collection, not fact agreement, precision, or an optimal target."""
    manifest = {(r["doc_id"], r["para_id"]): r for r in index["source_manifest"]}
    reference_locations = set(reference_locations) & set(manifest)
    links = defaultdict(set)
    for edge in index["similar_edges"]:
        a, b = edge["left_text_sha256"], edge["right_text_sha256"]
        links[a].add(b)
        links[b].add(a)
    pairs = {"exact": set(), "similar_nonidentical": set(), "all_at_threshold": set()}
    for location in sorted(reference_locations):
        text_id = manifest[location]["text_sha256"]
        for target_id in (text_id, *sorted(links[text_id])):
            for target in index["text_buckets"][target_id]["members"]:
                key = (target["doc_id"], target["para_id"])
                if key == location or key[0] == location[0]:
                    continue
                category = "exact" if target_id == text_id else "similar_nonidentical"
                pairs[category].add((location, key))
                pairs["all_at_threshold"].add((location, key))
    output = {}
    for name, values in pairs.items():
        collected = [(a, b) for a, b in sorted(values) if b in reference_locations]
        output[name] = {"gold_anchor_locations_with_mirrors": len({a for a, _ in values}),
            "directed_anchor_to_mirror_location_pairs": len(values),
            "mirror_pairs_also_collected_by_gold": len(collected),
            "collection_rate": _fraction(len(collected), len(values)),
            "pairs": [{"anchor": {"doc_id": a[0], "para_id": a[1]},
                "mirror": {"doc_id": b[0], "para_id": b[1]}, "mirror_location_collected": b in reference_locations}
                for a, b in sorted(values)]}
    return {"definition": "For each distinct valid gold doc/para anchor, enumerate direct cross-document raw-paragraph mirrors. Numerator counts directed pairs whose target doc/para occurs anywhere in evidence_samples. Multiple facts at a location do not multiply pairs. No fact_des semantic equivalence, conclusion usefulness or material eligibility is asserted.",
        "valid_gold_anchor_locations": len(reference_locations), "source_key_proxy_not_semantic_precision_or_recall": True,
        "not_runtime_selection_rule": True, **output}


def shared_fragment_probe(paragraphs, documents, evidence, fragment):
    """Two collection definitions: unchanged declared key vs exact quote span."""
    raw = find_literal_fragment_mirror_candidates(paragraphs, [fragment])["fragments"][0]
    locations = {(p["doc_id"], p["para_id"]): p for p in paragraphs}
    by_doc = {d["doc_id"]: d for d in documents}
    annotated = []
    for occurrence in raw["occurrences"]:
        key = (occurrence["doc_id"], occurrence["para_id"])
        para = locations[key]
        absolute_start = para["char_start"] + occurrence["start_cp"]
        absolute_end = para["char_start"] + occurrence["end_cp"]
        covered = []
        for ref in evidence:
            if ref.get("doc_id") != key[0]:
                continue
            start, end = ref.get("char_start"), ref.get("char_end")
            content = by_doc[key[0]]["content"]
            if (type(start) is int and type(end) is int and 0 <= start < end <= len(content)
                    and content[start:end] == ref.get("quote") and start <= absolute_start < absolute_end <= end):
                covered.append({"fact_id": ref.get("fact_id"), "declared_doc_id": ref.get("doc_id"),
                    "declared_para_id": ref.get("para_id"), "crosses_from_different_declared_para": ref.get("para_id") != key[1],
                    "reference_identity_repaired": False})
        annotated.append({**occurrence, "paragraph_codepoints": len(para["text"]),
            "declared_gold_anchor": any((ref.get("doc_id"), ref.get("para_id")) == key for ref in evidence),
            "verified_gold_document_quote_spans_covering_fragment": covered})
    unique = {(r["doc_id"], r["para_id"]): r for r in annotated}
    pairs = []
    for i, left in enumerate(sorted(unique)):
        for right in sorted(unique)[i + 1:]:
            a, b = locations[left]["text"], locations[right]["text"]
            score = character_similarity(a, b)
            pairs.append({"left": {"doc_id": left[0], "para_id": left[1]}, "right": {"doc_id": right[0], "para_id": right[1]},
                "whole_paragraph_exact": a == b, "whole_paragraph_character_similarity": score,
                "whole_paragraph_similarity_at_least_0_9": score >= 0.9})
    return {"fragment": fragment, "match_rule": raw.get("match_rule", "exact_literal_fragment"), "locations": annotated,
        "whole_paragraph_pairwise_comparison": pairs,
        "declared_location_collection": _fraction(sum(r["declared_gold_anchor"] for r in unique.values()), len(unique)),
        "literal_reference_quote_content_collection": _fraction(sum(bool(r["verified_gold_document_quote_spans_covering_fragment"]) for r in unique.values()), len(unique)),
        "meaning": "A cross-paragraph reference can cover a fragment in a later para without changing its original declared para_id. These two rates must not be conflated with each other, whole-paragraph mirrors or semantic P/R."}


def audit_topic(title, documents, evidence, conclusions, *, shared_fragments=()):
    paragraphs = raw_paragraphs(documents)
    by_location = {(p["doc_id"], p["para_id"]): p for p in paragraphs}
    by_doc = {doc["doc_id"]: doc for doc in documents}
    counts = Counter((row.get("doc_id"), row.get("para_id")) for row in evidence)
    quote_rows = []
    for offset, row in enumerate(evidence):
        key = (row.get("doc_id"), row.get("para_id"))
        para = by_location.get(key)
        quote = row.get("quote")
        valid_quote = isinstance(quote, str) and bool(quote)
        position = para["text"].find(quote) if para and valid_quote else -1
        document = by_doc.get(key[0])
        start, end = row.get("char_start"), row.get("char_end")
        doc_span_exact = bool(document and valid_quote and type(start) is int and type(end) is int
            and 0 <= start < end <= len(document["content"]) and document["content"][start:end] == quote)
        quote_rows.append({"reference_index": offset, "fact_id": row.get("fact_id"), "doc_id": key[0], "para_id": key[1],
            "location_exists": para is not None, "quote_codepoints": len(quote) if isinstance(quote, str) else None,
            "quote_sha256": sha256(quote.encode()).hexdigest() if isinstance(quote, str) else None,
            "literal_in_declared_paragraph": position >= 0, "first_match_start_cp": position if position >= 0 else None,
            "starts_at_paragraph_start": position == 0,
            "equals_whole_paragraph": bool(para and quote == para["text"]),
            "declared_document_char_span_exact": doc_span_exact,
            "cross_2_to_5": cross_paragraph_matches(row, document, key[1]) if document else []})
    index = build_paragraph_mirror_index(paragraphs)
    fields = sorted({key for row in conclusions for key in row})
    labels = {"verdict_label": Counter(), "resolution_type": Counter()}
    label_paths = {name: Counter() for name in labels}
    def walk(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in labels:
                    labels[key][str(item)] += 1
                    label_paths[key][".".join((*path, key))] += 1
                walk(item, (*path, key))
        elif isinstance(value, list):
            for item in value:
                walk(item, (*path, "[]"))
    walk(conclusions)
    fact_ids = Counter(row.get("fact_id") for row in evidence)
    reference_ids = [ref.get("fact_id") for row in conclusions for name in ("supporting_facts", "conflicting_facts") for ref in row.get(name, []) if isinstance(ref, dict)]
    return {"title": title, "format": FORMAT, "offline_reference_diagnostics_only": True,
        "counts": {"documents": len(documents), "paragraphs": len(paragraphs), "reference_evidence_rows": len(evidence), "conclusions": len(conclusions)},
        "F1_same_doc_para_reference_multiplicity": {"location_count": len(counts), "rows_per_location_histogram": _distribution(counts.values()),
            "multiple_reference_locations": [{"doc_id": key[0], "para_id": key[1], "rows": count,
                "fact_ids": [r.get("fact_id") for r in evidence if (r.get("doc_id"), r.get("para_id")) == key]}
                for key, count in sorted(counts.items(), key=lambda item: tuple(str(value) for value in item[0])) if count > 1]},
        "F2_quote_position": {"reference_rows": len(evidence), **{name: sum(row[name] for row in quote_rows) for name in (
            "location_exists", "literal_in_declared_paragraph", "starts_at_paragraph_start", "equals_whole_paragraph", "declared_document_char_span_exact")}, "rows": quote_rows},
        "F3_cross_paragraph_literal": {"policy": "Declared start paragraph plus next 1-4 paragraphs by document codepoint order; exact raw joins and single-newline joins reported separately, also raw document slice. Identity not repaired.",
            "rows_locatable_by_any_method": sum(bool(row["cross_2_to_5"]) for row in quote_rows),
            "nonliteral_declared_paragraph_but_cross_locatable": sum(not row["literal_in_declared_paragraph"] and bool(row["cross_2_to_5"]) for row in quote_rows),
            "matches": [row for row in quote_rows if row["cross_2_to_5"]]},
        "F4_exact_raw_paragraph_and_document_links": {"cross_document_exact_paragraph_groups": index["exact_groups"],
            "known_exact_content_or_explicit_url_document_groups": _doc_groups(documents),
            "meaning": "Raw-text similarity and exact document linkage are different signals; neither proves source truth or independent votes."},
        "F5_similar_raw_paragraph_candidates": {"metric": SIMILARITY, "threshold": 0.9,
            "direct_nonidentical_text_edges": len(index["similar_edges"]), "transitive_candidate_components": len(index["components"]),
            "whitespace_case_punctuation_normalized": False, "direct_edges_not_transitive_equivalence": True,
            "index": index},
        "F6_gold_mirror_location_collection": mirror_reference_coverage(index, counts),
        "shared_literal_fragment_probes": [shared_fragment_probe(paragraphs, documents, evidence, fragment) for fragment in shared_fragments],
        "F7_source_credibility": {"document_level_distribution": _distribution(doc.get("source_credibility") for doc in documents),
            "meaning": "Supplied metadata, not recalibrated or independently authenticated by this audit."},
        "F8_publish_time": {"document_level_categories": _distribution(publish_time_category(doc.get("publish_time")) for doc in documents),
            "original_value_distribution": _distribution(doc.get("publish_time") for doc in documents)},
        "F9_conclusion_fields_and_labels": {"top_level_field_presence": {field: sum(field in row for row in conclusions) for field in fields},
            "top_level_field_types": {field: _distribution(type(row[field]).__name__ for row in conclusions if field in row) for field in fields},
            **{name + "_distribution": dict(sorted(values.items())) for name, values in labels.items()},
            "label_paths": {name: dict(sorted(values.items())) for name, values in label_paths.items()},
            "duplicate_fact_ids": {str(key): count for key, count in fact_ids.items() if count > 1},
            "conclusion_reference_occurrences": len(reference_ids), "reference_fact_ids_missing_from_evidence": sorted({str(fid) for fid in reference_ids if fid not in fact_ids}),
            "nested_conflicting_rows_are_not_inferred_conflict_pairs": True}}


def extract_docx(path):
    with ZipFile(path) as archive:
        tree = ElementTree.fromstring(archive.read("word/document.xml"))
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = ["".join(node.text or "" for node in p.findall(".//w:t", namespace)) for p in tree.findall(".//w:p", namespace)]
    pattern = re.compile(r"doc_id|para_id|F1|精确率|准确率|召回率|四项|证据条目|证据原文|规模加权|冲突|置信|结论一致|分母|字段|逐字|无法定位|语义|开发专题|每条结论")
    indices = set()
    for index, text in enumerate(paragraphs):
        if pattern.search(text):
            indices.update(range(max(0, index - 1), min(len(paragraphs), index + 2)))
    return {"file": str(Path(path).resolve()), "sha256": _hash(path), "paragraph_count": len(paragraphs),
        "selected_paragraphs": [{"paragraph_index": i, "text": paragraphs[i]} for i in sorted(indices)],
        "extraction": "Literal OOXML paragraph text; selection is keyword plus adjacent paragraph, not inferred instructions."}


def write_audit(source_dir, output_dir, official_documents=(), *, shared_fragments=()):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() or output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("audit_output_must_be_new_outside_sources")
    folders = sorted(p.parent for p in source_dir.glob("*/raw_documents.json"))
    if not folders:
        raise ValueError("no_sample_topics_found")
    paths = [folder / name for folder in folders for name in ("raw_documents.json", "evidence_samples.json", "conclusions.json")]
    official_documents = [Path(path).resolve() for path in official_documents]
    paths.extend(official_documents)
    before = {str(path): _hash(path) for path in paths}
    results = []
    for folder in folders:
        result = audit_topic(folder.name, *[_load(folder / name) for name in ("raw_documents.json", "evidence_samples.json", "conclusions.json")], shared_fragments=shared_fragments)
        results.append(result)
        print(json.dumps({"topic": folder.name, "counts": result["counts"], "audit_complete": True}, ensure_ascii=False), flush=True)
    excerpts = [extract_docx(path) for path in official_documents]
    after = {str(path): _hash(path) for path in paths}
    if before != after:
        raise ValueError("sample_sources_changed_during_audit")
    output_dir.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    for i, result in enumerate(results, 1):
        write(f"topic_{i:02d}.json", result)
    write("official-document-extracts.json", excerpts)
    summary = {"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
        "offline_reference_audit_only": True, "model_calls": 0, "runtime_database_or_index_used": False,
        "source_files_sha256_before": before, "source_files_sha256_after": after, "sources_unchanged": True,
        "code_sha256": {str(Path(__file__).resolve()): _hash(__file__), str(ROOT / "app/domain/paragraph_mirrors.py"): _hash(ROOT / "app/domain/paragraph_mirrors.py")},
        "similarity_metric": SIMILARITY, "threshold": 0.9,
        "topics": [{"title": result["title"], "file": f"topic_{i:02d}.json", "counts": result["counts"],
            "literal_rows": result["F2_quote_position"]["literal_in_declared_paragraph"],
            "whole_paragraph_rows": result["F2_quote_position"]["equals_whole_paragraph"],
            "cross_paragraph_nonliteral_rows": result["F3_cross_paragraph_literal"]["nonliteral_declared_paragraph_but_cross_locatable"],
            "mirror_collection": {name: result["F6_gold_mirror_location_collection"][name]["collection_rate"] for name in ("exact", "similar_nonidentical", "all_at_threshold")},
            "shared_fragment_collection": [{"fragment": probe["fragment"], "declared_location_collection": probe["declared_location_collection"],
                "literal_reference_quote_content_collection": probe["literal_reference_quote_content_collection"]} for probe in result["shared_literal_fragment_probes"]]}
            for i, result in enumerate(results, 1)],
        "boundaries": ["F1-F9 labels are audit sections, not F1 quality scores.",
            "Mirror collection rate is a raw location selection proxy, not semantic precision/recall, not source independence and not comparable numerically with current 9/82 semantic precision.",
            "Gold is read only here, never inserted into runtime input, prompts, retrieval, model cache or application database.",
            "No original para_id, quote or answer is corrected; exact joins/newline joins/raw slices are reported separately.",
            "The mirror module is not wired into production and never creates/deletes evidence or independent votes."]}
    write("summary.json", summary)
    lines = ["# 六专题段落规则确定性审计", "", "仅离线读取公开样例及原参考，0模型调用，原文件SHA前后不变。F1—F9为审计项编号，不是质量F1。", "",
        "相似度：原始Unicode码点的 normalized indel = 2×LCS/(长度和)，阈值0.9；空白/大小写/标点均不归一化。近似连通组只用于导航，扩展仅取直接达阈值的边。", "",
        "|专题|文档/原段|参考条目|段内逐字|整段引用|跨2–5段补充定位|镜像键收录（已收录/候选）|", "|---|---:|---:|---:|---:|---:|---:|"]
    for topic in summary["topics"]:
        c = topic["counts"]; rate = topic["mirror_collection"]["all_at_threshold"]
        lines.append(f"|{topic['title']}|{c['documents']}/{c['paragraphs']}|{c['reference_evidence_rows']}|{topic['literal_rows']}|{topic['whole_paragraph_rows']}|{topic['cross_paragraph_nonliteral_rows']}|{rate['numerator']}/{rate['denominator']}|")
    lines.extend(["", "镜像键收录率定义：对每个不同的、有效的参考doc/para，列举跨文档直接镜像段；分子为目标doc/para也被任一参考条目收录的有向对数。它不检查断言、主体、时点、数值或口径的一致性，不能与当前语义P直接比较。", "",
        "默认建议：镜像可以扩展候选，但不直接正式入池；所有扩展保留每个原始doc/para，分别按相同事实与用途规则评分、核验材料资格并使用可调额度。source_group去重只影响投票，不得据此删除证据条目。完整相同文本也不证明相同来源时间/归属或题目用途。低整段镜像收录率不能否定共享句子候选扩展；两种规则分别列出。", "",
        "原文跨段只记录从声明起始段至邻接段的逐字定位：无分隔符连接、单换行连接、保留原文间隔的连续doc切片分栏；不把doc_016#p5等参考自动改成p6。详见逐专题F3。", "",
        "官方口径：评分docx先精确doc_id+para_id再判断同事实四维；段落键收录不能冒充官方条目P/R。原始参考重复、跨段和字段差异保留。结论支持规模加权及冲突成对结构的正文/附录/真实样例差异，见official-document-extracts.json与独立评分器说明。", ""])
    for result in results:
        for probe in result["shared_literal_fragment_probes"]:
            if not probe["locations"]:
                continue
            a, b = probe["declared_location_collection"], probe["literal_reference_quote_content_collection"]
            lines.extend([f"## 共享原句：{result['title']}", "", f"原句：{probe['fragment']}", "",
                f"声明doc/para键收录 {a['numerator']}/{a['denominator']}；逐字参考quote实际覆盖 {b['numerator']}/{b['denominator']}。后者允许原参考从前一声明段跨至后段，但原para_id不修。",
                "", "|原始位置|整段码点|gold声明键|被逐字参考quote覆盖|", "|---|---:|---:|---:|"])
            for occurrence in probe["locations"]:
                refs = ", ".join(str(row["fact_id"]) + "(" + str(row["declared_para_id"]) + ")" for row in occurrence["verified_gold_document_quote_spans_covering_fragment"])
                lines.append(f"|{occurrence['doc_id']} / {occurrence['para_id']}|{occurrence['paragraph_codepoints']}|{occurrence['declared_gold_anchor']}|{refs or '无'}|")
            lines.extend(["", "完整原段两两字符相似度、整段exact与>=0.9判断详见对应topic JSON 的 shared_literal_fragment_probes；共享原句相同不代表完整段相同。", ""])
    with (output_dir / "README.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-document", type=Path, action="append", default=[])
    parser.add_argument("--shared-fragment", action="append", default=[])
    args = parser.parse_args()
    summary = write_audit(args.source_dir, args.output_dir, args.official_document, shared_fragments=args.shared_fragment)
    print(json.dumps({"topics": len(summary["topics"]), "model_calls": 0, "sources_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
