"""v19 paragraph protocol: source-owned quotes, bounded ranking, replayable proof.

This is neither the v17 scope ledger nor an assertion of model infallibility.
All admitted paragraphs retain their own source keys and an independent field
review. Similarity metadata never proves fact equivalence or independent votes.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Literal
from uuid import NAMESPACE_URL, uuid5
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

VERSION = "paragraph-ranked-source-v19"
PROMPT_VERSION = "trusted-analysis-json-v19-paragraph-ranked"
ARGUMENTS = ("时间", "主体", "地点", "内容", "影响", "数值口径")


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class QueryGroup(Contract):
    group: Literal["X", "Y", "official"]
    terms: list[str] = Field(min_length=1, max_length=8)


class QueryOutput(Contract):
    groups: list[QueryGroup] = Field(min_length=3, max_length=3)


class RankedParagraph(Contract):
    paragraph_ref: str
    score: StrictInt = Field(ge=0, le=3)
    critical_qualifier: StrictBool
    stance: Literal["supports_X", "supports_Y", "official", "qualifier", "neutral"]
    reason: str = Field(min_length=1, max_length=600)


class RankOutput(Contract):
    decisions: list[RankedParagraph]


class Argument(Contract):
    时间: str | None
    主体: str | None
    地点: str | None
    内容: str | None
    影响: str | None
    数值口径: str | None


class ParagraphAnnotation(Contract):
    paragraph_ref: str
    fact_des: str = Field(min_length=1)
    argument: Argument
    entities: list[str]


class AnnotationOutput(Contract):
    annotations: list[ParagraphAnnotation]


class FieldChecks(Contract):
    时间: StrictBool
    主体: StrictBool
    地点: StrictBool
    内容: StrictBool
    影响: StrictBool
    数值口径: StrictBool


class ParagraphFieldDecision(Contract):
    paragraph_ref: str
    annotation_sha256: str
    fact_des_supported: StrictBool
    field_checks: FieldChecks
    entities_supported: StrictBool
    attribution_preserved: StrictBool
    reason: str = Field(min_length=1, max_length=600)


class FieldOutput(Contract):
    decisions: list[ParagraphFieldDecision]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def config_values(value=None):
    value = value or {}
    result = {"candidate_limit": value.get("candidate_limit", 60), "minimum_score": value.get("minimum_score", 2),
              "selection_cap": value.get("selection_cap", 10), "rank_batch_size": value.get("rank_batch_size", 12),
              "mirror_enabled": value.get("mirror_enabled", False), "mirror_candidate_limit": value.get("mirror_candidate_limit", 60)}
    bounds = {"candidate_limit": (40, 60), "minimum_score": (2, 3), "selection_cap": (1, 30), "rank_batch_size": (1, 20),
              "mirror_candidate_limit": (0, 120)}
    if (set(value) - set(result) or type(result["mirror_enabled"]) is not bool
            or any(type(result[k]) is not int or not lower <= result[k] <= upper for k, (lower, upper) in bounds.items())):
        raise ValueError("paragraph_ranked_config_invalid")
    return result


def project_question(question):
    if not isinstance(question, dict) or not isinstance(question.get("question"), str) or not question["question"].strip() or not question.get("id"):
        raise ValueError("paragraph_ranked_question_invalid")
    raw = question.get("raw_question_id", question.get("question_id"))
    if not isinstance(raw, str) or not raw:
        raise ValueError("paragraph_ranked_raw_question_id_invalid")
    return {"id": question["id"], "raw_question_id": raw, "question": question["question"]}


def source_manifest(paragraphs):
    seen, rawseen, rows = set(), set(), []
    for p in paragraphs:
        if (not isinstance(p, dict) or any(not isinstance(p.get(k), str) or not p[k] for k in ("id", "document_id", "raw_doc_id", "raw_para_id"))
                or not isinstance(p.get("text"), str)):
            raise ValueError("paragraph_ranked_source_invalid")
        key = p["raw_doc_id"], p["raw_para_id"]
        if p["id"] in seen or key in rawseen:
            raise ValueError("paragraph_ranked_duplicate_source")
        seen.add(p["id"]); rawseen.add(key)
        rows.append({k: deepcopy(p.get(k)) for k in ("id", "document_id", "raw_doc_id", "raw_para_id", "text", "title",
            "content_sha256", "content_url", "raw_publish_time", "parsed_publish_time", "time_precision")})
        for key in ("source_credibility_json", "source_credibility"):
            if key in p:
                rows[-1][key] = deepcopy(p[key])
    return rows


def query_catalog(question, output):
    question = project_question(question)
    value = QueryOutput.model_validate(output).model_dump()
    if [g["group"] for g in value["groups"]] != ["X", "Y", "official"]:
        raise ValueError("paragraph_ranked_query_groups")
    groups = []
    for group in value["groups"]:
        if any(not t.strip() or len(t) > 160 for t in group["terms"]) or len(set(group["terms"])) != len(group["terms"]):
            raise ValueError("paragraph_ranked_query_terms")
        # Every search carries the unabridged target and its literal qualifiers;
        # the model may expand terms but cannot silently replace the question.
        groups.append({**group, "full_original_question": question["question"],
            "object_time_scope_context": question["question"],
            "search_text": question["question"] + "\n" + " ".join(group["terms"])})
    return {"question_sha256": digest(question), "groups": groups}


def recall_paragraphs(question, paragraphs, queries, *, config=None, excluded_spans=()):
    """Title/body BM25 reciprocal-rank union; tokenize each source field once."""
    from app.domain.retrieval import _tokens
    config = config_values(config); manifest = source_manifest(paragraphs)
    if queries != query_catalog(question, {"groups": [{k: g[k] for k in ("group", "terms")} for g in queries["groups"]]}):
        raise ValueError("paragraph_ranked_query_binding")
    excluded = set()
    byid = {p["id"]: p for p in manifest}
    for entry in excluded_spans:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            raise ValueError("paragraph_ranked_exclusion_invalid")
        pid, begin, end, quote = entry; p = byid.get(pid)
        if (p is None or type(begin) is not int or type(end) is not int or not 0 <= begin < end <= len(p["text"])
                or p["text"][begin:end] != quote):
            raise ValueError("paragraph_ranked_exclusion_source_changed")
        excluded.add(pid)  # whole-paragraph quotations cannot crop around a human exclusion
    material_policy = source_material_policy(question, paragraphs)
    material_ids = set(material_policy["eligible_paragraph_ids"])
    eligible = [p for p in manifest if p["id"] not in excluded and p["id"] in material_ids and p["text"]]
    hits, fusion = defaultdict(list), defaultdict(float)
    for field in ("title", "text"):
        tokenized = [Counter(_source_tokens(p.get(field) or "")) for p in eligible]
        df = Counter(t for words in tokenized for t in words)
        avg = sum(sum(words.values()) for words in tokenized) / len(eligible) if eligible else 0
        for group in queries["groups"]:
            terms = set(_tokens(group["search_text"])); ranked = []
            for i, words in enumerate(tokenized):
                common = terms & words.keys()
                if not common:
                    continue
                length = sum(words.values()); score = sum(math.log(1 + (len(eligible) - df[t] + .5) / (df[t] + .5)) *
                    words[t] * 2.5 / (words[t] + 1.5 * (.25 + .75 * length / avg)) for t in common) if avg else 0
                ranked.append((score, i))
            ranked.sort(key=lambda pair: (-pair[0], pair[1]))
            for rank, (score, i) in enumerate(ranked[:config["candidate_limit"]], 1):
                pid = eligible[i]["id"]; fusion[pid] += 1 / (60 + rank)
                hits[pid].append({"group": group["group"], "field": field, "rank": rank, "bm25_score": score})
    order = {p["id"]: i for i, p in enumerate(manifest)}
    ids = sorted(fusion, key=lambda pid: (-fusion[pid], order[pid]))[:config["candidate_limit"]]
    base_ids = list(ids)
    expansion = {"enabled": False, "candidate_budget": config["mirror_candidate_limit"], "added_count": 0, "additions": []}
    if config["mirror_enabled"]:
        from app.domain.paragraph_mirror_recall import expand_base_mirrors
        expansion = expand_base_mirrors(eligible, base_ids, candidate_limit=config["mirror_candidate_limit"])
        ids.extend(row["paragraph_id"] for row in expansion["additions"])
    additions = {row["paragraph_id"]: row for row in expansion["additions"]}
    catalog = [{"paragraph_ref": "P" + str(i + 1), "paragraph_id": pid,
        "raw_doc_id": byid[pid]["raw_doc_id"], "raw_para_id": byid[pid]["raw_para_id"],
        "source_text_sha256": digest(byid[pid]["text"]), "fusion_score": fusion.get(pid), "retrieval_hits": hits.get(pid, []),
        "candidate_origin": "bm25_base" if pid in base_ids else "mirror_expansion",
        "mirror_origins": additions.get(pid, {}).get("origins", [])}
        for i, pid in enumerate(ids)]
    from app.domain.paragraph_mirrors import build_paragraph_mirror_index
    mirrors = build_paragraph_mirror_index([byid[pid] for pid in ids], threshold=.9)
    return {"candidate_catalog": catalog, "source_manifest_sha256": digest(manifest), "mirror_candidate_index": mirrors,
        "base_candidate_count": len(base_ids), "mirror_candidate_count": len(additions), "mirror_expansion": expansion,
        "source_material_policy": material_policy,
        "query_catalog_sha256": digest(queries), "excluded_spans_sha256": digest(sorted(map(list, excluded_spans))),
        "complete_source_scan": False, "raw_source_paragraph_count": len(manifest), "recalled_paragraph_count": len(catalog),
        "strategy": "title_body_bm25_rrf_union_bounded", "config": config}


def source_material_policy(question, paragraphs):
    """Qualify immutable source dates BEFORE recall/cap; never spend a slot on late/unknown material."""
    from app.domain.material_eligibility import extract_question_policy, _repository_catalog
    policy = extract_question_policy(project_question(question)["question"])
    applied = policy["mode"] == "publication_cutoff" and policy["extraction_status"] == "supported_literal_policy"
    if not applied:
        return {"applied": False, "policy": policy, "eligible_paragraph_ids": [p["id"] for p in paragraphs],
                "source_decisions": [], "counts": {"qualified": len(paragraphs), "excluded": 0, "unknown": 0}}
    catalog, bindings = _repository_catalog(project_question(question)["question"], paragraphs)
    decisions, eligible, counts = {}, [], {"qualified": 0, "excluded": 0, "unknown": 0}
    for paragraph in paragraphs:
        binding = bindings[paragraph["id"]]; sid = binding["source_id"]
        if sid not in decisions:
            decisions[sid] = catalog.assess(binding)
        decision = decisions[sid]
        if decision["status"] == "within_cutoff":
            eligible.append(paragraph["id"]); counts["qualified"] += 1
        elif decision["status"] == "unknown":
            counts["unknown"] += 1
        else:
            counts["excluded"] += 1
    return {"applied": True, "policy": policy, "eligible_paragraph_ids": eligible,
            "source_decisions": [decisions[key] for key in sorted(decisions)], "counts": counts}


@lru_cache(maxsize=20_000)
def _source_tokens(text):
    # Immutable in-memory token tuples only. No reference labels, decisions or
    # source rewrites are cached; source hashes are replayed independently.
    from app.domain.retrieval import _tokens
    return tuple(_tokens(text))


def paragraph_model_catalog(catalog, *, stage="rank"):
    """Project source-owned model input; retrieval/provenance stay in proof.

    Annotation and field checks may only use the literal paragraph, so they
    receive no title or publication timestamp from which to invent event facts.
    The optional rank timestamp is normalized source publication context only.
    This projection is idempotent for an already compact catalog and also
    protects direct provider callers from forwarding arbitrary extra fields.
    """
    from app.domain.source_metadata import clean_publish_time
    if not isinstance(stage, str) or stage not in {"rank", "annotate", "fields"} or not isinstance(catalog, list):
        raise ValueError("paragraph_ranked_model_catalog_invalid")
    rows, seen = [], set()
    for item in catalog:
        if not isinstance(item, dict):
            raise ValueError("paragraph_ranked_model_catalog_invalid")
        ref, text = item.get("paragraph_ref"), item.get("text")
        if (not isinstance(ref, str) or not re.fullmatch(r"P[1-9][0-9]*", ref)
                or ref in seen or not isinstance(text, str) or not text):
            raise ValueError("paragraph_ranked_model_catalog_invalid")
        seen.add(ref)
        row = {"paragraph_ref": ref, "text": text}
        if stage == "rank":
            title = item.get("title")
            if isinstance(title, str) and title.strip():
                row["title"] = title
            # Never forward an arbitrary raw/relative date or trust a caller's
            # status/normalized object. Reparse the actual scalar date source.
            raw_time = item.get("raw_publish_time", item.get("publication_time"))
            cleaned = clean_publish_time(raw_time)
            if cleaned["normalized"] is not None:
                row["publication_time"] = cleaned["normalized"]
        rows.append(row)
    return rows


def rank_batch_catalog(recall, paragraph_map, refs=None, *, stage="rank"):
    """Resolve stable P aliases locally and expose only the compact wire view.

    Full source keys, source hashes, retrieval hits, similarity and candidate
    origins remain in recall/proof. A changed source cannot be hidden by the
    projection: verify identity/hash against the internal catalog first.
    """
    if (not isinstance(recall, dict) or not isinstance(recall.get("candidate_catalog"), list)
            or not isinstance(paragraph_map, dict)):
        raise ValueError("paragraph_ranked_model_source_binding")
    aliases = [row.get("paragraph_ref") if isinstance(row, dict) else None for row in recall["candidate_catalog"]]
    if (any(not isinstance(ref, str) or not re.fullmatch(r"P[1-9][0-9]*", ref) for ref in aliases)
            or len(aliases) != len(set(aliases))):
        raise ValueError("paragraph_ranked_model_source_binding")
    if refs is not None and (not isinstance(refs, (list, tuple))
            or any(not isinstance(ref, str) or ref not in aliases for ref in refs)
            or len(refs) != len(set(refs))):
        raise ValueError("paragraph_ranked_model_source_binding")
    wanted = set(refs) if refs is not None else None
    rows = []
    for entry in recall["candidate_catalog"]:
        if wanted is not None and entry["paragraph_ref"] not in wanted:
            continue
        source_id = entry.get("paragraph_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("paragraph_ranked_model_source_binding")
        source = paragraph_map.get(source_id)
        if (not isinstance(source, dict) or source.get("id") != entry.get("paragraph_id")
                or not isinstance(source.get("text"), str)
                or any(not isinstance(source.get(key), str) or not source[key] for key in ("raw_doc_id", "raw_para_id"))
                or any(source.get(key) != entry.get(key) for key in ("raw_doc_id", "raw_para_id"))
                or digest(source["text"]) != entry.get("source_text_sha256")):
            raise ValueError("paragraph_ranked_model_source_binding")
        rows.append({"paragraph_ref": entry["paragraph_ref"], "text": source["text"],
                     "title": source.get("title"), "raw_publish_time": source.get("raw_publish_time")})
    return paragraph_model_catalog(rows, stage=stage)


def validate_rank_batch(output, expected_refs):
    value = RankOutput.model_validate(output).model_dump()["decisions"]
    ids = [r["paragraph_ref"] for r in value]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_refs):
        raise ValueError("paragraph_ranked_rank_coverage")
    if any(r["critical_qualifier"] and r["score"] < 2 for r in value):
        raise ValueError("paragraph_ranked_critical_qualifier_below_two")
    return sorted(value, key=lambda row: expected_refs.index(row["paragraph_ref"]))


def select_ranked(recall, decisions, *, config=None):
    config = config_values(config); refs = [r["paragraph_ref"] for r in recall["candidate_catalog"]]
    decisions = validate_rank_batch({"decisions": decisions}, refs)
    if any(r["critical_qualifier"] and r["score"] < 2 for r in decisions):
        raise ValueError("paragraph_ranked_critical_qualifier_below_two")
    order = {ref: i for i, ref in enumerate(refs)}
    selected = sorted((r for r in decisions if r["score"] >= config["minimum_score"]),
        key=lambda r: (-r["score"], -int(r["critical_qualifier"]), order[r["paragraph_ref"]]))[:config["selection_cap"]]
    return [r["paragraph_ref"] for r in selected]


def validate_annotations(output, selected_refs):
    rows = AnnotationOutput.model_validate(output).model_dump()["annotations"]
    ids = [r["paragraph_ref"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(selected_refs):
        raise ValueError("paragraph_ranked_one_annotation_per_selected_paragraph")
    if any(not r["fact_des"].strip() or any(not e.strip() for e in r["entities"]) or len(set(r["entities"])) != len(r["entities"]) for r in rows):
        raise ValueError("paragraph_ranked_annotation_empty_or_duplicate")
    return sorted(rows, key=lambda row: selected_refs.index(row["paragraph_ref"]))


def validate_fields(output, annotations):
    rows = FieldOutput.model_validate(output).model_dump()["decisions"]
    byid = {r["paragraph_ref"]: r for r in annotations}
    ids = [r["paragraph_ref"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(byid):
        raise ValueError("paragraph_ranked_field_coverage")
    for row in rows:
        if row["annotation_sha256"] != digest(byid[row["paragraph_ref"]]):
            raise ValueError("paragraph_ranked_field_annotation_binding")
        if not (row["fact_des_supported"] and row["entities_supported"] and row["attribution_preserved"] and all(row["field_checks"].values())):
            raise ValueError("paragraph_ranked_fields_not_verified")
    return sorted(rows, key=lambda row: list(byid).index(row["paragraph_ref"]))


def compile_ranked_result(question, paragraphs, *, queries, recall, ranks, selected_refs, annotations,
                          field_reviews, config=None, excluded_spans=()):
    """Pure replay: no source quote or acceptance flag is supplied by the model."""
    question = project_question(question); config = config_values(config); manifest = source_manifest(paragraphs)
    rebuilt = recall_paragraphs(question, paragraphs, queries, config=config, excluded_spans=excluded_spans)
    if recall != rebuilt or selected_refs != select_ranked(recall, ranks, config=config):
        raise ValueError("paragraph_ranked_recall_or_selection_replay")
    annotations = validate_annotations({"annotations": annotations}, selected_refs)
    fields = validate_fields({"decisions": field_reviews}, annotations)
    byid = {p["id"]: p for p in manifest}; refs = {r["paragraph_ref"]: r for r in recall["candidate_catalog"]}
    # Exact paragraph copies across sources keep distinct records, but contribute
    # one known-identity group. Unrelated groups are not proven independent.
    groups = defaultdict(set); parents = {p["document_id"]: p["document_id"] for p in manifest}
    def root(doc):
        while parents[doc] != doc:
            parents[doc] = parents[parents[doc]]; doc = parents[doc]
        return doc
    def join(docs):
        docs = list(docs)
        for doc in docs[1:]:
            a, b = sorted((root(docs[0]), root(doc))); parents[b] = a
    known_documents = defaultdict(set)
    for p in manifest:
        for name in ("content_sha256", "content_url"):
            if p.get(name):
                known_documents[(name, p[name])].add(p["document_id"])
    for docs in known_documents.values():
        join(sorted(docs))
    for ref in selected_refs:
        p = byid[refs[ref]["paragraph_id"]]; groups[digest(p["text"])].add(p["document_id"])
    for docs in groups.values():
        join(sorted(docs))
    # Direct near-mirror correlation is not proof of distinct provenance.
    # Keep every source record and its own ranking/field decision, while
    # conservatively preventing a known near copy from adding an independent
    # confidence vote. Components here constrain votes, never expand recall or
    # assert that every pair in a component has equivalent facts/provenance.
    mirror_index = recall["mirror_candidate_index"]
    for edge in mirror_index["similar_edges"]:
        docs = {byid[member["paragraph_id"]]["document_id"]
            for side in ("left_text_sha256", "right_text_sha256")
            for member in mirror_index["text_buckets"][edge[side]]["members"]}
        join(sorted(docs))
    components = defaultdict(list)
    for doc in sorted(parents):
        components[root(doc)].append(doc)
    evidence = []
    from app.domain.source_metadata import source_credibility_value
    for annotation in annotations:
        p = byid[refs[annotation["paragraph_ref"]]["paragraph_id"]]; quote = p["text"]
        identity = str(uuid5(NAMESPACE_URL, "trust-judge:v19:" + digest({"question": question, "paragraph": p["id"], "annotation": annotation})))
        grade, grade_basis = source_credibility_value(p)
        evidence.append({"id": identity, "paragraph_id": p["id"], "raw_doc_id": p["raw_doc_id"], "raw_para_id": p["raw_para_id"],
            "start_cp": 0, "end_cp": len(quote), "quote": quote, "fact_des": annotation["fact_des"],
            "argument": deepcopy(annotation["argument"]), "entities": list(annotation["entities"]),
            "grade_score": grade, "grade_basis": grade_basis, "source_level": "unknown", "quote_verified": True, "paragraph_fields_verified": True,
            "interpretation_origin": "model", "model_span_corrected": False, "span_location_method": "program_whole_original_paragraph",
            "source_group_id": "PG" + digest(components[root(p["document_id"])]),
            "source_group_basis": "conservative_doc_identity_exact_or_direct_mirror_correlation_not_proven_independence"})
    from app.domain.material_eligibility import apply_workflow_material_eligibility
    qualified, material = apply_workflow_material_eligibility(question, paragraphs, evidence)
    proof = {"version": VERSION, "question_sha256": digest(question), "source_manifest_sha256": digest(manifest), "config": config,
        "query_catalog": deepcopy(queries), "recall": deepcopy(recall), "rank_decisions": deepcopy(ranks),
        "selected_refs": list(selected_refs), "annotations": annotations, "field_reviews": fields,
        "excluded_spans_sha256": digest(sorted(map(list, excluded_spans))), "qualified_evidence_sha256": digest(qualified),
        "operator_exclusions": sorted(map(list, excluded_spans)),
        "one_record_per_selected_source_key": True, "atomic_fact_splitting_performed": False,
        "formal_paragraph_count": len(qualified), "selected_paragraph_count": len(selected_refs), "all_sources_scanned": False}
    proof["proof_sha256"] = digest(proof)
    return {"evidence": qualified, "verification_protocol": VERSION, "paragraph_ranked_proof": proof, "material_eligibility": material}


def validate_ranked_result(result, question, paragraphs, *, expected_config=None, excluded_spans=None):
    if result.get("verification_protocol") != VERSION or any(k in result for k in ("strict_scope_review", "quote_faithful_bank", "purpose_review")):
        raise ValueError("paragraph_ranked_protocol_not_legacy_scope")
    proof = result.get("paragraph_ranked_proof")
    if not isinstance(proof, dict) or proof.get("version") != VERSION:
        raise ValueError("paragraph_ranked_complete_proof_required")
    if excluded_spans is None:
        # Standalone source replay can validate the declared immutable spans.
        # Repository additionally binds them to trusted origin-run overlays.
        excluded_spans = proof.get("operator_exclusions", [])
    config = config_values(proof.get("config"))
    if expected_config is not None and config != config_values(expected_config):
        raise ValueError("paragraph_ranked_config_binding")
    replayed = compile_ranked_result(question, paragraphs, queries=proof["query_catalog"], recall=proof["recall"],
        ranks=proof["rank_decisions"], selected_refs=proof["selected_refs"], annotations=proof["annotations"],
        field_reviews=proof["field_reviews"], config=config, excluded_spans=excluded_spans)
    if any(result.get(k) != replayed[k] for k in replayed):
        raise ValueError("paragraph_ranked_formal_projection_replay")
    return replayed
