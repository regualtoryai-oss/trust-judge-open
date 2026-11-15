"""Pure OFFLINE two-interpretation matching; no file/model/runtime access.

The caller must first verify CLOSED generation and cache files. This module
binds semantic decisions to exact question/fact/source payloads, not file truth.
No embedding score, same source text or unknown edge is a confirmed fact match.
"""
from collections import defaultdict
from hashlib import sha256
import json
import re

FACT_FIELDS = ("doc_id", "para_id", "quote", "fact_des", "argument", "entities",
               "start_cp", "end_cp", "char_start", "char_end")


def fingerprint(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def location(row):
    key = row.get("doc_id"), row.get("para_id")
    return key if all(isinstance(x, str) and x for x in key) else None


def units(rows, policy):
    result, groups = [], {}
    for index, row in enumerate(rows):
        key = location(row)
        if policy == "unique_paragraphs" and key is not None and key in groups:
            result[groups[key]]["row_indices"].append(index)
        else:
            if key is not None: groups[key] = len(result)
            result.append({"source": key, "row_indices": [index]})
    return result


def prepare_pairs(question, submitted, reference, source_contexts, *, record_policy="raw_records"):
    """Freeze feasible within-question pairs, including previously omitted IDs.

    source_contexts maps original (doc_id,para_id) to bound literal context text.
    Cross-paragraph quotations may need a larger program-bound context. Missing
    context/claim stays unknown and is not a ready-to-send judge request.
    """
    text = question.get("question") if isinstance(question, dict) else question
    if not isinstance(text, str) or not text.strip() or record_policy not in {"raw_records", "unique_paragraphs"}:
        raise ValueError("matching_original_question_and_explicit_record_policy_required")
    if (not isinstance(submitted, list) or not isinstance(reference, list) or not isinstance(source_contexts, dict)
            or any(not isinstance(r, dict) for r in submitted + reference)):
        raise ValueError("matching_final_record_lists_and_source_contexts_required")
    left, right = units(submitted, record_policy), units(reference, record_policy)
    lu = {i:u for u,row in enumerate(left) for i in row["row_indices"]}
    ru = {i:u for u,row in enumerate(right) for i in row["row_indices"]}
    pairs = []
    for i, predicted in enumerate(submitted):
        for j, gold in enumerate(reference):
            a,b = location(predicted), location(gold)
            if a is None or b is None: continue  # unlocated rows remain denominators
            contexts = []
            for key in (a,b):
                value = source_contexts.get(key)
                if value is not None and not isinstance(value,str): raise ValueError("matching_literal_context_text_required")
                contexts.append({"doc_id":key[0],"para_id":key[1],"text":value})
            p = {k:predicted[k] for k in FACT_FIELDS if k in predicted}; g = {k:gold[k] for k in FACT_FIELDS if k in gold}
            payload = {"question":text,"submitted_source":contexts[0],"reference_source":contexts[1],"submitted":p,"reference":g}
            legacy = ({"question":text,"source_paragraph":contexts[0],"submitted":p,"reference":g} if a == b else None)
            ready = all(isinstance(row.get("fact_des"),str) and bool(row["fact_des"].strip())
                and isinstance(row.get("quote"),str) and bool(row["quote"])
                and isinstance(context["text"],str) and row["quote"] in context["text"] for row,context in zip((p,g),contexts))
            pairs.append({"pair_ref":f"E{len(pairs)+1}","submitted_index":i,"reference_index":j,
                "submitted_unit":lu[i],"reference_unit":ru[j],"same_id":a==b,
                "payload_sha256":fingerprint(payload),"legacy_payload_sha256":fingerprint(legacy) if legacy else None,
                "ready_for_semantic_judge":ready,"not_ready_reason":None if ready else "missing_claim_or_literal_source_context"})
    return {"format":"offline-two-interpretation-pair-catalog-v1","record_policy":record_policy,
        "question_sha256":fingerprint(text),"submitted_records_sha256":fingerprint(submitted),"reference_records_sha256":fingerprint(reference),
        "submitted_units":left,"reference_units":right,"pairs":pairs,
        "catalog_sha256":fingerprint([text,submitted,reference,sorted((k,v) for k,v in source_contexts.items()),record_policy]),
        "contains_reference_material":True,"permitted_runtime_use":False,"new_model_calls":0,
        "closed_file_provenance_verified_here":False}


def _decision_label(records, vote_policy):
    """Contradictory immutable labels are unknown, never best-of-cache."""
    if not records: return "unknown"
    grouped = defaultdict(list)
    for row in records:
        identity = fingerprint([row["judge"],row["prompt_version"],row["prompt_sha256"]])
        grouped[identity].append(row)
    labels = []
    for rows in grouped.values():
        if vote_policy == "legacy_single":
            values = {r["verdict"] for r in rows if r.get("sampling") is None}
            labels.append(next(iter(values)) if len(values)==1 else "unknown")
        else:
            slots = []
            for index in (1,2,3):
                values = {r["verdict"] for r in rows if r.get("sampling") is not None and r["sampling"]["sample_index"]==index}
                slots.append(next(iter(values)) if len(values)==1 else "unknown")
            labels.append(next((label for label in ("match","no_match","uncertain") if slots.count(label)>=2),"unknown")
                if "unknown" not in slots else "unknown")
    distinct = set(labels)
    value = next(iter(distinct)) if len(distinct)==1 else "unknown"
    return value if value in {"match","no_match"} else "unknown"


def _maximum_matching(left_count, right_count, edges):
    adjacent = defaultdict(list)
    for a,b in sorted(set(edges)): adjacent[a].append(b)
    occupied = {}
    def visit(a, seen):
        for b in adjacent[a]:
            if b in seen: continue
            seen.add(b)
            if b not in occupied or visit(occupied[b],seen): occupied[b]=a; return True
        return False
    for a in range(left_count): visit(a,set())
    return [{"submitted_unit":a,"reference_unit":b} for b,a in sorted(occupied.items())]


def _metrics(matches, submitted, reference):
    return {"matches":matches,"submitted_units":submitted,"reference_units":reference,
        "precision":matches/submitted if submitted else None,"recall":matches/reference if reference else None,
        "f1":2*matches/(submitted+reference) if submitted+reference else None}


def compare_pairs(catalog, closed_decisions=(), *, vote_policy="legacy_single"):
    """Use exact payload hashes only; CLOSED file verification belongs upstream.

    Each decision must include payload_sha256, verdict, judge, prompt_version,
    prompt_sha256, closed_provenance:[{file,sha256}], and optional sampling.
    Provenance descriptors are retained/validated structurally, not opened here.
    """
    if catalog.get("format") != "offline-two-interpretation-pair-catalog-v1" or vote_policy not in {"legacy_single","majority3"}:
        raise ValueError("matching_pair_catalog_and_vote_policy_required")
    owners=[]
    for side in ("submitted_units","reference_units"):
        owner={}
        for index,unit in enumerate(catalog[side]):
            for row in unit["row_indices"]:
                if type(row)is not int or row<0 or row in owner:raise ValueError("matching_unique_unit_row_indices_required")
                owner[row]=index
        if sorted(owner)!=list(range(len(owner))):raise ValueError("matching_complete_unit_row_indices_required")
        owners.append(owner)
    seen=set()
    for index,pair in enumerate(catalog["pairs"],1):
        key=pair["submitted_index"],pair["reference_index"]
        if (key in seen or key[0]not in owners[0] or key[1]not in owners[1] or pair["pair_ref"]!=f"E{index}"
                or owners[0][key[0]]!=pair["submitted_unit"] or owners[1][key[1]]!=pair["reference_unit"]
                or type(pair["same_id"])is not bool):raise ValueError("matching_original_pair_unit_scope_required")
        a=catalog["submitted_units"][pair["submitted_unit"]]["source"]
        b=catalog["reference_units"][pair["reference_unit"]]["source"]
        if a is None or b is None or pair["same_id"]!=(a==b):raise ValueError("matching_pair_source_identity_changed")
        seen.add(key)
    known = {h for p in catalog["pairs"] for h in (p["payload_sha256"],p["legacy_payload_sha256"]) if h}
    by_hash = defaultdict(list)
    for row in closed_decisions:
        if (row.get("payload_sha256") not in known or row.get("verdict") not in {"match","no_match","uncertain"}
                or not isinstance(row.get("judge"),dict) or not row["judge"] or not isinstance(row.get("prompt_version"),str)
                or not isinstance(row.get("prompt_sha256"),str) or not re.fullmatch(r"[0-9a-f]{64}",row["prompt_sha256"])
                or not isinstance(row.get("closed_provenance"),list) or not row["closed_provenance"]
                or any(set(p)!={"file","sha256"} or not isinstance(p["file"],str) or not re.fullmatch(r"[0-9a-f]{64}",p["sha256"]) for p in row["closed_provenance"])):
            raise ValueError("matching_exact_payload_closed_decision_binding_required")
        sample = row.get("sampling")
        if sample is not None and (not isinstance(sample,dict) or sample.get("protocol")!="three-independent-cache-slots-v1"
                or type(sample.get("sample_index"))is not int or sample["sample_index"]not in {1,2,3} or sample.get("sample_count")!=3):
            raise ValueError("matching_explicit_three_slot_sampling_required")
        by_hash[row["payload_sha256"]].append(row)
    classified=[]; edges={"I":{"lower":set(),"upper":set()},"II":{"lower":set(),"upper":set()}}
    for pair in catalog["pairs"]:
        records = by_hash[pair["payload_sha256"]]+(by_hash[pair["legacy_payload_sha256"]] if pair["legacy_payload_sha256"] else [])
        # An exact hash cannot cure an incomplete source/claim context. ID-only
        # interpretation I remains separate from semantic qualification.
        label = _decision_label(records,vote_policy) if pair["ready_for_semantic_judge"] else "unknown"
        edge=(pair["submitted_unit"],pair["reference_unit"])
        for interpretation in ("I","II"):
            if label=="match" or interpretation=="I" and pair["same_id"]:
                edges[interpretation]["lower"].add(edge);edges[interpretation]["upper"].add(edge)
            elif label=="unknown":edges[interpretation]["upper"].add(edge)
        classified.append({**pair,"semantic_label":label,"bound_cache_records":len(records),
            "pending_judge":label=="unknown" and pair["ready_for_semantic_judge"]})
    n,m=len(catalog["submitted_units"]),len(catalog["reference_units"]); result={}
    for interpretation,graph in edges.items():
        result[interpretation]={}
        for bound,eligible in graph.items():
            matching=_maximum_matching(n,m,eligible)
            result[interpretation][bound]={"metrics":_metrics(len(matching),n,m),"matching":matching}
        result[interpretation]["point_estimate_available"]=graph["lower"]==graph["upper"]
    return {"format":"offline-two-interpretation-matching-bounds-v1","record_policy":catalog["record_policy"],
        "interpretations":result,"pair_diagnostics":classified,"vote_policy":vote_policy,
        "counts":{"feasible_pairs":len(classified),"same_id_pairs":sum(p["same_id"]for p in classified),
            "cross_id_pairs":sum(not p["same_id"]for p in classified),"semantic_unknown_pairs":sum(p["semantic_label"]=="unknown"for p in classified),
            "pending_judge_pairs":sum(p["pending_judge"]for p in classified),
            "pending_for_I_cross_id":sum(p["pending_judge"]and not p["same_id"]for p in classified),
            "pending_for_II_all_ids":sum(p["pending_judge"]for p in classified),
            "unlocated_submitted_units":sum(u["source"]is None for u in catalog["submitted_units"]),
            "unlocated_reference_units":sum(u["source"]is None for u in catalog["reference_units"])},
        "upper_scope":"located_source_pair_space_only; unlocated rows remain denominators, not a bound on all unknowable facts",
        "unknown_is_confirmed_match":False,"same_text_implies_same_source":False,"global_maximum_one_to_one":True,
        "upper_bound_is_measured_precision_or_recall":False,"official_score":False,"closed_file_provenance_verified_here":False,
        "single_vote_is_three_samples":False,"new_model_calls":0,"permitted_runtime_use":False}
