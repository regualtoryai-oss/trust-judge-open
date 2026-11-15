"""Raw-only, rank-first multi-view complete-parent delivery.

Existing BM25 provides lexical navigation. There is no model, BGE claim,
semantic filter, agent loop, source shortening or implicit neighbour expansion.
Raw IDs, rather than import UUIDs, own all ties and token-cost payloads.
"""
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
import re

import tiktoken

from app.domain.question_recall import _bm25, question_query_plan
from app.domain.source_bundle_tools import _raw_only
from app.domain.source_bundles import build_source_bundles

POLICY = {
    "policy": "rank_first_fair_literal_multiview_complete_parent_pack_v2",
    "retrieval_mode": "local_full_corpus_lexical_multiview",
    "question": "unchanged_full_question_context_and_literal_focus_views",
    "maximum_question_views": 5, "maximum_recall_pool": 180,
    "literal_options": "binary_or_multiple_punctuation_options_without_numeric_branches",
    "lexical_channels": "complete_parent_focus_complete_parent_full_and_body_eligible_complete_title_equal_family_rrf60",
    "BM25_scoring_unit": "complete_original_parent_not_best_short_window",
    "BM25_windowed": False,
    "semantic_rank_used": False, "BGE_used": False,
    "selection": "rank_first_round_robin_views_without_length_density",
    "qualifier_lane": "whole_question_body_candidates_with_literal_uncertainty_scope_or_negation",
    "qualifier_is_necessary_evidence": False,
    "neighbors_added": False,
    "exact_mirrors": "distinct_original_IDs_preserved_with_explicit_same_text_groups",
    "tokenizer": "cl100k_base", "payload_proxy_token_budget": 8000,
    "maximum_provided_parents": 60,
    "payload_cost": "canonical_json_complete_text_title_raw_doc_para_IDs_only_no_UUID",
    "oversized_parent": "skip_whole_parent_record_priority_and_budget_gap_never_truncate",
    "source_runs": "one_complete_original_parent_per_run_no_unprovided_gap_claim",
    "unprovided_context": "unknown_not_irrelevant",
    "navigation_score_is_semantic_relevance": False,
    "proxy_budget_is_native_request_budget": False,
}
_CONTRAST = re.compile(r"还是|抑或|或者|而非|或(?=[^者])")
_OPTIONS = re.compile(r"、|[；;]|还是|抑或|或者|而非|或(?=[^者])")
_QUALIFIER = re.compile(r"初步|暂未|暂时|尚未|未经|未确认|未证实|仍待|有待|截至|不含|包含|口径|并非|不是|否认|仅限|草案|拟议")
_TAIL = re.compile(r"[，,](?:以哪|哪一|哪方|谁|如何|这是否|应以哪).*$")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode("utf8")).hexdigest()


def question_views(question):
    """Literal alternatives can be one word; the full question is never lost.

    Full conditions accompany every view as context. They are not claimed to
    have been semantically understood or enforced by lexical matching.
    """
    text=question.get("question") if isinstance(question,dict) else question
    if not isinstance(text,str) or not text.strip():raise ValueError("multiview_original_question_required")
    candidates=[("whole_question",text)]
    # Literal list punctuation covers multi-option statistics/scopes as well
    # as binary questions; numeric values themselves never control branches.
    if _OPTIONS.search(text):
        options=[_TAIL.sub("",part).strip(" ，,；;？?。") for part in _OPTIONS.split(text)]
        options=[part for part in options if part]
        for index,focus in enumerate(options):
            origin=("literal_left" if index==0 else "literal_right") if len(options)==2 else "literal_option"
            candidates.append((origin,focus))
    for row in question_query_plan(text)["facets"]:
        if row["origin"]=="literal_clause":candidates.append(("literal_clause",row["text"]))
    views=[];seen=set()
    for origin,focus in candidates:
        if not focus or focus in seen:continue
        seen.add(focus)
        views.append({"view_id":"V"+str(len(views)+1),"origin":origin,"focus":focus,
                      "full_original_question":text,"conditions_semantically_verified":False})
        if len(views)>=POLICY["maximum_question_views"]:break
    return views


def _rrf(ranked):
    return {index:1/(60+rank) for rank,(_,index) in enumerate(ranked,1)}


def _body_family(rows,query):
    result=defaultdict(float)
    for kind in ("word","bigram"):
        for index,score in _rrf(_bm25(rows,query,kind,windowed=False)).items():result[index]+=score/2
    return dict(result)


def _rank_views(rows,views):
    whole=_body_family(rows,views[0]["full_original_question"])
    title_rows=[{**r,"text":r["title"]} for r in rows]
    result=[]
    for view in views:
        focus=_body_family(rows,view["focus"]) if view["focus"]!=views[0]["focus"] else whole
        title=_body_family(title_rows,view["focus"])
        eligible=set(whole)|set(focus)
        channels=[focus,whole] if view["focus"]!=views[0]["focus"] else [whole]
        channels.append({i:score for i,score in title.items() if i in eligible})
        fused={i:sum(channel.get(i,0) for channel in channels)/len(channels) for i in eligible}
        order=sorted(fused,key=lambda i:(-fused[i],rows[i]["raw_doc_id"],rows[i]["raw_para_id"]))
        result.append({**view,"ranked_paragraph_ids":[rows[i]["id"] for i in order],
            "scores_are_navigation_only":True,"body_matched_parents":len(eligible)})
    qualifier_order=[i for i in sorted(whole,key=lambda i:(-whole[i],rows[i]["raw_doc_id"],rows[i]["raw_para_id"]))
                     if _QUALIFIER.search(rows[i]["text"])]
    if qualifier_order:
        result.append({"view_id":"Q","origin":"corpus_literal_qualifier_lane","focus":views[0]["focus"],
            "full_original_question":views[0]["full_original_question"],
            "conditions_semantically_verified":False,"scores_are_navigation_only":True,
            "body_matched_parents":len(qualifier_order),"ranked_paragraph_ids":[rows[i]["id"] for i in qualifier_order]})
    return result


def _fair_order(lists):
    cursor=[0]*len(lists);seen=set();result=[];first_view={}
    while True:
        advanced=False
        for i,items in enumerate(lists):
            while cursor[i]<len(items) and items[cursor[i]] in seen:cursor[i]+=1
            if cursor[i]>=len(items):continue
            pid=items[cursor[i]];cursor[i]+=1;advanced=True;seen.add(pid)
            result.append(pid);first_view[pid]=i
        if not advanced:return result,first_view


def pack_original_parents(raw,tools,document_exclusions):
    """Compatibility tuple for the existing source-book adapter.

    ``tools`` is intentionally not invoked: ranking reuses the local BM25
    implementation over the full validated corpus, not a three-page facade.
    The receipt truthfully records zero SourceBundleTools search calls.
    """
    _raw_only((raw["documents"],raw["paragraphs"],raw["question"]))
    build_source_bundles(raw["documents"],raw["paragraphs"],[],max_paragraphs=5)
    documents={d["id"]:d for d in raw["documents"]}
    rows=sorted(({**r,"title":documents[r["document_id"]].get("title","")} for r in raw["paragraphs"] if r["text"].strip()),
                key=lambda r:(r["raw_doc_id"],r["raw_para_id"]))
    if len({(r["raw_doc_id"],r["raw_para_id"]) for r in rows})!=len(rows):raise ValueError("multiview_duplicate_raw_source")
    views=_rank_views(rows,question_views(raw["question"]))
    fair_order,first_view=_fair_order([v["ranked_paragraph_ids"] for v in views])
    pool=fair_order[:POLICY["maximum_recall_pool"]]
    by_id={r["id"]:r for r in raw["paragraphs"]}
    encoding=tiktoken.get_encoding(POLICY["tokenizer"])
    records=[];selected=[];used=0
    for rank,pid in enumerate(pool,1):
        row=by_id[pid];doc=documents[row["document_id"]]
        payload={"doc_id":row["raw_doc_id"],"para_id":row["raw_para_id"],"title":doc.get("title",""),"text":row["text"]}
        cost=max(1,len(encoding.encode(_json(payload),disallowed_special=())))
        excluded=any(item["document_id"]==row["document_id"] and item["start_cp"]<row["document_end_cp"]
                     and row["document_start_cp"]<item["end_cp"] for item in document_exclusions)
        if excluded:reason="operator_exclusion"
        elif len(selected)>=POLICY["maximum_provided_parents"]:reason="selection_capacity"
        elif used+cost>POLICY["payload_proxy_token_budget"]:reason="pooled_over_budget"
        else:reason="provided_complete_original_parent";selected.append(pid);used+=cost
        signals=[{"view_id":v["view_id"],"rank":v["ranked_paragraph_ids"].index(pid)+1} for v in views if pid in v["ranked_paragraph_ids"]]
        records.append({"paragraph_id":pid,"raw_doc_id":row["raw_doc_id"],"raw_para_id":row["raw_para_id"],
            "pool_rank":rank,"priority_view_id":views[first_view[pid]]["view_id"],"ranking_signals":signals,
            "payload_proxy_tokens":cost,"payload_sha256":_digest(payload),"operator_excluded":excluded,
            "provided":reason=="provided_complete_original_parent","reason":reason,
            "priority_is_length_density":False,"source_text_sha256":sha256(row["text"].encode("utf8")).hexdigest()})
    selected_set=set(selected);decisions={r["paragraph_id"]:r for r in records}
    unprovided=[{"paragraph_id":r["id"],"raw_doc_id":r["raw_doc_id"],"raw_para_id":r["raw_para_id"],
        "reason":decisions[r["id"]]["reason"] if r["id"] in decisions else "outside_recall_pool",
        "context_state":"unknown_not_irrelevant"} for r in raw["paragraphs"] if r["id"] not in selected_set]
    mirror_groups=defaultdict(list)
    for r in raw["paragraphs"]:mirror_groups[sha256(r["text"].encode("utf8")).hexdigest()].append(r["id"])
    mirrors=[{"source_text_sha256":key,"paragraph_ids":ids,"same_text_is_independent_sources":False}
             for key,ids in sorted(mirror_groups.items()) if len(ids)>1]
    navigation={"strategy":POLICY["policy"],"views":views,"BGE_used":False,"preview_text_sent_to_model":False}
    receipt={"format":"complete-parent-token-pack-receipt-v2","policy":deepcopy(POLICY),
        "question_sha256":_digest(raw["question"]),"tokenizer_package_version":version("tiktoken"),
        "retrieval_mode":POLICY["retrieval_mode"],"search_page_count":0,"search_pages_sha256":_digest(navigation),
        "recall_pool_paragraph_ids":pool,"recall_pool_count":len(pool),"candidate_records":records,
        "packing_priority_ids":list(pool),"provided_paragraph_ids":selected,"provided_parent_count":len(selected),
        "provided_payload_proxy_tokens":used,"unprovided_reason_records":unprovided,"source_text_shortened":False,
        "question_views":views,"exact_mirror_groups":mirrors,
        "priority_not_provided":[{k:r[k] for k in ("paragraph_id","raw_doc_id","raw_para_id","pool_rank","priority_view_id","payload_proxy_tokens","reason")}
                                 for r in records if not r["provided"]],
        "model_calls":0,"reference_data_accessed":False,"semantic_verified":False,"BGE_used":False}
    return selected,navigation,receipt
