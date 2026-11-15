"""Pure synthetic final-record/pair tests; no files, reference data or judges."""
from copy import deepcopy
import pytest
from scripts.two_stage_evidence_matching import prepare_pairs, compare_pairs


def fact(doc="a",para="p",text="合成记录声称程序尚未完成。"):
    return {"doc_id":doc,"para_id":para,"quote":text,"fact_des":"程序状态仍未完成","argument":{},"entities":[]}


def catalog(left,right,policy="raw_records"):
    sources={(r["doc_id"],r["para_id"]):r["quote"]for r in left+right if "doc_id"in r}
    return prepare_pairs("程序是否完成？",left,right,sources,record_policy=policy)


def decision(pair,label="match",sample=None,legacy=False):
    return {"payload_sha256":pair["legacy_payload_sha256"]if legacy else pair["payload_sha256"],"verdict":label,
        "judge":{"provider":"synthetic","model":"synthetic"},"prompt_version":"synthetic-v1","prompt_sha256":"a"*64,
        "closed_provenance":[{"file":"synthetic-closed-receipt.json","sha256":"b"*64}],"sampling":sample}


def count(result,interpretation,bound):return result["interpretations"][interpretation][bound]["metrics"]["matches"]


def test_cross_id_same_fact_is_only_confirmed_with_exact_decision():
    c=catalog([fact("one")],[fact("two")]);p=c["pairs"][0]
    assert not p["same_id"] and p["legacy_payload_sha256"]is None and p["ready_for_semantic_judge"]
    unknown=compare_pairs(c);assert count(unknown,"I","lower")==0 and count(unknown,"I","upper")==1
    result=compare_pairs(c,[decision(p)])
    assert count(result,"I","lower")==count(result,"II","lower")==1
    assert not result["same_text_implies_same_source"] and not result["closed_file_provenance_verified_here"]


def test_same_id_semantic_conflict_separates_two_interpretations():
    c=catalog([fact()],[fact()]);result=compare_pairs(c,[decision(c["pairs"][0],"no_match")])
    assert count(result,"I","lower")==1 and count(result,"II","lower")==0
    assert count(result,"II","upper")==0


def test_global_augmenting_rematch_avoids_greedy_same_id_lock():
    c=catalog([fact("a"),fact("b")],[fact("a"),fact("c")])
    ds=[decision(p)for p in c["pairs"]if (p["submitted_index"],p["reference_index"])in {(0,1),(1,0)}]
    result=compare_pairs(c,ds)
    assert count(result,"I","lower")==count(result,"II","lower")==2
    assert len({r["reference_unit"]for r in result["interpretations"]["I"]["lower"]["matching"]})==2


def test_unknown_and_unlocated_reference_keep_explicit_bounds_and_denominators():
    c=catalog([fact("a")],[fact("b"),{"_unresolved_reference":"synthetic-missing"}])
    r=compare_pairs(c);m=r["interpretations"]["II"]["upper"]["metrics"]
    assert m["reference_units"]==2 and m["matches"]==1 and m["recall"]==.5
    assert count(r,"II","lower")==0 and not r["upper_bound_is_measured_precision_or_recall"]


def test_duplicate_raw_and_unique_paragraph_policies_are_distinct():
    left=[fact(),fact()];right=[fact(),fact()]
    raw=compare_pairs(catalog(left,right));unique=compare_pairs(catalog(left,right,"unique_paragraphs"))
    assert count(raw,"I","lower")==2 and count(unique,"I","lower")==1
    mirrors=compare_pairs(catalog([fact("a"),fact("b")],[fact("a"),fact("b")],"unique_paragraphs"))
    assert mirrors["interpretations"]["I"]["lower"]["metrics"]["submitted_units"]==2


def test_conflicting_closed_cache_labels_are_unknown_not_best_of():
    c=catalog([fact("a")],[fact("b")]);p=c["pairs"][0]
    r=compare_pairs(c,[decision(p),decision(p,"no_match"),decision(p)])
    assert r["pair_diagnostics"][0]["semantic_label"]=="unknown"
    assert count(r,"I","lower")==0 and count(r,"I","upper")==1


def test_legacy_temperature_zero_single_is_not_three_votes():
    c=catalog([fact()],[fact()]);p=c["pairs"][0];old=decision(p,legacy=True)
    assert count(compare_pairs(c,[old]),"II","lower")==1
    r=compare_pairs(c,[old],vote_policy="majority3")
    assert count(r,"II","lower")==0 and r["counts"]["semantic_unknown_pairs"]==1
    assert not r["single_vote_is_three_samples"]


def test_explicit_three_slots_and_no_partial_set_fabrication():
    c=catalog([fact("a")],[fact("b")]);p=c["pairs"][0]
    ds=[decision(p,label,{"protocol":"three-independent-cache-slots-v1","sample_index":i,"sample_count":3})
        for i,label in [(1,"match"),(2,"match"),(3,"no_match")]]
    assert count(compare_pairs(c,ds,vote_policy="majority3"),"II","lower")==1
    assert count(compare_pairs(c,ds[:2],vote_policy="majority3"),"II","lower")==0
    ds.append(decision(p,"no_match",{"protocol":"three-independent-cache-slots-v1","sample_index":1,"sample_count":3}))
    assert count(compare_pairs(c,ds,vote_policy="majority3"),"II","lower")==0


def test_exact_payload_and_question_or_context_changes_invalidate_old_label():
    c=catalog([fact("a")],[fact("b")]);old=decision(c["pairs"][0]);wrong=deepcopy(old);wrong["payload_sha256"]="0"*64
    with pytest.raises(ValueError):compare_pairs(c,[wrong])
    new=prepare_pairs("改变后的完整原题？",[fact("a")],[fact("b")],{("a","p"):fact()["quote"],("b","p"):fact()["quote"]})
    with pytest.raises(ValueError):compare_pairs(new,[old])
    new=prepare_pairs("程序是否完成？",[fact("a")],[fact("b")],{("a","p"):fact()["quote"]+"新增限定",("b","p"):fact()["quote"]})
    with pytest.raises(ValueError):compare_pairs(new,[old])


def test_missing_full_source_is_unknown_and_not_ready_for_judge():
    c=prepare_pairs("程序是否完成？",[fact("a")],[fact("b")],{})
    r=compare_pairs(c)
    assert r["counts"]["semantic_unknown_pairs"]==1 and r["counts"]["pending_judge_pairs"]==0
    assert count(r,"II","upper")==1
    assert count(compare_pairs(c,[decision(c["pairs"][0])]),"II","lower")==0


def test_same_id_incomplete_context_keeps_I_hit_but_cannot_confirm_II_from_cache():
    c=prepare_pairs("程序是否完成？",[fact()],[fact()],{})
    r=compare_pairs(c,[decision(c["pairs"][0])])
    assert count(r,"I","lower")==1 and count(r,"II","lower")==0
    assert r["counts"]["pending_judge_pairs"]==0


def test_pair_scope_tamper_does_not_create_an_ID_hit():
    c=catalog([fact("a")],[fact("b")]);c["pairs"][0]["same_id"]=True
    with pytest.raises(ValueError,match="source_identity"):compare_pairs(c)


def test_pair_catalog_survives_json_roundtrip_without_losing_source_scope():
    import json
    c=json.loads(json.dumps(catalog([fact()],[fact()])))
    assert count(compare_pairs(c),"I","lower")==1
