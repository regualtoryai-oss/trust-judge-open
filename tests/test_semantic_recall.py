"""Synthetic local dense retrieval tests; no optional model download/import."""
from copy import deepcopy
import json

import numpy as np
import pytest

from app.domain.semantic_recall import (SemanticRecallUnavailable,FastEmbedLocal,build_index,
    fuse_candidates,rank_dense,source_manifest,window_catalog)


def paragraphs():
    return [{'id':'P1','raw_doc_id':'d1','raw_para_id':'p1','text':'甲方尚未确认。'*9},
            {'id':'P2','raw_doc_id':'d2','raw_para_id':'p1','text':'乙方公布最终数值。'},
            {'id':'P3','raw_doc_id':'d2','raw_para_id':'p2','text':'同一来源中的不同段落。'}]


def fixture_index():
    rows=paragraphs();windows=window_catalog(rows,window_chars=32,overlap_chars=8)
    vectors=np.array([[1.,0.]if w['paragraph_id']=='P2'else[0.,1.]for w in windows])
    return rows,windows,vectors,build_index(rows,windows,vectors,model_fingerprint='pinned-revision+hash',window_chars=32,overlap_chars=8)


def test_windows_are_original_halfopen_spans_and_every_parent_is_preserved():
    rows=paragraphs();before=deepcopy(rows);windows=window_catalog(rows,window_chars=32,overlap_chars=8)
    byid={r['id']:r for r in rows}
    assert rows==before and len(windows)>len(rows)
    for window in windows:
        assert window['text']==byid[window['paragraph_id']]['text'][window['start_cp']:window['end_cp']]
    assert {w['paragraph_id']for w in windows}=={'P1','P2','P3'}
    assert len({w['window_id']for w in windows})==len(windows)


def test_dense_collapses_windows_to_parent_once_and_binds_raw_text():
    rows,windows,vectors,index=fixture_index()
    candidates=rank_dense(index,[[3.,0.]],limit=3,paragraphs=rows)
    assert [r['id']for r in candidates]==['P2','P1','P3']
    assert candidates[0]['dense_cosine']==pytest.approx(1.)
    assert len({r['id']for r in candidates})==3
    changed=deepcopy(rows);changed[0]['text']+='改变'
    with pytest.raises(ValueError,match='source_changed'):rank_dense(index,[[1.,0.]],paragraphs=changed)
    vectors[0]=[999,999]
    assert rank_dense(index,[[1.,0.]],limit=1)[0]['id']=='P2'


def test_index_rejects_window_id_or_span_or_source_tamper():
    rows,windows,vectors,index=fixture_index()
    for key,value in [('paragraph_id','OTHER'),('text','伪造'),('start_cp',99)]:
        changed=deepcopy(windows);changed[0][key]=value
        with pytest.raises(ValueError,match='source_binding'):build_index(rows,changed,vectors,model_fingerprint='revision',window_chars=32,overlap_chars=8)


@pytest.mark.parametrize('query',[[[float('nan'),1]],[[0.,0.]],[[1.,0.,0.]],[[float('inf'),1]]])
def test_invalid_query_vectors_fail_closed(query):
    _,_,_,index=fixture_index()
    with pytest.raises(ValueError):rank_dense(index,query)


def test_model_and_content_changes_invalidate_binding_and_arrays_are_readonly():
    rows,windows,vectors,index=fixture_index()
    second=build_index(rows,windows,vectors,model_fingerprint='different-model',window_chars=32,overlap_chars=8)
    assert index.binding_sha256!=second.binding_sha256
    with pytest.raises(ValueError):index._vectors[0]=[1,1]
    descriptor=index.descriptor;descriptor['model_fingerprint']='changed'
    assert index.descriptor['model_fingerprint']=='pinned-revision+hash'


def test_rrf_fuses_parents_and_has_stable_ties_without_credibility_votes():
    rows=paragraphs()
    result=fuse_candidates(rows,[[{'id':'P1'},{'id':'P2'}],[{'id':'P2'},{'id':'P1'},{'id':'P3'}]],limit=3)
    assert [r['id']for r in result]==['P1','P2','P3']
    assert result[0]['fusion_score']==result[1]['fusion_score']
    assert len(result[0]['retrieval_signals'])==2
    with pytest.raises(ValueError):fuse_candidates(rows,[[{'id':'P1'},{'id':'P1'}]])
    with pytest.raises(ValueError):fuse_candidates(rows,[[{'id':'gold-id'}]])
    with pytest.raises(ValueError,match='source_mismatch'):fuse_candidates(rows,[[{'id':'P1','raw_doc_id':'wrong'}]])


def test_no_model_is_implicitly_fetched_when_missing(tmp_path):
    with pytest.raises(SemanticRecallUnavailable,match='not_prepared'):FastEmbedLocal(tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_incomplete_or_unapproved_model_manifest_never_loads_optional_runtime(tmp_path):
    marker=tmp_path/'local-model-manifest.json'
    marker.write_text(json.dumps({'repo':'intfloat/multilingual-e5-small','revision':'revision','files':[]}),encoding='utf-8')
    with pytest.raises(SemanticRecallUnavailable,match='not_allowlisted'):FastEmbedLocal(tmp_path)


def test_source_manifest_excludes_annotations_answers_and_duplicate_location():
    rows=paragraphs();rows[0].update({'gold':'never-index','evidence_samples':{'secret':'never-index'}})
    assert 'never-index'not in json.dumps(source_manifest(rows))
    with pytest.raises(ValueError,match='duplicate'):source_manifest([rows[0],deepcopy(rows[0])])


def test_empty_text_has_no_window_but_invalid_parameters_are_rejected():
    rows=paragraphs();rows[0]['text']=' '
    assert 'P1'not in {w['paragraph_id']for w in window_catalog(rows)}
    with pytest.raises(ValueError):window_catalog(rows,window_chars=20)
    with pytest.raises(ValueError):window_catalog(rows,overlap_chars=320)
