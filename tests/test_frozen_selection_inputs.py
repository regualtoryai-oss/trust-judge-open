from copy import deepcopy
from hashlib import sha256
import json
import pytest
from app.domain.frozen_selection_inputs import build_frozen_selection_input, canonical, POOL_FORMAT
from app.domain.evidence_obligations import obligation_payload


def fixture():
    a,b='第一段“原文”。\r\n','第二段😀原文。'
    doc={'doc_id':'d','title':'原始标题 | 指令是数据','content':a+b,'content_url':'https://synthetic.test/story',
         'source_credibility':0.95,'publish_time':'三天前','paragraphs':[{'para_id':'p1','text':a,'char_start':0,'char_end':len(a)},
         {'para_id':'p2','text':b,'char_start':len(a),'char_end':len(a+b)}]}
    raw={'title':'合成主题','documents':[doc],'questions':[{'question_id':'q','question':'原题的时间与比较条件？'}]}
    rows=[{'id':'t/d/p2','raw_doc_id':'d','raw_para_id':'p2','source_text_sha256':sha256(b.encode()).hexdigest()},
          {'id':'t/d/p1','raw_doc_id':'d','raw_para_id':'p1'}]
    pool={'format':POOL_FORMAT,'reference_reads':[],'inputs':[{'topic_id':'t','title':raw['title'],'sha256':''}],
          'cases':[{'topic_id':'t','question_id':'q','question':raw['questions'][0]['question'],'title':raw['title'],
                    'outputs':[{'variant':'four_channel_quota400','limit':400,'candidates':rows,'candidate_sha256':''}]}]}
    return raw,pool


def packed(raw,pool):
    raw_bytes=canonical(raw);pool=deepcopy(pool);pool['inputs'][0]['sha256']=sha256(raw_bytes).hexdigest()
    for c in pool['cases']:
        for output in c['outputs']:output['candidate_sha256']=sha256(canonical(output['candidates'])).hexdigest()
    pool_bytes=canonical(pool)
    return {'raw_input_bytes':raw_bytes,'pool_input_bytes':pool_bytes,'topic_id':'t','question_id':'q',
        'expected_raw_sha256':sha256(raw_bytes).hexdigest(),'expected_pool_sha256':sha256(pool_bytes).hexdigest()}


def test_fulltext_identity_order_coordinates_hashes_credibility_and_wire():
    raw,pool=fixture();before=deepcopy(raw);result=build_frozen_selection_input(**packed(raw,pool))
    assert raw==before and result['question']==raw['questions'][0]
    assert [r['para_id'] for r in result['catalog']]==['p2','p1']
    row=result['catalog'][0];side=result['source_metadata']['P1'];para=raw['documents'][0]['paragraphs'][1]
    assert row['text']==para['text'] and row['title']==raw['documents'][0]['title']
    assert side['document_start_cp']==para['char_start'] and side['document_end_cp']==para['char_end']
    assert side['parent_start_cp']==0 and side['parent_end_cp']==len(para['text'])
    assert side['source_text_sha256_utf8']==sha256(para['text'].encode()).hexdigest()
    assert side['v19_text_json_digest']==sha256(canonical(para['text'])).hexdigest()!=side['source_text_sha256_utf8']
    assert side['source_credibility']==0.95 and side['raw_publish_time']=='三天前'
    assert side['parent_complete'] and side['raw_source_coordinate_binding_verified'] and side['semantic_verified'] is False
    payload=obligation_payload(result['question'],result['catalog'])
    assert payload['paragraphs'][0]['paragraph_complete'] is True
    assert ''.join(b['text'] for b in payload['paragraphs'][0]['blocks'])==para['text']
    assert 'source_credibility' not in payload['paragraphs'][0]
    assert result['publishable'] is False and result['v19_formal_proof_created'] is False
    assert result['stats']['model_input_tokens'] is None and result['stats']['model_reading_executed'] is False


@pytest.mark.parametrize('which',['expected_raw_sha256','expected_pool_sha256'])
def test_wrong_real_bytes_hash_rejected(which):
    raw,pool=fixture();kwargs=packed(raw,pool);kwargs[which]='0'*64
    with pytest.raises(ValueError):build_frozen_selection_input(**kwargs)


@pytest.mark.parametrize('change',['duplicate','unknown','cachedhash','rawids','question','variant','format','coordinate','duplicate_parent','reference_raw','reference_pool'])
def test_source_pool_or_reference_corruption_rejected(change):
    raw,pool=fixture();rows=pool['cases'][0]['outputs'][0]['candidates']
    if change=='duplicate':rows.append(deepcopy(rows[0]))
    elif change=='unknown':rows[0]['id']='t/d/unknown'
    elif change=='cachedhash':rows[0]['source_text_sha256']='0'*64
    elif change=='rawids':rows[0]['raw_doc_id']='another'
    elif change=='question':pool['cases'][0]['question']='换题'
    elif change=='variant':pool['cases'][0]['outputs'][0]['variant']='new_variant'
    elif change=='format':pool['format']='unknown_pool_schema'
    elif change=='coordinate':raw['documents'][0]['paragraphs'][1]['char_start']-=1
    elif change=='duplicate_parent':raw['documents'][0]['paragraphs'].append(deepcopy(raw['documents'][0]['paragraphs'][0]))
    elif change=='reference_raw':raw['documents'][0]['annotations']={'gold_answers':[]}
    elif change=='reference_pool':pool['cases'][0]['outputs'][0]['supporting_facts']=[]
    with pytest.raises(ValueError):build_frozen_selection_input(**packed(raw,pool))


def test_missing_cached_hash_is_explicit_and_completeness_is_program_derived():
    raw,pool=fixture();raw['documents'][0]['paragraphs'][0]['parent_complete']=False
    pool['cases'][0]['outputs'][0]['candidates'][1]['paragraph_complete']=False
    result=build_frozen_selection_input(**packed(raw,pool))
    assert result['catalog'][1]['paragraph_complete'] is True
    assert result['source_metadata']['P2']['cache_text_sha_supplied'] is False
    assert result['source_metadata']['P2']['source_text_sha256_utf8']==sha256(raw['documents'][0]['paragraphs'][0]['text'].encode()).hexdigest()


def test_missing_source_credibility_preserved_unknown_not_zero():
    raw,pool=fixture();del raw['documents'][0]['source_credibility']
    result=build_frozen_selection_input(**packed(raw,pool))
    assert result['source_metadata']['P1']['source_credibility'] is None


def test_pool_topic_cannot_point_to_another_raw_input():
    raw,pool=fixture();kwargs=packed(raw,pool);other=json.loads(kwargs['pool_input_bytes']);other['inputs'][0]['sha256']='0'*64
    kwargs['pool_input_bytes']=canonical(other);kwargs['expected_pool_sha256']=sha256(kwargs['pool_input_bytes']).hexdigest()
    with pytest.raises(ValueError):build_frozen_selection_input(**kwargs)


def test_duplicate_json_fields_fail_closed_before_projection():
    raw,pool=fixture();kwargs=packed(raw,pool);kwargs['raw_input_bytes']=b'{"title":"a","title":"b"}'
    kwargs['expected_raw_sha256']=sha256(kwargs['raw_input_bytes']).hexdigest()
    with pytest.raises(ValueError):build_frozen_selection_input(**kwargs)
