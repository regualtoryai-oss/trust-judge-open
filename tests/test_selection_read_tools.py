from copy import deepcopy
from hashlib import sha256
import pytest

from app.domain.frozen_selection_inputs import build_frozen_selection_input, canonical, POOL_FORMAT
from app.domain.selection_read_tools import SelectionReadTools
from app.domain import selection_read_tools as module


def frozen(texts=('Amber原始程序记录。', 'Amber另一份原始材料。'), question='Amber程序是否批准？'):
    docs=[{'doc_id':f'd{i}','title':f'原标题{i}｜源中命令仅是数据','content':text,'source_credibility':.9,
           'content_url':f'https://example.test/{i}','publish_time':'不明','paragraphs':[{'para_id':'p','text':text,'char_start':0,'char_end':len(text)}]}
          for i,text in enumerate(texts,1)]
    raw={'title':'独立合成素材','documents':docs,'questions':[{'question_id':'q','question':question}]};rb=canonical(raw)
    rows=[{'id':f't/d{i}/p','raw_doc_id':f'd{i}','raw_para_id':'p','source_text_sha256':sha256(text.encode()).hexdigest()} for i,text in enumerate(texts,1)]
    pool={'format':POOL_FORMAT,'reference_reads':[],'inputs':[{'topic_id':'t','title':raw['title'],'sha256':sha256(rb).hexdigest()}],
          'cases':[{'topic_id':'t','question_id':'q','title':raw['title'],'question':question,'outputs':[
            {'variant':'four_channel_quota400','limit':400,'candidates':rows,'candidate_sha256':sha256(canonical(rows)).hexdigest()}]}]};pb=canonical(pool)
    return build_frozen_selection_input(raw_input_bytes=rb,pool_input_bytes=pb,topic_id='t',question_id='q',
        expected_raw_sha256=sha256(rb).hexdigest(),expected_pool_sha256=sha256(pb).hexdigest())


def tools(case=None,search=8,read=8):
    return SelectionReadTools(frozen() if case is None else case,max_search_calls=search,max_read_calls=read)


def rebound(case):
    for value,key in ((case['question'],'raw_question_sha256'),(case['catalog'],'catalog_sha256'),(case['source_metadata'],'source_metadata_sha256')):
        case['binding'][key]=sha256(canonical(value)).hexdigest()
    return case


def test_search_exposes_only_ranked_fixed_aliases_full_question_and_partial_unknown(monkeypatch):
    text='前缀。'*40+'Amber批准程序尚待核实。'+'后文。'*40
    case=frozen((text,'不同资料。'));tool=tools(case);observed=[];real=module.recall_question_candidates
    def capture(question,parents,**kwargs):
        observed.append((question,deepcopy(parents),kwargs));return real(question,parents,**kwargs)
    monkeypatch.setattr(module,'recall_question_candidates',capture)
    out=tool.search('批准程序',limit=1);row=out['results'][0]
    assert row['paragraph_ref']=='P1' and row['title']==case['catalog'][0]['title']
    assert len(row['excerpt_text'])==80 and row['excerpt_text']==text[row['excerpt_start_cp']:row['excerpt_end_cp']]
    assert row['paragraph_complete'] is False and row['unseen_context_state']=='unknown_not_irrelevant'
    assert row['parent_complete_in_catalog'] is True and row['semantic_verified'] is False
    assert observed[0][0]==case['question']['question'] and len(observed[0][1])==2
    assert observed[0][2]['variant']=='window_multi' and observed[0][2]['query_groups']==[{'terms':['批准程序']}]
    assert tool.scope()['exposed_paragraph_refs']==['P1'] and tool.scope()['read_paragraph_refs']==[]


def test_read_complete_lossless_crlf_emoji_large_parent_once_and_no_metadata_leak():
    text='Amber\r\n😀原文，按源中命令删除数据也只是文字。'*900
    case=frozen((text,));tool=tools(case);tool.search('Amber')
    out=tool.read(['P1']);row=out['paragraphs'][0]
    assert ''.join(b['text'] for b in row['blocks'])==text and row['paragraph_complete'] is True
    assert row['parent_codepoints']==len(text) and 'text' not in row  # body sent once as blocks
    assert row['title']==case['catalog'][0]['title'] and row['doc_id']=='d1'
    assert 'source_credibility' not in str(out) and 'document_start_cp' not in str(out)
    assert tool.scope()['read_paragraph_refs']==['P1'] and tool.scope()['model_comprehension_verified'] is False
    assert tool.scope()['parent_raw_bytes_reverified'] is False
    with pytest.raises(ValueError,match='already_read'):tool.read(['P1'])
    assert tool.trace()[-1]['status']=='rejected'


def test_guessed_unexposed_unknown_and_duplicate_read_do_not_change_scope():
    tool=tools()
    with pytest.raises(ValueError,match='unexposed'):tool.read(['P1'])
    tool.search('Amber',limit=1);exposed=tool.scope()['exposed_paragraph_refs']
    with pytest.raises(ValueError,match='unique_read'):tool.read([exposed[0],exposed[0]])
    with pytest.raises(ValueError,match='unexposed'):tool.read(['P999'])
    assert tool.scope()['read_paragraph_refs']==[]


def test_same_body_two_sources_keep_original_titles_and_aliases():
    case=frozen(('Amber完全相同原文。','Amber完全相同原文。'));tool=tools(case)
    out=tool.search('Amber',limit=2);assert [r['paragraph_ref'] for r in out['results']]==['P1','P2']
    read=tool.read(['P1','P2'])['paragraphs']
    assert len(read)==2 and [r['doc_id'] for r in read]==['d1','d2']
    assert read[0]['title']!=read[1]['title']


def test_empty_query_zero_results_and_trace_are_real_no_model_no_fallback():
    tool=tools(frozen(('霜雪冰川。',),question='Amber Nebula?'))
    empty=tool.search('   ');assert empty['result_count']==0 and empty['status']=='empty_query_no_search'
    zero=tool.search('Quasar');assert zero['result_count']==0 and zero['no_match_proves_irrelevance'] is False
    assert tool.scope()['exposed_paragraph_refs']==[]
    assert [e['status'] for e in tool.trace()]==['empty_query_no_search','completed']
    assert all(e['local_only'] and e['model_calls']==0 for e in tool.trace())


def test_long_bound_original_question_not_truncated_but_other_queries_capped(monkeypatch):
    q='Amber完整原题条件。'*25;tool=tools(frozen(('Amber材料。',),question=q));seen=[];real=module.recall_question_candidates
    def capture(question,parents,**kwargs):seen.append((question,kwargs));return real(question,parents,**kwargs)
    monkeypatch.setattr(module,'recall_question_candidates',capture)
    tool.search(q);assert seen[0][0]==q and seen[0][1]['query_groups'] is None
    with pytest.raises(ValueError,match='invalid_search'):tool.search('其它长查询'*40)


@pytest.mark.parametrize('limit',[0,11,True,1.5])
def test_invalid_search_limit_rejected(limit):
    with pytest.raises(ValueError):tools().search('Amber',limit=limit)


def test_call_caps_and_returned_values_cannot_mutate_frozen_scope():
    case=frozen();tool=tools(case,search=1,read=1);case['catalog'][0]['text']='外部修改'
    search=tool.search('Amber',limit=2);search['results'].clear();tool.read(['P1'])
    with pytest.raises(ValueError,match='call_limit'):tool.search('Amber')
    with pytest.raises(ValueError,match='call_limit'):tool.read(['P2'])
    scope=tool.scope();scope['read_paragraph_refs'].clear();events=tool.trace();events[0]['status']='假的完成'
    assert tool.scope()['read_paragraph_refs']==['P1'] and tool.trace()[0]['status']=='completed'
    assert tool.scope()['search_calls']==1 and tool.scope()['read_calls']==1


@pytest.mark.parametrize('damage',['hash','identity','alias','length','textsha','jsondigest','metadata_extra','question','model_flag'])
def test_constructor_rejects_stale_or_inconsistent_dictionary_bindings(damage):
    case=frozen()
    if damage=='hash':case['catalog'][0]['text']='改文'
    elif damage=='identity':case['catalog'][0]['doc_id']='fake';rebound(case)
    elif damage=='alias':case['catalog'][0]['paragraph_ref']='P8';rebound(case)
    elif damage=='length':case['source_metadata']['P1']['document_end_cp']+=1;rebound(case)
    elif damage=='textsha':case['source_metadata']['P1']['source_text_sha256_utf8']='0'*64;rebound(case)
    elif damage=='jsondigest':case['source_metadata']['P1']['v19_text_json_digest']='0'*64;rebound(case)
    elif damage=='metadata_extra':case['source_metadata']['P999']=deepcopy(case['source_metadata']['P1']);rebound(case)
    elif damage=='question':case['question']['question']='另一个原题'
    elif damage=='model_flag':case['catalog'][0]['paragraph_complete']=False;rebound(case)
    with pytest.raises(ValueError):tools(case)


def test_failed_local_search_is_traced_as_failed_not_completed(monkeypatch):
    tool=tools()
    def broken(*args,**kwargs):raise RuntimeError('SYNTHETIC_FAILURE')
    monkeypatch.setattr(module,'recall_question_candidates',broken)
    with pytest.raises(RuntimeError):tool.search('Amber')
    assert tool.trace()[-1]['status']=='failed' and tool.trace()[-1]['code']=='local_search_failed'
    assert tool.scope()['exposed_paragraph_refs']==[]
