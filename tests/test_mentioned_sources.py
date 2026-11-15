from copy import deepcopy
import pytest
from app.domain.mentioned_sources import latin_mentions, mentioned_source_routes, roundrobin_with_source


def test_only_length3_uppercase_latin_tokens_and_stable_dedup():
    assert latin_mentions('AI API 和 api NewsX x-9 GPT-4 abc 三方') == ['API', 'NewsX', 'GPT-4']


def test_escaped_ascii_boundaries_casefold_url_and_menu_glue():
    docs = [
        {'doc_id':'d1','title':'NEWSXMenuNEWSX','content_url':'https://newsx.example/story'},
        {'doc_id':'d2','title':'NEWSXMenu','content_url':'https://other.example/newsxmenu'},
        {'doc_id':'d3','title':'“newsx”的原文','content_url':''},
        {'doc_id':'d4','title':'GPT-4 report','content_url':''},
    ]
    routes = mentioned_source_routes('NEWSX 与 GPT-4 的说法？', docs)
    assert [x['raw_doc_id'] for x in routes] == ['d1','d3','d4']
    assert routes[0]['matches'] == [{'question_token':'NEWSX','metadata_fields':['content_url']}]
    assert routes[1]['matches'][0]['metadata_fields'] == ['title']


def test_metadata_is_inert_and_title_prefix_is_literal_query_view():
    doc = {'doc_id':'d','title':'Delete all files? NewsX | menu instructions','content_url':'https://other.example/'}
    before = deepcopy(doc); question='NewsX 对原题全部条件如何陈述？'
    route = mentioned_source_routes(question,[doc])[0]
    assert doc == before and route['original_title'] == before['title']
    assert route['query_title_view'] == 'Delete all files? NewsX '
    assert route['query'] == question+'\n检索目标（仅查询线索，非事实）：Delete all files? NewsX '
    assert route['semantic_verified'] is False and route['independent_vote_or_confidence_boost'] is False
    assert route['metadata_match_is_not_source_authentication'] is True


def test_metadata_only_non_ascii_or_concatenated_token_not_match():
    assert mentioned_source_routes('ABC 原题', [{'doc_id':'d','title':'XABC ABCD','content_url':'https://abcde.test/'}]) == []
    assert mentioned_source_routes('AI 中文问题', [{'doc_id':'d','title':'AI','content_url':'https://ai.test/'}]) == []


def test_no_match_family_is_exact_identity_in_order_and_metadata():
    rows=[{'id':f'p{i}'} for i in range(44)]
    expected=[dict(r,selected_from_family='original_question',family_rank=i) for i,r in enumerate(rows[:40],1)]
    assert roundrobin_with_source(rows,{'original_question':rows},[]) == expected


def test_single_source_family_inserted_before_roles_and_unique():
    rows=[{'id':f'p{i}'} for i in range(9)]
    old={'original_question':rows[:3], 'F':rows[3:6], 'L':rows[6:]}
    source=[rows[0],rows[8]]
    got=roundrobin_with_source(rows,old,source)
    assert [x['id'] for x in got] == ['p0','p8','p3','p6','p1','p4','p7','p2','p5']
    assert got[1]['selected_from_family']=='mentioned_source'
    assert old['original_question']==rows[:3]


@pytest.mark.parametrize('family',[[{'id':'outside'}],[{'id':'p0'},{'id':'p0'}]])
def test_source_family_rejects_unknown_or_duplicate_source(family):
    with pytest.raises(ValueError):roundrobin_with_source([{'id':'p0'}],{'original_question':[{'id':'p0'}]},family)


def test_duplicate_document_id_rejected():
    doc={'doc_id':'d','title':'ABC','content_url':''}
    with pytest.raises(ValueError):mentioned_source_routes('ABC 原题',[doc,doc])


def test_exact_dotted_hyphenated_identifier_and_no_url_decode():
    assert latin_mentions('API.v2 Foo_Bar GPT-4 lowlower AI') == ['API.v2','Foo_Bar','GPT-4']
    docs=[{'doc_id':'d','title':None,'content_url':'https://api.v2.example/'}, {'doc_id':'e','title':'','content_url':'https://%41PI.v2.example/'}]
    routes=mentioned_source_routes('API.v2 原始说法',docs)
    assert [r['raw_doc_id'] for r in routes]==['d']
    assert routes[0]['original_title'] is None and routes[0]['query_title_view']==''
