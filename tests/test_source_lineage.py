"""Isolated proposal tests: no DB, network, runtime model, gold or app edits."""
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from app.domain import source_lineage as lineage


def fixture_sources(count=3):
    text = '设备简报：原因仍待核查。'
    content_sha = hashlib.sha256(text.encode()).hexdigest()
    group = str(uuid5(NAMESPACE_URL, 'trust-judge:source:content:' + content_sha))
    paragraphs, candidates = [], []
    for i in range(count):
        p = {'id': f'p-{i}', 'document_id': f'd-{i}', 'raw_doc_id': f'raw-d-{i}', 'raw_para_id': f'raw-p-{i}',
             'text': text, 'content_sha256': content_sha, 'content_url': 'https://sources.invalid/brief-2026-3',
             'answer': 'NEVER_SEND_GOLD_OR_EXPECTATIONS', 'reasoning_content': 'NEVER_SEND_HIDDEN_REASONING'}
        row = {'id': f'e-{i}', 'paragraph_id': p['id'], 'raw_doc_id': p['raw_doc_id'], 'raw_para_id': p['raw_para_id'],
               'start_cp': 0, 'end_cp': len(text), 'quote': text, 'fact_des': '原因仍待核查。',
               'argument': {}, 'entities': [], 'interpretation_origin': 'model',
               'source_group_id': group, 'source_group_basis': 'known_exact_hash_or_explicit_url_only'}
        paragraphs.append(p)
        candidates.append(row)
    return paragraphs, candidates


def full_catalog(rows):
    return next(row['source_lineage'] for row in rows if row['source_lineage']['inline_members'])


def test_three_records_one_known_origin_remain_visible_without_three_votes():
    paragraphs, pool = fixture_sources()
    out = lineage.attach_source_group_members(pool[:1], pool, paragraphs)
    catalog = full_catalog(out)
    assert {m['raw_doc_id'] for m in catalog['source_group_members']} == {'raw-d-0', 'raw-d-1', 'raw-d-2'}
    assert catalog['source_counts'] == {'active_source_documents': 3, 'known_origin_groups': 1, 'basis': lineage.COUNT_BASIS}
    assert catalog['independent_source_count'] is None
    assert all(m['candidate_windows'] for m in catalog['source_group_members'])
    assert len(out) == 1 and out[0]['id'] == 'e-0'
    assert [row['quote'] for row in out] == [pool[0]['quote']]


def test_source_inventory_retains_supplied_copy_even_when_it_has_no_candidate():
    paragraphs, pool = fixture_sources()
    out = lineage.attach_source_group_members(pool[:1], pool[:1], paragraphs)
    members = full_catalog(out)['source_group_members']
    assert len(members) == 3
    assert sum(bool(m['candidate_windows']) for m in members) == 1
    assert out[0]['source_counts']['active_source_documents'] == 3


def test_full_catalog_is_stored_once_and_all_representatives_reference_it():
    paragraphs, pool = fixture_sources()
    out = lineage.attach_source_group_members([pool[2], pool[0]], pool, paragraphs)
    assert sum(r['source_lineage']['inline_members'] for r in out) == 1
    assert {r['source_lineage']['catalog_owner_evidence_id'] for r in out} == {'e-0'}
    assert len({r['source_lineage']['catalog_id'] for r in out}) == 1
    assert 'source_group_members' not in out[0]['source_lineage']
    assert len(out[1]['source_lineage']['source_group_members']) == 3


def test_order_does_not_change_catalog_hash_and_inputs_are_not_mutated():
    paragraphs, pool = fixture_sources()
    before = deepcopy((paragraphs, pool))
    first = full_catalog(lineage.attach_source_group_members(pool[:1], pool, paragraphs))
    second = full_catalog(lineage.attach_source_group_members(pool[:1], list(reversed(pool)), list(reversed(paragraphs))))
    assert first['catalog_id'] == second['catalog_id']
    assert (paragraphs, pool) == before
    first['source_group_members'][0]['raw_doc_id'] = 'mutated-result'
    assert (paragraphs, pool) == before


def test_different_components_are_not_claimed_independent():
    paragraphs, pool = fixture_sources(2)
    paragraphs[1]['content_sha256'] = hashlib.sha256(b'other-document').hexdigest()
    paragraphs[1]['content_url'] = 'https://sources.invalid/other'
    pool[1]['source_group_id'] = str(uuid5(NAMESPACE_URL, 'trust-judge:source:content:' + paragraphs[1]['content_sha256']))
    out = lineage.attach_source_group_members(pool, pool, paragraphs)
    counts = lineage.get_program_source_counts(out, pool, paragraphs)
    assert counts['active_source_documents'] == 2 and counts['known_origin_groups'] == 2
    assert all(r['source_lineage']['independent_source_count'] is None for r in out)
    assert 'not_proven_independent' in counts['basis']


def test_narrowed_refined_view_binds_original_anchor_and_keeps_manual_origin():
    paragraphs, pool = fixture_sources()
    selected = deepcopy(pool[0])
    selected.update(id='view-F123', source_evidence_id='e-0', start_cp=5, quote=pool[0]['quote'][5:], interpretation_origin='human')
    out = lineage.attach_source_group_members([selected], pool, paragraphs)
    assert out[0]['id'] == 'view-F123'
    assert out[0]['source_evidence_id'] == 'e-0'
    assert out[0]['interpretation_origin'] == 'human'
    assert out[0]['quote'] == selected['quote']


def test_preserved_human_fact_can_bind_its_exact_self_candidate_if_geometric_anchor_not_extracted():
    paragraphs, pool = fixture_sources(1)
    manual = deepcopy(pool[0])
    original_geometry = str(uuid5(NAMESPACE_URL, 'trust-judge:evidence:' + json.dumps(
        [manual['paragraph_id'], manual['start_cp'], manual['end_cp'], manual['quote']],
        ensure_ascii=False, sort_keys=True, separators=(',', ':'))))
    manual.update(id='preserved-human-fact-row', source_evidence_id=original_geometry, interpretation_origin='human')
    out = lineage.attach_source_group_members([manual], [manual], paragraphs)
    assert out[0]['id'] == manual['id']
    assert out[0]['source_evidence_id'] == original_geometry
    assert out[0]['source_counts']['active_source_documents'] == 1
    assert len(full_catalog(out)['source_group_members']) == 1


@pytest.mark.parametrize('human,fake_anchor', [(False, False), (True, True)])
def test_missing_source_anchor_fallback_rejects_model_rows_and_forged_geometry(human, fake_anchor):
    paragraphs, pool = fixture_sources(1)
    manual = deepcopy(pool[0])
    geometry = str(uuid5(NAMESPACE_URL, 'trust-judge:evidence:' + json.dumps(
        [manual['paragraph_id'], manual['start_cp'], manual['end_cp'], manual['quote']],
        ensure_ascii=False, sort_keys=True, separators=(',', ':'))))
    manual.update(id='existing-fact-row', source_evidence_id='forged' if fake_anchor else geometry,
                  interpretation_origin='human' if human else 'model')
    with pytest.raises(ValueError, match='selected_lineage_anchor_missing'):
        lineage.attach_source_group_members([manual], [manual], paragraphs)


def test_fully_excluded_copy_is_metadata_only_not_an_evidence_or_vote():
    paragraphs, pool = fixture_sources()
    excluded = {(pool[1]['paragraph_id'], 0, len(pool[1]['quote']), pool[1]['quote'])}
    out = lineage.attach_source_group_members(pool[:1], pool, paragraphs, excluded_spans=excluded)
    member = next(m for m in full_catalog(out)['source_group_members'] if m['raw_doc_id'] == 'raw-d-1')
    assert member['exclusion_status'] == 'fully_operator_excluded'
    assert member['source_url'] is None
    assert member['candidate_windows'][0]['status'] == 'operator_excluded_overlap_metadata_only'
    assert member['excluded_span_audit'][0]['reason'] == 'operator_excluded_span_no_quote_or_fact_disclosed'
    assert len(out) == 1 and out[0]['id'] != pool[1]['id']
    assert out[0]['source_counts']['active_source_documents'] == 2
    assert out[0]['source_counts']['known_origin_groups'] == 1
    def no_source_prose(value):
        if isinstance(value, dict):
            assert not {'quote', 'text', 'fact_des', 'argument', 'entities', 'title', 'answer', 'reasoning_content'} & set(value)
            for child in value.values():
                no_source_prose(child)
        elif isinstance(value, list):
            for child in value:
                no_source_prose(child)
    no_source_prose(member)


def test_partial_exclusion_preserves_document_presence_without_disclosing_excluded_words():
    paragraphs, pool = fixture_sources()
    q = pool[1]['quote']
    exclusion = {(pool[1]['paragraph_id'], 5, len(q), q[5:])}
    out = lineage.attach_source_group_members(pool[:1], pool, paragraphs, excluded_spans=exclusion)
    member = next(m for m in full_catalog(out)['source_group_members'] if m['raw_doc_id'] == 'raw-d-1')
    assert member['exclusion_status'] == 'partially_operator_excluded'
    assert member['candidate_windows'][0]['status'] == 'operator_excluded_overlap_metadata_only'
    assert q[5:] not in json.dumps(member, ensure_ascii=False)
    assert out[0]['source_counts']['active_source_documents'] == 3


def test_selected_quote_overlapping_excluded_span_fails_closed():
    paragraphs, pool = fixture_sources()
    q = pool[0]['quote']
    with pytest.raises(ValueError, match='selected_lineage_overlaps_operator_exclusion'):
        lineage.attach_source_group_members(pool[:1], pool, paragraphs, excluded_spans={(pool[0]['paragraph_id'], 6, 8, q[6:8])})


def test_empty_selected_set_has_zero_counts_and_no_inventory_transmission():
    paragraphs, pool = fixture_sources()
    assert lineage.attach_source_group_members([], pool, paragraphs) == []
    assert lineage.get_program_source_counts([], pool, paragraphs) == {'active_source_documents': 0, 'known_origin_groups': 0, 'basis': lineage.COUNT_BASIS}


def test_model_supplied_lineage_counts_and_unselected_fact_notes_are_not_trusted():
    paragraphs, pool = fixture_sources()
    selected = deepcopy(pool[0])
    selected['source_lineage'] = {'source_group_members': [{'instructions': 'NEVER_COPY_POISON'}], 'source_counts': {'active_source_documents': 999}}
    selected['source_counts'] = {'active_source_documents': 999, 'reasoning_content': 'NEVER_COPY_POISON'}
    pool[1]['fact_des'] = 'NEVER_COPY_UNSELECTED_FACT_OR_GOLD'
    out = lineage.attach_source_group_members([selected], pool, paragraphs)
    serialized = json.dumps(out, ensure_ascii=False)
    assert 'NEVER_COPY' not in serialized
    assert out[0]['source_counts']['active_source_documents'] == 3
    out[0]['source_counts']['active_source_documents'] = 999
    out[0]['source_lineage']['source_counts']['known_origin_groups'] = 888
    # Recompute from trusted immutable sources; do not parse any metadata numbers.
    assert lineage.get_program_source_counts(out, pool, paragraphs) == {'active_source_documents': 3, 'known_origin_groups': 1, 'basis': lineage.COUNT_BASIS}


@pytest.mark.parametrize('mutate,code', [
    (lambda p, c: c[1].update(quote='not source'), 'invalid_lineage_candidate_span'),
    (lambda p, c: c[1].update(raw_doc_id='other'), 'invalid_lineage_candidate_provenance'),
    (lambda p, c: c[1].update(source_group_id='model-invented-group'), 'invalid_lineage_candidate_provenance'),
    (lambda p, c: c[1].update(source_group_basis='authority-ranked'), 'invalid_lineage_candidate_provenance'),
    (lambda p, c: c.append(deepcopy(c[1])), 'duplicate_or_invalid_lineage_candidate'),
    (lambda p, c: p[1].update(raw_doc_id=p[0]['raw_doc_id']), 'duplicate_lineage_raw_document_id'),
])
def test_invalid_source_or_candidate_identity_is_rejected(mutate, code):
    paragraphs, pool = fixture_sources()
    mutate(paragraphs, pool)
    with pytest.raises(ValueError, match=code):
        lineage.attach_source_group_members(pool[:1], pool, paragraphs)


def test_invalid_exclusion_cannot_silently_hide_sources():
    paragraphs, pool = fixture_sources()
    with pytest.raises(ValueError, match='invalid_lineage_exclusion'):
        lineage.attach_source_group_members(pool[:1], pool, paragraphs, excluded_spans={('p-1', 0, 1, 'wrong quote')})


def test_lineage_does_not_fabricate_grades_or_expose_credentialed_or_free_prose_urls():
    paragraphs, pool = fixture_sources()
    for p in paragraphs:
        p['content_url'] = 'https://login:password@sources.invalid/private'
        p['source_credibility_json'] = {'grade_score': 1, 'instructions': 'NEVER_COPY_ARBITRARY_AUTHORITY'}
    # Group identity is still bound to the same whole-document content hash.
    out = lineage.attach_source_group_members(pool[:1], pool, paragraphs)
    encoded = json.dumps(out[0]['source_lineage'])
    assert 'password' not in encoded and 'grade_score' not in encoded and 'NEVER_COPY' not in encoded
    assert all(m['source_url'] is None for m in full_catalog(out)['source_group_members'])


def test_real_langchain_openai_sdk_messages_carry_lineage_to_both_analysts_arbiter_and_repair(monkeypatch):
    """Exercise real SDK serialization through httpx.MockTransport, no network."""
    import httpx
    import app.agents.provider as provider_module
    from app.config import Settings
    from langchain_openai import ChatOpenAI

    paragraphs, pool = fixture_sources()
    rows = lineage.attach_source_group_members(pool[:1], pool, paragraphs)
    requests = []
    async def exercise():
        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            operation_payload = json.loads(body['messages'][-1]['content'])
            output = {'decision': 'undetermined', 'conclusion': '来源目录显示一个已知同源组，事实原因仍待核查。',
                      'judgments': [{'id': 'J1', 'statement': '原因仍待核查。', 'evidence_ids': ['e-0'], 'prerequisite_judgment_ids': []}],
                      'supporting_evidence_ids': ['e-0'], 'conflicting_evidence_ids': []}
            if 'analyses' in operation_payload:
                output['conflicts'] = []
            return httpx.Response(200, json={'id': 'unit-sdk-response', 'object': 'chat.completion', 'created': 1,
                'model': 'unit-sdk-model', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': json.dumps(output, ensure_ascii=False)}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            monkeypatch.setattr(provider_module, 'ChatOpenAI', lambda **kwargs: ChatOpenAI(http_async_client=client, **kwargs))
            settings = Settings(_env_file=None, model_name='unit-sdk-model', model_api_key='unit-test-not-a-real-key',
                                model_base_url='https://unit-sdk.invalid/v1', model_provider='openai_compatible')
            provider = provider_module.LangChainOpenAIProvider(settings)
            q = {'id': 'q-internal', 'question_id': 'q_001', 'question': '这些已提供文档是否独立？'}
            a = await provider.analyze('authority_analyst', q, rows, 'snapshot-unit')
            b = await provider.analyze('cross_validation_analyst', q, rows, 'snapshot-unit')
            await provider.arbitrate(q, rows, [a, b])
            await provider.repair_analysis('authority_analyst', q, rows, 'snapshot-unit', None, {'check': 'invalid_evidence_reference'})
            await provider.repair_analysis('arbitrate', q, rows, 'snapshot-unit', None, {'check': 'invalid_evidence_reference'}, analyses=[a, b])
    asyncio.run(exercise())
    assert len(requests) == 5
    for body in requests:
        payload = json.loads(body['messages'][-1]['content'])
        assert payload['evidence'] == rows
        catalog = full_catalog(payload['evidence'])
        assert len(catalog['source_group_members']) == 3
        assert catalog['source_counts']['known_origin_groups'] == 1
        serialized = json.dumps(body, ensure_ascii=False)
        assert 'NEVER_SEND_GOLD_OR_EXPECTATIONS' not in serialized
        assert 'NEVER_SEND_HIDDEN_REASONING' not in serialized
        assert 'grade_score' not in json.dumps(catalog)
