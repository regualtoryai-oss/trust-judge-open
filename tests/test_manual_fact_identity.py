"""Manual per-fact identity must survive a shared immutable quote.

These are explicit synthetic data-flow tests. The injected provider performs
no model/HTTP calls. Its fixed scope decisions only exercise the production
strict identity path; this is not semantic-quality evaluation.
"""
from copy import deepcopy
import hashlib
import json
from uuid import NAMESPACE_URL, uuid5

import pytest

from app.domain.strict_verification import build_fact_views, build_scope_request, resolve_scope_verification
from app.workflows.analysis import run_analysis


TEXT = '甲设备停机3台，乙设备停机4台。'
A = '甲设备停机3台。'
B = '乙设备停机4台。'


def uid(value):
    return str(uuid5(NAMESPACE_URL, 'trust-judge:' + value))


def annotation(statement):
    return {'paragraph_id': 'manual-p', 'start_cp': 0, 'end_cp': len(TEXT),
            'quote': TEXT, 'fact_des': statement, 'argument': {}, 'entities': []}


def manual_fact(statement):
    anchor = uid('evidence:' + json.dumps(['manual-p', 0, len(TEXT), TEXT],
        ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    canonical = {'id': anchor, 'raw_doc_id': 'raw-manual-doc', 'raw_para_id': 'raw-manual-p1', **annotation(statement)}
    view = build_fact_views([canonical])[0]
    return {**canonical, 'id': uid('verified-fact:' + view['id']),
            'source_evidence_id': anchor, 'fact_view_id': view['id']}


class NoModelFixtureProvider:
    def __init__(self, candidates):
        self.candidates = candidates

    async def extract(self, question, paragraphs):
        return deepcopy(self.candidates)

    async def analyze(self, role, question, evidence, snapshot_id):
        return {'decision': 'undetermined', 'conclusion': '此合成测试只检查数据流。',
                'judgments': [], 'supporting_evidence_ids': [], 'conflicting_evidence_ids': []}

    async def arbitrate(self, question, evidence, analyses):
        return {**await self.analyze('unused', question, evidence, 'unused'), 'conflicts': []}

    async def verify_evidence_scope(self, question, evidence):
        payload, _, views = build_scope_request(question, evidence)
        decisions = {f'C{i + 1}': {'fact_entailment': 'entailed', 'target_relation': 'direct_answer',
            'target_question_text': question['question'], 'qualifier_dimension': None,
            'requires_unstated_bridge': False, 'target_fact_present_but_annotation_incomplete': False,
            'reason': '显式合成固定决定，只检查身份与数据流。'} for i in range(len(views))}
        return resolve_scope_verification({'candidate_catalog_id': payload['candidate_catalog_id'], 'decisions': decisions}, question, evidence)


async def execute(tmp_path, automatic, manual, overrides=None):
    paragraphs = [{'id': 'manual-p', 'document_id': 'manual-d', 'raw_doc_id': 'raw-manual-doc',
        'raw_para_id': 'raw-manual-p1', 'text': TEXT,
        'content_sha256': hashlib.sha256(TEXT.encode()).hexdigest(), 'content_url': None}]
    return await run_analysis({'id': 'manual-q', 'raw_question_id': 'q_001', 'question': '甲乙分别有多少设备停机？'},
        paragraphs, NoModelFixtureProvider(automatic), checkpoint_path=tmp_path / 'checkpoints.sqlite3',
        thread_id='explicit-manual-fact-identity-test', cancel_check=lambda: False, emit=lambda *_: None,
        manual_evidence=manual, interpretation_overrides=overrides or {},
        strict_scope_enabled=True)


@pytest.mark.asyncio
async def test_materialized_fact_uuid_interpretation_overlay_resolves_without_changing_quote(tmp_path):
    fact = manual_fact(A)
    revised = '甲设备的通报停机数是3台。'
    result = await execute(tmp_path, [annotation(A)], [fact], {fact['id']: {'fact_des': revised}})
    matching = [row for row in result['evidence'] if row['id'] == fact['id']]
    assert len(matching) == 1
    assert matching[0]['fact_des'] == revised
    assert matching[0]['quote'] == TEXT
    assert matching[0]['interpretation_origin'] == 'human'


@pytest.mark.asyncio
async def test_editing_one_fact_on_shared_quote_does_not_erase_other_automatic_fact(tmp_path):
    fact_a = manual_fact(A)
    revised = '甲设备的通报停机数是3台。'
    result = await execute(tmp_path, [annotation(A), annotation(B)], [fact_a], {fact_a['id']: {'fact_des': revised}})
    statements = {row['fact_des'] for row in result['evidence']}
    statements.update(view['fact_des'] for row in result['evidence'] for view in row.get('additional_interpretations', []))
    assert revised in statements
    assert B in statements
    assert all(row['quote'] == TEXT for row in result['evidence'])


@pytest.mark.asyncio
async def test_two_manual_facts_on_same_quote_are_not_last_write_wins(tmp_path):
    a, b = manual_fact(A), manual_fact(B)
    assert a['id'] != b['id'] and a['source_evidence_id'] == b['source_evidence_id']
    result = await execute(tmp_path, [], [a, b])
    human = [row for row in result['evidence'] if row['interpretation_origin'] == 'human']
    assert {row['id'] for row in human} == {a['id'], b['id']}
    assert {row['fact_des'] for row in human} == {A, B}
    assert len(human) == 2
    assert all(row['quote'] == TEXT for row in human)
