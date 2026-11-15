"""Candidate reuse is a new, bound quality job; never a new source scan.

Explicit synthetic SQLite fixtures only. No model, original samples or gold.
"""
from copy import deepcopy
import hashlib
import json
import sqlite3
from uuid import uuid4

import pytest

from app.domain.coverage import plan_source_batches
from app.domain.refine_input import (RefineInputInvalid, create_candidate_artifact,
    validate_candidate_artifact, digest)
from app.domain.snapshot import build_snapshot
from app.errors import ServiceError
from app.utils import loads
from test_repository import setup_repo, create, output


HISTORY = [{'operation': 'extract', 'input_tokens': 12, 'output_tokens': 8,
            'total_tokens': 20, 'usage_source': 'provider', 'cost': None}]
OVERLAYS = {'excluded': [], 'manual_evidence': [], 'interpretation_overrides': {}}


def complete_case(tmp_path, *, missing_ledger=False, incomplete_coverage=False):
    repo, topic_id = setup_repo(tmp_path, question_count=2)
    q = repo.questions(topic_id)[0]
    run = create(repo, topic_id, selected=[q['id']])
    assert repo.claim()['id'] == run['id']
    result = output(repo, topic_id, q, run['id'])
    for e in result['evidence']:
        e['interpretation_origin'] = 'model'
    if not missing_ledger:
        result['candidate_evidence'] = deepcopy(result['evidence'])
    paragraphs = repo.paragraphs(topic_id)
    config = repo.config()
    plan = plan_source_batches(q['question'], paragraphs,
        max_codepoints=config['source_batch_codepoints'], max_catalog_spans=config['source_batch_spans'])
    result['source_coverage'] = {**plan['coverage'], 'strategy': 'full_source_scan',
        'total_paragraphs': len(paragraphs), 'total_codepoints': sum(len(p['text']) for p in paragraphs),
        'complete': not incomplete_coverage,
        'completed_batches': len(plan['batches']), 'scanned_paragraphs': len(paragraphs),
        'scanned_codepoints': sum(len(p['text']) for p in paragraphs)}
    result['usage'] = deepcopy(HISTORY)
    revision = repo.revision(topic_id, run['target_revision_id'])
    snapshot = build_snapshot(repo.topic(topic_id), [q], {q['id']: result},
        revision_id=revision['id'], graph_revision=revision['revision_no'], run_id=run['id'])
    snapshot['scope'] = {'completed_question_ids': [q['id']], 'all_question_count': 2, 'is_complete_topic': False}
    snapshot['overlays'] = deepcopy(OVERLAYS)
    repo.publish(run['id'], snapshot, {q['id']: result})
    return repo, topic_id, q, repo.run(run['id']), snapshot, result, paragraphs


def artifact_fixture(tmp_path):
    repo, tid, q, run, snapshot, result, paragraphs = complete_case(tmp_path)
    artifact = create_candidate_artifact(topic_id=tid, source_revision_id=run['target_revision_id'],
        source_artifact_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
        source_manifest_sha256=repo.topic(tid)['source_manifest_sha256'], question=q,
        paragraphs=paragraphs, result=result, source_config=repo.config(), overlays=OVERLAYS, created_at='unit-fixture')
    args = {'topic_id': tid, 'question': q, 'paragraphs': paragraphs,
            'source_manifest_sha256': repo.topic(tid)['source_manifest_sha256'], 'overlays': OVERLAYS}
    return artifact, args


def resign(artifact):
    artifact['artifact_sha256'] = digest({k: v for k, v in artifact.items() if k != 'artifact_sha256'})


def test_artifact_is_deeply_copied_and_bound_to_complete_candidates(tmp_path):
    artifact, args = artifact_fixture(tmp_path)
    out = validate_candidate_artifact(artifact, **args)
    assert out == artifact
    out['candidate_evidence'][0]['fact_des'] = 'changed clone only'
    assert artifact['candidate_evidence'][0]['fact_des'] != out['candidate_evidence'][0]['fact_des']
    assert artifact['source_usage'] == HISTORY
    assert artifact['source_coverage']['complete'] is True


@pytest.mark.parametrize('key,value,check', [
    ('topic_id', 'foreign-topic', 'refine_question_binding'),
    ('source_manifest_sha256', 'f' * 64, 'refine_source_manifest'),
    ('overlays', {'excluded': [], 'manual_evidence': [], 'interpretation_overrides': {'different': {'fact_des': 'manual'}}}, 'refine_overlay_binding'),
])
def test_external_binding_mismatches_rejected(tmp_path, key, value, check):
    artifact, args = artifact_fixture(tmp_path)
    with pytest.raises(RefineInputInvalid) as error:
        validate_candidate_artifact(artifact, **{**args, key: value})
    assert error.value.check == check


@pytest.mark.parametrize('field,value', [('id', 'foreign-question'), ('raw_question_id', 'q_999'), ('question', '另一个问题')])
def test_question_internal_raw_id_and_text_are_bound(tmp_path, field, value):
    artifact, args = artifact_fixture(tmp_path)
    with pytest.raises(RefineInputInvalid, match='refine_question_binding'):
        validate_candidate_artifact(artifact, **{**args, 'question': {**args['question'], field: value}})


def test_raw_paragraph_metadata_and_text_changes_are_bound(tmp_path):
    artifact, args = artifact_fixture(tmp_path)
    for key, value in [('text', '原文改动'), ('raw_para_id', 'wrong-para'), ('content_url', 'https://other.invalid/')]:
        paragraphs = deepcopy(args['paragraphs'])
        paragraphs[0][key] = value
        with pytest.raises(RefineInputInvalid, match='refine_source_manifest'):
            validate_candidate_artifact(artifact, **{**args, 'paragraphs': paragraphs})


def test_arbitrary_candidate_mutation_rejected_even_before_semantic_review(tmp_path):
    artifact, args = artifact_fixture(tmp_path)
    artifact['candidate_evidence'][0]['fact_des'] = 'tampered annotation'
    with pytest.raises(RefineInputInvalid, match='refine_artifact_hash'):
        validate_candidate_artifact(artifact, **args)


@pytest.mark.parametrize('coverage_change', [{'complete': False}, {'scanned_paragraphs': 0}, {'completed_batches': 0}, {'plan_fingerprint': 'f' * 64}, {'source_fingerprint': 'f' * 64}])
def test_resigned_incomplete_or_wrong_coverage_still_rejected(tmp_path, coverage_change):
    artifact, args = artifact_fixture(tmp_path)
    artifact['source_coverage'].update(coverage_change)
    resign(artifact)
    with pytest.raises(RefineInputInvalid, match='refine_incomplete_coverage'):
        validate_candidate_artifact(artifact, **args)


def test_resigned_wrong_source_quote_rejected(tmp_path):
    artifact, args = artifact_fixture(tmp_path)
    artifact['candidate_evidence'][0]['quote'] = '失联9人'
    artifact['candidate_sha256'] = digest(artifact['candidate_evidence'])
    resign(artifact)
    with pytest.raises(RefineInputInvalid, match='refine_candidate_source'):
        validate_candidate_artifact(artifact, **args)


@pytest.mark.parametrize('missing,incomplete,code', [(True, False, 'refine_missing_candidate_ledger'), (False, True, 'refine_incomplete_coverage')])
def test_repository_rejects_historical_ready_without_full_input(tmp_path, missing, incomplete, code):
    repo, tid, q, source, *_ = complete_case(tmp_path, missing_ledger=missing, incomplete_coverage=incomplete)
    before = repo.all('SELECT id FROM runs')
    with pytest.raises(ServiceError) as error:
        repo.create_run(tid, {'base_revision_id': source['target_revision_id'], 'question_ids': [q['id']],
            'mode': 'quality-refine', 'source_revision_id': source['target_revision_id']}, 'bad-source', model_available=True)
    assert error.value.code == code
    assert repo.all('SELECT id FROM runs') == before


def test_new_review_can_use_old_ready_version_but_never_inherits_its_costs(tmp_path, monkeypatch):
    repo, tid, q, old, snapshot, *_ = complete_case(tmp_path)
    old_payload = repo.one('SELECT payload_json FROM revision_artifacts WHERE revision_id=?', (old['target_revision_id'],))['payload_json']
    old_config = repo.config()
    monkeypatch.setattr(repo, 'config', lambda: {**old_config, 'prompt_version': 'unit-explicit-next-quality-version', 'prompt_sha256': 'b' * 64})
    created = repo.create_run(tid, {'base_revision_id': old['target_revision_id'], 'question_ids': [q['id']],
        'mode': 'quality-refine', 'source_revision_id': old['target_revision_id']}, 'new-review', model_available=True)
    assert created['id'] != old['id'] and created['target_revision_id'] != old['target_revision_id']
    assert created['mode'] == 'quality-refine' and created['usage'] is None
    artifact = repo.refine_input(created['id'], q['id'])
    assert artifact['source_run_id'] == old['id'] and artifact['source_revision_id'] == old['target_revision_id']
    assert artifact['source_usage'] == HISTORY
    assert artifact['source_config']['prompt_version'] == old_config['prompt_version']
    assert created['derived_inputs'][0]['historical_source_usage'] == HISTORY
    new_usage = [{'operation': 'verify_evidence_scope', 'input_tokens': 3, 'output_tokens': 4, 'total_tokens': 7, 'usage_source': 'provider'}]
    repo.record_usage(created['id'], q['id'], new_usage)
    assert repo.run(created['id'])['usage'][q['id']][0]['total_tokens'] == 7
    assert repo.run(old['id'])['usage'][q['id']][0]['total_tokens'] == 20
    assert repo.one('SELECT payload_json FROM revision_artifacts WHERE revision_id=?', (old['target_revision_id'],))['payload_json'] == old_payload
    attached = next(e for e in repo.events(created['id']) if e['type'] == 'candidate_artifact_attached')
    assert attached['data']['new_source_scan_performed'] is False
    assert attached['data']['kind'] == 'new_quality_refine'
    with repo.connection() as db:
        for sql in ('UPDATE run_refine_inputs SET payload_json=\'{}\' WHERE run_id=?', 'DELETE FROM run_refine_inputs WHERE run_id=?'):
            with pytest.raises(sqlite3.IntegrityError, match='immutable_refine_input'):
                db.execute(sql, (created['id'],))


def test_refine_cannot_take_another_question_or_changed_operator_overlay(tmp_path):
    repo, tid, q, old, *_ = complete_case(tmp_path)
    other = repo.questions(tid)[1]
    body = {'base_revision_id': old['target_revision_id'], 'question_ids': [other['id']],
            'mode': 'quality-refine', 'source_revision_id': old['target_revision_id']}
    with pytest.raises(ServiceError) as error:
        repo.create_run(tid, body, 'wrong-question', model_available=True)
    assert error.value.code == 'refine_missing_candidate_ledger'
    body['question_ids'] = [q['id']]
    with pytest.raises(ServiceError) as error:
        repo.create_run(tid, body, 'changed-overlay', model_available=True, options={'overlays': {**OVERLAYS, 'excluded': [['new', 0, 1, 'x']]}})
    assert error.value.code == 'refine_overlay_binding'


def test_question_and_original_source_tables_remain_immutable(tmp_path):
    repo, tid, *_ = complete_case(tmp_path)
    with repo.connection() as db:
        for table, field in [('questions', 'question'), ('documents', 'content'), ('paragraphs', 'text')]:
            with pytest.raises(sqlite3.IntegrityError, match='immutable_'):
                db.execute(f'UPDATE {table} SET {field}=?', ('forbidden external alteration',))
