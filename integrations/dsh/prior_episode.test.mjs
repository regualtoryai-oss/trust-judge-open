import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { verifyPriorEpisode } from './prior_episode.mjs';
const sha = value => createHash('sha256').update(value).digest('hex');
async function fixture(mutation = {}) {
  const directory = await mkdtemp(resolve(tmpdir(), 'dsh-prior-'));
  const save = async (name, value) => { const file = resolve(directory, name), bytes = JSON.stringify(value); await writeFile(file, bytes); return { file, sha256: sha(bytes) }; };
  const rawCase = { question: { question: '程序是否完成？', question_id: 'q_synthetic' }, topic_title: '合成专题' };
  const raw = await save('raw.json', rawCase);
  const manifest = await save('manifest.json', { casePath: raw.file, priorKnownTokens: 0, episodeId: 'synthetic' });
  const rows = [{ episode_id: 'synthetic', sequence: 1, status: 'request_started', request_sequence: 1, wire_sha256: 'a' },
    { episode_id: 'synthetic', sequence: 2, status: 'usage_settled', request_sequence: 1, wire_sha256: 'a', provider_usage: { input_tokens: 80, output_tokens: 20, total_tokens: 100 }, known_episode_tokens_after: 100, unknown_usage_requests: 0 }];
  const journal = rows.map(JSON.stringify).join('\n'); await writeFile(resolve(directory, 'budget.jsonl'), journal);
  await save('launch-start.json', { manifest_sha256: manifest.sha256, fixture: false });
  const receipt = await save('closed-receipt.json', { closed: true, fixture: false, initializeOnly: false, unknown_usage_requests: 0,
    known_tokens: 100, model_requests: 1, manifest_sha256: manifest.sha256, case_sha256: raw.sha256,
    budget_audit_sha256: sha(journal), status: 'failed_model_turn', ...mutation });
  return { descriptor: { ...receipt, manifest }, options: { rawCase, root: directory, expectedKnownTokens: 100 }, save };
}
test('known failed calls carry their actual usage and all byte bindings without resetting', async () => {
  const f = await fixture(); f.options.rawCase.question = { question_id: 'q_synthetic', question: '程序是否完成？' };
  const result = await verifyPriorEpisode(f.descriptor, f.options);
  assert.equal(result.known_tokens, 100); assert.equal(result.prior_status, 'failed_model_turn'); assert.equal(result.bindings.length, 5);
});
test('unknown usage, mock receipts, claimed lower totals or a different question cannot carry', async () => {
  for (const mutation of [{ fixture: true }, { unknown_usage_requests: 1 }, { known_tokens: 0 }]) {
    const f = await fixture(mutation); await assert.rejects(verifyPriorEpisode(f.descriptor, f.options), /known_closed_real_prior_required/);
  }
  const f = await fixture(); f.options.rawCase.question.question = '另一个问题';
  await assert.rejects(verifyPriorEpisode(f.descriptor, f.options), /prior_question_identity_changed/);
});
test('changing a closed usage journal cannot mint a fresh budget', async () => {
  const f = await fixture(); await writeFile(resolve(f.options.root, 'budget.jsonl'), '');
  await assert.rejects(verifyPriorEpisode(f.descriptor, f.options), /prior_usage_binding_changed/);
});
