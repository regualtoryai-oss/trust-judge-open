/** Actual official SDK + runtime + native serializer, with a network-free provider fixture. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runComparison, verifyBudgetReady } from './run_comparison.mjs';
import { SOURCE_LOCAL_V2_INSTRUCTIONS } from './evidence_policy.mjs';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const runtimeAvailable = await access(resolve(root, '.runtime/dsh-official-r1/node_modules/@deepseek-ai/dsh-sdk-client/lib/index.js')).then(() => true, () => false);

async function fixture(maxOutputTokens = 3072, contextMode = undefined, selectionPolicy = undefined) {
  const relative = `.runtime/dsh-sdk-tests/${randomUUID()}`, dir = resolve(root, relative);
  await mkdir(dir, { recursive: true });
  const source = `${contextMode === 'staged-short-window' ? '原段背景。'.repeat(40) : ''}主管部门称事件仍在调查，尚未确认原因。`;
  const raw = { question: { question_id: 'fixture_q', question: '事件是否已确认？' }, topic_title: '合成测试',
    catalog: [{ paragraph_ref: 'P1', doc_id: 'synthetic_doc', para_id: 'synthetic_doc#p1', title: '测试通报', text: source,
      parent_complete: true, paragraph_complete: true }] };
  const bytes = JSON.stringify(raw);
  await writeFile(resolve(dir, 'raw.json'), bytes);
  const plan = [{ name: 'read_candidate_packet', arguments: {} }, { name: 'submit_evidence', arguments: {
    conclusion: '事件原因尚未确认。', verdict_label: 'partial_support', selected: [{ p: 'P1', roles: ['L'],
      fact_des: source, argument: { 时间: '', 主体: '主管部门', 地点: '', 内容: '事件仍在调查', 影响: '', 数值口径: '' }, entities: ['主管部门'] }],
    supporting: ['P1'], conflicting: [], basis: '原文明确说明尚未确认。' } }];
  const fixturePlan = contextMode === 'staged-short-window'
    ? [{ name: 'read_selected_sources', arguments: { refs: ['P1'] } }, ...plan.slice(1)]
    : contextMode ? plan.slice(1) : plan;
  await writeFile(resolve(dir, 'plan.json'), JSON.stringify(fixturePlan));
  const manifest = { format: 'dsh-comparison-manifest-v1', episodeId: `fixture-${randomUUID()}`, wholeLimit: 20000,
    priorKnownTokens: 0, maxCalls: contextMode === 'staged-short-window' ? 2 : 6, maxOutputTokens, safetyTokens: 512, model: 'deepseek-v4-pro', reasoningEffort: 'off',
    ...(contextMode ? { contextMode } : {}),
    ...(selectionPolicy ? { selectionPolicy } : {}),
    casePath: resolve(dir, 'raw.json'), caseSha256: createHash('sha256').update(bytes).digest('hex'),
    outputDirectory: `${relative}/run`, fixturePlan: resolve(dir, 'plan.json') };
  const manifestPath = resolve(dir, 'manifest.json');
  await writeFile(manifestPath, JSON.stringify(manifest));
  return { receipt: await runComparison({ manifestPath, fixture: true }), dir: resolve(dir, 'run'), source };
}

test('official DSH loop reads complete bounded packet and submits with no confirmation model call', { skip: !runtimeAvailable }, async () => {
  const { receipt, dir, source } = await fixture();
  assert.equal(receipt.status, 'closed_proposed_unverified');
  assert.equal(receipt.fixture, true);
  assert.equal(receipt.model_requests, 2);
  assert.equal(receipt.known_tokens, 300); // fixture usage only, never a real bill.
  assert.equal(receipt.unknown_usage_requests, 0);
  const artifact = JSON.parse(await readFile(resolve(dir, 'submission.json'), 'utf8'));
  assert.equal(artifact.evidence[0].quote, source);
  assert.equal(artifact.quality.publishable, false);
  const wires = (await readFile(resolve(dir, 'fixture-wire.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  assert.equal(wires.length, 2);
  assert.ok(JSON.stringify(wires[1].messages).includes(source));
  assert.deepEqual(wires[0].tools.sort(), ['read_candidate_packet', 'read_sources', 'search_sources', 'submit_evidence']);
});

test('actual SDK budget failure is CLOSED and sends zero provider requests', { skip: !runtimeAvailable }, async () => {
  const { receipt, dir } = await fixture(20000);
  assert.equal(receipt.status, 'budget_blocked');
  assert.equal(receipt.model_requests, 0);
  assert.equal(receipt.known_tokens, 0);
  assert.equal(receipt.submission_sha256, null);
  await assert.rejects(readFile(resolve(dir, 'fixture-wire.jsonl')));
});

test('official SDK complete-context mode presents all original data and exposes only terminal submit', { skip: !runtimeAvailable }, async () => {
  const { receipt, dir, source } = await fixture(3072, 'full-candidate-context');
  assert.equal(receipt.status, 'closed_proposed_unverified');
  assert.equal(receipt.model_requests, 1);
  assert.equal(receipt.known_tokens, 150);
  const wires = (await readFile(resolve(dir, 'fixture-wire.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  assert.equal(wires.length, 1);
  assert.deepEqual(wires[0].tools, ['submit_evidence']);
  assert.ok(JSON.stringify(wires[0].system).includes(source));
  assert.equal(receipt.unknown_usage_requests, 0);
});

test('official SDK forwards the frozen source-local policy literally in the same single request', { skip: !runtimeAvailable }, async () => {
  const { receipt, dir, source } = await fixture(6144, 'full-candidate-context', 'source-local-v2');
  assert.equal(receipt.status, 'closed_proposed_unverified');
  assert.equal(receipt.selection_policy, 'source-local-v2');
  assert.equal(receipt.selection_policy_sha256, createHash('sha256').update(SOURCE_LOCAL_V2_INSTRUCTIONS).digest('hex'));
  assert.equal(receipt.model_requests, 1);
  assert.equal(receipt.unknown_usage_requests, 0);
  const wires = (await readFile(resolve(dir, 'fixture-wire.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  assert.equal(wires.length, 1);
  assert.deepEqual(wires[0].tools, ['submit_evidence']);
  assert.ok(JSON.stringify(wires[0].system).includes(source));
  assert.ok(JSON.stringify(wires[0].system).includes(SOURCE_LOCAL_V2_INSTRUCTIONS));
});

test('official SDK staged business reads complete parent once, withdraws previews and verifies both actual native phases', { skip: !runtimeAvailable }, async () => {
  const { receipt, dir, source } = await fixture(6144, 'staged-short-window', 'source-local-v2');
  assert.equal(receipt.status, 'closed_proposed_unverified');
  assert.equal(receipt.model_requests, 2);
  assert.equal(receipt.known_tokens, 300); // fake usage, not a real bill.
  const wires = (await readFile(resolve(dir, 'fixture-wire.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  assert.equal(wires.length, 2);
  assert.deepEqual(wires[0].tools, ['read_selected_sources']);
  assert.deepEqual(wires[1].tools, ['submit_evidence']);
  assert.ok(wires[0].system.includes('candidate_previews'));
  assert.ok(!wires[1].system.includes('candidate_previews\":['));
  assert.ok(!wires[1].system.includes(source));
  const audit = (await readFile(resolve(dir, 'staged-presence.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(audit.map(row => row.phase), ['preview', 'final']);
  assert.ok(audit.every(row => row.native_presence_verified_before_admission === true && row.semantic_verified === false));
  const artifact = JSON.parse(await readFile(resolve(dir, 'submission.json'), 'utf8'));
  assert.equal(artifact.evidence[0].quote, source);
  assert.equal(artifact.scope.context_mode, 'staged-short-window');
  assert.equal(artifact.quality.publishable, false);
});

test('SDK readiness alone cannot authorize a prompt without the matching installed budget guard', async () => {
  const dir = resolve(root, `.runtime/dsh-sdk-tests/${randomUUID()}`); await mkdir(dir, { recursive: true });
  const path = resolve(dir, 'ready.json'), expected = { episodeId: 'expected', sourceBindingsSha256: 'a'.repeat(64) };
  await writeFile(path, JSON.stringify({ format: 'dsh-budget-ready-v1', episode_id: 'other', source_bindings_sha256: expected.sourceBindingsSha256, guarded_native_transport_installed: true }));
  await assert.rejects(verifyBudgetReady(path, expected), { code: 'budget_plugin_not_ready' });
  await writeFile(path, JSON.stringify({ format: 'dsh-budget-ready-v1', episode_id: expected.episodeId, source_bindings_sha256: expected.sourceBindingsSha256, guarded_native_transport_installed: false }));
  await assert.rejects(verifyBudgetReady(path, expected), { code: 'budget_plugin_not_ready' });
});
