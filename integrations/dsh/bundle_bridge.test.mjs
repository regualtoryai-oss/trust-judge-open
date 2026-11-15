import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createBundleBridge } from './bundle_bridge.mjs';
import { apply as evidenceApply, bindBundleGuard } from './bundle_evidence.mjs';
import { apply as guardApply } from './bundle_guard.mjs';
import { buildBundleFixturePatch } from './bundle_profile.mjs';
import { syntheticBundleCase } from './bundle_fixture.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha = value => createHash('sha256').update(value).digest('hex');
async function setup() {
  const dir = await mkdtemp(resolve(ROOT, '.runtime/bundle-bridge-test-'));
  const input = syntheticBundleCase(), casePath = resolve(dir, 'synthetic-source.json');
  await writeFile(casePath, JSON.stringify(input));
  return { dir, input, config: { casePath, caseSha256: sha(await readFile(casePath)),
    pythonExecutable: resolve(ROOT, '.venv/Scripts/python.exe'),
    scriptSha256: sha(await readFile(resolve(ROOT, 'scripts/source_bundle_tool.py'))) } };
}

test('controlled bridge executes real isolated Python with replayed read state', async () => {
  const { config, input } = await setup(), bridge = await createBundleBridge(config);
  try {
    const preview = await bridge.call('search', { query: input.question.question, limit: 60, exclude_paragraph_ids: [] });
    assert.equal(preview.results.length, 4);
    await assert.rejects(bridge.call('read', { anchor_paragraph_ids: ['opaque-source-0'], before: 0, after: 1 }), /unique_native_read_call_id_required/);
    const read = await bridge.call('read', { anchor_paragraph_ids: ['opaque-source-0'], before: 0, after: 1 }, { callId: 'actual-bridge-call-id' });
    assert.ok(read.bundles[0].quote.includes('<真实原始分隔😀>'));
    assert.equal(bridge.snapshot().reads[0].tool_use_id, 'actual-bridge-call-id');
    assert.equal(bridge.snapshot().native_model_delivery_verified, false);
    assert.equal(bridge.snapshot().history.length, 2);
    assert.ok(!JSON.stringify(bridge.snapshot().history).includes('第一原段完整尾部'));
  } finally { await bridge.close(); }
});

test('bound case changes, unknown paths and shell operations are rejected', async () => {
  const { config } = await setup(), bridge = await createBundleBridge(config);
  try {
    await assert.rejects(bridge.call('terminal', { command: 'anything' }), /unknown_operation/);
    await assert.rejects(bridge.call('search', { query: '项目', path: 'other-file' }), /unknown_or_missing_fields/);
    await writeFile(config.casePath, '{}');
    await assert.rejects(bridge.call('search', { query: '项目' }), /bound_source_changed/);
  } finally { await bridge.close(); }
});

test('business plugin cannot submit after backend read before actual native request proof', async () => {
  const { config, dir, input } = await setup(), episodeId = randomUUID(), slot = bindBundleGuard(episodeId), registered = new Map(), cleanups = [];
  const ctx = { trustJudgeBundleGuard: { episodeId }, systemPrompt: { section() {} },
    tools: { register(definition) { registered.set(definition.name, definition); return () => registered.delete(definition.name); } },
    effect(callback) { cleanups.push(callback()); } };
  const exec = callId => ({ callId, signal: new AbortController().signal, concludeTurn() { throw new Error('must_not_conclude'); } });
  try {
    await evidenceApply(ctx, { episodeId, bridge: config, submissionPath: resolve(dir, 'not-submitted.json') });
    assert.deepEqual([...registered.keys()], ['search_source_bundles']);
    await registered.get('search_source_bundles').execute({ query: input.question.question, limit: 60, exclude_paragraph_ids: [] }, exec('search'));
    await registered.get('read_source_bundles').execute({ anchor_paragraph_ids: ['opaque-source-0'], before: 0, after: 1 }, exec('read'));
    assert.ok(registered.has('submit_bundle_evidence'));
    await assert.rejects(registered.get('submit_bundle_evidence').execute({ selected: [] }, exec('submit')), /current_native_full_read_proof_required/);
    assert.equal(slot.getPresence().reads.length, 1);
  } finally {
    for (const cleanup of cleanups.reverse()) await cleanup?.();
    slot.dispose();
  }
});

test('new guard and profile explicitly refuse a real or unguarded transport', async () => {
  await assert.rejects(guardApply({}, { fixtureOnly: false }), /fixture_transport_required/);
  assert.throws(() => buildBundleFixturePatch({ bridge: {}, guardConfig: { fixtureOnly: false } }), /explicit_fixture_bindings_required/);
});

