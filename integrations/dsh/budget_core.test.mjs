import test from 'node:test';
import assert from 'node:assert/strict';
import { EpisodeBudgetGuard, nativeUsage } from './budget_core.mjs';

const body = extra => JSON.stringify({ model: 'deepseek-v4-pro', messages: [{ role: 'user', content: '原题' }],
  tools: [{ type: 'function', function: { name: 'read_sources', parameters: { type: 'object' } } }],
  thinking: { type: 'disabled' }, max_tokens: 512, ...extra });
const usage = { type: 'usage', usage: { inputTokens: 100, cacheReadTokens: 200, outputTokens: 30, reasoningTokens: 10 } };
const finish = { type: 'finish', reason: { kind: 'stop' } };
function fixture(extra = {}) {
  const events = []; let calls = 0;
  const guard = new EpisodeBudgetGuard({ episodeId: 'synthetic-episode', priorKnownTokens: 1000,
    maxCalls: 3, maxOutputTokens: 4096, sourceCheck: () => true, countInputTokens: async ({ body: wire }) => {
      assert.equal(wire.tools[0].function.name, 'read_sources'); return 2000;
    }, record: async event => { events.push(event); }, ...extra });
  const invoke = async function* () { calls++; yield usage; yield finish; };
  return { guard, events, invoke, calls: () => calls };
}
const consume = async iterator => { const values = []; for await (const item of iterator) values.push(item); return values; };

test('actual native messages and tools are counted, cache included once, reasoning not twice', async () => {
  const f = fixture(); await consume(f.guard.protectStream({ prepareWire: () => body(), invoke: f.invoke }));
  assert.equal(f.calls(), 1); assert.equal(f.guard.snapshot().known_episode_tokens, 1330);
  assert.equal(f.events.at(-1).provider_usage.input_tokens, 300);
  assert.equal(f.events[0].preflight_total_tokens, 4024);
  assert(!JSON.stringify(f.events).includes('原题'));
});
test('whole episode budget includes prior, full input, output and safety, before invoke', async () => {
  const f = fixture({ priorKnownTokens: 18000 });
  await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body(), invoke: f.invoke })), { code: 'whole_episode_budget_exceeded' });
  assert.equal(f.calls(), 0); assert.equal(f.guard.snapshot().unknown_usage_requests, 0);
});
test('unknown or heuristic counter result cannot invoke', async () => {
  for (const value of [undefined, null, -1, 1.1, '2000', NaN]) {
    const f = fixture({ countInputTokens: async () => value });
    await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body(), invoke: f.invoke })), { code: 'unknown_input_cost' });
    assert.equal(f.calls(), 0);
  }
});
test('source and native request changes block before invoke', async () => {
  const f = fixture(); let generation = 0;
  await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body({ temperature: generation++ }), invoke: f.invoke })), { code: 'native_request_changed' });
  assert.equal(f.calls(), 0);
  const g = fixture({ sourceCheck: () => false });
  await assert.rejects(consume(g.guard.protectStream({ prepareWire: () => body(), invoke: g.invoke })), { code: 'source_binding_changed' });
  assert.equal(g.calls(), 0);
});
test('reported usage is retained even when stream subsequently fails', async () => {
  const f = fixture(); const invoke = async function* () { yield usage; throw new Error('private body'); };
  await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body(), invoke })), { code: 'model_stream_failed' });
  assert.equal(f.guard.snapshot().known_episode_tokens, 1330); assert.equal(f.guard.snapshot().halted, true);
  assert.equal(f.guard.snapshot().unknown_usage_requests, 0);
  await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body(), invoke: f.invoke })), { code: 'episode_halted' });
});
test('missing usage and output truncation halt, without automatic retry', async () => {
  for (const chunks of [[finish], [usage, { type: 'finish', reason: { kind: 'max-tokens' } }]]) {
    const f = fixture(); const invoke = async function* () { yield* chunks; };
    await consume(f.guard.protectStream({ prepareWire: () => body(), invoke }));
    assert.equal(f.guard.snapshot().halted, true); assert.equal(f.guard.snapshot().started_calls, 1);
  }
});
test('duplicate usage snapshots replace instead of double charging', async () => {
  const f = fixture(); const invoke = async function* () { yield usage; yield usage; yield finish; };
  await consume(f.guard.protectStream({ prepareWire: () => body(), invoke }));
  assert.equal(f.guard.snapshot().known_episode_tokens, 1330);
});
test('unknown usage on a failed request freezes later calls', async () => {
  const f = fixture(); const invoke = async function* () { throw new Error('secret'); };
  await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body(), invoke })), { code: 'model_stream_failed' });
  assert.equal(f.guard.snapshot().unknown_usage_requests, 1); assert.equal(f.guard.snapshot().halted, true);
});
test('output cap and disabled thinking are explicit preflight requirements', async () => {
  for (const extra of [{ max_tokens: 0 }, { max_tokens: 8192 }, { thinking: { type: 'enabled' } }, { reasoning_effort: 'low' }]) {
    const f = fixture(); await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body(extra), invoke: f.invoke })));
    assert.equal(f.calls(), 0);
  }
});
test('failure of durable admission prevents all provider calls', async () => {
  const f = fixture({ record: async () => { throw new Error('disk full'); } });
  await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body(), invoke: f.invoke })), { code: 'model_stream_failed' });
  assert.equal(f.calls(), 0);
});
test('native usage validates strict nonnegative counts and total consistency', () => {
  for (const invalid of [{ inputTokens: -1, outputTokens: 2 }, { inputTokens: 1, outputTokens: 2, totalTokens: 4 },
    { inputTokens: 1, outputTokens: 2, reasoningTokens: 3 }]) assert.throws(() => nativeUsage(invalid));
});
test('invalid later usage preserves earlier known sample but marks uncertainty', async () => {
  const f = fixture(); const invoke = async function* () {
    yield usage; yield { type: 'usage', usage: { inputTokens: -1, outputTokens: 2 } };
  };
  await assert.rejects(consume(f.guard.protectStream({ prepareWire: () => body(), invoke })));
  assert.equal(f.guard.snapshot().known_episode_tokens, 1330);
  assert.equal(f.guard.snapshot().unknown_usage_requests, 1);
  assert.equal(f.guard.snapshot().halted, true);
});
test('provider exceeding reserved output halts even when reported usage is below episode cap', async () => {
  const f = fixture(); const invoke = async function* () {
    yield { type: 'usage', usage: { inputTokens: 1, outputTokens: 513 } }; yield finish;
  };
  await consume(f.guard.protectStream({ prepareWire: () => body(), invoke }));
  assert.equal(f.guard.snapshot().known_episode_tokens, 1514);
  assert.equal(f.guard.snapshot().halted, true);
});
