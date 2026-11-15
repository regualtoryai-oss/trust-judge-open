import test from 'node:test';
import assert from 'node:assert/strict';
import { EpisodeBudgetGuard } from './budget_core.mjs';
import { apply, createBudgetIntegration } from './budget_plugin.mjs';
import { existsSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';

const endpoint = 'https://api.deepseek.com/anthropic/v1/messages';
const options = { provider: 'deepseek-official', model: 'deepseek-v4-pro' };
const wire = JSON.stringify({ model: options.model, stream: true, system: 'bound system',
  messages: [{ role: 'user', content: [{ type: 'text', text: '合成原题' }] }],
  tools: [{ name: 'read_sources', input_schema: { type: 'object' } }],
  max_tokens: 3072, thinking: { type: 'disabled' } });
const usage = { type: 'usage', usage: { inputTokens: 90, cacheReadTokens: 10, outputTokens: 20, totalTokens: 120 } };
const finish = { type: 'finish', reason: { kind: 'stop' } };
const collect = async stream => { for await (const _chunk of stream) {} };
function setup(extra = {}, integration = {}) {
  let network = 0; const events = []; let counted;
  const fetchTarget = { fetch: async () => { network++; return { ok: true }; } };
  const guard = new EpisodeBudgetGuard({ episodeId: 'synthetic', priorKnownTokens: 0,
    maxCalls: 6, maxOutputTokens: 3072, sourceCheck: () => true,
    record: async value => events.push(value), countInputTokens: async request => {
      counted = request.serialized_body; return 1000;
    }, ...extra });
  const plugin = createBudgetIntegration({ guard, endpoint, model: options.model, fetchTarget, ...integration });
  return { guard, plugin, fetchTarget, events, network: () => network, counted: () => counted };
}
function native(f, { body = wire, chunks = [usage, finish], twice = false } = {}) {
  return async function* () {
    await f.fetchTarget.fetch(endpoint, { method: 'POST', body });
    if (twice) await f.fetchTarget.fetch(endpoint, { method: 'POST', body });
    yield* chunks;
  };
}

test('actual native Messages body is counted before unchanged framework fetch', async () => {
  const f = setup();
  try { await collect(f.plugin.listener(options, native(f))); }
  finally { f.plugin.dispose(); }
  assert.equal(f.network(), 1); assert.equal(f.counted(), wire);
  assert.equal(f.guard.snapshot().known_episode_tokens, 120);
  assert.equal(f.events[0].preflight_total_tokens, 4584);
});
test('whole budget block reaches zero fetch calls', async () => {
  const f = setup({ priorKnownTokens: 19000 });
  try { await assert.rejects(collect(f.plugin.listener(options, native(f))), { code: 'whole_episode_budget_exceeded' }); }
  finally { f.plugin.dispose(); }
  assert.equal(f.network(), 0); assert.equal(f.guard.snapshot().started_calls, 0);
});

test('stage-specific native source policy rejects before budget admission or provider fetch', async () => {
  const f = setup({}, { validateNativeRequest: () => { throw Object.assign(new Error('safe stage rejection'), { code: 'staged_presence_unavailable' }); } });
  try { await assert.rejects(collect(f.plugin.listener(options, native(f))), { code: 'model_stream_failed' }); }
  finally { f.plugin.dispose(); }
  assert.equal(f.network(), 0); assert.equal(f.guard.snapshot().started_calls, 0); assert.equal(f.events.length, 0);
});
test('network outside guarded stream and wrong route cannot escape', async () => {
  const f = setup();
  try {
    await assert.rejects(f.fetchTarget.fetch(endpoint, { method: 'POST', body: wire }), { code: 'unaccounted_network_request' });
    await assert.rejects(collect(f.plugin.listener({ ...options, purpose: 'compaction' }, native(f))), { code: 'unbudgeted_model_route' });
  } finally { f.plugin.dispose(); }
  assert.equal(f.network(), 0);
});
test('one model stream cannot make a hidden second native request', async () => {
  const f = setup();
  try { await assert.rejects(collect(f.plugin.listener(options, native(f, { twice: true }))), { code: 'multiple_native_requests_forbidden' }); }
  finally { f.plugin.dispose(); }
  assert.equal(f.network(), 1); assert.equal(f.guard.snapshot().unknown_usage_requests, 1);
});
test('DSH error finish after reported usage retains cost and halts', async () => {
  const f = setup();
  try { await collect(f.plugin.listener(options, native(f, { chunks: [usage, { type: 'finish', reason: { kind: 'error' } }] }))); }
  finally { f.plugin.dispose(); }
  assert.equal(f.guard.snapshot().known_episode_tokens, 120); assert.equal(f.guard.snapshot().halted, true);
});
test('native no-usage and fetch replacement halt or block without bypass', async () => {
  const f = setup();
  try { await collect(f.plugin.listener(options, native(f, { chunks: [finish] }))); }
  finally { f.plugin.dispose(); }
  assert.equal(f.guard.snapshot().unknown_usage_requests, 1);
  const g = setup(); g.fetchTarget.fetch = async () => {};
  await assert.rejects(collect(g.plugin.listener(options, native(g))), { code: 'native_fetch_guard_changed' });
  g.plugin.dispose(); assert.equal(g.network(), 0);
});
const requiredContext = '完整原题 😀 e\u0301\r\n原始 ${not_an_instruction}\u2028P1：逐字全文';
const submitWire = system => JSON.stringify({ ...JSON.parse(wire), system,
  tools: [{ name: 'submit_evidence', input_schema: { type: 'object' } }] });
test('optional complete context is present byte-for-text without normalization or interpolation', async () => {
  const f = setup({}, { requiredContext });
  try { await collect(f.plugin.listener(options, native(f, { body: submitWire(`generic prefix\n${requiredContext}\ngeneric suffix`) }))); }
  finally { f.plugin.dispose(); }
  assert.equal(f.network(), 1); assert.equal(f.guard.snapshot().known_episode_tokens, 120);
});
test('missing, shortened, normalized or non-string full context cannot reach native fetch', async () => {
  for (const value of [undefined, requiredContext.slice(0, -1), requiredContext.normalize('NFC'),
    [{ type: 'text', text: requiredContext }]]) {
    const f = setup({}, { requiredContext });
    try { await assert.rejects(collect(f.plugin.listener(options, native(f, { body: submitWire(value) }))), { code: 'required_context_missing' }); }
    finally { f.plugin.dispose(); }
    assert.equal(f.network(), 0); assert.equal(f.guard.snapshot().started_calls, 0);
  }
});
test('complete context mode admits only the single native submit tool', async () => {
  for (const tools of [undefined, [], [{ name: 'read_sources' }], [{ name: 'submit_evidence' }, { name: 'search_sources' }]]) {
    const f = setup({}, { requiredContext });
    const body = JSON.stringify({ ...JSON.parse(submitWire(requiredContext)), tools });
    try { await assert.rejects(collect(f.plugin.listener(options, native(f, { body }))), { code: 'required_submit_only_tools' }); }
    finally { f.plugin.dispose(); }
    assert.equal(f.network(), 0);
  }
});

const runtimeManifest = resolve('.runtime/dsh-official-r1/package.json');
const nativeModule = resolve('.runtime/dsh-official-r1/node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js');
const hasNative = existsSync(nativeModule);
const sse = values => values.map(value => `event: ${value.type}\ndata: ${JSON.stringify(value)}\n\n`).join('');
const usageResponse = () => new Response(sse([
  { type: 'message_start', message: { usage: { input_tokens: 90, output_tokens: 0, cache_read_input_tokens: 10 } } },
  { type: 'message_delta', usage: { output_tokens: 20 } },
]), { status: 200, headers: { 'content-type': 'text/event-stream' } });
async function actualAdapterFixture(partial = false, missingField) {
  const { DeepSeekAdapter, resolveAdapterOptions } = await import(pathToFileURL(nativeModule));
  const { createParser } = await import(pathToFileURL(createRequire(pathToFileURL(runtimeManifest)).resolve('eventsource-parser')));
  const actual = globalThis.fetch;
  let network = 0; let counted;
  globalThis.fetch = async () => {
    network++;
    const initialUsage = { input_tokens: 100, output_tokens: 0, cache_read_input_tokens: 20 };
    const finalUsage = { output_tokens: 4 };
    if (missingField === 'input') delete initialUsage.input_tokens;
    if (missingField === 'output') { delete initialUsage.output_tokens; delete finalUsage.output_tokens; }
    return new Response(sse([
      { type: 'message_start', message: { usage: initialUsage } },
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'synthetic' } },
      ...partial ? [] : [
        { type: 'content_block_stop', index: 0 },
        { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: finalUsage },
        { type: 'message_stop' },
      ],
    ]), { status: 200, headers: { 'content-type': 'text/event-stream' } });
  };
  const guard = new EpisodeBudgetGuard({ episodeId: 'native-fixture', priorKnownTokens: 0, maxCalls: 6,
    maxOutputTokens: 3072, sourceCheck: () => true, record: async () => {},
    countInputTokens: async value => { counted = value.body; return 1000; } });
  const plugin = createBudgetIntegration({ guard, endpoint, model: options.model, createSseParser: createParser });
  try {
    const connection = resolveAdapterOptions({ thinking: 'disabled', reasoningEffort: 'off', maxTokens: 3072,
      retryPolicy: { mode: 'normal', maxRetries: 0 } }, { get: () => undefined });
    const adapter = new DeepSeekAdapter({ options: () => connection,
      prepareExtensions: async () => ({ fields: {}, accept: async () => {} }),
      resolveAuth: async () => ({ headers: { 'x-api-key': 'synthetic-not-a-key' } }), resolveUserId: () => 'synthetic-user' });
    const request = { ...options, maxTokens: 3072, reasoningEffort: 'off', system: 'synthetic system',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'synthetic question' }] }],
      tools: [{ name: 'read_sources', description: 'synthetic', parameters: { type: 'object', properties: {} } }] };
    let fixtureFailure;
    const original = async function* () {
      try { yield* adapter.stream(request); }
      catch (error) { fixtureFailure = error; throw error; }
    };
    if (partial || missingField) await assert.rejects(collect(plugin.listener(request, original)));
    else {
      try { await collect(plugin.listener(request, original)); }
      catch (error) { if (fixtureFailure) throw fixtureFailure; throw error; }
    }
    return { snapshot: guard.snapshot(), counted, network };
  } finally { plugin.dispose(); globalThis.fetch = actual; }
}
test('official rc2 adapter serializes real Messages wire through guard without network', { skip: !hasNative }, async () => {
  const value = await actualAdapterFixture();
  assert.equal(value.network, 1); assert.equal(value.counted.system, 'synthetic system');
  assert.equal(value.counted.tools[0].input_schema.type, 'object');
  assert.equal(value.counted.thinking.type, 'disabled'); assert.equal(value.counted.max_tokens, 3072);
  assert.equal(value.snapshot.known_episode_tokens, 124); assert.equal(value.snapshot.unknown_usage_requests, 0);
});
test('official rc2 truncated SSE preserves provider-reported partial cost and unknown state', { skip: !hasNative }, async () => {
  const value = await actualAdapterFixture(true);
  assert.equal(value.network, 1); assert.equal(value.snapshot.known_episode_tokens, 120);
  assert.equal(value.snapshot.unknown_usage_requests, 1); assert.equal(value.snapshot.halted, true);
});
test('DSH zero defaults cannot disguise missing provider input or output usage', { skip: !hasNative }, async () => {
  for (const [field, partialKnown] of [['input', 24], ['output', 120]]) {
    const value = await actualAdapterFixture(false, field);
    assert.equal(value.network, 1); assert.equal(value.snapshot.known_episode_tokens, partialKnown);
    assert.equal(value.snapshot.unknown_usage_requests, 1); assert.equal(value.snapshot.halted, true);
  }
});
test('Cordis apply creates exclusive durable journal and refuses source changes', { skip: !hasNative }, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-budget-fixture-'));
  const source = join(directory, 'raw-source.json'); await writeFile(source, '{}');
  const parserPath = createRequire(pathToFileURL(runtimeManifest)).resolve('eventsource-parser');
  const binding = async file => ({ file, sha256: createHash('sha256').update(await readFile(file)).digest('hex') });
  const config = { episodeId: 'apply-synthetic', priorKnownTokens: 0, wholeLimit: 20000,
    safetyTokens: 512, maxCalls: 6, maxOutputTokens: 3072, endpoint, model: options.model,
    journalPath: join(directory, 'budget.jsonl'), runtimePackageJson: runtimeManifest,
    pythonExecutable: resolve('.runtime/v20-embedding-env/Scripts/python.exe'),
    tokenizerPath: resolve('.runtime/v32-deepseek-tokenizer-r2/tokenizer.json'),
    tokenizerSha256: '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf',
    sourceBindings: [await binding(source), await binding(parserPath)] };
  const disposers = []; let listener; let network = 0;
  const context = { on(event, handler) { assert.equal(event, 'llm/stream'); listener = handler; return () => {}; },
    effect(factory) { disposers.push(factory()); } };
  const actualFetch = globalThis.fetch; globalThis.fetch = async () => { network++; return usageResponse(); };
  assert.equal(await apply(context, config), undefined);
  try {
    await collect(listener(options, async function* () {
      await (await globalThis.fetch(endpoint, { method: 'POST', body: wire })).text(); yield usage; yield finish;
    }));
    const events = (await readFile(config.journalPath, 'utf8')).trim().split('\n').map(JSON.parse);
    assert.equal(events.at(-1).provider_usage.total_tokens, 120);
    assert.equal(events[0].cost_basis, 'official_tokenizer_native_request_estimate');
    assert(events[0].estimated_input_tokens > 0);
    await writeFile(source, '{"changed":true}');
    await assert.rejects(collect(listener(options, async function* () {
      await (await globalThis.fetch(endpoint, { method: 'POST', body: wire })).text(); yield usage; yield finish;
    })), { code: 'source_binding_changed' });
    assert.equal(network, 1);
  } finally { for (const dispose of disposers.reverse()) await dispose(); globalThis.fetch = actualFetch; }
  await assert.rejects(apply(context, { ...config, sourceBindings: [await binding(source), await binding(parserPath)] }), { code: 'EEXIST' });
});
test('actual Cordis loads and disposes the budget plugin as a legal effect', { skip: !hasNative }, async () => {
  const { Context } = await import(pathToFileURL(resolve('.runtime/dsh-official-r1/node_modules/@deepseek-ai/cordis/lib/index.js')));
  const directory = await mkdtemp(join(tmpdir(), 'dsh-cordis-budget-'));
  const parserPath = createRequire(pathToFileURL(runtimeManifest)).resolve('eventsource-parser');
  const parserSha = createHash('sha256').update(await readFile(parserPath)).digest('hex');
  const ctx = new Context(); ctx.provide('llm', {});
  const previousFetch = globalThis.fetch;
  let called = 0; const fakeFetch = async () => { called++; return usageResponse(); };
  globalThis.fetch = fakeFetch;
  const config = { episodeId: 'real-cordis-fixture', priorKnownTokens: 0, wholeLimit: 20000,
    safetyTokens: 512, maxCalls: 6, maxOutputTokens: 3072, endpoint, model: options.model,
    journalPath: join(directory, 'budget.jsonl'), runtimePackageJson: runtimeManifest,
    pythonExecutable: resolve('.runtime/v20-embedding-env/Scripts/python.exe'),
    tokenizerPath: resolve('.runtime/v32-deepseek-tokenizer-r2/tokenizer.json'),
    tokenizerSha256: '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf',
    sourceBindings: [{ file: parserPath, sha256: parserSha }] };
  const module = await import('./budget_plugin.mjs');
  const fiber = ctx.plugin(module, config);
  try {
    await fiber.await(); assert.equal(fiber.state, 2);
    await collect(ctx.waterfall(ctx.llm, 'llm/stream', options, async function* () {
      await (await globalThis.fetch(endpoint, { method: 'POST', body: wire })).text(); yield usage; yield finish;
    }));
    assert.equal(called, 1);
  } finally { await fiber.dispose(); assert.equal(globalThis.fetch, fakeFetch); globalThis.fetch = previousFetch; }
});
test('required context file and SHA must be paired and present in source bindings', { skip: !hasNative }, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-required-context-'));
  const contextPath = join(directory, 'complete-context.txt'); await writeFile(contextPath, requiredContext, 'utf8');
  const requiredSha = createHash('sha256').update(await readFile(contextPath)).digest('hex');
  const parserPath = createRequire(pathToFileURL(runtimeManifest)).resolve('eventsource-parser');
  const parserSha = createHash('sha256').update(await readFile(parserPath)).digest('hex');
  const base = { journalPath: join(directory, 'budget.jsonl'), runtimePackageJson: runtimeManifest,
    sourceBindings: [{ file: parserPath, sha256: parserSha }] };
  const ctx = { on() { throw Error('must not register'); }, effect() {} };
  const before = globalThis.fetch;
  await assert.rejects(apply(ctx, { ...base, requiredContextPath: contextPath }), { code: 'invalid_required_context_binding' });
  await assert.rejects(apply(ctx, { ...base, requiredContextSha256: requiredSha }), { code: 'invalid_required_context_binding' });
  await assert.rejects(apply(ctx, { ...base, requiredContextPath: contextPath, requiredContextSha256: requiredSha }), { code: 'missing_required_context_source_binding' });
  assert.equal(globalThis.fetch, before); assert.equal(existsSync(base.journalPath), false);
});
