import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { apply, createEvidenceBackend, toolDefinitions, TOOL_NAMES, readSettledPacketBudget, buildCompleteContext } from './evidence_tools.mjs';
import { selectionPolicySection } from './evidence_policy.mjs';

const sha = text => createHash('sha256').update(text).digest('hex');
function fixture() {
  return { question: { question_id: 'q_synthetic', question: '项目延期，是施工方拒绝施工还是管理方尚未批准？' }, catalog: [
    { paragraph_ref: 'P1', doc_id: 'd1', para_id: 'p1', title: '项目登记', text: '登记表记载：申请已受理，但批准尚未完成。', paragraph_complete: true, parent_complete: true },
    { paragraph_ref: 'P2', doc_id: 'd2', para_id: 'p1', title: '施工方说明', text: '施工方称其愿意施工，并否认曾拒绝开工。', paragraph_complete: true, parent_complete: true },
    { paragraph_ref: 'P3', doc_id: 'd3', para_id: 'p1', title: '历史项目', text: '过去的项目有40人参与。', paragraph_complete: true, parent_complete: true },
    { paragraph_ref: 'P4', doc_id: 'd4', para_id: 'p1', title: '空原段', text: '', paragraph_complete: true, parent_complete: true },
  ], source_metadata: { P1: { source_credibility: 0.9, raw_publish_time: 'N天前' } } };
}
const argument = () => ({ 时间: '', 主体: '管理方', 地点: '', 内容: '受理并不等于批准', 影响: '', 数值口径: '' });
const selected = (p = 'P1') => ({ p, roles: ['P', 'L'], fact_des: '申请已受理，批准尚未完成。', argument: argument(), entities: ['管理方'] });
const finalArgs = () => ({ conclusion: '公开信息只确认申请已受理，尚不能确认最终原因。', verdict_label: 'partial_support', selected: [selected()], supporting: ['P1'], conflicting: [], basis: '程序状态仍有限定。' });
function fetched() { const backend = createEvidenceBackend(fixture()); backend.search({ query: '申请 批准 施工 拒绝 40', limit: 10 }); backend.read({ refs: ['P1', 'P2', 'P3'] }); return backend; }

test('only four business tools are registered, with no quote/coordinate inputs', () => {
  const definitions = toolDefinitions(createEvidenceBackend(fixture()));
  assert.deepEqual(definitions.map(row => row.name), TOOL_NAMES);
  assert.ok(definitions.every(row => row.isConcurrencySafe() === false));
  const wire = JSON.stringify(definitions.map(row => row.parameters));
  assert.ok(!wire.includes('char_start')); assert.ok(!wire.includes('"quote"'));
});
test('search returns exact contiguous codepoint views and does not expose bodies by default', () => {
  const raw = fixture(); raw.catalog[0].text = '😀 '.repeat(70) + '申请已经提交，尚未完成批准。' + ' 后文未知'.repeat(40);
  const backend = createEvidenceBackend(raw), result = backend.search({ query: '申请', limit: 1 }).results[0];
  assert.equal(result.p, 'P1'); assert.equal(result.text, Array.from(raw.catalog[0].text).slice(result.start_cp, result.end_cp).join(''));
  assert.ok(Array.from(result.text).length <= 80); assert.equal(result.before_unknown, true); assert.equal(result.after_unknown, true);
});
test('search missing term has zero results, no fallback to unrelated paragraphs', () => {
  const backend = createEvidenceBackend(fixture()), result = backend.search({ query: '火星轨道' });
  assert.equal(result.returned, 0); assert.equal(result.unseen_context, 'unknown_not_irrelevant');
});
test('query/title relevance does not prove facts or use credibility filtering', () => {
  const raw = fixture(); raw.source_metadata.P2 = { source_credibility: 0.01 };
  const backend = createEvidenceBackend(raw), result = backend.search({ query: '施工方 拒绝' });
  assert.ok(result.results.some(row => row.p === 'P2')); assert.equal(result.semantic_verified, false);
  assert.ok(!JSON.stringify(result).includes('source_credibility'));
});
test('read requires search-exposed refs, returns full single-copy original blocks', () => {
  const backend = createEvidenceBackend(fixture()); assert.throws(() => backend.read({ refs: ['P1'] }), /unknown_or_duplicate/);
  backend.search({ query: '申请' }); const result = backend.read({ refs: ['P1'] });
  assert.deepEqual(result.paragraphs[0].blocks, [['S1', fixture().catalog[0].text]]);
  assert.equal(JSON.stringify(result).split(fixture().catalog[0].text).length, 2);
  assert.equal(result.model_comprehension_verified, false);
});
test('same body at different original identities is retained', () => {
  const raw = fixture(); raw.catalog[1].text = raw.catalog[0].text;
  const backend = createEvidenceBackend(raw); backend.search({ query: '申请', limit: 10 });
  assert.deepEqual(backend.read({ refs: ['P1', 'P2'] }).paragraphs.map(row => row.p), ['P1', 'P2']);
});
test('duplicate original IDs and duplicate aliases are rejected independently', () => {
  const raw = fixture(); raw.catalog[1].doc_id = 'd1'; assert.throws(() => createEvidenceBackend(raw), /unique_original/);
  const other = fixture(); other.catalog[1].paragraph_ref = 'P1'; assert.throws(() => createEvidenceBackend(other), /unique_source_ref/);
});
test('incomplete source catalog and changed source hash fail closed', () => {
  const raw = fixture(); raw.catalog[0].parent_complete = false; assert.throws(() => createEvidenceBackend(raw), /full_original/);
  const other = fixture(); other.source_metadata.P1.source_text_sha256_utf8 = sha('changed'); assert.throws(() => createEvidenceBackend(other), /sha256_mismatch/);
});
test('valid submission uses original IDs and entire original quote, receipt remains small', () => {
  const backend = fetched(), receipt = backend.submit(finalArgs()), artifact = backend.snapshot().submission;
  assert.equal(receipt.status, 'submitted_unverified'); assert.equal(receipt.publishable, false);
  assert.equal(artifact.evidence[0].quote, fixture().catalog[0].text);
  assert.equal(artifact.evidence[0].doc_id, 'd1'); assert.equal(artifact.evidence[0].para_id, 'p1');
  assert.equal(artifact.evidence[0].source_credibility, 0.9); assert.equal(artifact.evidence[0].raw_publish_time, 'N天前');
  assert.equal(artifact.conclusion.supporting_facts[0].fact_id, 'fact_001');
  assert.equal(artifact.quality.semantic_verified, false); assert.equal(artifact.quality.independent_ab_performed, false);
  assert.ok(!JSON.stringify(receipt).includes(fixture().catalog[0].text));
});
test('submission cannot select an unread or invented ref', () => {
  const backend = createEvidenceBackend(fixture()); backend.search({ query: '申请' });
  assert.throws(() => backend.submit(finalArgs()), /unknown_or_duplicate/);
  const other = fetched(), args = finalArgs(); args.selected[0].p = 'P999'; assert.throws(() => other.submit(args), /unknown_or_duplicate/);
});
test('one source only once; duplicate or unknown roles rejected', () => {
  for (const roles of [['D', 'D'], ['background'], []]) { const backend = fetched(), args = finalArgs(); args.selected[0].roles = roles; assert.throws(() => backend.submit(args), /invalid_proposed_roles/); }
  const backend = fetched(), args = finalArgs(); args.selected.push(selected()); assert.throws(() => backend.submit(args), /unknown_or_duplicate/);
});
test('all support/conflict refs must be selected, conflict code uses official labels', () => {
  const backend = fetched(), args = finalArgs(); args.conflicting = [{ p: 'P2', resolution_type: 'Other' }]; assert.throws(() => backend.submit(args), /unknown_or_duplicate/);
  args.conflicting = [{ p: 'P1', resolution_type: 'newest' }]; assert.throws(() => backend.submit(args), /invalid_resolution/);
});
test('no fabricated support is needed for a valid undetermined result', () => {
  const backend = createEvidenceBackend(fixture());
  backend.submit({ conclusion: '公开信息不足，无法判定。', verdict_label: 'undetermined', selected: [], supporting: [], conflicting: [], basis: '' });
  assert.equal(backend.snapshot().submission.evidence.length, 0);
});
test('asserted conclusion cannot have empty proposed support', () => {
  const backend = fetched(), args = finalArgs(); args.supporting = []; assert.throws(() => backend.submit(args), /requires_proposed_support/);
});
test('model-written quote/offset/source fields are rejected rather than repaired', () => {
  for (const extra of [{ quote: 'invented' }, { doc_id: 'invented' }, { char_start: 0 }]) {
    const backend = fetched(), args = finalArgs(); Object.assign(args.selected[0], extra); assert.throws(() => backend.submit(args), /invalid_tool_arguments/);
  }
});
test('time must be literal or unknown and new numeric literals are rejected', () => {
  const backend = fetched(), args = finalArgs(); args.selected[0].argument.时间 = '今年'; assert.throws(() => backend.submit(args), /time_must_be_literal/);
  args.selected[0].argument.时间 = ''; args.selected[0].fact_des = '有99人等待批准'; assert.throws(() => backend.submit(args), /annotation_number/);
  args.selected[0] = selected(); args.conclusion = '共有99人获批'; assert.throws(() => backend.submit(args), /conclusion_number/);
});
test('conclusion numeric guard uses cited sources, not merely any fetched parent', () => {
  const backend = fetched(), args = finalArgs(); args.conclusion = '40人涉及这次批准'; assert.throws(() => backend.submit(args), /conclusion_number/);
});
test('empty parent remains representable but cannot become evidence', () => {
  const raw = fixture(); raw.catalog[3].title = '申请'; const backend = createEvidenceBackend(raw); backend.search({ query: '申请', limit: 10 });
  assert.deepEqual(backend.read({ refs: ['P4'] }).paragraphs[0].blocks, []);
  const args = finalArgs(); args.selected[0].p = 'P4'; args.supporting = ['P4']; assert.throws(() => backend.submit(args), /empty_parent/);
});
test('aborted tools record safe failure, no source state mutation', () => {
  const backend = createEvidenceBackend(fixture()), controller = new AbortController(); controller.abort();
  assert.throws(() => backend.search({ query: '申请' }, controller.signal), /tool_aborted/);
  assert.deepEqual(backend.snapshot().exposed_refs, []); assert.equal(backend.snapshot().trace[0].code, 'tool_aborted');
});
test('successful submit is final and snapshot mutation cannot change source binding', () => {
  const backend = fetched(); backend.submit(finalArgs()); const snapshot = backend.snapshot(); snapshot.submission.evidence[0].quote = 'changed';
  assert.equal(backend.snapshot().submission.evidence[0].quote, fixture().catalog[0].text);
  assert.throws(() => backend.search({ query: '申请' }), /already_submitted/);
});
test('plugin registration verifies bytes, writes only a separate exclusive artifact and never stdout', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-tools-synthetic-'));
  try {
    const casePath = join(directory, 'source-case.json'), submissionPath = join(directory, 'submission.json'), bytes = JSON.stringify(fixture());
    await writeFile(casePath, bytes); const registered = [], sections = [];
    const ctx = { tools: { register: definition => registered.push(definition) }, systemPrompt: { section: section => sections.push(section) } };
    await assert.rejects(apply(ctx, { casePath, caseSha256: sha('wrong') }), /bytes_sha256_mismatch/);
    const result = await apply(ctx, { casePath, caseSha256: sha(bytes), submissionPath }); assert.equal(result, undefined); assert.equal(registered.length, 4); assert.equal(sections.length, 1);
    const signal = new AbortController().signal;
    let concluded = false;
    await registered.find(row => row.name === 'search_sources').execute({ query: '申请' }, { signal });
    await registered.find(row => row.name === 'read_sources').execute({ refs: ['P1'] }, { signal });
    await registered.find(row => row.name === 'submit_evidence').execute(finalArgs(), { signal, concludeTurn() { concluded = true; } });
    const artifact = JSON.parse(await readFile(submissionPath, 'utf8')); assert.equal(artifact.evidence[0].quote, fixture().catalog[0].text);
    assert.equal(artifact.quality.publishable, false);
    assert.equal(concluded, true);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
test('actual official Cordis accepts plugin apply lifecycle and supported schemas without model services', async () => {
  const runtime = resolve('.runtime/dsh-official-r1/node_modules/@deepseek-ai');
  const { Context } = await import(pathToFileURL(join(runtime, 'cordis/lib/index.js')).href);
  const { assertSupportedJsonSchema } = await import(pathToFileURL(join(runtime, 'dsh-tools/lib/index.js')).href);
  const directory = await mkdtemp(join(tmpdir(), 'dsh-cordis-synthetic-'));
  const ctx = new Context(), registered = [];
  ctx.provide('tools', { register(definition) { assertSupportedJsonSchema(definition.parameters); assertSupportedJsonSchema(definition.output.schema); registered.push(definition); } });
  ctx.provide('systemPrompt', { section() {} });
  try {
    const casePath = join(directory, 'source-case.json'), bytes = JSON.stringify(fixture()); await writeFile(casePath, bytes);
    const plugin = await import('./evidence_tools.mjs');
    const fiber = ctx.plugin(plugin, { casePath, caseSha256: sha(bytes) }); await fiber.await();
    assert.equal(registered.length, 4); await fiber.dispose();
  } finally { await ctx.fiber.dispose(); await rm(directory, { recursive: true, force: true }); }
});
const packetBudget = () => ({ known_episode_tokens: 1000, packet_capacity_tokens: 13904, whole_limit: 20000,
  reserved_final_output_tokens: 3072, safety_tokens: 512, wrapper_reserve_tokens: 512, native_preflight_still_required: true });
test('complete candidate packet preserves every original full parent exactly once with no count cap', async () => {
  const raw = fixture(); let counted;
  const backend = createEvidenceBackend(raw, { getPacketBudget: packetBudget, countPacketTokens(text) { counted = text; return 7114; } });
  const result = await backend.packet({});
  assert.equal(result.status, 'candidate_packet_complete'); assert.deepEqual(result.paragraphs, raw.catalog.map(row => [row.paragraph_ref, row.title, row.text]));
  assert.equal(counted, JSON.stringify({ columns: result.columns, paragraphs: result.paragraphs }));
  assert.equal(result.minimal_packet_tokens, 7114); assert.equal(result.candidate_scope_only, true); assert.equal(result.unprovided_context_unknown, true);
  assert.equal(result.model_context_delivery_verified, false); assert.deepEqual(backend.snapshot().local_fetched_full_refs, ['P1', 'P2', 'P3', 'P4']);
  assert.equal(backend.snapshot().trace[0].official_tokenizer_result_tokens, 7114);
});
test('packet too large returns no text and cannot establish exposed/read refs', async () => {
  const backend = createEvidenceBackend(fixture(), { getPacketBudget: () => ({ ...packetBudget(), packet_capacity_tokens: 10 }), countPacketTokens: () => 11 });
  const result = await backend.packet({}); assert.equal(result.status, 'packet_too_large'); assert.ok(!Object.hasOwn(result, 'paragraphs'));
  assert.deepEqual(backend.snapshot().exposed_refs, []); assert.deepEqual(backend.snapshot().local_fetched_full_refs, []);
  assert.throws(() => backend.submit(finalArgs()), /unknown_or_duplicate/);
});
test('packet accepts exact dynamic boundary and never silently chooses a fitting subset', async () => {
  const backend = createEvidenceBackend(fixture(), { getPacketBudget: () => ({ ...packetBudget(), packet_capacity_tokens: 50 }), countPacketTokens: () => 50 });
  assert.equal((await backend.packet({})).paragraphs.length, fixture().catalog.length);
  let calls = 0;
  const other = createEvidenceBackend(fixture(), { getPacketBudget: packetBudget, countPacketTokens: () => { calls++; return 20; } });
  await other.packet({}); const repeated = await other.packet({});
  assert.equal(calls, 1); assert.equal(repeated.status, 'packet_already_returned'); assert.ok(!Object.hasOwn(repeated, 'paragraphs'));
});
test('unknown/bool/negative/fractional packet costs or missing budget fail closed with backward compatible constructor', async () => {
  assert.equal((await createEvidenceBackend(fixture()).packet({})).status, 'packet_budget_unknown');
  for (const measured of [undefined, null, true, -1, 1.5, NaN, Infinity]) {
    const backend = createEvidenceBackend(fixture(), { getPacketBudget: packetBudget, countPacketTokens: () => measured });
    assert.equal((await backend.packet({})).status, 'packet_counter_unavailable'); assert.deepEqual(backend.snapshot().local_fetched_full_refs, []);
  }
  const backend = createEvidenceBackend(fixture(), { getPacketBudget: packetBudget, countPacketTokens: () => { throw new Error('synthetic'); } });
  assert.equal((await backend.packet({})).status, 'packet_counter_unavailable');
});
test('cancellation during packet preflight never exposes parents', async () => {
  const controller = new AbortController();
  const backend = createEvidenceBackend(fixture(), { getPacketBudget: packetBudget, countPacketTokens: () => { controller.abort(); return 1; } });
  await assert.rejects(backend.packet({}, controller.signal), /tool_aborted/); assert.deepEqual(backend.snapshot().local_fetched_full_refs, []);
});
test('same exact search window and full returned parents become compact references', async () => {
  const backend = createEvidenceBackend(fixture()); const first = backend.search({ query: '申请' });
  assert.equal(typeof first.results[0].text, 'string');
  assert.deepEqual(backend.search({ query: '申请' }).results[0], { p: 'P1', reused: 'same_exact_window_previously_returned' });
  backend.read({ refs: ['P1'] }); assert.deepEqual(backend.search({ query: '批准' }).results[0], { p: 'P1', reused: 'full_parent_previously_returned' });
  const all = createEvidenceBackend(fixture(), { getPacketBudget: packetBudget, countPacketTokens: () => 1 }); await all.packet({});
  assert.ok(all.search({ query: '施工' }).results.every(row => row.reused === 'full_parent_previously_returned' && !Object.hasOwn(row, 'text')));
});
test('new query positions remain visible, exact-body mirrors are not merged across P identities', () => {
  const raw = fixture(); raw.catalog[0].text = '申请' + '空白'.repeat(90) + '批准'; raw.catalog[1].text = raw.catalog[0].text;
  const backend = createEvidenceBackend(raw), first = backend.search({ query: '申请', limit: 10 }), second = backend.search({ query: '批准', limit: 10 });
  assert.equal(first.results.length, 2); assert.equal(second.results.length, 2);
  assert.notEqual(first.results[0].start_cp, second.results[0].start_cp); assert.ok(second.results.every(row => typeof row.text === 'string'));
});
test('settled packet budget counts all prior usage, pending/unknown/halted journals do not produce capacity', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-packet-budget-synthetic-'));
  const journalPath = join(directory, 'journal.jsonl');
  const config = { journalPath, episodeId: 'synthetic', priorKnownTokens: 0, wholeLimit: 20000, maxOutputTokens: 3072, safetyTokens: 512 };
  const rows = [
    { status: 'admitted', request_sequence: 1, known_episode_tokens_before: 0 },
    { status: 'request_started', request_sequence: 1 },
    { status: 'usage_settled', request_sequence: 1, provider_usage: { total_tokens: 1430 }, known_episode_tokens_after: 1430, unknown_usage_requests: 0, halted: false, response_completed: true },
  ].map((row, i) => ({ episode_id: 'synthetic', sequence: i + 1, ...row }));
  const save = async value => writeFile(journalPath, value.map(row => JSON.stringify(row)).join('\n') + '\n');
  try {
    await save(rows); const result = await readSettledPacketBudget(config);
    assert.equal(result.known_episode_tokens, 1430); assert.equal(result.packet_capacity_tokens, 13044);
    await save(rows.slice(0, 2)); assert.equal(await readSettledPacketBudget(config), null);
    for (const mutation of [{ status: 'usage_unknown' }, { halted: true }, { unknown_usage_requests: 1 }, { provider_usage: { total_tokens: true } }, { known_episode_tokens_after: 0 }, { response_completed: false }]) {
      await save([...rows.slice(0, 2), { ...rows[2], ...mutation }]); assert.equal(await readSettledPacketBudget(config), null);
    }
    await save([]); assert.equal(await readSettledPacketBudget(config), null);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
test('complete context is one exact deterministic source string with full question and doc-shared titles', () => {
  const raw = fixture(); raw.question.question += ' 保留原题限定。';
  raw.catalog[1].doc_id = raw.catalog[0].doc_id; raw.catalog[1].para_id = 'p2'; raw.catalog[1].title = raw.catalog[0].title;
  raw.catalog[0].text = '\n 😀\t原文 {{unknown_variable}}\r\n' + '完整内容。'.repeat(1000);
  const text = buildCompleteContext(raw), parsed = JSON.parse(text);
  assert.equal(parsed.question, raw.question.question); assert.equal(parsed.titles.length, 3);
  assert.deepEqual(parsed.paragraphs.map(row => row[0]), raw.catalog.map(row => row.paragraph_ref));
  assert.deepEqual(parsed.paragraphs.map(row => row[2]), raw.catalog.map(row => row.text));
  assert.equal(parsed.paragraphs[0][1], parsed.paragraphs[1][1]); assert.equal(text, buildCompleteContext(raw));
  assert.equal(parsed.candidate_scope_only, true); assert.equal(parsed.outside_context_unknown, true);
  assert.ok(!text.includes('source_credibility'));
});
test('inconsistent titles within one original doc are rejected rather than silently overwritten', () => {
  const raw = fixture(); raw.catalog[1].doc_id = raw.catalog[0].doc_id; raw.catalog[1].para_id = 'p2';
  assert.throws(() => buildCompleteContext(raw), /inconsistent_original_document_title/);
});
test('full-context mode prepares refs but does not claim a local read or model delivery', async () => {
  const backend = createEvidenceBackend(fixture(), { contextMode: 'full-candidate-context' });
  assert.deepEqual(toolDefinitions(backend).map(row => row.name), ['submit_evidence']);
  const snapshot = backend.snapshot(); assert.deepEqual(snapshot.source_context_prepared_refs, ['P1', 'P2', 'P3', 'P4']);
  assert.deepEqual(snapshot.local_fetched_full_refs, []); assert.equal(snapshot.model_context_delivery_verified, false);
  assert.equal(snapshot.model_comprehension_verified, false);
  assert.throws(() => backend.search({ query: '申请' }), /tool_not_available/);
  assert.throws(() => backend.read({ refs: ['P1'] }), /tool_not_available/);
  await assert.rejects(backend.packet({}), /tool_not_available/);
  backend.submit(finalArgs()); const artifact = backend.snapshot().submission;
  assert.equal(artifact.evidence[0].quote, fixture().catalog[0].text); assert.equal(artifact.scope.source_context_prepared, true);
  assert.equal(artifact.scope.model_context_delivery_verified, false); assert.deepEqual(artifact.scope.local_fetched_full_refs, []);
  assert.equal(artifact.quality.publishable, false); assert.equal(artifact.quality.semantic_verified, false);
});
test('full-context plugin registers only submit and literal source-data section with interpolation disabled', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-full-context-synthetic-'));
  try {
    const raw = fixture(); raw.catalog[0].text += ' {{no_substitution}}';
    const casePath = join(directory, 'raw.json'), bytes = JSON.stringify(raw); await writeFile(casePath, bytes);
    const registered = [], sections = [];
    await apply({ tools: { register: value => registered.push(value) }, systemPrompt: { section: value => sections.push(value) } },
      { casePath, caseSha256: sha(bytes), contextMode: 'full-candidate-context' });
    assert.deepEqual(registered.map(row => row.name), ['submit_evidence']);
    const data = sections.find(row => row.name === 'trust-judge-source-data');
    assert.equal(data.interpolate, false); assert.equal(data.text, buildCompleteContext(raw));
    const { renderPrompt } = await import(pathToFileURL(resolve('.runtime/dsh-official-r1/node_modules/@deepseek-ai/dsh-system-prompt/lib/index.js')).href);
    assert.equal(renderPrompt({ sections: [data], variables: {} }), data.text);
    assert.ok(!sections.find(row => row.name === 'trust-judge-evidence').text.includes('First call read_candidate_packet'));
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('compact full-context policy replaces duplicate instructions while retaining every original source and exact submit schema', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-full-context-compact-'));
  try {
    const raw = fixture(); raw.catalog[0].text = '\n😀 {{literal}}\r\n' + raw.catalog[0].text.repeat(100);
    const casePath = join(directory, 'raw.json'), bytes = JSON.stringify(raw); await writeFile(casePath, bytes);
    const load = async selectionPolicy => {
      const tools = [], sections = [];
      await apply({ tools: { register: value => tools.push(value) }, systemPrompt: { section: value => sections.push(value) } },
        { casePath, caseSha256: sha(bytes), contextMode: 'full-candidate-context', selectionPolicy });
      return { tools, sections };
    };
    const compact = await load('source-local-v3-compact'), ordinary = await load('source-local-v3');
    assert.equal(compact.sections.length, 2); assert.equal(ordinary.sections.length, 3);
    assert.deepEqual(compact.sections[0], selectionPolicySection('source-local-v3-compact'));
    assert.equal(compact.sections[1].order, 1); assert.equal(compact.sections[1].interpolate, false);
    assert.equal(compact.sections[1].text, buildCompleteContext(raw));
    assert.equal(compact.sections[1].text, ordinary.sections[1].text);
    assert.deepEqual(JSON.parse(compact.sections[1].text).paragraphs.map(row => row[2]), raw.catalog.map(row => row.text));
    const schemas = tools => tools.map(({ name, description, parameters, output }) => ({ name, description, parameters, schema: output.schema }));
    assert.deepEqual(schemas(compact.tools), schemas(ordinary.tools));
    let concluded = 0;
    await compact.tools[0].execute(finalArgs(), { signal: new AbortController().signal, concludeTurn() { concluded++; } });
    assert.equal(concluded, 1);
    await assert.rejects(apply({ tools: { register() { assert.fail('must not register'); } }, systemPrompt: { section() { assert.fail('must not add sections'); } } },
      { casePath, caseSha256: sha(bytes), contextMode: 'tools', selectionPolicy: 'source-local-v3-compact' }), /compact_policy_requires_full_context/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
test('terminal submit needs official concludeTurn and never concludes when persistence fails', async () => {
  const backend = fetched(), definition = toolDefinitions(backend).find(row => row.name === 'submit_evidence'), signal = new AbortController().signal;
  await assert.rejects(definition.execute(finalArgs(), { signal }), /official_dsh_conclude_turn_required/);
  assert.equal(backend.snapshot().submission, null);
  let concluded = false;
  const failing = toolDefinitions(backend, { onSubmit: () => { throw new Error('synthetic_disk_error'); } }).find(row => row.name === 'submit_evidence');
  await assert.rejects(failing.execute(finalArgs(), { signal, concludeTurn() { concluded = true; } }), /synthetic_disk_error/);
  assert.equal(concluded, false);
});

test('exact question designations identify the subject without borrowing question counts as facts', () => {
  const raw = fixture(); raw.question.question = 'ZX482事故是否造成157人伤亡，QX91部件是否受损？';
  const args = finalArgs(); args.verdict_label = 'undetermined'; args.conclusion = 'ZX482事故原因尚未确定，QX91部件情况尚未确定。';
  const backend = createEvidenceBackend(raw, { contextMode: 'full-candidate-context' }); backend.submit(args);
  assert.deepEqual(backend.snapshot().submission.quality.question_designations_not_numeric_claims, ['ZX482', 'QX91']);
  for (const text of ['ZX482事故造成157人伤亡。', 'ZX482事故发生于2026年。', 'ZX483事故原因未知。', 'ZX482事故中482人受伤。']) {
    const next = createEvidenceBackend(raw, { contextMode: 'full-candidate-context' });
    assert.throws(() => next.submit({ ...args, conclusion: text }), /conclusion_number_not_in_cited_sources/);
  }
});

test('isolated submit ends once, retains audit, and never silently fabricates annotations', async () => {
  const { isolateAnnotations } = await import('./annotation_isolation.mjs');
  const raw = fixture(), backend = createEvidenceBackend(raw, { contextMode: 'full-candidate-context' });
  const args = finalArgs(); args.selected[0].argument.时间 = '未知原段没有这个时间';
  args.selected.push({ ...selected('P2'), roles: ['unknown'], fact_des: '' });
  let audit, persisted, ended = 0;
  const definition = toolDefinitions(backend, {
    prepareSubmission: original => isolateAnnotations(original, raw), onAnnotationAudit: value => { audit = value; },
    onSubmit: value => { persisted = value; },
  }).find(row => row.name === 'submit_evidence');
  const result = await definition.execute(args, { signal: new AbortController().signal, concludeTurn() { ended++; } });
  assert.equal(ended, 1); assert.equal(result.evidence_count, 1); assert.equal(result.annotation_quarantine_count, 1);
  assert.equal(persisted.evidence[0].argument.时间, ''); assert.equal(persisted.quality.optional_time_cleared_count, 1);
  assert.equal(persisted.quality.publishable, false); assert.equal(persisted.quality.semantic_verified, false);
  assert.equal(audit.original_args.selected[0].argument.时间, '未知原段没有这个时间');
  assert.deepEqual(persisted.scope.original_proposed_refs, ['P1', 'P2']);
  assert.equal(args.selected[0].argument.时间, '未知原段没有这个时间');
});

test('quarantined relations and unread sources cannot be bypassed through submission preparation', async () => {
  const { isolateAnnotations } = await import('./annotation_isolation.mjs');
  const raw = fixture(), signal = new AbortController().signal;
  for (const linkedQuarantine of [true, false]) {
    const backend = createEvidenceBackend(raw, { contextMode: linkedQuarantine ? 'full-candidate-context' : 'tools' });
    const args = finalArgs(); if (linkedQuarantine) args.selected[0].fact_des = '';
    let saved = false, ended = false;
    const tool = toolDefinitions(backend, { prepareSubmission: original => isolateAnnotations(original, raw), onSubmit() { saved = true; } }).find(row => row.name === 'submit_evidence');
    await assert.rejects(tool.execute(args, { signal, concludeTurn() { ended = true; } }), linkedQuarantine ? /referenced_annotation_quarantined/ : /unknown_or_duplicate/);
    assert.equal(saved, false); assert.equal(ended, false); assert.equal(backend.snapshot().submission, null);
  }
});
