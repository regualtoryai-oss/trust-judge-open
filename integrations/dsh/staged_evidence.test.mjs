import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { apply, sourceWindow, buildStagedCandidateContext, createStagedEvidenceBackend, bindStagedAdmission, getStagedPresence } from './staged_evidence.mjs';

const sha = value => createHash('sha256').update(value).digest('hex');
const approved = () => ({ admitted: true, whole_limit: 20000, preflight_total_tokens: 5000, native_next_request_still_required: true });
const raw = () => ({ question: { question_id: 'q_synthetic', question: '延期是施工方拒绝，还是管理方尚未批准？' }, catalog: [
  { paragraph_ref: 'P1', doc_id: 'd1', para_id: 'p1', title: '完整原标题 {{literal}}', text: '\n🦊登记记录显示：申请已受理，尚未批准。\r\n', paragraph_complete: true, parent_complete: true },
  { paragraph_ref: 'P2', doc_id: 'd2', para_id: 'p7', title: '施工方声明', text: '施工方称愿意开工，并否认拒绝施工。', paragraph_complete: true, parent_complete: true },
  { paragraph_ref: 'P3', doc_id: 'd3', para_id: 'p2', title: '背景', text: '往年项目已结束。', paragraph_complete: true, parent_complete: true },
], source_metadata: { P1: { source_credibility: 0.9, raw_publish_time: 'N天前' } } });
const argument = () => ({ 时间: '', 主体: '管理方', 地点: '', 内容: '申请受理与批准的程序状态', 影响: '', 数值口径: '' });
const args = p => ({ conclusion: '公开记录显示尚未批准，不能由此断言延期责任。', verdict_label: 'partial_support',
  selected: [{ p, roles: ['P', 'L'], fact_des: '登记记录显示已受理，尚未批准。', argument: argument(), entities: ['管理方'] }],
  supporting: [p], conflicting: [], basis: '受理不等于批准。' });
const signal = () => new AbortController().signal;

test('short views remain exact codepoint windows and can locate a late original-question term', () => {
  const text = '🦊开场'.repeat(50) + '管理方尚未批准' + '结束'.repeat(50);
  const view = sourceWindow(text, raw().question.question, 30);
  assert.equal(view.text, Array.from(text).slice(view.start_cp, view.end_cp).join(''));
  assert.equal(view.end_cp - view.start_cp, 30); assert.ok(view.text.includes('尚未批准'));
  assert.equal(view.before_unknown, true); assert.equal(view.after_unknown, true);
  assert.throws(() => sourceWindow(text, '原题', true), /preview_configuration/);
});

test('candidate section keeps full question, literal titles and all original mirror identities', () => {
  const input = raw(); input.question.question += ' 原题完整限定。'; input.catalog[2].text = input.catalog[0].text;
  const body = buildStagedCandidateContext(input, { previewCodepoints: 10 }), parsed = JSON.parse(body);
  assert.equal(parsed.question, input.question.question); assert.equal(parsed.titles[0][1], input.catalog[0].title);
  assert.deepEqual(parsed.candidate_previews.map(row => row[0]), ['P1', 'P2', 'P3']);
  for (const row of parsed.candidate_previews) assert.equal(row[2], Array.from(input.catalog.find(p => p.paragraph_ref === row[0]).text).slice(row[3], row[4]).join(''));
  assert.equal(parsed.preview_omitted_context, 'unknown_not_irrelevant'); assert.ok(body.includes('{{literal}}'));
  assert.ok(!body.includes('source_credibility')); assert.ok(!body.includes('N天前'));
});

test('raw hash, full parent and original title integrity reuse existing validation', () => {
  const input = raw(); input.source_metadata.P1.source_text_sha256_utf8 = sha('changed');
  assert.throws(() => createStagedEvidenceBackend(input), /source_text_sha256_mismatch/);
  const incomplete = raw(); incomplete.catalog[0].parent_complete = false;
  assert.throws(() => createStagedEvidenceBackend(incomplete), /full_original_parent/);
  const duplicate = raw(); duplicate.catalog[1].doc_id = 'd1'; duplicate.catalog[1].title = '不同原标题';
  assert.throws(() => createStagedEvidenceBackend(duplicate), /inconsistent_original_document_title/);
});

test('full read admission receives complete selected parents and cannot silently shrink a rejected selection', async () => {
  let request;
  const backend = createStagedEvidenceBackend(raw(), { admitRead: value => { request = value; return { admitted: false, code: 'whole_episode_budget_exceeded' }; } });
  await assert.rejects(backend.read({ refs: ['P1', 'P2'] }), /whole_episode_budget_exceeded/);
  assert.deepEqual(request.selected_refs, ['P1', 'P2']); assert.deepEqual(request.full_read_result.paragraphs.map(row => row[2]), raw().catalog.slice(0, 2).map(row => row.text));
  assert.equal(backend.snapshot().stage, 'select'); assert.deepEqual(backend.snapshot().selected_full_read_refs, []);
  assert.equal(backend.presence().full_read_result_json, null); assert.equal(backend.presence().phase, 'preview');
  assert.throws(() => backend.submit(args('P1')), /full_original_sources_required/);
});

test('unknown, duplicate, missing native policy and incomplete admission receipts fail closed', async () => {
  const backend = createStagedEvidenceBackend(raw(), { admitRead: approved });
  for (const refs of [['P9'], ['P1', 'P1']]) await assert.rejects(backend.read({ refs }), /unknown_or_duplicate/);
  await assert.rejects(createStagedEvidenceBackend(raw()).read({ refs: ['P1'] }), /native_staged_admission_unavailable/);
  for (const receipt of [null, { ...approved(), native_next_request_still_required: false }, { ...approved(), preflight_total_tokens: 20001 }, { ...approved(), preflight_total_tokens: true }]) {
    const invalid = createStagedEvidenceBackend(raw(), { admitRead: () => receipt });
    await assert.rejects(invalid.read({ refs: ['P1'] }), /native_staged_read_not_admitted/);
    assert.equal(invalid.presence().phase, 'preview');
  }
});

test('cancellation while awaiting read admission leaves sources unread and candidate stage intact', async () => {
  const controller = new AbortController();
  const backend = createStagedEvidenceBackend(raw(), { admitRead: () => { controller.abort(); return approved(); } });
  await assert.rejects(backend.read({ refs: ['P1'] }, controller.signal), /tool_aborted/);
  assert.deepEqual(backend.snapshot().selected_full_read_refs, []); assert.equal(backend.snapshot().submission, null);
  assert.equal(backend.presence().phase, 'preview');
});

test('one admitted read removes preview data and returns every selected complete parent once only', async () => {
  const input = raw(), backend = createStagedEvidenceBackend(input, { admitRead: approved, previewCodepoints: 4 });
  const before = backend.dataContext(), result = await backend.read({ refs: ['P2', 'P1'] });
  assert.deepEqual(result.paragraphs, [['P2', input.catalog[1].title, input.catalog[1].text], ['P1', input.catalog[0].title, input.catalog[0].text]]);
  assert.equal(JSON.stringify(result).split(input.catalog[1].text).length - 1, 1);
  assert.notEqual(backend.dataContext(), before); assert.equal(Object.hasOwn(JSON.parse(backend.dataContext()), 'candidate_previews'), false);
  const presence = backend.presence(); assert.equal(presence.phase, 'final'); assert.equal(presence.preview_context, before);
  assert.equal(presence.post_read_context, backend.dataContext()); assert.equal(presence.full_read_result_json, JSON.stringify(result));
  assert.deepEqual(presence.read_refs, ['P2', 'P1']); assert.equal(result.model_context_delivery_verified, false);
  await assert.rejects(backend.read({ refs: ['P3'] }), /staged_read_once_only/);
});

test('submit may select only fully read parents and original quote IDs/offsets remain program-owned', async () => {
  const input = raw(), backend = createStagedEvidenceBackend(input, { admitRead: approved });
  await backend.read({ refs: ['P1'] });
  assert.throws(() => backend.submit(args('P2')), /unknown_or_duplicate/);
  backend.submit(args('P1')); const artifact = backend.snapshot().submission, fact = artifact.evidence[0];
  assert.equal(fact.doc_id, 'd1'); assert.equal(fact.para_id, 'p1'); assert.equal(fact.quote, input.catalog[0].text);
  assert.equal(fact.parent_end_cp, Array.from(input.catalog[0].text).length); assert.equal(fact.source_credibility, 0.9);
  assert.equal(artifact.scope.context_mode, 'staged-short-window'); assert.equal(artifact.scope.source_count, 3);
  assert.deepEqual(artifact.scope.local_fetched_full_refs, ['P1']); assert.equal(artifact.scope.read_selection_budget.native_next_request_still_required, true);
  assert.equal(artifact.scope.read_selection_budget.admission_basis, 'read_payload_estimate_negative_screen_not_native_wire');
  assert.equal(artifact.scope.read_selection_budget.future_request_fit_guaranteed, false);
  for (const key of ['semantic_verified', 'publishable', 'independent_ab_performed']) assert.equal(artifact.quality[key], false);
  assert.equal(artifact.scope.model_context_delivery_verified, false);
  const snapshot = backend.snapshot(); snapshot.submission.evidence[0].quote = 'mutated';
  assert.equal(backend.snapshot().submission.evidence[0].quote, input.catalog[0].text);
});

test('no source is fabricated to make an admitted empty-read undetermined proposal succeed', async () => {
  const backend = createStagedEvidenceBackend(raw(), { admitRead: approved });
  await backend.read({ refs: [] }); backend.submit({ conclusion: '提供范围内无法判定。', verdict_label: 'undetermined', selected: [], supporting: [], conflicting: [], basis: '' });
  assert.deepEqual(backend.snapshot().submission.evidence, []); assert.equal(backend.snapshot().submission.quality.publishable, false);
});

test('registry requires unique native policy binding and never claims a staged state before business apply', () => {
  const id = randomUUID(), unbind = bindStagedAdmission(id, approved);
  assert.equal(getStagedPresence(id), null); assert.throws(() => bindStagedAdmission(id, approved), /unique_staged_admission/);
  unbind(); assert.equal(getStagedPresence(id), null); bindStagedAdmission(id, approved)();
});

test('Cordis adapter switches sole native business tool, uses literal dynamic section, and concludes only after exclusive persistence', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-staged-synthetic-')), id = randomUUID(), unbind = bindStagedAdmission(id, approved);
  try {
    const bytes = JSON.stringify(raw()), casePath = join(directory, 'source-case.json'), submissionPath = join(directory, 'submission.json');
    await writeFile(casePath, bytes); const registered = new Map(), sections = [], effects = [];
    const ctx = { trustJudgeStagedAdmission: { episodeId: id }, tools: { register(definition) { assert.ok(!registered.has(definition.name)); registered.set(definition.name, definition); return () => registered.delete(definition.name); } },
      systemPrompt: { section(value) { sections.push(value); } }, effect(value) { effects.push(value); } };
    const result = await apply(ctx, { casePath, caseSha256: sha(bytes), submissionPath, episodeId: id, previewCodepoints: 8 });
    assert.equal(result, undefined); assert.deepEqual([...registered.keys()], ['read_selected_sources']);
    const data = sections.find(row => row.name === 'trust-judge-staged-data'); assert.equal(data.interpolate, false);
    assert.ok(data.text().includes('candidate_previews')); assert.ok(sections.some(row => row.name === 'trust-judge-source-local-v2'));
    await registered.get('read_selected_sources').execute({ refs: ['P1'] }, { signal: signal() });
    assert.deepEqual([...registered.keys()], ['submit_evidence']); assert.equal(Object.hasOwn(JSON.parse(data.text()), 'candidate_previews'), false);
    assert.equal(getStagedPresence(id).phase, 'final'); let concluded = 0;
    await registered.get('submit_evidence').execute(args('P1'), { signal: signal(), concludeTurn() { concluded++; } });
    assert.equal(concluded, 1); assert.equal(JSON.parse(await readFile(submissionPath, 'utf8')).scope.context_mode, 'staged-short-window');
    effects.forEach(effect => effect()()); assert.equal(getStagedPresence(id), null);
  } finally { unbind(); await rm(directory, { recursive: true, force: true }); }
});

test('actual official schema validator accepts staged business tool definitions without model or network services', async () => {
  const { assertSupportedJsonSchema } = await import(pathToFileURL(resolve('.runtime/dsh-official-r1/node_modules/@deepseek-ai/dsh-tools/lib/index.js')).href);
  const backend = createStagedEvidenceBackend(raw(), { admitRead: approved });
  const definition = backend.submitDefinition(); assertSupportedJsonSchema(definition.parameters); assertSupportedJsonSchema(definition.output.schema);
  assert.ok(!JSON.stringify(definition.parameters).includes('"quote"')); assert.ok(!JSON.stringify(definition.parameters).includes('"doc_id"'));
});

test('missing official concludeTurn and persistence failure never conclude or pretend to publish', async () => {
  const backend = createStagedEvidenceBackend(raw(), { admitRead: approved }); await backend.read({ refs: ['P1'] });
  await assert.rejects(backend.submitDefinition().execute(args('P1'), { signal: signal() }), /official_dsh_conclude_turn_required/);
  assert.equal(backend.snapshot().submission, null); let concluded = false;
  const definition = backend.submitDefinition({ onSubmit() { throw new Error('synthetic_disk_failure'); } });
  await assert.rejects(definition.execute(args('P1'), { signal: signal(), concludeTurn() { concluded = true; } }), /synthetic_disk_failure/);
  assert.equal(concluded, false); assert.equal(backend.snapshot().submission.quality.publishable, false);
});

test('official Cordis required service orders asynchronous budget initialization before staged business', async () => {
  const { Context } = await import(pathToFileURL(resolve('.runtime/dsh-official-r1/node_modules/@deepseek-ai/cordis/lib/index.js')).href);
  const plugin = await import('./staged_evidence.mjs');
  const directory = await mkdtemp(join(tmpdir(), 'dsh-staged-cordis-order-')), id = randomUUID(), ctx = new Context(), registered = new Map(), sections = [];
  ctx.provide('tools', { register(definition) { registered.set(definition.name, definition); return () => registered.delete(definition.name); } });
  ctx.provide('systemPrompt', { section(value) { sections.push(value); } });
  let release; const barrier = new Promise(resolve => { release = resolve; });
  try {
    const bytes = JSON.stringify(raw()), casePath = join(directory, 'source-case.json'); await writeFile(casePath, bytes);
    const budgetFiber = ctx.plugin({ name: 'synthetic-async-budget', async apply(budgetCtx) {
      await barrier; const unbind = bindStagedAdmission(id, approved);
      budgetCtx.provide('trustJudgeStagedAdmission', { episodeId: id }); budgetCtx.effect(() => unbind);
    } });
    const businessFiber = ctx.plugin(plugin, { casePath, caseSha256: sha(bytes), episodeId: id });
    assert.equal(registered.size, 0); assert.equal(sections.length, 0); assert.equal(getStagedPresence(id), null);
    release(); await budgetFiber.await(); await businessFiber.await();
    assert.deepEqual([...registered.keys()], ['read_selected_sources']); assert.equal(getStagedPresence(id).phase, 'preview');
  } finally { await ctx.fiber.dispose(); await rm(directory, { recursive: true, force: true }); }
});

test('an admission service from a different episode cannot activate business state', async () => {
  const id = randomUUID();
  await assert.rejects(apply({ trustJudgeStagedAdmission: { episodeId: 'different' } }, {
    casePath: resolve('source-case.json'), caseSha256: 'a'.repeat(64), episodeId: id,
  }), /native_staged_admission_service_mismatch/);
});

test('v3 isolation clears only nonliteral optional time and quarantines an unreferenced invalid row', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-staged-v3-isolation-')), id = randomUUID(), unbind = bindStagedAdmission(id, approved);
  try {
    const input = raw(), bytes = JSON.stringify(input), casePath = join(directory, 'source-case.json');
    const submissionPath = join(directory, 'submission.json'), annotationAuditPath = join(directory, 'annotation-audit.json');
    await writeFile(casePath, bytes); const registered = new Map(), sections = [];
    const ctx = { trustJudgeStagedAdmission: { episodeId: id }, tools: { register(definition) { registered.set(definition.name, definition); return () => registered.delete(definition.name); } },
      systemPrompt: { section(value) { sections.push(value); } } };
    await apply(ctx, { casePath, caseSha256: sha(bytes), submissionPath, episodeId: id, selectionPolicy: 'source-local-v3', annotationAuditPath });
    assert.ok(sections.some(row => row.name === 'trust-judge-source-local-v3'));
    await registered.get('read_selected_sources').execute({ refs: ['P1', 'P2'] }, { signal: signal() });
    const proposal = args('P1'); proposal.selected[0].argument.时间 = '昨日';
    proposal.selected.push({ ...structuredClone(proposal.selected[0]), p: 'P2', roles: [], fact_des: '' });
    proposal.conflicting = [{ p: 'P1', resolution_type: 'Other' }];
    const original = structuredClone(proposal); let concluded = 0;
    await registered.get('submit_evidence').execute(proposal, { signal: signal(), concludeTurn() { concluded++; } });
    const audit = JSON.parse(await readFile(annotationAuditPath, 'utf8')), artifact = JSON.parse(await readFile(submissionPath, 'utf8'));
    assert.equal(concluded, 1); assert.deepEqual(proposal, original);
    assert.equal(audit.validation.status, 'validated'); assert.deepEqual(audit.original_args, original);
    assert.deepEqual(audit.accepted_ids, ['P1']); assert.deepEqual(audit.quarantine.map(row => row.p), ['P2']);
    assert.equal(audit.accepted_args.selected[0].argument.时间, '');
    assert.deepEqual(audit.accepted_args.supporting, original.supporting); assert.deepEqual(audit.accepted_args.conflicting, original.conflicting);
    assert.equal(artifact.evidence[0].quote, input.catalog[0].text);
    assert.equal(artifact.quality.annotation_isolation_applied, true); assert.equal(artifact.quality.annotation_quarantine_count, 1);
    assert.equal(artifact.quality.publishable, false); assert.deepEqual(artifact.scope.local_fetched_full_refs, ['P1', 'P2']);
  } finally { unbind(); await rm(directory, { recursive: true, force: true }); }
});

test('v3 isolation never salvages an unread source or deletes a relation to a quarantined source', async () => {
  for (const failure of ['unread', 'referenced-quarantine']) {
    const directory = await mkdtemp(join(tmpdir(), 'dsh-staged-v3-gate-')), id = randomUUID(), unbind = bindStagedAdmission(id, approved);
    try {
      const bytes = JSON.stringify(raw()), casePath = join(directory, 'source-case.json');
      const submissionPath = join(directory, 'submission.json'), annotationAuditPath = join(directory, 'annotation-audit.json');
      await writeFile(casePath, bytes); const registered = new Map();
      const ctx = { trustJudgeStagedAdmission: { episodeId: id }, tools: { register(definition) { registered.set(definition.name, definition); return () => registered.delete(definition.name); } }, systemPrompt: { section() {} } };
      await apply(ctx, { casePath, caseSha256: sha(bytes), submissionPath, episodeId: id, selectionPolicy: 'source-local-v3', annotationAuditPath });
      await registered.get('read_selected_sources').execute({ refs: ['P1'] }, { signal: signal() });
      const proposal = args(failure === 'unread' ? 'P2' : 'P1');
      if (failure === 'referenced-quarantine') proposal.selected[0].fact_des = '';
      let concluded = false;
      await assert.rejects(registered.get('submit_evidence').execute(proposal, { signal: signal(), concludeTurn() { concluded = true; } }),
        failure === 'unread' ? /unknown_or_duplicate_source_ref/ : /referenced_annotation_quarantined/);
      assert.equal(concluded, false); assert.equal(getStagedPresence(id).stage, 'submit');
      await assert.rejects(readFile(submissionPath), { code: 'ENOENT' });
      if (failure === 'referenced-quarantine') {
        const audit = JSON.parse(await readFile(annotationAuditPath, 'utf8'));
        assert.equal(audit.validation.status, 'blocked_by_quarantined_relation'); assert.deepEqual(audit.original_args.supporting, ['P1']);
      } else await assert.rejects(readFile(annotationAuditPath), { code: 'ENOENT' });
    } finally { unbind(); await rm(directory, { recursive: true, force: true }); }
  }
});

test('v3 explicit strict policy retains strict literal checks and isolation requires an independent audit path', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dsh-staged-v3-strict-')), id = randomUUID(), unbind = bindStagedAdmission(id, approved);
  try {
    const bytes = JSON.stringify(raw()), casePath = join(directory, 'source-case.json'), submissionPath = join(directory, 'submission.json');
    await writeFile(casePath, bytes); const registered = new Map();
    const ctx = { trustJudgeStagedAdmission: { episodeId: id }, tools: { register(definition) { registered.set(definition.name, definition); return () => registered.delete(definition.name); } }, systemPrompt: { section() {} } };
    const config = { casePath, caseSha256: sha(bytes), submissionPath, episodeId: id, selectionPolicy: 'source-local-v3' };
    for (const annotationAuditPath of [undefined, 'relative.json', casePath, submissionPath]) {
      await assert.rejects(apply(ctx, { ...config, annotationAuditPath }), /separate_staged_annotation_audit_required/);
      assert.equal(registered.size, 0);
    }
    await apply(ctx, { ...config, annotationPolicy: 'strict' });
    await registered.get('read_selected_sources').execute({ refs: ['P1'] }, { signal: signal() });
    const proposal = args('P1'); proposal.selected[0].argument.时间 = '昨日';
    await assert.rejects(registered.get('submit_evidence').execute(proposal, { signal: signal(), concludeTurn() {} }), /time_must_be_literal_or_empty/);
    await assert.rejects(readFile(submissionPath), { code: 'ENOENT' });
  } finally { unbind(); await rm(directory, { recursive: true, force: true }); }
});
