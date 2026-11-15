import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { SOURCE_LOCAL_V2_INSTRUCTIONS } from './evidence_policy.mjs';
import { FIT_POLICY, FIXTURE_PLAN, validateIndex, makeFixtureManifest, parseCli,
  runFixtureCase, runNativeFitSuite, extractFitDiagnostic } from './native_fit_suite.mjs';

const hash = 'a'.repeat(64);
const digest = text => createHash('sha256').update(text).digest('hex');
const context = 'synthetic complete context';
const index = () => ({ format: 'frozen-local-rerank-native-fit-inputs-v1', cases: Array.from({ length: 18 }, (_, i) => ({
  topic_id: `t${Math.floor(i / 3)}`, question_id: `q${i % 3}`, variants: ['baseline_hybrid40', 'title_bge_hybrid_rrf40'].map(label => ({ label, file: join(tmpdir(), `${i}-${label}.json`), sha256: hash })) })) });
const event = (input, status = 'admitted') => ({ status, estimated_input_tokens: input, known_episode_tokens_before: 0,
  reserved_output_tokens: 6144, safety_tokens: 512, preflight_total_tokens: input + 6656, wire_sha256: hash,
  cost_basis: 'official_tokenizer_native_request_estimate', ...(status === 'blocked' ? { code: 'whole_episode_budget_exceeded' } : {}) });
const receipt = (fits = true) => ({ closed: true, fixture: true, unknown_usage_requests: 0, cleanup_error: null,
  context_mode: 'full-candidate-context', complete_context_sha256: digest(context), selection_policy: 'source-local-v2',
  selection_policy_sha256: digest(SOURCE_LOCAL_V2_INSTRUCTIONS),
  status: fits ? 'closed_proposed_unverified' : 'budget_blocked', model_requests: fits ? 1 : 0,
  known_tokens: fits ? 150 : 0, evidence_count: 0, submission_sha256: fits ? hash : null });

test('all eighteen cases require both fixed variants; no duplicate question or missing binding', () => {
  assert.equal(validateIndex(index()).cases.length, 18);
  for (const mutate of [i => i.cases.pop(), i => i.cases[1] = i.cases[0], i => i.cases[0].variants.pop(),
    i => i.cases[0].variants[0].sha256 = null, i => i.cases[0].variants[0].file = join(tmpdir(), '.env')]) {
    const i = index(); mutate(i); assert.throws(() => validateIndex(i));
  }
});
test('fixed manifest preserves source SHA and v2, complete-context, one-call policy', () => {
  const caseBinding = index().cases[0].variants[0];
  const m = makeFixtureManifest({ ordinal: 1, label: caseBinding.label, case: caseBinding }, { outputDirectory: 'synthetic', fixturePlan: 'synthetic-plan' });
  for (const [key, value] of Object.entries(FIT_POLICY)) assert.equal(m[key], value);
  assert.equal(m.casePath, caseBinding.file); assert.equal(m.caseSha256, hash);
  assert.deepEqual(FIXTURE_PLAN.map(p => p.name), ['submit_evidence']);
  assert.equal(FIXTURE_PLAN[0].arguments.verdict_label, 'undetermined');
  assert.deepEqual(FIXTURE_PLAN[0].arguments.selected, []);
});
test('CLI has no paid mode, fixture toggle, source override, or missing review authorization', () => {
  for (const args of [['paid'], ['run'], ['run-fixtures', '--preparation', 'x'],
    ['run-fixtures', '--preparation', 'x', '--reviewed-preparation-sha', hash, '--fixture', 'false'],
    ['run-fixtures', '--preparation', 'x', '--reviewed-preparation-sha', hash, '--execute-paid', 'true']]) {
    assert.throws(() => parseCli(args));
  }
  assert.equal(parseCli(['run-fixtures', '--preparation', 'x', '--reviewed-preparation-sha', hash]).mode, 'run-fixtures');
});
test('native tokenizer threshold includes complete input plus 6144 output and 512 safety', () => {
  assert.equal(extractFitDiagnostic(receipt(), [event(13344)]).fit, true);
  const blocked = extractFitDiagnostic(receipt(false), [event(13345, 'blocked')]);
  assert.equal(blocked.fit, false); assert.equal(blocked.preflight_total_tokens, 20001);
  assert.equal(blocked.quality_metrics, null); assert.equal(blocked.real_api_requests, 0);
});
test('paid or unknown receipts, altered output reservation and extra requests cannot become fit results', () => {
  for (const change of [r => r.fixture = false, r => r.closed = false, r => r.unknown_usage_requests = 1,
    r => r.model_requests = 2, r => r.evidence_count = 1, r => r.known_tokens = 160]) {
    const r = receipt(); change(r); assert.throws(() => extractFitDiagnostic(r, [event(1000)]));
  }
  const p = event(1000); p.reserved_output_tokens = 512;
  assert.throws(() => extractFitDiagnostic(receipt(), [p]));
  assert.throws(() => extractFitDiagnostic(receipt(), [event(1000), event(1000)]));
});
test('test-injected runner receives only fixture:true and preserves selectionPolicy', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'dsh-native-fit-'));
  try {
    const manifestPath = join(dir, 'manifest.json');
    await writeFile(manifestPath, JSON.stringify(makeFixtureManifest({ ordinal: 1, label: 'baseline_hybrid40', case: { file: join(dir, 'raw.json'), sha256: hash } }, { outputDirectory: dir, fixturePlan: join(dir, 'plan.json') })));
    let calls = 0, checks = 0;
    const result = await runFixtureCase({ manifestPath, checkSources: async () => { checks++; }, runner: async args => {
      calls++; assert.deepEqual(args, { manifestPath, fixture: true, initializeOnly: false });
      const manifest = JSON.parse(await readFile(args.manifestPath));
      assert.equal(manifest.selectionPolicy, 'source-local-v2');
      assert.equal(manifest.contextMode, 'full-candidate-context');
      await writeFile(join(dir, 'budget.jsonl'), `${JSON.stringify(event(9000))}\n`);
      await writeFile(join(dir, 'complete-context.txt'), context);
      await writeFile(join(dir, 'profile.patch.yml'), JSON.stringify([{ insert: [{ id: 'trust-judge-evidence-tools', config: {
        contextMode: manifest.contextMode, selectionPolicy: manifest.selectionPolicy, caseSha256: hash } }] }]));
      await writeFile(join(dir, 'fixture-wire.jsonl'), `${JSON.stringify({ system: `${context}\n${SOURCE_LOCAL_V2_INSTRUCTIONS}`, tools: ['submit_evidence'], max_tokens: 6144 })}\n`);
      return receipt();
    } });
    assert.equal(calls, 1); assert.equal(checks, 2); assert.equal(result.estimated_input_tokens, 9000);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
test('source mutation or altered policy is rejected before injected SDK launch', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'dsh-native-fit-'));
  try {
    const manifestPath = join(dir, 'manifest.json');
    const manifest = makeFixtureManifest({ ordinal: 1, label: 'baseline_hybrid40', case: { file: join(dir, 'raw.json'), sha256: hash } }, { outputDirectory: dir, fixturePlan: join(dir, 'plan.json') });
    await writeFile(manifestPath, JSON.stringify(manifest)); let calls = 0;
    const runner = async () => { calls++; return receipt(); };
    await assert.rejects(runFixtureCase({ manifestPath, checkSources: async () => { throw new Error('frozen_file_changed'); }, runner }), /frozen_file_changed/);
    manifest.selectionPolicy = 'baseline'; await writeFile(manifestPath, JSON.stringify(manifest));
    await assert.rejects(runFixtureCase({ manifestPath, checkSources: async () => {}, runner }), /fixture_policy_changed/);
    assert.equal(calls, 0);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
test('suite requires fixture-only admission and exact reviewed preparation hash before any import', async () => {
  await assert.rejects(runNativeFitSuite({ confirmFixtureOnly: false }), /explicit_fixture_only_required/);
  const dir = await mkdtemp(join(tmpdir(), 'dsh-native-fit-'));
  try {
    const preparationPath = join(dir, 'preparation.json'); await writeFile(preparationPath, '{}');
    await assert.rejects(runNativeFitSuite({ preparationPath, reviewedPreparationSha256: hash, confirmFixtureOnly: true }), /frozen_file_changed/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
