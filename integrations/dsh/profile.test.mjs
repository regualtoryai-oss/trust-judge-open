import test from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { buildPatch, DSH_PROFILE, DSH_VERSION, DISABLED_PROFILE_ROWS } from './profile.mjs';
function fixture() { return { casePath: resolve('synthetic-raw.json'), caseSha256: '1'.repeat(64), submissionPath: resolve('synthetic-output.json'), sessionRoot: resolve('synthetic-sessions'), budgetPluginPath: resolve('integrations/dsh/budget_plugin.mjs'),
  budgetConfig: { episodeId: 'synthetic', priorKnownTokens: 0, wholeLimit: 20000, safetyTokens: 512, maxCalls: 6, maxOutputTokens: 3072, endpoint: 'https://api.deepseek.com/anthropic/v1/messages' } }; }
test('official SDK profile patch disables shell/fs/web-like extras and all extra model work', () => {
  const patch = buildPatch(fixture()); assert.equal(DSH_PROFILE, 'sdk-minimal'); assert.equal(DSH_VERSION, '0.2.1-rc.2');
  for (const id of ['persistent-bash', 'persistent-pwsh', 'tool-fs', 'compaction-basic', 'session-title', 'llm-retry']) assert.ok(patch.some(row => row.id === id && row.disabled === true));
  assert.ok(!DISABLED_PROFILE_ROWS.includes('session-projection'));
  assert.ok(!DISABLED_PROFILE_ROWS.includes('working-directory')); assert.ok(!DISABLED_PROFILE_ROWS.includes('filesystem'));
  const provider = patch.find(row => row.id === 'llm-deepseek').config;
  assert.equal(provider.reasoningEffort, 'off'); assert.equal(provider.thinking, 'disabled'); assert.equal(provider.retryPolicy.maxRetries, 0);
  const insert = patch.find(row => row.insert).insert; assert.equal(insert.length, 2); assert.ok(insert.every(row => row.name.startsWith('file:')));
});
test('profile cannot be built without an explicit matching whole-episode budget', () => {
  for (const alteration of [{ budgetConfig: undefined }, { budgetConfig: { ...fixture().budgetConfig, priorKnownTokens: null } }, { budgetConfig: { ...fixture().budgetConfig, wholeLimit: 40000 } }, { budgetConfig: { ...fixture().budgetConfig, maxOutputTokens: 8192 } }, { budgetConfig: { ...fixture().budgetConfig, endpoint: 'https://elsewhere.invalid/v1/messages' } }]) assert.throws(() => buildPatch({ ...fixture(), ...alteration }));
});
test('profile does not read environment or attach secrets and detaches caller configuration', () => {
  const args = fixture(), patch = buildPatch(args); args.budgetConfig.wholeLimit = 1;
  assert.equal(patch.find(row => row.insert).insert[0].config.wholeLimit, 20000);
  assert.ok(!JSON.stringify(patch).includes('DEEPSEEK_API_KEY='));
  assert.throws(() => buildPatch({ ...fixture(), casePath: 'relative.json' }), /absolute_bound/);
});
test('packet counter and settled journal settings are copied automatically from the existing budget config', () => {
  const args = fixture(); Object.assign(args.budgetConfig, { journalPath: resolve('synthetic-journal.jsonl'), pythonExecutable: resolve('synthetic-python.exe'), tokenizerPath: resolve('synthetic-tokenizer.json'), tokenizerSha256: 'a'.repeat(64) });
  const business = buildPatch(args).find(row => row.insert).insert.find(row => row.id === 'trust-judge-evidence-tools').config;
  assert.deepEqual(business.packetCounter, { pythonExecutable: args.budgetConfig.pythonExecutable, tokenizerPath: args.budgetConfig.tokenizerPath, tokenizerSha256: args.budgetConfig.tokenizerSha256 });
  assert.equal(business.packetBudget.journalPath, args.budgetConfig.journalPath); assert.equal(business.packetBudget.wholeLimit, 20000);
});
test('full-context mode is explicit and requires a source-presence guard before profile creation', () => {
  const args = fixture();
  assert.throws(() => buildPatch({ ...args, contextMode: 'full-candidate-context' }), /native_presence_gate/);
  Object.assign(args.budgetConfig, { requiredContextPath: resolve('synthetic-source-context.txt'), requiredContextSha256: 'b'.repeat(64) });
  const config = buildPatch({ ...args, contextMode: 'full-candidate-context' }).find(row => row.insert).insert.find(row => row.id === 'trust-judge-evidence-tools').config;
  assert.equal(config.contextMode, 'full-candidate-context');
  assert.throws(() => buildPatch({ ...args, contextMode: 'unknown' }), /unknown_context_mode/);
});

test('staged mode requires native presence policy, source-local rules and exactly two model calls', () => {
  const args = { ...fixture(), contextMode: 'staged-short-window', selectionPolicy: 'source-local-v2' };
  assert.throws(() => buildPatch(args), /staged_native_presence_gate/);
  Object.assign(args.budgetConfig, { stagedContext: true, stagedContextAuditPath: resolve('synthetic-staged-audit.jsonl'), maxCalls: 2 });
  const business = buildPatch(args).find(row => row.insert).insert[1];
  assert.ok(business.name.endsWith('/staged_evidence.mjs'));
  assert.equal(business.config.episodeId, 'synthetic');
  assert.equal(business.config.previewCodepoints, 160);
  assert.equal(business.config.annotationPolicy, 'strict');
  assert.ok(!Object.hasOwn(business.config, 'annotationAuditPath'));
  assert.ok(!Object.hasOwn(business.config, 'packetBudget'));
  assert.throws(() => buildPatch({ ...args, selectionPolicy: 'baseline' }), /staged_native_presence_gate/);
});

test('staged v3 uses the existing isolation backend by default with explicit independent audit output', () => {
  const args = { ...fixture(), contextMode: 'staged-short-window', selectionPolicy: 'source-local-v3' };
  Object.assign(args.budgetConfig, { stagedContext: true, stagedContextAuditPath: resolve('synthetic-staged-audit.jsonl'), maxCalls: 2 });
  const business = buildPatch(args).find(row => row.insert).insert[1].config;
  assert.equal(business.selectionPolicy, 'source-local-v3'); assert.equal(business.annotationPolicy, 'isolate-invalid-metadata-v1');
  assert.equal(business.annotationAuditPath, resolve('submission-annotation-audit.json'));
  assert.notEqual(business.annotationAuditPath, business.submissionPath);
  const strict = buildPatch({ ...args, annotationPolicy: 'strict' }).find(row => row.insert).insert[1].config;
  assert.equal(strict.annotationPolicy, 'strict'); assert.ok(!Object.hasOwn(strict, 'annotationAuditPath'));
  assert.throws(() => buildPatch({ ...args, annotationPolicy: 'unknown' }), /invalid_annotation_policy_for_context/);
  args.budgetConfig.maxCalls = 3;
  assert.throws(() => buildPatch(args), /staged_native_presence_gate_required/);
});

test('annotation isolation is explicit in full source-local profile and can remain strict', () => {
  const args = { ...fixture(), contextMode: 'full-candidate-context', selectionPolicy: 'source-local-v2' };
  Object.assign(args.budgetConfig, { requiredContextPath: resolve('synthetic-context.txt'), requiredContextSha256: 'a'.repeat(64) });
  const config = buildPatch(args).find(row => row.insert).insert[1].config;
  assert.equal(config.annotationPolicy, 'isolate-invalid-metadata-v1'); assert.ok(config.annotationAuditPath.endsWith('submission-annotation-audit.json'));
  assert.equal(buildPatch({ ...args, selectionPolicy: 'source-local-v3' }).find(row => row.insert).insert[1].config.annotationPolicy, 'isolate-invalid-metadata-v1');
  const strict = buildPatch({ ...args, annotationPolicy: 'strict' }).find(row => row.insert).insert[1].config;
  assert.equal(strict.annotationPolicy, 'strict'); assert.ok(!Object.hasOwn(strict, 'annotationAuditPath'));
  assert.throws(() => buildPatch({ ...fixture(), annotationPolicy: 'isolate-invalid-metadata-v1' }), /invalid_annotation_policy/);
});
