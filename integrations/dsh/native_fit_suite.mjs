/** Zero-real-API fit diagnostics through the official SDK, serializer and existing budget guard. */
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname, relative, isAbsolute, basename } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { SOURCE_LOCAL_V2_INSTRUCTIONS } from './evidence_policy.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const encode = value => `${JSON.stringify(value, null, 2)}\n`;
const fail = code => { throw new Error(code); };
const labels = Object.freeze(['baseline_hybrid40', 'title_bge_hybrid_rrf40']);
export const FIT_POLICY = Object.freeze({ fixture: true, initializeOnly: false, contextMode: 'full-candidate-context',
  selectionPolicy: 'source-local-v2', maxCalls: 1, maxOutputTokens: 6144, wholeLimit: 20000, safetyTokens: 512, priorKnownTokens: 0 });
export const FIXTURE_PLAN = Object.freeze([{ name: 'submit_evidence', arguments: { conclusion: '公开信息层面不可判定。',
  verdict_label: 'undetermined', selected: [], supporting: [], conflicting: [], basis: '' } }]);
function safePath(file) {
  if (typeof file !== 'string' || !isAbsolute(file) || /(^|[\\/])(?:\.env(?:\.|$)|credentials(?:[\\/]|$)|gold(?:[\\/]|$)|evidence_samples(?:[\\/]|$)|conclusions(?:[\\/]|$))/i.test(file)) fail('raw_only_path_required');
  return resolve(file);
}
async function bound(item) {
  const file = safePath(item?.file);
  if (!/^[a-f0-9]{64}$/.test(item.sha256 ?? '')) fail('sha256_required');
  const bytes = await readFile(file);
  if (digest(bytes) !== item.sha256) fail('frozen_file_changed');
  return bytes;
}
const descriptor = async file => ({ file: safePath(resolve(file)), sha256: digest(await readFile(file)) });
const writeNew = (file, value) => writeFile(file, encode(value), { flag: 'wx' });

export function validateIndex(index) {
  if (index?.format !== 'frozen-local-rerank-native-fit-inputs-v1' || !Array.isArray(index.cases) || index.cases.length !== 18) fail('eighteen_frozen_cases_required');
  const seen = new Set();
  for (const row of index.cases) {
    if (typeof row.topic_id !== 'string' || typeof row.question_id !== 'string' || !row.topic_id || !row.question_id) fail('case_identity_required');
    const id = JSON.stringify([row.topic_id, row.question_id]);
    if (seen.has(id)) fail('duplicate_case');
    seen.add(id);
    if (!Array.isArray(row.variants) || row.variants.length !== 2 || new Set(row.variants.map(v => v.label)).size !== 2
      || row.variants.some(v => !labels.includes(v.label) || !/^[a-f0-9]{64}$/.test(v.sha256 ?? ''))) fail('two_fixed_variants_required');
    row.variants.forEach(v => safePath(v.file));
  }
  return index;
}
function validateCase(value) {
  if (typeof value?.question?.question !== 'string' || typeof value.question.question_id !== 'string'
    || !Array.isArray(value.catalog) || value.catalog.length !== 40) fail('forty_complete_raw_parents_required');
  const refs = new Set();
  for (const p of value.catalog) {
    if (!/^P[1-9]\d*$/.test(p.paragraph_ref ?? '') || refs.has(p.paragraph_ref) || typeof p.text !== 'string'
      || typeof p.title !== 'string' || typeof p.doc_id !== 'string' || typeof p.para_id !== 'string'
      || p.paragraph_complete !== true || p.parent_complete !== true) fail('complete_raw_parent_required');
    refs.add(p.paragraph_ref);
    if (Object.keys(p).some(k => !['paragraph_ref', 'text', 'title', 'doc_id', 'para_id', 'paragraph_complete', 'parent_complete'].includes(k))) fail('raw_catalog_only');
  }
  if (Object.keys(value).some(k => /^(?:gold|reference|evidence|conclusions|scores?|answers?)$/i.test(k))) fail('raw_case_only');
}
export function makeFixtureManifest(entry, { outputDirectory, fixturePlan }) {
  return { format: 'dsh-comparison-manifest-v1', episodeId: `native-fit-${entry.ordinal}-${entry.label}`,
    model: 'deepseek-v4-pro', reasoningEffort: 'off', ...FIT_POLICY,
    casePath: entry.case.file, caseSha256: entry.case.sha256, outputDirectory, fixturePlan };
}
function validateManifest(manifest) {
  for (const [key, value] of Object.entries(FIT_POLICY)) if (manifest[key] !== value) fail('fixture_policy_changed');
  if (manifest.format !== 'dsh-comparison-manifest-v1' || manifest.model !== 'deepseek-v4-pro' || manifest.reasoningEffort !== 'off') fail('fixture_policy_changed');
}

/** Prepare performs only local SHA/shape checks and writes manifests; never initializes an SDK. */
export async function prepareNativeFitSuite({ indexPath, indexSha256, outputDirectory, selectionPolicyPath, selectionPolicySha256 }) {
  const indexBinding = { file: safePath(indexPath), sha256: indexSha256 };
  const index = validateIndex(JSON.parse(await bound(indexBinding)));
  const policyFile = resolve(ROOT, 'integrations/dsh/evidence_policy.mjs');
  if (resolve(selectionPolicyPath ?? '') !== policyFile) fail('actual_profile_policy_source_required');
  const policyBinding = { file: policyFile, sha256: selectionPolicySha256 };
  await bound(policyBinding);
  const output = safePath(outputDirectory), prefix = `${resolve(ROOT, '.runtime')}/`.replaceAll('\\', '/');
  if (!output.replaceAll('\\', '/').startsWith(prefix)) fail('isolated_runtime_output_required');
  const entries = [];
  for (const [ordinal, row] of index.cases.entries()) for (const label of labels) {
    const variant = row.variants.find(v => v.label === label), source = JSON.parse(await bound(variant));
    validateCase(source);
    if (source.question.question_id !== row.question_id) fail('question_identity_changed');
    entries.push({ ordinal: ordinal + 1, topic_id: row.topic_id, question_id: row.question_id, label, case: { file: variant.file, sha256: variant.sha256 } });
  }
  await mkdir(output); // Exclusive directory: no fixture/result reuse or promotion.
  const planFile = resolve(output, 'fixture-plan.json'); await writeNew(planFile, FIXTURE_PLAN);
  for (const entry of entries) {
    const manifestFile = resolve(output, `${entry.ordinal}-${entry.label}.manifest.json`);
    await writeNew(manifestFile, makeFixtureManifest(entry, { outputDirectory: relative(ROOT, resolve(output, `${entry.ordinal}-${entry.label}`)), fixturePlan: planFile }));
    entry.manifest = await descriptor(manifestFile);
  }
  const runtime = resolve(ROOT, '.runtime/dsh-official-r1');
  const parserFile = createRequire(pathToFileURL(resolve(runtime, 'package.json'))).resolve('eventsource-parser');
  const sources = ['native_fit_suite.mjs', 'run_comparison.mjs', 'fixture_preload.mjs', 'profile.mjs', 'evidence_policy.mjs',
    'evidence_tools.mjs', 'budget_plugin.mjs', 'budget_core.mjs', 'budget_counter.mjs'].map(name => resolve(ROOT, 'integrations/dsh', name));
  sources.push(parserFile, resolve(runtime, 'package-lock.json'), resolve(runtime, 'node_modules/@deepseek-ai/dsh/lib/bin.js'),
    resolve(runtime, 'node_modules/@deepseek-ai/dsh-sdk-client/lib/index.js'), resolve(runtime, 'node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js'),
    resolve(ROOT, '.runtime/v32-deepseek-tokenizer-r2/tokenizer.json'));
  const preparation = { format: 'dsh-native-fit-fixture-preparation-v1', fixture: true, real_api_requests_allowed: false,
    policy: FIT_POLICY, source_index: indexBinding, selection_policy_source: policyBinding, fixture_plan: await descriptor(planFile),
    implementation_bindings: await Promise.all(sources.map(descriptor)), entries,
    paid_api_calls: 0, reference_reads: [], fit_is_quality: false, fixture_submission_must_not_be_promoted: true };
  const file = resolve(output, 'fit-preparation.json'); await writeNew(file, preparation);
  return descriptor(file);
}

/** Only guard preflight data are interpreted. Fixture usage150 is not a paid bill or evidence output. */
export function extractFitDiagnostic(receipt, rows) {
  if (receipt?.fixture !== true || receipt.closed !== true || receipt.unknown_usage_requests !== 0 || receipt.cleanup_error !== null
    || receipt.context_mode !== 'full-candidate-context' || !/^[a-f0-9]{64}$/.test(receipt.complete_context_sha256 ?? '')
    || receipt.selection_policy !== 'source-local-v2' || receipt.selection_policy_sha256 !== digest(SOURCE_LOCAL_V2_INSTRUCTIONS)) fail('closed_fixture_required');
  const attempts = rows.filter(r => ['admitted', 'blocked'].includes(r.status));
  if (attempts.length !== 1) fail('one_native_preflight_required');
  const p = attempts[0], fits = p.status === 'admitted';
  if (!Number.isSafeInteger(p.estimated_input_tokens) || p.estimated_input_tokens < 0 || p.known_episode_tokens_before !== 0
    || p.reserved_output_tokens !== 6144 || p.safety_tokens !== 512
    || p.preflight_total_tokens !== p.estimated_input_tokens + 6144 + 512
    || p.cost_basis !== 'official_tokenizer_native_request_estimate') fail('native_preflight_policy_changed');
  if (fits && (p.preflight_total_tokens > 20000 || receipt.model_requests !== 1 || receipt.status !== 'closed_proposed_unverified'
    || receipt.known_tokens !== 150 || receipt.evidence_count !== 0)) fail('fixture_execution_inconsistent');
  if (!fits && (p.preflight_total_tokens <= 20000 || p.code !== 'whole_episode_budget_exceeded'
    || receipt.status !== 'budget_blocked' || receipt.model_requests !== 0 || receipt.known_tokens !== 0 || receipt.submission_sha256 !== null)) fail('fixture_block_inconsistent');
  return { fixture: true, fit: fits, estimated_input_tokens: p.estimated_input_tokens, preflight_total_tokens: p.preflight_total_tokens,
    native_wire_sha256: p.wire_sha256, estimate_not_billing_guarantee: true, real_api_requests: 0, quality_metrics: null };
}

/** Injection exists for synthetic tests. The production call site always forces fixture:true. */
export async function runFixtureCase({ manifestPath, checkSources, runner }) {
  await checkSources();
  const manifest = JSON.parse(await readFile(manifestPath)); validateManifest(manifest);
  const receipt = await runner({ manifestPath, fixture: true, initializeOnly: false });
  await checkSources();
  const rows = (await readFile(resolve(ROOT, manifest.outputDirectory, 'budget.jsonl'), 'utf8')).split(/\r?\n/).filter(Boolean).map(JSON.parse);
  const diagnostic = extractFitDiagnostic(receipt, rows);
  const context = await readFile(resolve(ROOT, manifest.outputDirectory, 'complete-context.txt'));
  if (digest(context) !== receipt.complete_context_sha256) fail('complete_context_receipt_changed');
  const patch = JSON.parse(await readFile(resolve(ROOT, manifest.outputDirectory, 'profile.patch.yml'), 'utf8'));
  const plugins = patch.flatMap(row => row.insert ?? []), business = plugins.find(p => p.id === 'trust-judge-evidence-tools')?.config;
  if (business?.contextMode !== 'full-candidate-context' || business.selectionPolicy !== 'source-local-v2'
    || business.caseSha256 !== manifest.caseSha256) fail('actual_fixture_profile_changed');
  if (diagnostic.fit) {
    const wires = (await readFile(resolve(ROOT, manifest.outputDirectory, 'fixture-wire.jsonl'), 'utf8')).split(/\r?\n/).filter(Boolean).map(JSON.parse);
    if (wires.length !== 1 || typeof wires[0].system !== 'string' || !wires[0].system.includes(context.toString('utf8'))
      || !wires[0].system.includes(SOURCE_LOCAL_V2_INSTRUCTIONS) || JSON.stringify(wires[0].tools) !== '["submit_evidence"]'
      || wires[0].max_tokens !== 6144) fail('actual_native_fixture_context_changed');
  }
  return diagnostic;
}

export async function runNativeFitSuite({ preparationPath, reviewedPreparationSha256, confirmFixtureOnly }) {
  if (confirmFixtureOnly !== true) fail('explicit_fixture_only_required');
  const prepBinding = { file: safePath(preparationPath), sha256: reviewedPreparationSha256 };
  const prep = JSON.parse(await bound(prepBinding));
  if (prep.format !== 'dsh-native-fit-fixture-preparation-v1' || prep.fixture !== true || prep.real_api_requests_allowed !== false
    || prep.entries?.length !== 36 || JSON.stringify(prep.policy) !== JSON.stringify(FIT_POLICY)) fail('frozen_fixture_preparation_required');
  const guards = [prepBinding, prep.source_index, prep.selection_policy_source, prep.fixture_plan, ...prep.implementation_bindings,
    ...prep.entries.flatMap(e => [e.case, e.manifest])];
  const checkSources = async () => { for (const item of guards) await bound(item); };
  await checkSources();
  validateIndex(JSON.parse(await bound(prep.source_index)));
  if (JSON.stringify(JSON.parse(await bound(prep.fixture_plan))) !== JSON.stringify(FIXTURE_PLAN)) fail('only_empty_undetermined_fixture_allowed');
  const output = dirname(prepBinding.file);
  await writeNew(resolve(output, 'fit-execution-start.json'), { fixture: true, preparation_sha256: reviewedPreparationSha256, real_api_requests_allowed: false });
  const { runComparison } = await import('./run_comparison.mjs');
  const results = [];
  for (const label of labels) {
    const group = prep.entries.filter(e => e.label === label);
    if (group.length !== 18) fail('eighteen_cases_per_group_required');
    for (let i = 0; i < group.length; i += 2) {
      const batch = await Promise.all(group.slice(i, i + 2).map(async entry => ({ topic_id: entry.topic_id, question_id: entry.question_id, label,
        ...await runFixtureCase({ manifestPath: entry.manifest.file, checkSources, runner: runComparison }) })));
      results.push(...batch);
      process.stdout.write(`${JSON.stringify({ fixture: true, completed: results.length, total: 36 })}\n`);
    }
  }
  await checkSources();
  const report = { format: 'dsh-native-fit-fixture-report-v1', closed: true, fixture: true, preparation_sha256: reviewedPreparationSha256,
    real_api_requests: 0, quality_metrics: null, fixture_submission_must_not_be_promoted: true,
    groups: labels.map(label => ({ label, cases: 18, fit_count: results.filter(r => r.label === label && r.fit).length })), results };
  const file = resolve(output, 'native-fit-report.json'); await writeNew(file, report); return descriptor(file);
}

function parseCli(argv) {
  const [mode, ...rest] = argv, options = {};
  const allowed = mode === 'prepare' ? ['index', 'index-sha', 'output', 'policy-source', 'policy-sha']
    : mode === 'run-fixtures' ? ['preparation', 'reviewed-preparation-sha'] : fail('only_prepare_or_run_fixtures_allowed');
  for (let i = 0; i < rest.length; i += 2) {
    const key = rest[i]?.replace(/^--/, '');
    if (!rest[i]?.startsWith('--') || !allowed.includes(key) || options[key] !== undefined || !rest[i + 1] || rest[i + 1].startsWith('--')) fail('unknown_or_missing_cli_argument');
    options[key] = rest[i + 1];
  }
  if (allowed.some(key => !options[key])) fail('required_cli_argument_missing');
  return { mode, options };
}
export { parseCli };
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  Promise.resolve().then(() => {
    const { mode, options: o } = parseCli(process.argv.slice(2));
    return mode === 'prepare' ? prepareNativeFitSuite({ indexPath: o.index, indexSha256: o['index-sha'], outputDirectory: o.output,
      selectionPolicyPath: o['policy-source'], selectionPolicySha256: o['policy-sha'] })
      : runNativeFitSuite({ preparationPath: o.preparation, reviewedPreparationSha256: o['reviewed-preparation-sha'], confirmFixtureOnly: true });
  }).then(value => process.stdout.write(encode(value))).catch(error => { process.stderr.write(`${basename(fileURLToPath(import.meta.url))}: ${error.message}\n`); process.exitCode = 1; });
}
