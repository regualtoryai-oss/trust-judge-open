/** Fixed raw-data integration check only. No paid switch and no answer references. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runComparison } from './run_comparison.mjs';
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha = data => createHash('sha256').update(data).digest('hex');
const save = (file, value) => writeFile(file, `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
const INDEX_SHA = '702eea717c3456f491b922c090ca434eeb681b574241726d7fa603a1a1396168';

export async function runFixedStagedFixture(outputDirectory) {
  const dir = resolve(outputDirectory);
  if (!dir.startsWith(`${resolve(ROOT, '.runtime')}\\`) && !dir.startsWith(`${resolve(ROOT, '.runtime')}/`)) throw new Error('isolated_runtime_output_required');
  const indexBytes = await readFile(resolve(ROOT, '.runtime/title-aware-rerank-r1/native-fit-index.json'));
  if (sha(indexBytes) !== INDEX_SHA) throw new Error('frozen_raw_index_changed');
  const cases = JSON.parse(indexBytes).cases.filter(c => c.topic_id === '04' && c.question_id === 'q_002');
  if (cases.length !== 1) throw new Error('fixed_case_required');
  const source = cases[0].variants.find(v => v.label === 'title_bge_hybrid_rrf40');
  const bytes = await readFile(source.file);
  if (sha(bytes) !== source.sha256) throw new Error('frozen_raw_case_changed');
  const raw = JSON.parse(bytes);
  if (raw.catalog.length !== 40) throw new Error('forty_original_candidates_required');
  // Mechanical read plan, fixed by raw source order, not a model or reference choice.
  const selected = raw.catalog.slice(0, 6);
  const plan = [{ name: 'read_selected_sources', arguments: { refs: selected.map(p => p.paragraph_ref) } },
    { name: 'submit_evidence', arguments: { conclusion: '此为无网接口测试，不提供实际研判结论。', verdict_label: 'undetermined',
      selected: selected.map(p => ({ p: p.paragraph_ref, roles: ['L'], fact_des: '模拟注释，仅验证工具协议；不构成事实概括。',
        argument: { 时间: '', 主体: '', 地点: '', 内容: '', 影响: '', 数值口径: '' }, entities: [] })),
      supporting: [], conflicting: [], basis: '预先声明的假供应商响应；不用于质量评分或发布。' } }];
  await mkdir(dir);
  const planPath = resolve(dir, 'fixture-plan.json'); await save(planPath, plan);
  const manifest = { format: 'dsh-comparison-manifest-v1', episodeId: 'fixed04-staged-raw-fixture', wholeLimit: 20000, priorKnownTokens: 0,
    model: 'deepseek-v4-pro', reasoningEffort: 'off', maxCalls: 2, maxOutputTokens: 6144, safetyTokens: 512,
    contextMode: 'staged-short-window', selectionPolicy: 'source-local-v2', previewCodepoints: 160,
    casePath: source.file, caseSha256: source.sha256, outputDirectory: resolve(dir, 'run'), fixturePlan: planPath };
  const manifestPath = resolve(dir, 'manifest.json'); await save(manifestPath, manifest);
  const receipt = await runComparison({ manifestPath, fixture: true });
  const run = resolve(dir, 'run');
  const ledger = (await readFile(resolve(run, 'budget.jsonl'), 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  const result = { format: 'fixed-staged-raw-fixture-report-v1', fixture: true, closed: true, case: '04/q_002',
    original_candidate_count: 40, fixed_simulated_read_count: 6, read_choice_basis: 'first_six_original_source_order_not_model_or_reference',
    status: receipt.status, source_case_sha256: source.sha256, closed_receipt_sha256: sha(await readFile(resolve(run, 'closed-receipt.json'))),
    native_estimated_input_tokens: ledger.filter(r => r.status === 'admitted').map(r => r.estimated_input_tokens),
    fixture_usage_tokens_not_real_bill: receipt.known_tokens, real_api_calls: 0, paid_tokens: 0, quality_metrics: null,
    simulation_does_not_prove_real_question_budget_fit: true, published: false };
  await save(resolve(dir, 'raw-fixture-report.json'), result);
  if (receipt.status !== 'closed_proposed_unverified' || receipt.model_requests !== 2 || receipt.evidence_count !== 6) throw new Error('fixture_did_not_complete');
  return result;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('fixture_output_directory_only');
  runFixedStagedFixture(process.argv[2]).then(r => process.stdout.write(`${JSON.stringify(r)}\n`))
    .catch(e => { process.stderr.write(`${e.name}: ${e.message}\n`); process.exitCode = 1; });
}
