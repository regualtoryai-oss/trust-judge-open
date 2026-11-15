/** Official SDK native serialization fixture only. Cannot launch paid traffic. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildBundleFixturePatch } from './bundle_profile.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha = value => createHash('sha256').update(value).digest('hex');
const save = (path, value) => writeFile(path, `${JSON.stringify(value, null, 2)}\n`, { flag: 'wx' });
const canonical = value => JSON.stringify(value && typeof value === 'object'
  ? Array.isArray(value) ? value.map(item => JSON.parse(canonical(item)))
    : Object.fromEntries(Object.keys(value).sort().map(key => [key, JSON.parse(canonical(value[key]))])) : value);

export function syntheticBundleCase() {
  const pieces = [
    `项目受理记录😀。${'申请受理不是批准。'.repeat(24)}第一原段完整尾部。`,
    `项目尚未作出批准决定。${'须核验主管部门后续程序。'.repeat(20)}第二原段完整尾部🚀。`,
    `项目后来出现新的材料。${'材料记载的是补充申请。'.repeat(22)}第三原段完整尾部。`,
    `项目审批状态仍待核验。${'公开资料未给出批准日期。'.repeat(20)}第四原段完整尾部😀。`,
  ];
  let content = '原始文档前缀|';
  const paragraphs = [], manifest = [];
  pieces.forEach((text, index) => {
    const start = Array.from(content).length, id = `opaque-source-${index}`;
    content += text;
    const end = Array.from(content).length;
    paragraphs.push({ id, document_id: 'synthetic-document-internal', raw_doc_id: '原文档', raw_para_id: `原段-${index}`,
      text, document_start_cp: start, document_end_cp: end });
    manifest.push({ id, raw_para_id: `原段-${index}`, start_cp: start, end_cp: end, text_sha256: sha(text) });
    content += '\r\n<真实原始分隔😀>'; // Retain the actual document bytes, never join guessed separators.
  });
  return { format: 'frozen-source-bundle-case-v1', question: { question_id: 'synthetic-only-q', question: '项目是否已经获得主管部门批准？' },
    documents: [{ id: 'synthetic-document-internal', raw_doc_id: '原文档', title: '合成项目程序记录', content,
      content_sha256: sha(content), paragraph_manifest: manifest, paragraph_manifest_sha256: sha(canonical(manifest)) }],
    paragraphs, limits: { max_search_calls: 2, max_read_calls: 2, max_read_codepoints: 10000, max_read_payload_bytes: 100000 } };
}

export async function runBundleFixture(outputDirectory) {
  const dir = resolve(outputDirectory), normalized = dir.replaceAll('\\', '/');
  if (!normalized.startsWith(`${resolve(ROOT, '.runtime').replaceAll('\\', '/')}/`)) throw new Error('bundle_fixture_new_runtime_directory_required');
  await mkdir(dir, { recursive: false });
  const input = syntheticBundleCase(), casePath = resolve(dir, 'synthetic-raw.json');
  await save(casePath, input);
  const selected = [[1, 1], [2, 3]].map(([start, end]) => ({
    start_paragraph_id: input.paragraphs[start].id, end_paragraph_id: input.paragraphs[end].id,
    fact_des: '预先声明的合成注释，仅供工具合同验证，不代表模型研判。',
    argument: Object.fromEntries(['时间', '主体', '地点', '内容', '影响', '数值口径'].map(key => [key, key === '内容' ? input.paragraphs[start].text : ''])),
    entities: [], proposed_roles: ['L'],
  }));
  const plan = [
    { name: 'search_source_bundles', arguments: { query: input.question.question, limit: 60, exclude_paragraph_ids: [] } },
    { name: 'read_source_bundles', arguments: { anchor_paragraph_ids: [input.paragraphs[0].id], before: 0, after: 1 } },
    { name: 'read_source_bundles', arguments: { anchor_paragraph_ids: [input.paragraphs[2].id], before: 0, after: 1 } },
    { name: 'submit_bundle_evidence', arguments: { selected } },
  ];
  const planPath = resolve(dir, 'fixture-plan.json');
  await save(planPath, plan);
  const runtime = resolve(ROOT, '.runtime/dsh-official-r1'), pkg = resolve(runtime, 'package.json');
  const parser = createRequire(pathToFileURL(pkg)).resolve('eventsource-parser');
  const sdkFile = resolve(runtime, 'node_modules/@deepseek-ai/dsh-sdk-client/lib/index.js');
  for (const pkgName of ['dsh', 'dsh-sdk-client', 'dsh-llm-deepseek']) {
    assert.equal(JSON.parse(await readFile(resolve(runtime, `node_modules/@deepseek-ai/${pkgName}/package.json`))).version, '0.2.1-rc.2');
  }
  const files = [casePath, planPath, parser, sdkFile, resolve(runtime, 'package-lock.json'),
    resolve(runtime, 'node_modules/@deepseek-ai/dsh/lib/bin.js'),
    resolve(runtime, 'node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js'),
    ...['bundle_bridge.mjs', 'bundle_evidence.mjs', 'bundle_presence.mjs', 'bundle_guard.mjs', 'bundle_profile.mjs', 'bundle_fixture_preload.mjs',
      'bundle_fixture.mjs', 'profile.mjs', 'evidence_policy.mjs', 'budget_plugin.mjs', 'budget_core.mjs', 'budget_counter.mjs'].map(file => resolve(ROOT, 'integrations/dsh', file)),
    ...['scripts/source_bundle_tool.py', 'app/domain/source_bundle_tools.py', 'app/domain/source_bundles.py',
      'app/domain/frozen_selection_inputs.py', 'app/domain/question_recall.py', 'app/domain/query_context.py',
      'app/domain/retrieval.py', 'app/domain/semantic_recall.py', 'app/domain/source_catalog.py', 'app/domain/numeric_literals.py'].map(file => resolve(ROOT, file)),
  ];
  const bindings = await Promise.all(files.map(async file => ({ file, sha256: sha(await readFile(file)) })));
  const guardConfig = { fixtureOnly: true, endpoint: 'https://api.deepseek.com/anthropic/v1/messages', model: 'deepseek-v4-pro',
    runtimePackageJson: pkg, sourceBindings: bindings, journalPath: resolve(dir, 'fixture-budget.jsonl'), presencePath: resolve(dir, 'native-presence.jsonl'),
    readinessPath: resolve(dir, 'guard-ready.json'), episodeId: 'synthetic-bundle-two-burst-only',
    priorKnownTokens: 0, wholeLimit: 40000, safetyTokens: 512, maxCalls: 4, maxOutputTokens: 1024,
    pythonExecutable: resolve(ROOT, '.runtime/v20-embedding-env/Scripts/python.exe'),
    tokenizerPath: resolve(ROOT, '.runtime/v32-deepseek-tokenizer-r2/tokenizer.json'),
    tokenizerSha256: '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf' };
  const bridge = { casePath, caseSha256: sha(await readFile(casePath)), pythonExecutable: resolve(ROOT, '.venv/Scripts/python.exe'),
    scriptSha256: sha(await readFile(resolve(ROOT, 'scripts/source_bundle_tool.py'))), journalPath: resolve(dir, 'business-transcript.jsonl') };
  const patchFile = resolve(dir, 'profile.patch.yml'), submissionPath = resolve(dir, 'proposed.json');
  await save(patchFile, buildBundleFixturePatch({ bridge, submissionPath, sessionRoot: resolve(dir, 'sessions'), guardConfig }));
  const env = Object.fromEntries(['SystemRoot', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT']
    .filter(key => process.env[key] !== undefined).map(key => [key, process.env[key]]));
  Object.assign(env, { DEEPSEEK_API_KEY: 'local-bundle-fixture-not-a-real-credential', TRUST_DSH_BUNDLE_FIXTURE: '1',
    TRUST_DSH_BUNDLE_FIXTURE_PLAN: planPath, TRUST_DSH_BUNDLE_FIXTURE_WIRES: resolve(dir, 'native-wire.jsonl'),
    NODE_OPTIONS: `--import=${pathToFileURL(resolve(ROOT, 'integrations/dsh/bundle_fixture_preload.mjs')).href}` });
  const { DeepSeekHarness } = await import(pathToFileURL(sdkFile));
  const harness = new DeepSeekHarness({ dshBin: resolve(runtime, 'node_modules/@deepseek-ai/dsh/lib/bin.js'), profile: 'sdk-minimal',
    patches: [patchFile], dshHome: resolve(dir, 'home'), cwd: dir, processCwd: runtime, env,
    provider: 'deepseek-official', model: 'deepseek-v4-pro', reasoningEffort: 'off', maxTokens: 1024,
    initializeTimeoutMs: 60000, requestTimeoutMs: 60000 });
  let turnEnd;
  try {
    await harness.start();
    const ready = JSON.parse(await readFile(guardConfig.readinessPath));
    assert.equal(ready.native_guard_installed, true); assert.equal(ready.fixture_only, true);
    assert.equal(ready.source_bindings_sha256, sha(JSON.stringify(bindings)));
    await harness.run('合成工具协议测试：搜索、读两批完整原文，再提交预先声明的未验注释。', {
      onNotification(notification) { const event = notification.params?.event; if (event?.type === 'turn/end') turnEnd = event.data?.reason; },
    });
  } finally { await harness.close(); }
  const wires = (await readFile(resolve(dir, 'native-wire.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  const presence = (await readFile(guardConfig.presencePath, 'utf8')).trim().split('\n').map(JSON.parse);
  const proposed = JSON.parse(await readFile(submissionPath));
  const ledger = (await readFile(guardConfig.journalPath, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.equal(turnEnd?.kind, 'completed'); assert.equal(wires.length, 4);
  assert.deepEqual(presence.map(row => row.read_receipts.length), [0, 0, 1, 2]);
  assert.deepEqual(wires[0].tools.map(row => row.name), ['search_source_bundles']);
  assert.deepEqual(wires[3].tools.map(row => row.name), ['submit_bundle_evidence']);
  assert.equal(proposed.status, 'proposed_unverified'); assert.equal(proposed.publishable, false);
  assert.equal(proposed.evidence.length, 2); assert.equal(proposed.source_bytes_present_in_submission_request, true);
  for (const evidence of proposed.evidence) {
    assert.equal(evidence.quote, Array.from(input.documents[0].content).slice(evidence.start_cp, evidence.end_cp).join(''));
    assert.equal(evidence.native_model_delivery_verified, false); // Backend provenance remains distinct from native receipt.
  }
  const settled = ledger.filter(row => row.status === 'usage_settled');
  assert.equal(settled.length, 4); assert.ok(settled.every(row => row.response_completed && row.unknown_usage_requests === 0));
  for (const binding of bindings) assert.equal(sha(await readFile(binding.file)), binding.sha256);
  const receipt = { format: 'official-dsh-source-bundle-fixture-v1', closed: true, fixture: true,
    real_api_requests: 0, paid_tokens: 0, simulated_provider_requests: 4, simulated_usage_tokens_not_bill: 600,
    quality_metrics: null, actual_native_read_counts: presence.map(row => row.read_receipts.length),
    original_source_bundle_count: 2, proposed_evidence_count: 2, semantic_verified: false, publishable: false,
    costs_remaining_unproven: ['real_question_budget_fit', 'remaining_A_B_arbitration_reserve', 'real_selection_quality'],
    repeated_full_history_input_risk: true, native_wire_sha256: sha(await readFile(resolve(dir, 'native-wire.jsonl'))),
    native_presence_sha256: sha(await readFile(guardConfig.presencePath)), submission_sha256: sha(await readFile(submissionPath)),
    source_bindings: bindings };
  await save(resolve(dir, 'closed-receipt.json'), receipt);
  return receipt;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 3) throw new Error('bundle_fixture_output_directory_only');
  runBundleFixture(process.argv[2]).then(receipt => process.stdout.write(`${JSON.stringify({ closed: receipt.closed, real_api_requests: 0,
    simulated_provider_requests: receipt.simulated_provider_requests, quality_metrics: null })}\n`))
    .catch(error => { process.stderr.write(`${error.name}: ${error.message}\n`); process.exitCode = 1; });
}

