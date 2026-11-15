/** Launch the existing pinned DSH SDK/runtime. No agent or retry loop is implemented here. */
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { buildPatch as officialProfilePatch } from './profile.mjs';
import { buildCompleteContext } from './evidence_tools.mjs';
import { selectionPolicySection } from './evidence_policy.mjs';
import { verifyPriorEpisode } from './prior_episode.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const json = value => `${JSON.stringify(value, null, 2)}\n`;
const writeNew = (path, value) => writeFile(path, json(value), { flag: 'wx' });

/** Safe cause codes stay separate from a later call-limit/transport termination. */
export function safeToolErrorCodes(trace) {
  return [...new Set(trace.filter(row => row.event_type === 'tool/result' && row.is_error === true)
    .flatMap(row => (row.content ?? []).flatMap(block => {
      const match = typeof block.text === 'string' ? block.text.match(/^Error: ([a-z][a-z_]{0,95})$/) : null;
      return match ? [match[1]] : [];
    })))];
}

/** SDK startup can succeed while a plugin failed; no prompt may precede explicit guard readiness. */
export async function verifyBudgetReady(path, { episodeId, sourceBindingsSha256 }) {
  let ready;
  for (let i = 0; i < 50; i++) {
    ready = await readFile(path, 'utf8').then(JSON.parse).catch(() => null);
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  if (ready?.format !== 'dsh-budget-ready-v1' || ready.episode_id !== episodeId
      || ready.source_bindings_sha256 !== sourceBindingsSha256 || ready.guarded_native_transport_installed !== true) {
    const error = new Error('budget_plugin_not_ready'); error.code = 'budget_plugin_not_ready'; throw error;
  }
}

export async function runComparison({ manifestPath, fixture = false, initializeOnly = false }) {
  const manifestBytes = await readFile(manifestPath);
  const manifest = JSON.parse(manifestBytes);
  if (manifest.format !== 'dsh-comparison-manifest-v1' || manifest.wholeLimit !== 20000
      || !Number.isSafeInteger(manifest.priorKnownTokens) || manifest.priorKnownTokens < 0 || manifest.priorKnownTokens > 20000
      || manifest.model !== 'deepseek-v4-pro' || manifest.reasoningEffort !== 'off') throw new Error('invalid_frozen_manifest');
  // Reject unknown policies before any runtime, source or credential is loaded.
  const policy = selectionPolicySection(manifest.selectionPolicy ?? 'baseline');
  const annotationPolicy = manifest.annotationPolicy ?? ((manifest.contextMode === 'full-candidate-context' && ['source-local-v2', 'source-local-v3', 'source-local-v3-compact'].includes(manifest.selectionPolicy))
    || (manifest.contextMode === 'staged-short-window' && manifest.selectionPolicy === 'source-local-v3') ? 'isolate-invalid-metadata-v1' : 'strict');
  if (!['strict', 'isolate-invalid-metadata-v1'].includes(annotationPolicy)
      || (annotationPolicy !== 'strict' && !['full-candidate-context', 'staged-short-window'].includes(manifest.contextMode))) throw new Error('invalid_annotation_policy_for_context');
  const output = resolve(ROOT, manifest.outputDirectory);
  if (!output.startsWith(`${resolve(ROOT, '.runtime')}\\`) && !output.startsWith(`${resolve(ROOT, '.runtime')}/`)) throw new Error('isolated_output_required');
  await mkdir(output, { recursive: true });
  await writeNew(resolve(output, 'launch-start.json'), { manifest_sha256: digest(manifestBytes), fixture, initializeOnly, started_at: new Date().toISOString() });
  const rawCasePath = resolve(ROOT, manifest.casePath), caseBytes = await readFile(rawCasePath);
  if (digest(caseBytes) !== manifest.caseSha256) throw new Error('raw_case_binding_changed');
  const rawCase = JSON.parse(caseBytes);
  if (manifest.priorKnownTokens > 0 && !manifest.priorClosedReceipt) throw new Error('bound_prior_receipt_required');
  if (manifest.priorKnownTokens === 0 && manifest.priorClosedReceipt) throw new Error('nonzero_prior_receipt_usage_required');
  const prior = manifest.priorKnownTokens > 0 ? await verifyPriorEpisode(manifest.priorClosedReceipt,
    { rawCase, root: ROOT, expectedKnownTokens: manifest.priorKnownTokens }) : { bindings: [] };
  const staged = manifest.contextMode === 'staged-short-window';
  const runtime = resolve(ROOT, '.runtime/dsh-official-r1');
  const sdkPath = resolve(runtime, 'node_modules/@deepseek-ai/dsh-sdk-client/lib/index.js');
  for (const name of ['dsh', 'dsh-sdk-client', 'dsh-llm-deepseek']) {
    const pkg = JSON.parse(await readFile(resolve(runtime, `node_modules/@deepseek-ai/${name}/package.json`), 'utf8'));
    if (pkg.version !== '0.2.1-rc.2') throw new Error('pinned_runtime_version_required');
  }
  const sdk = await import(pathToFileURL(sdkPath));
  const patchPath = resolve(output, 'profile.patch.yml');
  const submissionPath = resolve(output, 'submission.json');
  const auditPath = resolve(output, 'budget.jsonl');
  let contextPath = null, contextSha = null;
  if (manifest.contextMode === 'full-candidate-context') {
    const completeContext = buildCompleteContext(rawCase);
    contextPath = resolve(output, 'complete-context.txt'); contextSha = digest(completeContext);
    await writeFile(contextPath, completeContext, { flag: 'wx' });
  }
  const endpointBase = 'https://api.deepseek.com/anthropic';
  const runtimePackageJson = resolve(runtime, 'package.json');
  const parserPath = createRequire(pathToFileURL(runtimePackageJson)).resolve('eventsource-parser');
  const boundFiles = [rawCasePath, parserPath, sdkPath, ...(contextPath ? [contextPath] : []),
    resolve(runtime, 'package-lock.json'),
    resolve(runtime, 'node_modules/@deepseek-ai/dsh/lib/bin.js'),
    resolve(runtime, 'node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js'),
    ...['evidence_tools.mjs', 'annotation_isolation.mjs', 'submission_shape.mjs', 'evidence_policy.mjs', 'prior_episode.mjs', 'profile.mjs', 'budget_plugin.mjs', 'budget_core.mjs', 'budget_counter.mjs', 'run_comparison.mjs',
      ...(staged ? ['staged_evidence.mjs', 'staged_presence.mjs'] : [])]
      .map(name => resolve(ROOT, 'integrations/dsh', name))];
  const sourceBindings = [...await Promise.all(boundFiles.map(async file => ({ file, sha256: digest(await readFile(file)) }))), ...prior.bindings];
  const budgetConfig = { endpoint: `${endpointBase}/v1/messages`, model: manifest.model,
    readinessPath: resolve(output, 'budget-ready.json'),
    ...(staged ? { stagedContext: true, stagedContextAuditPath: resolve(output, 'staged-presence.jsonl') } : {}),
    ...(contextPath ? { requiredContextPath: contextPath, requiredContextSha256: contextSha } : {}),
    runtimePackageJson, sourceBindings, journalPath: auditPath, episodeId: manifest.episodeId,
    priorKnownTokens: manifest.priorKnownTokens, maxCalls: manifest.maxCalls,
    maxOutputTokens: manifest.maxOutputTokens, wholeLimit: manifest.wholeLimit, safetyTokens: manifest.safetyTokens,
    pythonExecutable: resolve(ROOT, '.runtime/v20-embedding-env/Scripts/python.exe'),
    tokenizerPath: resolve(ROOT, '.runtime/v32-deepseek-tokenizer-r2/tokenizer.json'),
    tokenizerSha256: '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf' };
  const patch = officialProfilePatch({ casePath: rawCasePath, caseSha256: manifest.caseSha256,
    ...(manifest.contextMode ? { contextMode: manifest.contextMode } : {}),
    ...(manifest.selectionPolicy ? { selectionPolicy: manifest.selectionPolicy } : {}),
    annotationPolicy,
    ...(staged ? { previewCodepoints: manifest.previewCodepoints ?? 160 } : {}),
    submissionPath, sessionRoot: resolve(output, 'sessions'),
    budgetPluginPath: resolve(ROOT, 'integrations/dsh/budget_plugin.mjs'), budgetConfig,
    maxOutputTokens: manifest.maxOutputTokens, endpointBase });
  await writeFile(patchPath, typeof patch === 'string' ? patch : json(patch), { flag: 'wx' });
  const safeEnv = Object.fromEntries(['SystemRoot', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT']
    .filter(key => process.env[key] !== undefined).map(key => [key, process.env[key]]));
  let key = 'local-fixture-not-a-real-credential';
  if (!fixture && !initializeOnly) {
    const dotenv = await readFile(resolve(ROOT, '.env'), 'utf8');
    const match = dotenv.match(/^TRUST_MODEL_API_KEY\s*=\s*(.*?)\s*$/m);
    key = match?.[1]?.replace(/^(['"])(.*)\1$/, '$2');
    if (!key || !key.startsWith('sk-')) throw new Error('configured_key_missing');
  }
  safeEnv.DEEPSEEK_API_KEY = key;
  if (fixture) {
    safeEnv.NODE_OPTIONS = `--import=${pathToFileURL(resolve(ROOT, 'integrations/dsh/fixture_preload.mjs')).href}`;
    safeEnv.TRUST_DSH_FIXTURE_PLAN = resolve(ROOT, manifest.fixturePlan);
    safeEnv.TRUST_DSH_FIXTURE_WIRES = resolve(output, 'fixture-wire.jsonl');
  }
  const harness = new sdk.DeepSeekHarness({
    dshBin: resolve(runtime, 'node_modules/@deepseek-ai/dsh/lib/bin.js'),
    profile: 'sdk-minimal', patches: [patchPath],
    dshHome: resolve(output, 'home'), cwd: output, processCwd: runtime,
    env: safeEnv, provider: 'deepseek-official', model: manifest.model,
    reasoningEffort: manifest.reasoningEffort, maxTokens: manifest.maxOutputTokens,
    initializeTimeoutMs: 60000, requestTimeoutMs: 240000,
  });
  const trace = [], start = performance.now();
  let status = 'failed', error = null, cleanupError = null, finalResponse = '', sessionId = null, turnEnd = null;
  try {
    await harness.start();
    await verifyBudgetReady(budgetConfig.readinessPath, { episodeId: manifest.episodeId, sourceBindingsSha256: digest(JSON.stringify(sourceBindings)) });
    if (initializeOnly) status = 'initialized_without_prompt';
    else {
      const prompt = staged
        ? `专题：${rawCase.topic_title ?? ''}\n问题：${rawCase.question.question}\n先用read_selected_sources一次选择需要完整阅读的P地址；收到完整原段后调用submit_evidence。首轮只是逐字短窗，前后未知；保留双方说法、关键前提与限定，不据缺失推定不存在。`
        : manifest.selectionPolicy === 'source-local-v3-compact'
        ? '请审阅完整资料并调用submit_evidence。'
        : manifest.contextMode === 'full-candidate-context'
        ? `专题：${rawCase.topic_title ?? ''}\n问题：${rawCase.question.question}\n完整候选原文已在数据区。请审阅后只调用submit_evidence提交最终结构，兼顾正反证据、关键前提和限定条件。注释简明；时间原文没有就留空，数值不得补造。候选池外的材料保持未知。`
        : `专题：${rawCase.topic_title ?? ''}\n问题：${rawCase.question.question}\n请用原始资料工具研判，最终调用submit_evidence提交。检索兼顾题目各方说法、前提和限定条件；所有未看全文的段落保持未知。注释简明，时间原文没有就留空。`;
      const result = await harness.run(prompt, { onNotification(notification) {
        const event = notification.params?.event;
        const detail = {};
        if (event?.type === 'tool/call') Object.assign(detail, { tool: event.data?.name, arguments: event.data?.arguments });
        if (event?.type === 'tool/result') Object.assign(detail, { call_id: event.data?.message?.toolCallId,
          is_error: event.data?.message?.isError,
          ...(event.data?.message?.isError ? { content: event.data.message.content } : {}) });
        if (event?.type === 'turn/end') { turnEnd = event.data?.reason; detail.reason = turnEnd; }
        trace.push({ elapsed_ms: Math.round(performance.now() - start), method: notification.method,
          event_type: event?.type ?? event?.kind ?? null, ...detail });
      } });
      finalResponse = result.finalResponse; sessionId = result.sessionId;
      status = 'idle';
    }
  } catch (caught) {
    // SDK exceptions may contain arbitrary stderr; never print or store a credential.
    error = { name: caught.name, code: caught.code ?? null,
      message: String(caught.message).replace(/sk-[A-Za-z0-9_-]+/g, '[REDACTED]').slice(0, 12000) };
  } finally {
    try { await harness.close(); }
    catch (caught) { cleanupError = { name: caught.name, code: caught.code ?? null }; }
  }
  const usageRows = (await readFile(auditPath, 'utf8').catch(() => '')).split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
  const settlements = usageRows.filter(row => row.status === 'usage_settled' || row.status === 'usage_unknown');
  const startedRequests = usageRows.filter(row => row.status === 'request_started').length;
  const knownTokens = settlements.reduce((total, row) => total + (row.provider_usage?.total_tokens ?? row.partial_provider_usage?.total_tokens ?? 0), manifest.priorKnownTokens);
  const unknown = settlements.filter(row => row.status === 'usage_unknown').length + Math.max(0, startedRequests - settlements.length);
  const artifactBytes = await readFile(submissionPath).catch(() => null);
  const artifact = artifactBytes ? JSON.parse(artifactBytes) : null;
  const annotationAuditBytes = await readFile(resolve(output, 'submission-annotation-audit.json')).catch(() => null);
  const budgetBlock = usageRows.findLast(row => row.status === 'blocked');
  const stagedRows = staged ? (await readFile(resolve(output, 'staged-presence.jsonl'), 'utf8').catch(() => ''))
    .split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line)) : [];
  const stagedFailure = stagedRows.findLast(row => row.status === 'rejected_before_admission');
  const sdkStatus = status;
  if (!initializeOnly) status = unknown ? 'failed_usage_unknown' : budgetBlock ? 'budget_blocked'
    : stagedFailure ? 'context_validation_failed'
    : cleanupError ? 'failed_cleanup' : artifact && turnEnd?.kind === 'completed' ? 'closed_proposed_unverified' : status === 'idle' ? 'closed_without_submission' : status;
  if (status === 'closed_without_submission' && turnEnd?.kind === 'error') status = 'failed_model_turn';
  const summaryPath = resolve(output, 'sdk-summary.json');
  await writeNew(summaryPath, { status, sdkStatus, error, cleanupError, sessionId, turnEnd, finalResponse, trace });
  let officialSessionLog = null;
  async function findSessionLog(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true }).catch(() => [])) {
      const file = resolve(directory, entry.name);
      if (entry.isDirectory()) await findSessionLog(file);
      else if (entry.name === 'session.v4.jsonl' && dirname(file).endsWith(sessionId ?? '__missing__')) {
        if (officialSessionLog) throw new Error('ambiguous_official_session_log');
        officialSessionLog = { file, sha256: digest(await readFile(file)) };
      }
    }
  }
  if (sessionId) await findSessionLog(resolve(output, 'sessions'));
  const receipt = { format: 'dsh-comparison-closed-v1', closed: true, fixture, initializeOnly,
    manifest_sha256: digest(manifestBytes), case_sha256: manifest.caseSha256,
    runtime_version: '0.2.1-rc.2', sdk_version: '0.2.1-rc.2', model: manifest.model,
    thinking: 'disabled', native_protocol: 'DeepSeek Messages',
    sdk_summary_sha256: digest(await readFile(summaryPath)), official_session_log: officialSessionLog,
    prior_known_tokens: manifest.priorKnownTokens, prior_closed_receipt: manifest.priorClosedReceipt ?? null,
    context_mode: manifest.contextMode ?? 'tools', complete_context_sha256: contextSha,
    ...(staged ? { staged_presence_audit_sha256: digest(await readFile(resolve(output, 'staged-presence.jsonl')).catch(() => Buffer.alloc(0))) } : {}),
    selection_policy: manifest.selectionPolicy ?? 'baseline',
    selection_policy_sha256: policy ? digest(policy.text) : null,
    annotation_policy: annotationPolicy,
    annotation_audit_sha256: annotationAuditBytes ? digest(annotationAuditBytes) : null,
    annotation_quarantine_count: artifact?.quality?.annotation_quarantine_count ?? 0,
    optional_time_cleared_count: artifact?.quality?.optional_time_cleared_count ?? 0,
    status, sdk_status: sdkStatus, error_present: error !== null || turnEnd?.kind === 'error' || !!stagedFailure, cleanup_error: cleanupError,
    failure_code: stagedFailure?.code ?? budgetBlock?.code ?? error?.code ?? (turnEnd?.kind === 'error' ? 'model_turn_failed' : null),
    tool_validation_error_codes: safeToolErrorCodes(trace),
    turn_end: turnEnd, budget_block_code: budgetBlock?.code ?? null,
    model_requests: startedRequests, known_tokens: knownTokens, unknown_usage_requests: unknown,
    elapsed_seconds: (performance.now() - start) / 1000,
    submission_sha256: artifactBytes ? digest(artifactBytes) : null,
    evidence_count: artifact?.evidence?.length ?? 0, published: false,
    reference_read_during_generation: false, final_support_semantics_verified: false,
    budget_audit_sha256: digest(await readFile(auditPath).catch(() => Buffer.alloc(0))),
  };
  await writeNew(resolve(output, 'closed-receipt.json'), receipt);
  return receipt;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const manifestPath = resolve(process.argv[2] ?? '');
  runComparison({ manifestPath, fixture: process.argv.includes('--fixture'), initializeOnly: process.argv.includes('--initialize-only') })
    .then(value => process.stdout.write(json(value)))
    .catch(error => { process.stderr.write(`${error.name}: ${String(error.message).replace(/sk-[A-Za-z0-9_-]+/g, '[REDACTED]').slice(0, 2000)}\n`); process.exitCode = 1; });
}
