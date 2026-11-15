/** One explicit workbench selection episode in the official SDK. No custom loop. */
import { readFile, writeFile, mkdir, readdir, stat } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildBundleWorkbenchPatch } from './bundle_profile.mjs';
import { digest, canonical, fail, loadWorkbenchManifest, readBound } from './bundle_workbench_contract.mjs';
import { publicBundleToolFailure } from './bundle_public_errors.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const save = (file, value) => writeFile(file, `${JSON.stringify(value)}\n`, { flag: 'wx' });
const bind = async file => ({ file, sha256: digest(await readFile(file)) });
const jsonLines = async file => (await readFile(file, 'utf8').catch(() => '')).split(/\r?\n/).filter(Boolean).map(JSON.parse);
const safe = error => ({ name: /^[A-Za-z0-9_]+$/.test(error?.name ?? '') ? error.name : 'Error',
  code: /^[a-z][a-z0-9_]{0,100}$/.test(error?.code ?? '') ? error.code : 'bundle_workbench_runtime_failed' });
const notify = (event, data = {}) => process.stdout.write(`${JSON.stringify({ type: 'bundle_workbench_event', event, data })}\n`);
async function ready(path, predicate) {
  for (let count = 0; count < 100; count++) {
    const value = await readFile(path, 'utf8').then(JSON.parse).catch(() => null);
    if (value) { if (!predicate(value)) fail('bundle_workbench_readiness_mismatch'); return value; }
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  fail('bundle_workbench_readiness_missing');
}

export async function runBundleWorkbench(manifestPath, manifestSha256) {
  const authorized = await loadWorkbenchManifest(manifestPath, manifestSha256);
  const { manifest, raw, parentTokens, effectiveFrontStop, maxNativeRequests } = authorized;
  const output = resolve(manifest.output_directory);
  const authorizedDirectory = resolve(dirname(manifestPath)).replaceAll('\\', '/').toLowerCase();
  if (!output.replaceAll('\\', '/').toLowerCase().startsWith(`${authorizedDirectory}/`)) fail('bundle_workbench_isolated_authorized_output_required');
  await mkdir(output, { recursive: false });
  await save(resolve(output, 'launch-start.json'), { manifest: { file: manifestPath, sha256: manifestSha256 },
    execution_mode: manifest.execution_mode, question_id: raw.question.question_id, started_at: new Date().toISOString() });
  const runtime = resolve(ROOT, '.runtime/dsh-official-r1'), packageJson = resolve(runtime, 'package.json');
  const sdkPath = resolve(runtime, 'node_modules/@deepseek-ai/dsh-sdk-client/lib/index.js');
  const parser = createRequire(pathToFileURL(packageJson)).resolve('eventsource-parser');
  for (const name of ['dsh', 'dsh-sdk-client', 'dsh-llm-deepseek']) {
    if (JSON.parse(await readFile(resolve(runtime, `node_modules/@deepseek-ai/${name}/package.json`))).version !== '0.2.1-rc.2') fail('bundle_workbench_pinned_runtime_required');
  }
  const sources = [{ file: manifestPath, sha256: manifestSha256 }, manifest.raw_case, manifest.documents, manifest.paragraphs, manifest.previous_usage];
  for (const [relative, archived] of Object.entries(manifest.source_archives)) {
    const current = resolve(ROOT, relative);
    if (digest(await readFile(current)) !== archived.sha256) fail('bundle_workbench_archived_executable_changed');
    sources.push({ file: current, sha256: archived.sha256 }, archived);
  }
  for (const file of [parser, sdkPath, resolve(runtime, 'package-lock.json'), resolve(runtime, 'node_modules/@deepseek-ai/dsh/lib/bin.js'),
    resolve(runtime, 'node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js')]) sources.push(await bind(file));
  if (manifest.fixture_plan) sources.push(manifest.fixture_plan);
  const sourceBindings = [...new Map(sources.map(row => [row.file, row])).values()];
  const bridge = { casePath: manifest.raw_case.file, caseSha256: manifest.raw_case.sha256,
    pythonExecutable: resolve(ROOT, '.venv/Scripts/python.exe'), scriptSha256: digest(await readFile(resolve(ROOT, 'scripts/source_bundle_tool.py'))),
    journalPath: resolve(output, 'business-transcript.jsonl'), submitRequestPath: resolve(output, 'submit-request.json'),
    toolSubmissionPath: resolve(output, 'tool-submission.json'), sourceBindings };
  const guardConfig = { authorization: { file: manifestPath, sha256: manifestSha256 }, endpoint: 'https://api.deepseek.com/anthropic/v1/messages',
    model: manifest.model, runtimePackageJson: packageJson, sourceBindings, episodeId: manifest.authorization.episode_id,
    priorKnownTokens: parentTokens, wholeLimit: effectiveFrontStop, maxCalls: maxNativeRequests,
    maxOutputTokens: Math.max(...Object.values(manifest.output_capacity)), safetyTokens: manifest.safety_tokens,
    journalPath: resolve(output, 'budget.jsonl'), presencePath: resolve(output, 'native-presence.jsonl'),
    requestIndexPath: resolve(output, 'native-request-index.jsonl'), readinessPath: resolve(output, 'guard-ready.json'),
    pythonExecutable: resolve(ROOT, '.runtime/v20-embedding-env/Scripts/python.exe'),
    tokenizerPath: resolve(ROOT, '.runtime/v32-deepseek-tokenizer-r2/tokenizer.json'),
    tokenizerSha256: '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf' };
  const submissionPath = resolve(output, 'artifact.json'), businessReady = resolve(output, 'business-ready.json');
  const patch = buildBundleWorkbenchPatch({ bridge, guardConfig, submissionPath, sessionRoot: resolve(output, 'sessions'),
    businessReadinessPath: businessReady, outputCapacity: manifest.output_capacity, compactHandles: manifest.compact_handles === true });
  const patchPath = resolve(output, 'profile.patch.yml'); await save(patchPath, patch);
  const env = Object.fromEntries(['SystemRoot', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT']
    .filter(key => process.env[key] !== undefined).map(key => [key, process.env[key]]));
  if (manifest.execution_mode === 'fixture') {
    Object.assign(env, { DEEPSEEK_API_KEY: 'explicit-workbench-fixture-not-a-credential', TRUST_DSH_BUNDLE_FIXTURE: '1',
      TRUST_DSH_BUNDLE_FIXTURE_PLAN: manifest.fixture_plan.file, TRUST_DSH_BUNDLE_FIXTURE_WIRES: resolve(output, 'fixture-native-wire.jsonl'),
      NODE_OPTIONS: `--import=${pathToFileURL(resolve(ROOT, 'integrations/dsh/bundle_fixture_preload.mjs')).href}` });
  } else {
    if (typeof process.env.TRUST_BUNDLE_WORKBENCH_KEY !== 'string' || !process.env.TRUST_BUNDLE_WORKBENCH_KEY) fail('bundle_workbench_configured_key_missing');
    env.DEEPSEEK_API_KEY = process.env.TRUST_BUNDLE_WORKBENCH_KEY;
  }
  const { DeepSeekHarness } = await import(pathToFileURL(sdkPath));
  const harness = new DeepSeekHarness({ dshBin: resolve(runtime, 'node_modules/@deepseek-ai/dsh/lib/bin.js'), profile: 'sdk-minimal',
    patches: [patchPath], dshHome: resolve(output, 'home'), cwd: output, processCwd: runtime, env,
    provider: 'deepseek-official', model: manifest.model, reasoningEffort: 'off', maxTokens: guardConfig.maxOutputTokens,
    initializeTimeoutMs: 60000, requestTimeoutMs: manifest.timeout_seconds * 1000 });
  let turnEnd, error = null, cancelled = false, closePromise, sessionId = null;
  const trace = [];
  const close = () => closePromise ??= harness.close();
  const cancel = () => { cancelled = true; void close().catch(() => {}); };
  const timer = setTimeout(cancel, manifest.timeout_seconds * 1000);
  let observedUsageEvents = 0, observedNavigation = false, polling = false;
  const poll = async () => {
    if (polling) return;
    polling = true;
    try {
      await stat(resolve(output, 'cancel-requested.json')).then(cancel, () => {});
      if (!observedNavigation) {
        const file = resolve(output, 'program-navigation.json');
        const bytes = await readFile(file).catch(() => null);
        const navigation = bytes ? JSON.parse(bytes) : null;
        if (navigation?.format === 'program-source-bundle-navigation-archive-v1' && navigation.status === 'completed'
            && navigation.case_sha256 === manifest.raw_case.sha256 && navigation.model_calls === 0) {
          notify('prefetch_completed', { tool: 'search_source_bundles', call_id: 'program-original-question-navigation',
            candidate_count: navigation.candidate_count, completed_at: navigation.completed_at, model_calls: 0,
            navigation_archive: { file, sha256: digest(bytes) } });
          observedNavigation = true;
        }
      }
      const text = await readFile(guardConfig.journalPath, 'utf8').catch(() => '');
      const completeLines = text.split('\n').slice(0, -1);
      for (const line of completeLines.slice(observedUsageEvents)) {
        if (!line) continue;
        const row = JSON.parse(line);
        if (row.status === 'request_started') notify('native_request_started', { request_sequence: row.request_sequence,
          estimated_input_tokens: row.estimated_input_tokens, reserved_output_tokens: row.reserved_output_tokens });
        if (['usage_settled', 'usage_unknown'].includes(row.status)) notify('native_usage_settled', {
          request_sequence: row.request_sequence, known_usage: row.status === 'usage_settled',
          provider_usage: row.provider_usage, known_episode_tokens: row.known_episode_tokens_after,
          unknown_usage_requests: row.unknown_usage_requests, response_completed: row.response_completed, halted: row.halted });
      }
      observedUsageEvents = completeLines.length;
    } finally { polling = false; }
  };
  const watcher = setInterval(() => { void poll().catch(() => {}); }, 100);
  process.once('SIGINT', cancel); process.once('SIGTERM', cancel);
  try {
    await harness.start();
    await ready(guardConfig.readinessPath, value => value.format === 'dsh-source-bundle-workbench-ready-v1'
      && value.authorization_sha256 === manifestSha256 && value.source_bindings_sha256 === digest(JSON.stringify(sourceBindings))
      && value.native_guard_installed === true && value.episode_id === manifest.authorization.episode_id);
    await ready(businessReady, value => value.format === 'dsh-source-bundle-business-ready-v1'
      && value.case_sha256 === manifest.raw_case.sha256 && value.episode_id === manifest.authorization.episode_id);
    if (cancelled) fail('bundle_workbench_cancelled');
    notify('selection_started', { native_max_requests: maxNativeRequests, parent_tokens: parentTokens, front_stop: effectiveFrontStop,
      tail_reserved_tokens: manifest.tail_reserved_tokens });
    const result = await harness.run('按完整原题搜索原始资料，完整阅读所需来源后提交未验的证据范围与注释。覆盖竞争说法、关键前提及限定；不得补造来源、时间或数字。', {
      onNotification(notification) {
        const event = notification.params?.event;
        if (event?.type === 'turn/end') turnEnd = event.data?.reason;
        if (event?.type === 'tool/call') {
          trace.push({ type: 'tool/call', name: event.data?.name, arguments: event.data?.arguments, call_id: event.data?.callId });
          notify('tool_started', { tool: event.data?.name, call_id: event.data?.callId });
        }
        if (event?.type === 'tool/result') {
          trace.push({ type: 'tool/result', call_id: event.data?.message?.toolCallId, is_error: event.data?.message?.isError,
            ...(event.data?.message?.isError ? { content: event.data.message.content } : {}) });
          notify('tool_completed', { call_id: event.data?.message?.toolCallId, failed: event.data?.message?.isError === true,
            ...(event.data?.message?.isError === true ? publicBundleToolFailure(event.data.message.content) : {}),
            deferred: event.data?.message?.isError === true && ['DEFERRED:', 'NOT_READ:'].some(prefix => JSON.stringify(event.data.message.content).includes(prefix)) });
        }
      },
    });
    sessionId = result.sessionId;
  } catch (caught) { error = safe(caught); }
  finally {
    clearInterval(watcher); clearTimeout(timer); process.removeListener('SIGINT', cancel); process.removeListener('SIGTERM', cancel);
    try { await close(); } catch (caught) { error ??= safe(caught); }
    await poll();
  }
  const ledger = await jsonLines(guardConfig.journalPath), started = ledger.filter(row => row.status === 'request_started');
  const settled = ledger.filter(row => ['usage_settled', 'usage_unknown'].includes(row.status));
  const requestRows = await jsonLines(guardConfig.requestIndexPath);
  const nativeRequests = started.map(row => requestRows.find(item => item.native_request.sha256 === row.wire_sha256)?.native_request).filter(Boolean);
  const completeUsage = started.length === settled.length && settled.every(row => row.status === 'usage_settled' && row.unknown_usage_requests === 0
    && row.provider_usage && ['input_tokens', 'output_tokens', 'total_tokens'].every(key => Number.isSafeInteger(row.provider_usage[key]) && row.provider_usage[key] >= 0)
    && row.provider_usage.input_tokens + row.provider_usage.output_tokens === row.provider_usage.total_tokens);
  const newTokens = settled.reduce((sum, row) => sum + (row.provider_usage?.total_tokens ?? row.partial_provider_usage?.total_tokens ?? 0), 0);
  const artifact = await readFile(submissionPath, 'utf8').then(JSON.parse).catch(() => null);
  let excluded = false;
  const byId = new Map(raw.paragraphs.map(row => [row.id, row]));
  for (const evidence of artifact?.evidence ?? []) for (const span of manifest.excluded_spans) {
    const original = byId.get(span[0]);
    if (original.document_id === evidence.document_id && original.document_start_cp + span[1] < evidence.end_cp
        && evidence.start_cp < original.document_start_cp + span[2]) excluded = true;
  }
  const presenceRows = await jsonLines(guardConfig.presencePath);
  const blocked = ledger.findLast(row => row.status === 'blocked');
  const presenceFailure = presenceRows.findLast(row => row.status === 'rejected_before_admission');
  const incompleteResponse = settled.some(row => row.halted === true || row.response_completed !== true);
  const lastBusinessFailure = trace.filter(row => row.type === 'tool/result' && row.is_error === true)
    .map(row => publicBundleToolFailure(row.content).failure_code).filter(Boolean).at(-1) ?? null;
  const status = cancelled ? 'cancelled' : !completeUsage ? 'failed_usage_unknown' : excluded ? 'failed_excluded_span'
    : blocked ? 'budget_blocked' : presenceFailure ? 'failed_native_presence' : incompleteResponse ? 'failed_native_response' : error ? 'failed'
    : artifact && turnEnd?.kind === 'completed' && nativeRequests.length === started.length ? 'succeeded' : 'failed_without_submission';
  const usageReport = { new_request_count: started.length, new_tokens: newTokens, parent_tokens: parentTokens,
    whole_episode_tokens: parentTokens + newTokens, known_usage_complete: completeUsage,
    real_api_requests: manifest.execution_mode === 'fixture' ? 0 : started.length,
    simulated_usage_not_bill: manifest.execution_mode === 'fixture', front_token_stop: effectiveFrontStop,
    reserved_tail_tokens: manifest.tail_reserved_tokens, whole_token_stop: manifest.whole_token_stop };
  await save(resolve(output, 'usage-report.json'), usageReport);
  await save(resolve(output, 'sdk-summary.json'), { status, turnEnd, error, cancelled, session_id: sessionId, trace });
  const receipt = { format: 'dsh-source-bundle-closed-receipt-v1', status, episode_closed: true,
    source_binding: artifact?.source_binding ?? (await readFile(businessReady, 'utf8').then(JSON.parse).catch(() => null))?.source_binding ?? null,
    raw_case_sha256: manifest.raw_case.sha256, artifact_sha256: artifact ? digest(await readFile(submissionPath)) : null,
    native_requests: nativeRequests, ...usageReport, execution_mode: manifest.execution_mode,
    failure_code: cancelled ? 'bundle_workbench_cancelled' : !completeUsage ? 'experiment_usage_unknown'
      : excluded ? 'bundle_workbench_operator_excluded_span_selected' : blocked?.code ?? presenceFailure?.code ?? error?.code
      ?? (incompleteResponse ? 'bundle_workbench_native_response_incomplete' : null)
      ?? (status === 'succeeded' ? null : 'bundle_workbench_submission_missing'),
    source_bindings: sourceBindings, model: manifest.model, publishable: false,
    last_business_failure_code: lastBusinessFailure };
  await save(resolve(output, 'closed-receipt.json'), receipt);
  notify('selection_closed', { status, ...usageReport, failure_code: receipt.failure_code });
  return receipt;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 4) throw new Error('bundle_workbench_bound_manifest_arguments_required');
  runBundleWorkbench(resolve(process.argv[2]), process.argv[3]).then(receipt => {
    process.stdout.write(`${JSON.stringify({ type: 'bundle_workbench_result', status: receipt.status })}\n`);
    if (receipt.status !== 'succeeded') process.exitCode = 2;
  }).catch(error => { process.stderr.write(`${JSON.stringify(safe(error))}\n`); process.exitCode = 2; });
}

