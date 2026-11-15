/** Fixed Python business subprocess; never a model-facing shell or filesystem tool. */
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, open, writeFile } from 'node:fs/promises';
import { isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildHandleManifest, handleReadArguments, handleReadView, handleNavigationView,
  handleSubmitParameters, handleSubmitArguments, parseUniqueNativeArguments, stableHandleJson } from './bundle_handles.mjs';

const SCRIPT = fileURLToPath(new URL('../../scripts/source_bundle_tool.py', import.meta.url));
const sha = value => createHash('sha256').update(value).digest('hex');
const clone = value => structuredClone(value);
const fail = code => { const error = new Error(code); error.code = code; throw error; };
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const safeCode = value => typeof value === 'string' && /^(?:source_bundle(?:_tools)?|bundle_bridge)_[a-z0-9_]+$/.test(value);

/** Bind one immutable raw case and program. Full result text exists only in read results, not replay history. */
export async function createBundleBridge(config) {
  const allowed = ['casePath', 'caseSha256', 'pythonExecutable', 'scriptSha256', 'journalPath', 'timeoutMs',
    'submitRequestPath', 'toolSubmissionPath', 'sourceBindings'];
  if (!object(config) || Object.keys(config).some(key => !allowed.includes(key))
      || ![config.casePath, config.pythonExecutable].every(value => typeof value === 'string' && isAbsolute(value))
      || ![config.caseSha256, config.scriptSha256].every(value => /^[0-9a-f]{64}$/.test(value ?? ''))
      || (config.journalPath !== undefined && (!isAbsolute(config.journalPath) || config.journalPath === config.casePath))) fail('bundle_bridge_invalid_configuration');
  for (const path of [config.submitRequestPath, config.toolSubmissionPath].filter(value => value !== undefined)) {
    if (!isAbsolute(path) || [config.casePath, config.journalPath].includes(path)) fail('bundle_bridge_separate_submission_paths_required');
  }
  if (config.submitRequestPath && config.submitRequestPath === config.toolSubmissionPath) fail('bundle_bridge_separate_submission_paths_required');
  const bindings = clone(config.sourceBindings ?? []);
  if (!Array.isArray(bindings) || bindings.some(row => !isAbsolute(row?.file ?? '') || !/^[a-f0-9]{64}$/.test(row?.sha256 ?? ''))) fail('bundle_bridge_source_bindings_required');
  const timeoutMs = config.timeoutMs ?? 15000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) fail('bundle_bridge_invalid_timeout');
  const history = [], reads = [], events = [];
  let scope, busy = false, submitted = false;
  const journal = config.journalPath ? await open(config.journalPath, 'wx') : null;
  const env = Object.fromEntries(['SystemRoot', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP']
    .filter(key => process.env[key] !== undefined).map(key => [key, process.env[key]]));
  async function invoke(operation, args, signal) {
    if (signal?.aborted) fail('bundle_bridge_aborted');
    if (sha(await readFile(config.casePath)) !== config.caseSha256 || sha(await readFile(SCRIPT)) !== config.scriptSha256) fail('bundle_bridge_bound_source_changed');
    for (const row of bindings) if (sha(await readFile(row.file)) !== row.sha256) fail('bundle_bridge_bound_source_changed');
    const request = JSON.stringify({ operation, arguments: args, history });
    if (Buffer.byteLength(request) > 2000000) fail('bundle_bridge_request_too_large');
    const stdout = await new Promise((resolve, reject) => {
      const child = execFile(config.pythonExecutable, ['-I', SCRIPT, config.casePath, config.caseSha256],
        { env, windowsHide: true, timeout: timeoutMs, maxBuffer: 10000000, encoding: 'utf8', signal },
        (error, stdout) => {
          if (error && !stdout) { reject(Object.assign(new Error('bundle_bridge_subprocess_failed'), { code: 'bundle_bridge_subprocess_failed' })); return; }
          resolve(stdout);
        });
      child.stdin.on('error', () => {});
      child.stdin.end(request, 'utf8');
    });
    let response;
    try { response = JSON.parse(stdout); } catch { fail('bundle_bridge_invalid_subprocess_response'); }
    if (!object(response) || response.format !== 'source-bundle-tool-response-v1'
        || response.case_sha256 !== config.caseSha256 || !['ok', 'rejected'].includes(response.status)
        || !object(response.history_entry) || response.model_calls !== 0 || response.publishable !== false) {
      fail(safeCode(response?.error_code) ? response.error_code : 'bundle_bridge_unbound_subprocess_response');
    }
    return response;
  }
  async function call(operation, args, { signal, callId } = {}) {
    if (busy || submitted) fail(submitted ? 'bundle_bridge_already_submitted' : 'bundle_bridge_concurrent_call_forbidden');
    if (!['search', 'plan', 'read', 'cite', 'submit'].includes(operation) || !object(args)) fail('bundle_bridge_unknown_operation');
    if (operation === 'read' && (typeof callId !== 'string' || !callId || reads.some(row => row.tool_use_id === callId))) fail('bundle_bridge_unique_native_read_call_id_required');
    busy = true;
    try {
      const response = await invoke(operation, args, signal);
      if (signal?.aborted) fail('bundle_bridge_aborted');
      const entry = clone(response.history_entry);
      if (operation !== 'submit') history.push(entry);
      scope = response.scope;
      const event = { sequence: events.length + 1, operation, status: response.status,
        history_entry: entry, ...(callId ? { tool_use_id: callId } : {}), case_sha256: config.caseSha256 };
      events.push(event);
      if (journal) { await journal.write(`${JSON.stringify(event)}\n`); await journal.sync(); }
      if (response.status !== 'ok') fail(safeCode(response.error_code) ? response.error_code : 'bundle_bridge_tool_rejected');
      if (operation === 'read') reads.push({ tool_use_id: callId, arguments: clone(args), result: clone(response.result) });
      if (operation === 'submit') {
        if (config.submitRequestPath) await writeFile(config.submitRequestPath, `${JSON.stringify({ operation, arguments: args, history })}\n`, { flag: 'wx' });
        if (config.toolSubmissionPath) await writeFile(config.toolSubmissionPath, `${JSON.stringify(response)}\n`, { flag: 'wx' });
        submitted = true;
      }
      return clone(response.result);
    } finally { busy = false; }
  }
  try { scope = (await invoke('inspect', {})).result; }
  catch (error) { await journal?.close(); throw error; }
  return { call,
    snapshot: () => ({ scope: clone(scope), history: clone(history), reads: clone(reads), events: clone(events), submitted,
      model_calls: 0, publishable: false, native_model_delivery_verified: false }),
    close: () => journal?.close() };
}

/** Compact native projection over the unchanged UUID business operations. */
export async function createBundleHandleBridge(bridge, config, navigation, paths) {
  if (!object(paths) || ![paths.handlesPath, paths.nativeProjectionsPath].every(path => isAbsolute(path ?? ''))
      || paths.handlesPath === paths.nativeProjectionsPath || [paths.handlesPath, paths.nativeProjectionsPath].includes(config.casePath)) fail('bundle_handles_separate_archives_required');
  const bytes = await readFile(config.casePath);
  if (sha(bytes) !== config.caseSha256) fail('bundle_bridge_bound_source_changed');
  const raw = JSON.parse(bytes), search = bridge.snapshot().history.find(row => row.operation === 'search' && row.status === 'ok');
  if (!search) fail('bundle_handles_bound_navigation_required');
  const planned = await bridge.call('plan', { anchor_paragraph_ids: navigation.results.map(row => row.paragraph_id), before: 1, after: 3 },
    { callId: 'program-source-bundle-handle-planning' });
  const manifest = buildHandleManifest(raw, navigation, planned,
    { case_sha256: config.caseSha256, backend_navigation_sha256: search.result_sha256 });
  const manifestBytes = `${stableHandleJson(manifest)}\n`, handlesSha256 = sha(manifestBytes);
  await writeFile(paths.handlesPath, manifestBytes, { flag: 'wx' });
  const journal = await open(paths.nativeProjectionsPath, 'wx'), nativeReads = [];
  async function validate(args, rawArguments) {
    const parsed = parseUniqueNativeArguments(rawArguments);
    if (stableHandleJson(parsed) !== stableHandleJson(args)) fail('bundle_handles_raw_arguments_changed');
    if (sha(await readFile(paths.handlesPath)) !== handlesSha256) fail('bundle_handles_projection_archive_changed');
  }
  async function record(value) { await journal.write(`${stableHandleJson(value)}\n`); await journal.sync(); }
  return { manifest: clone(manifest), manifestSha256: handlesSha256,
    navigationView: withdrawn => handleNavigationView(manifest, navigation, withdrawn),
    submitParameters: () => handleSubmitParameters(manifest, bridge.snapshot().reads),
    nativeReads: () => clone(nativeReads),
    async read(args, { signal, callId, rawArguments }) {
      await validate(args, rawArguments);
      const business = handleReadArguments(manifest, args);
      const canonical = await bridge.call('read', business, { signal, callId });
      const view = handleReadView(manifest, canonical, args);
      const row = { operation: 'read', tool_use_id: callId, raw_native_arguments: rawArguments, native_arguments: clone(args),
        business_arguments: business, canonical_result: canonical, canonical_result_sha256: bridge.snapshot().history.at(-1).result_sha256,
        native_result: view, native_read_view_sha256: sha(stableHandleJson(view)), handles_sha256: handlesSha256 };
      await record(row); nativeReads.push(row);
      return view;
    },
    async submit(args, { signal, callId, rawArguments }) {
      await validate(args, rawArguments);
      const business = handleSubmitArguments(manifest, bridge.snapshot().reads, args);
      const canonical = await bridge.call('submit', business, { signal, callId });
      await record({ operation: 'submit', tool_use_id: callId, raw_native_arguments: rawArguments, native_arguments: clone(args),
        business_arguments: business, canonical_result: canonical, canonical_result_sha256: sha(stableHandleJson(canonical)), handles_sha256: handlesSha256 });
      return canonical;
    },
    close: () => journal.close() };
}

