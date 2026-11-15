/** Cordis budget policy for the official DSH loop and native Messages transport. */
import { AsyncLocalStorage } from 'node:async_hooks';
import { createHash } from 'node:crypto';
import { open, readFile, writeFile } from 'node:fs/promises';
import { isAbsolute, basename, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { BudgetError, EpisodeBudgetGuard } from './budget_core.mjs';
import { createOfficialTokenizerCounter } from './budget_counter.mjs';

export const name = 'trust-judge-episode-budget';
export const inject = ['llm'];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const fail = code => { throw new BudgetError(code); };

/**
 * Installs only request policy, not an agent loop. Native final request bytes are
 * admitted immediately before fetch; DSH's own translated chunks settle usage.
 * fetchTarget is injectable solely for no-network integration fixtures.
 */
export function createBudgetIntegration({ guard, endpoint, model, createSseParser, requiredContext, validateNativeRequest, fetchTarget = globalThis }) {
  const target = new URL(endpoint);
  if (target.protocol !== 'https:' || target.username || target.password || target.search || target.hash
    || target.hostname !== 'api.deepseek.com' || target.pathname !== '/anthropic/v1/messages'
    || typeof model !== 'string' || !model) fail('invalid_native_endpoint');
  if (requiredContext !== undefined && (typeof requiredContext !== 'string' || requiredContext.length === 0)) {
    fail('invalid_required_context');
  }
  const storage = new AsyncLocalStorage();
  const originalFetch = fetchTarget.fetch;
  if (typeof originalFetch !== 'function') fail('missing_native_fetch');
  let disposed = false;
  const wrappedFetch = async (url, init) => {
    if (disposed) fail('budget_plugin_disposed');
    const state = storage.getStore();
    // No Models, Files, auxiliary services or unaccounted side calls in this profile.
    if (!state || String(url) !== endpoint || init?.method !== 'POST' || typeof init.body !== 'string') {
      fail('unaccounted_network_request');
    }
    let body;
    try { body = JSON.parse(init.body); } catch { fail('invalid_native_request'); }
    if (body.model !== model) fail('native_model_changed');
    if (requiredContext !== undefined) {
      if (typeof body.system !== 'string' || !body.system.includes(requiredContext)) fail('required_context_missing');
      if (!Array.isArray(body.tools) || body.tools.length !== 1 || body.tools[0]?.name !== 'submit_evidence') {
        fail('required_submit_only_tools');
      }
    }
    if (validateNativeRequest) await validateNativeRequest(body);
    const receipt = await state.beforeWire(init.body);
    if (sha(Buffer.from(init.body, 'utf8')) !== receipt.wire_sha256) fail('native_request_changed');
    const response = await originalFetch.call(fetchTarget, url, init);
    if (!response.body || !response.ok) return response;
    if (typeof createSseParser !== 'function') fail('missing_sse_usage_observer');
    const decoder = new TextDecoder();
    const parser = createSseParser({ onEvent(event) {
      // Inspect only provider usage buckets; never persist model text or reasoning.
      if (event.event && !['message_start', 'message_delta'].includes(event.event)) return;
      let value;
      try { value = JSON.parse(event.data); } catch { return; }
      const sample = value.type === 'message_start' ? value.message?.usage : value.type === 'message_delta' ? value.usage : undefined;
      if (!sample || typeof sample !== 'object') return;
      for (const [wire, local] of Object.entries({ input_tokens: 'inputTokens', output_tokens: 'outputTokens',
        cache_read_input_tokens: 'cacheReadTokens', cache_creation_input_tokens: 'cacheWriteTokens' })) {
        if (sample[wire] !== undefined) {
          state.observedUsage[local] = sample[wire]; state.reportedFields.add(local);
        }
      }
      state.observeProviderUsage(state.observedUsage);
    } });
    const tapped = response.body.pipeThrough(new TransformStream({
      transform(bytes, controller) { parser.feed(decoder.decode(bytes, { stream: true })); controller.enqueue(bytes); },
      flush() { parser.feed(decoder.decode()); },
    }));
    return new Response(tapped, { status: response.status, statusText: response.statusText, headers: response.headers });
  };
  fetchTarget.fetch = wrappedFetch;
  const listener = (options, next) => guard.protectNativeStream({ invoke: async function* (beforeWire, observeProviderUsage) {
    if (disposed || fetchTarget.fetch !== wrappedFetch) fail('native_fetch_guard_changed');
    if (options.provider !== 'deepseek-official' || options.model !== model || options.purpose !== undefined) {
      fail('unbudgeted_model_route');
    }
    const state = { beforeWire, observeProviderUsage, observedUsage: { inputTokens: 0, outputTokens: 0 }, reportedFields: new Set() };
    const iterator = storage.run(state, () => next()[Symbol.asyncIterator]());
    try {
      while (true) {
        const value = await storage.run(state, () => iterator.next());
        if (value.done) return;
        if (value.value?.type === 'usage' && typeof createSseParser === 'function'
          && (!state.reportedFields.has('inputTokens') || !state.reportedFields.has('outputTokens'))) {
          fail('unknown_provider_usage');
        }
        yield value.value;
      }
    } finally {
      if (iterator.return) await storage.run(state, () => iterator.return());
    }
  } });
  return { listener, guard, dispose() {
    disposed = true;
    if (fetchTarget.fetch === wrappedFetch) fetchTarget.fetch = originalFetch;
  } };
}

/** One new, non-reusable journal and explicit prior usage are required per episode. */
export async function apply(ctx, config) {
  if (!config || !Array.isArray(config.sourceBindings) || config.sourceBindings.length === 0
    || !isAbsolute(config.journalPath ?? '') || !isAbsolute(config.runtimePackageJson ?? '')) fail('invalid_budget_configuration');
  const bindings = config.sourceBindings.map(item => {
    if (!isAbsolute(item.file ?? '') || !/^[a-f0-9]{64}$/.test(item.sha256 ?? '')
      || /^\.env(?:\.|$)/i.test(basename(item.file))) fail('invalid_source_binding');
    return Object.freeze({ ...item });
  });
  const sourceCheck = async () => {
    for (const item of bindings) if (sha(await readFile(item.file)) !== item.sha256) return false;
    return true;
  };
  if (!await sourceCheck()) fail('source_binding_changed');
  let requiredContext;
  if (config.requiredContextPath !== undefined || config.requiredContextSha256 !== undefined) {
    if (!isAbsolute(config.requiredContextPath ?? '') || !/^[a-f0-9]{64}$/.test(config.requiredContextSha256 ?? '')) {
      fail('invalid_required_context_binding');
    }
    const identity = file => process.platform === 'win32' ? resolve(file).toLowerCase() : resolve(file);
    if (!bindings.some(item => identity(item.file) === identity(config.requiredContextPath)
      && item.sha256 === config.requiredContextSha256)) fail('missing_required_context_source_binding');
    const contextBytes = await readFile(config.requiredContextPath);
    if (sha(contextBytes) !== config.requiredContextSha256) fail('required_context_binding_changed');
    try { requiredContext = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(contextBytes); }
    catch { fail('invalid_required_context'); }
    if (requiredContext.length === 0) fail('invalid_required_context');
  }
  const parserPath = createRequire(pathToFileURL(config.runtimePackageJson)).resolve('eventsource-parser');
  const identity = file => process.platform === 'win32' ? resolve(file).toLowerCase() : resolve(file);
  if (!bindings.some(item => identity(item.file) === identity(parserPath))) fail('missing_parser_source_binding');
  const { createParser } = await import(pathToFileURL(parserPath));
  const counter = createOfficialTokenizerCounter(config);
  const journal = await open(config.journalPath, 'wx');
  let integration;
  let unbindStaged;
  let stagedJournal;
  try {
    const guard = new EpisodeBudgetGuard({ episodeId: config.episodeId, priorKnownTokens: config.priorKnownTokens,
      wholeLimit: config.wholeLimit, safetyTokens: config.safetyTokens, maxCalls: config.maxCalls,
      maxOutputTokens: config.maxOutputTokens, countInputTokens: counter, sourceCheck,
      record: async event => { await journal.write(`${JSON.stringify(event)}\n`); await journal.sync(); } });
    let validateNativeRequest;
    if (config.stagedContext === true) {
      if (requiredContext !== undefined || !isAbsolute(config.stagedContextAuditPath ?? '')) fail('invalid_staged_context_configuration');
      for (const file of ['staged_evidence.mjs', 'staged_presence.mjs']) {
        if (!bindings.some(item => identity(item.file) === identity(fileURLToPath(new URL(file, import.meta.url))))) fail('missing_staged_source_binding');
      }
      const { bindStagedAdmission, getStagedPresence } = await import('./staged_evidence.mjs');
      const { verifyStagedNativePresence, screenStagedRead } = await import('./staged_presence.mjs');
      stagedJournal = await open(config.stagedContextAuditPath, 'wx');
      unbindStaged = bindStagedAdmission(config.episodeId, readRequest => screenStagedRead({ guard, countTokens: counter,
        readRequest, maxOutputTokens: config.maxOutputTokens, safetyTokens: config.safetyTokens }));
      ctx.provide('trustJudgeStagedAdmission', { episodeId: config.episodeId });
      validateNativeRequest = async body => {
        try {
          const proof = verifyStagedNativePresence(body, getStagedPresence(config.episodeId));
          await stagedJournal.write(`${JSON.stringify(proof)}\n`); await stagedJournal.sync();
        } catch (error) {
          const code = error instanceof BudgetError ? error.code : 'staged_context_policy_failed';
          await stagedJournal.write(`${JSON.stringify({ status: 'rejected_before_admission', code })}\n`); await stagedJournal.sync();
          throw new BudgetError(code);
        }
      };
    }
    integration = createBudgetIntegration({ guard, endpoint: config.endpoint, model: config.model,
      createSseParser: createParser, requiredContext, validateNativeRequest });
    const remove = ctx.on('llm/stream', integration.listener);
    ctx.effect(() => async () => {
      integration.dispose();
      if (typeof remove === 'function') remove();
      unbindStaged?.(); await stagedJournal?.close();
      await journal.close();
    });
    if (config.readinessPath !== undefined) {
      if (!isAbsolute(config.readinessPath)) fail('invalid_budget_readiness_path');
      await writeFile(config.readinessPath, `${JSON.stringify({ format: 'dsh-budget-ready-v1', episode_id: config.episodeId,
        source_bindings_sha256: sha(JSON.stringify(config.sourceBindings)), guarded_native_transport_installed: true })}\n`, { flag: 'wx' });
    }
    // Cordis treats a returned value as a cleanup effect; no ordinary object may escape.
  } catch (error) {
    integration?.dispose(); unbindStaged?.(); await stagedJournal?.close(); await journal.close(); throw error;
  }
}
