/** Explicit workbench native guard. Owns transport policy, not an agent loop. */
import { open, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createBudgetIntegration } from './budget_plugin.mjs';
import { BudgetError, EpisodeBudgetGuard } from './budget_core.mjs';
import { createOfficialTokenizerCounter } from './budget_counter.mjs';
import { bindBundleGuard } from './bundle_evidence.mjs';
import { verifyBundleNativePresence } from './bundle_presence.mjs';
import { digest, loadWorkbenchManifest, readBound, fail } from './bundle_workbench_contract.mjs';

export const name = 'trust-judge-source-bundle-workbench-guard';
export const inject = ['llm'];

export async function apply(ctx, config) {
  const authorized = await loadWorkbenchManifest(config.authorization.file, config.authorization.sha256);
  const { manifest, effectiveFrontStop, maxNativeRequests, parentTokens } = authorized;
  const marker = globalThis[Symbol.for('trust-judge:bundle-fixture-transport')] === true;
  if ((manifest.execution_mode === 'fixture') !== marker) fail('bundle_workbench_execution_transport_mismatch');
  if (config.episodeId !== manifest.authorization.episode_id || config.wholeLimit !== effectiveFrontStop
      || config.priorKnownTokens !== parentTokens || config.maxCalls !== maxNativeRequests
      || config.maxOutputTokens !== Math.max(...Object.values(manifest.output_capacity))) fail('bundle_workbench_guard_budget_mismatch');
  const bindings = structuredClone(config.sourceBindings);
  if (!Array.isArray(bindings) || !bindings.some(row => row.file === config.authorization.file && row.sha256 === config.authorization.sha256)) fail('bundle_workbench_authorization_binding_required');
  const sourceCheck = async () => {
    try { for (const row of bindings) await readBound(row); return true; } catch { return false; }
  };
  if (!await sourceCheck()) fail('bundle_workbench_source_binding_changed');
  const tokenizer = createOfficialTokenizerCounter(config);
  const parserPath = createRequire(pathToFileURL(config.runtimePackageJson)).resolve('eventsource-parser');
  if (!bindings.some(row => row.file === parserPath)) fail('bundle_workbench_parser_binding_required');
  const { createParser } = await import(pathToFileURL(parserPath));
  const journal = await open(config.journalPath, 'wx'), presence = await open(config.presencePath, 'wx');
  const requestIndex = await open(config.requestIndexPath, 'wx');
  const originalFetch = globalThis.fetch, slot = bindBundleGuard(config.episodeId);
  const bodies = new Map();
  let ordinal = 0, integration, remove;
  const countInputTokens = async request => {
    ordinal++;
    const file = resolve(manifest.output_directory, `native-request-${String(ordinal).padStart(2, '0')}.json`);
    await writeFile(file, request.serialized_body, { flag: 'wx' });
    const binding = { file, sha256: digest(Buffer.from(request.serialized_body, 'utf8')) };
    const estimate = await tokenizer(request);
    bodies.set(binding.sha256, { ordinal, binding });
    await requestIndex.write(`${JSON.stringify({ ordinal, native_request: binding, estimated_input_tokens: estimate })}\n`);
    await requestIndex.sync();
    return estimate;
  };
  const archiveFetch = async (url, init) => {
    const row = bodies.get(digest(Buffer.from(init.body, 'utf8')));
    if (!row) throw new BudgetError('bundle_workbench_native_request_not_archived');
    const response = await originalFetch(url, init);
    const prefix = resolve(manifest.output_directory, `native-response-${String(row.ordinal).padStart(2, '0')}`);
    const raw = await open(`${prefix}.sse`, 'wx');
    await writeFile(`${prefix}-http.json`, `${JSON.stringify({ status: response.status, request: row.binding,
      credentials_or_headers_logged: false })}\n`, { flag: 'wx' });
    if (!response.body) { await raw.close(); return response; }
    let closed = false;
    const close = async complete => {
      if (closed) return;
      closed = true;
      await raw.sync(); await raw.close();
      await writeFile(`${prefix}-closed.json`, `${JSON.stringify({ response_stream_complete: complete,
        raw_response_sha256: digest(await readFile(`${prefix}.sse`)) })}\n`, { flag: 'wx' });
    };
    const reader = response.body.getReader();
    const stream = new ReadableStream({
      async pull(controller) {
        try {
          const part = await reader.read();
          if (part.done) { await close(true); controller.close(); }
          else { await raw.write(part.value); controller.enqueue(part.value); }
        } catch (error) { await close(false); controller.error(error); }
      },
      async cancel(reason) { try { await reader.cancel(reason); } finally { await close(false); } },
    });
    return new Response(stream, { status: response.status, statusText: response.statusText, headers: response.headers });
  };
  globalThis.fetch = archiveFetch;
  try {
    const guard = new EpisodeBudgetGuard({ episodeId: config.episodeId, priorKnownTokens: parentTokens,
      wholeLimit: effectiveFrontStop, safetyTokens: manifest.safety_tokens, maxCalls: maxNativeRequests,
      maxOutputTokens: config.maxOutputTokens, countInputTokens, sourceCheck,
      record: async event => { await journal.write(`${JSON.stringify(event)}\n`); await journal.sync(); } });
    integration = createBudgetIntegration({ guard, endpoint: config.endpoint, model: manifest.model, createSseParser: createParser,
      validateNativeRequest: async body => {
        try {
          const state = slot.getPresence(), names = body.tools?.map(row => row.name) ?? [];
          const expected = names.includes('submit_bundle_evidence') ? manifest.output_capacity.submission : manifest.output_capacity.navigation;
          if (body.max_tokens !== expected) throw new BudgetError('bundle_presence_phase_output_reserve_changed');
          const proof = verifyBundleNativePresence(body, state);
          await presence.write(`${JSON.stringify(proof)}\n`); await presence.sync(); slot.recordProof(state, proof);
        } catch (error) {
          slot.clearProof();
          const code = /^bundle_presence_[a-z_]+$/.test(error?.code ?? '') ? error.code : 'bundle_presence_policy_failed';
          await presence.write(`${JSON.stringify({ status: 'rejected_before_admission', code })}\n`); await presence.sync();
          throw new BudgetError(code);
        }
      } });
    remove = ctx.on('llm/stream', integration.listener);
    ctx.provide('trustJudgeBundleGuard', { episodeId: config.episodeId });
    await writeFile(config.readinessPath, `${JSON.stringify({ format: 'dsh-source-bundle-workbench-ready-v1',
      episode_id: config.episodeId, authorization_sha256: config.authorization.sha256,
      source_bindings_sha256: digest(JSON.stringify(bindings)), case_sha256: manifest.raw_case.sha256,
      native_guard_installed: true, execution_mode: manifest.execution_mode,
      whole_token_stop: manifest.whole_token_stop, effective_front_stop: effectiveFrontStop,
      reserved_tail_tokens: manifest.tail_reserved_tokens, parent_tokens: parentTokens })}\n`, { flag: 'wx' });
    ctx.effect(() => async () => {
      integration.dispose(); if (globalThis.fetch === archiveFetch) globalThis.fetch = originalFetch;
      remove?.(); slot.dispose(); await journal.close(); await presence.close(); await requestIndex.close();
    });
  } catch (error) {
    integration?.dispose(); if (globalThis.fetch === archiveFetch) globalThis.fetch = originalFetch;
    remove?.(); slot.dispose(); await journal.close(); await presence.close(); await requestIndex.close(); throw error;
  }
}

