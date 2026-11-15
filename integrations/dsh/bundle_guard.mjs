/** Fixture-only native budget/presence adapter; reuse official stream accounting. */
import { open, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { isAbsolute } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createBudgetIntegration } from './budget_plugin.mjs';
import { BudgetError, EpisodeBudgetGuard } from './budget_core.mjs';
import { createOfficialTokenizerCounter } from './budget_counter.mjs';
import { bindBundleGuard } from './bundle_evidence.mjs';
import { verifyBundleNativePresence } from './bundle_presence.mjs';

export const name = 'trust-judge-source-bundle-guard';
export const inject = ['llm'];
const sha = value => createHash('sha256').update(value).digest('hex');
const fail = code => { const error = new Error(code); error.code = code; throw error; };

export async function apply(ctx, config) {
  if (config?.fixtureOnly !== true || globalThis[Symbol.for('trust-judge:bundle-fixture-transport')] !== true) fail('bundle_guard_fixture_transport_required');
  if (!Array.isArray(config.sourceBindings) || !config.sourceBindings.length
      || ![config.journalPath, config.presencePath, config.runtimePackageJson, config.readinessPath].every(value => isAbsolute(value ?? ''))) fail('bundle_guard_bound_configuration_required');
  const bindings = structuredClone(config.sourceBindings);
  if (bindings.some(row => !isAbsolute(row.file ?? '') || !/^[a-f0-9]{64}$/.test(row.sha256 ?? ''))) fail('bundle_guard_source_bindings_required');
  const sourceCheck = async () => (await Promise.all(bindings.map(async row => sha(await readFile(row.file)) === row.sha256))).every(Boolean);
  if (!await sourceCheck()) fail('bundle_guard_source_binding_changed');
  const counter = createOfficialTokenizerCounter(config);
  const parserPath = createRequire(pathToFileURL(config.runtimePackageJson)).resolve('eventsource-parser');
  if (!bindings.some(row => row.file === parserPath)) fail('bundle_guard_bound_parser_required');
  const { createParser } = await import(pathToFileURL(parserPath));
  const journal = await open(config.journalPath, 'wx'), presenceJournal = await open(config.presencePath, 'wx');
  const slot = bindBundleGuard(config.episodeId);
  let integration, remove;
  try {
    const guard = new EpisodeBudgetGuard({ episodeId: config.episodeId, priorKnownTokens: config.priorKnownTokens,
      wholeLimit: config.wholeLimit, safetyTokens: config.safetyTokens, maxCalls: config.maxCalls,
      maxOutputTokens: config.maxOutputTokens, countInputTokens: counter, sourceCheck,
      record: async event => { await journal.write(`${JSON.stringify(event)}\n`); await journal.sync(); } });
    integration = createBudgetIntegration({ guard, endpoint: config.endpoint, model: config.model, createSseParser: createParser,
      validateNativeRequest: async body => {
        try {
          const state = slot.getPresence();
          const proof = verifyBundleNativePresence(body, state);
          await presenceJournal.write(`${JSON.stringify(proof)}\n`); await presenceJournal.sync();
          slot.recordProof(state, proof);
        } catch (error) {
          slot.clearProof();
          const code = typeof error?.code === 'string' && /^bundle_presence_[a-z_]+$/.test(error.code)
            ? error.code : 'bundle_presence_policy_failed';
          await presenceJournal.write(`${JSON.stringify({ status: 'rejected_before_admission', code })}\n`); await presenceJournal.sync();
          throw new BudgetError(code);
        }
      } });
    remove = ctx.on('llm/stream', integration.listener);
    ctx.provide('trustJudgeBundleGuard', { episodeId: config.episodeId });
    await writeFile(config.readinessPath, `${JSON.stringify({ format: 'dsh-bundle-fixture-ready-v1', episode_id: config.episodeId,
      source_bindings_sha256: sha(JSON.stringify(bindings)), fixture_only: true, native_guard_installed: true })}\n`, { flag: 'wx' });
    ctx.effect(() => async () => { integration.dispose(); remove?.(); slot.dispose(); await journal.close(); await presenceJournal.close(); });
  } catch (error) {
    integration?.dispose(); remove?.(); slot.dispose(); await journal.close(); await presenceJournal.close(); throw error;
  }
}

