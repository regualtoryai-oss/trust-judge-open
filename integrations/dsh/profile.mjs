/** Generate an invocation patch for the official 0.2.1-rc.2 sdk-minimal profile. No process or model calls. */
import { isAbsolute, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { selectionPolicySection } from './evidence_policy.mjs';

export const DSH_VERSION = '0.2.1-rc.2';
export const DSH_PROFILE = 'sdk-minimal';
// Retain session-projection, working-directory and its fs service: rc.2 SDK/core require them.
// No filesystem or working-directory tools are exposed to the model.
export const DISABLED_PROFILE_ROWS = [
  'deepseek-llm-api-extensions', 'session-log-deepseek', 'plugin-package-inventory-deepseek',
  'sandbox', 'attachment-local', 'tool-fs', 'tool-working-directory',
  'token-meter', 'compaction-basic', 'image-offload', 'sandbox-policy', 'subprocess', 'pty',
  'terminal-bash', 'terminal-pwsh', 'session-title', 'mcp-resources', 'llm-retry', 'jobs',
  'persistent-bash', 'persistent-pwsh',
];
function absolute(value) { return typeof value === 'string' && isAbsolute(value); }
function strictInteger(value) { return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0; }
function fail(code) { throw new Error(code); }

/** Pass returned rows to an SDK `patches` YAML file (JSON serialization is valid YAML). */
export function buildPatch({ casePath, caseSha256, submissionPath, sessionRoot, budgetPluginPath, budgetConfig, packetCounter, contextMode = 'tools', selectionPolicy = 'baseline', annotationPolicy, previewCodepoints = 160, maxOutputTokens = 3072, endpointBase = 'https://api.deepseek.com/anthropic' }) {
  selectionPolicySection(selectionPolicy); // Validate before any launch; baseline contributes no extra bytes.
  if (!['tools', 'full-candidate-context', 'staged-short-window'].includes(contextMode)) fail('unknown_context_mode');
  if (![casePath, submissionPath, sessionRoot, budgetPluginPath].every(absolute) || casePath === submissionPath || !/^[a-f0-9]{64}$/.test(caseSha256 ?? '')) fail('absolute_bound_case_and_output_paths_required');
  if (!budgetConfig || budgetConfig.wholeLimit !== 20000 || !strictInteger(budgetConfig.priorKnownTokens) || !strictInteger(budgetConfig.safetyTokens) || budgetConfig.safetyTokens < 512 || !strictInteger(budgetConfig.maxCalls) || budgetConfig.maxCalls < 1 || !strictInteger(maxOutputTokens) || maxOutputTokens < 1 || budgetConfig.maxOutputTokens !== maxOutputTokens) fail('whole_episode_budget_plugin_required');
  if (contextMode === 'full-candidate-context' && (!absolute(budgetConfig.requiredContextPath) || !/^[a-f0-9]{64}$/.test(budgetConfig.requiredContextSha256 ?? ''))) fail('full_context_native_presence_gate_required');
  const staged = contextMode === 'staged-short-window';
  const resolvedAnnotationPolicy = annotationPolicy ?? ((contextMode === 'full-candidate-context' && ['source-local-v2', 'source-local-v3', 'source-local-v3-compact'].includes(selectionPolicy))
    || (staged && selectionPolicy === 'source-local-v3') ? 'isolate-invalid-metadata-v1' : 'strict');
  if (!['strict', 'isolate-invalid-metadata-v1'].includes(resolvedAnnotationPolicy)
      || (resolvedAnnotationPolicy !== 'strict' && !['full-candidate-context', 'staged-short-window'].includes(contextMode))) fail('invalid_annotation_policy_for_context');
  const isolateAnnotations = resolvedAnnotationPolicy === 'isolate-invalid-metadata-v1';
  if (staged && (budgetConfig.stagedContext !== true || !absolute(budgetConfig.stagedContextAuditPath)
      || typeof budgetConfig.episodeId !== 'string' || !['source-local-v2', 'source-local-v3'].includes(selectionPolicy)
      || budgetConfig.maxCalls !== 2
      || !strictInteger(previewCodepoints) || previewCodepoints < 1)) fail('staged_native_presence_gate_required');
  if (typeof endpointBase !== 'string' || !endpointBase.startsWith('https://') || budgetConfig.endpoint !== `${endpointBase.replace(/\/$/, '')}/v1/messages`) fail('budget_and_provider_endpoint_must_match');
  const businessPluginPath = fileURLToPath(new URL(staged ? './staged_evidence.mjs' : './evidence_tools.mjs', import.meta.url));
  const configuredCounter = packetCounter ?? (budgetConfig.pythonExecutable && budgetConfig.tokenizerPath && budgetConfig.tokenizerSha256 ? {
    pythonExecutable: budgetConfig.pythonExecutable, tokenizerPath: budgetConfig.tokenizerPath, tokenizerSha256: budgetConfig.tokenizerSha256 } : null);
  const packetBudget = { journalPath: budgetConfig.journalPath, episodeId: budgetConfig.episodeId, priorKnownTokens: budgetConfig.priorKnownTokens,
    wholeLimit: budgetConfig.wholeLimit, maxOutputTokens: budgetConfig.maxOutputTokens, safetyTokens: budgetConfig.safetyTokens };
  return [
    ...DISABLED_PROFILE_ROWS.map(id => ({ id, disabled: true })),
    { id: 'llm-deepseek', config: { baseURL: endpointBase, thinking: 'disabled', reasoningEffort: 'off', maxTokens: maxOutputTokens,
      retryPolicy: { mode: 'normal', maxRetries: 0 }, streamIdleTimeoutMs: 120000 } },
    { id: 'system-prompt', config: { includeHarnessIdentity: false, includeRuntimeContext: false, personaPrefix: 'You evaluate disputed claims using original sources and explicit uncertainty.' } },
    { id: 'agent-loop', config: { agents: [], maxParallelToolCalls: 1 } },
    { id: 'sessions', config: { root: sessionRoot, compression: 'none' } },
    { id: 'sdk-jsonrpc-server', config: { maxTokensAsSuccess: false } },
    { insert: [
      { id: 'trust-judge-episode-budget', name: pathToFileURL(budgetPluginPath).href, config: structuredClone(budgetConfig) },
      { id: 'trust-judge-evidence-tools', name: pathToFileURL(businessPluginPath).href, config: staged
        ? { casePath, caseSha256, submissionPath, episodeId: budgetConfig.episodeId, selectionPolicy, previewCodepoints,
          annotationPolicy: resolvedAnnotationPolicy,
          ...(isolateAnnotations ? { annotationAuditPath: resolve(dirname(submissionPath), 'submission-annotation-audit.json') } : {}) }
        : { casePath, caseSha256, submissionPath, contextMode,
        ...(selectionPolicy !== 'baseline' ? { selectionPolicy } : {}),
        annotationPolicy: resolvedAnnotationPolicy,
        ...(isolateAnnotations ? { annotationAuditPath: resolve(dirname(submissionPath), 'submission-annotation-audit.json') } : {}),
        packetBudget, ...(configuredCounter ? { packetCounter: structuredClone(configuredCounter) } : {}) } },
    ] },
  ];
}
