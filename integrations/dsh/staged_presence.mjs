/** Native-message presence policy only; owns neither a serializer nor an agent loop. */
import { createHash } from 'node:crypto';
import { BudgetError } from './budget_core.mjs';
const fail = code => { throw new BudgetError(code); };
const sha = value => createHash('sha256').update(value).digest('hex');
const copies = (text, value) => text.split(value).length - 1;
const textBlocks = content => typeof content === 'string' ? [content]
  : Array.isArray(content) ? content.filter(b => b?.type === 'text' && typeof b.text === 'string').map(b => b.text) : [];

/** Source strings come from the bound business backend, never model arguments. */
export function verifyStagedNativePresence(body, stage) {
  if (!stage) fail('staged_presence_unavailable');
  if (!['preview', 'final'].includes(stage.phase) || typeof stage.preview_context !== 'string' || !stage.preview_context) fail('invalid_staged_context_state');
  if (typeof body?.system !== 'string' || !Array.isArray(body.messages)) fail('invalid_staged_native_body');
  const toolName = stage.phase === 'preview' ? 'read_selected_sources' : 'submit_evidence';
  if (!Array.isArray(body.tools) || body.tools.length !== 1 || body.tools[0]?.name !== toolName) fail('staged_tool_contract_changed');
  if (stage.phase === 'preview') {
    if (copies(body.system, stage.preview_context) !== 1 || stage.full_read_result_json !== null
        || !Array.isArray(stage.read_refs) || stage.read_refs.length) fail('staged_preview_missing');
  } else {
    if (typeof stage.post_read_context !== 'string' || !stage.post_read_context
        || copies(body.system, stage.post_read_context) !== 1 || body.system.includes(stage.preview_context)
        || typeof stage.full_read_result_json !== 'string' || !stage.full_read_result_json
        || !Array.isArray(stage.read_refs)) fail('staged_final_context_missing');
    const calls = body.messages.flatMap(m => m.role === 'assistant' && Array.isArray(m.content)
      ? m.content.filter(b => b.type === 'tool_use' && b.name === 'read_selected_sources') : []);
    if (calls.length !== 1 || typeof calls[0].id !== 'string'
        || JSON.stringify(calls[0].input?.refs) !== JSON.stringify(stage.read_refs)) fail('staged_read_call_missing');
    const results = body.messages.flatMap(m => m.role === 'user' && Array.isArray(m.content)
      ? m.content.filter(b => b.type === 'tool_result' && b.tool_use_id === calls[0].id && b.is_error !== true) : []);
    if (results.length !== 1 || textBlocks(results[0].content).filter(s => s === stage.full_read_result_json).length !== 1) fail('staged_full_read_result_missing');
    const allText = [body.system, ...body.messages.flatMap(m => textBlocks(m.content)),
      ...body.messages.flatMap(m => Array.isArray(m.content) ? m.content.flatMap(b => b.type === 'tool_result' ? textBlocks(b.content) : []) : [])];
    if (allText.some(s => s.includes(stage.preview_context))) fail('staged_previews_retained_in_history');
    if (allText.reduce((n, s) => n + copies(s, stage.full_read_result_json), 0) !== 1) fail('staged_full_read_result_duplicated');
  }
  return { phase: stage.phase, context_sha256: sha(stage.phase === 'preview' ? stage.preview_context : stage.post_read_context),
    full_read_result_sha256: stage.full_read_result_json ? sha(stage.full_read_result_json) : null,
    read_refs: [...stage.read_refs], native_presence_verified_before_admission: true, semantic_verified: false };
}

/** Cheap negative screen of full tool content; only actual native preflight can admit a request. */
export async function screenStagedRead({ guard, countTokens, readRequest, maxOutputTokens, safetyTokens }) {
  const state = guard.snapshot();
  if (state.halted || state.in_flight || state.unknown_usage_requests || state.started_calls !== 1) fail('staged_read_requires_settled_first_call');
  const payload = JSON.stringify(readRequest.full_read_result);
  const estimate = await countTokens({ serialized_body: payload });
  if (!Number.isSafeInteger(estimate) || estimate < 0) fail('unknown_read_payload_cost');
  const total = state.known_episode_tokens + estimate + maxOutputTokens + safetyTokens;
  return { admitted: total <= state.whole_limit, whole_limit: state.whole_limit, preflight_total_tokens: total,
    ...(total > state.whole_limit ? { code: 'staged_read_payload_budget_exceeded' } : {}),
    read_payload_estimated_tokens: estimate, known_episode_tokens: state.known_episode_tokens,
    estimate_kind: 'canonical_read_payload_only_not_future_native_request',
    native_next_request_still_required: true, future_request_fit_guaranteed: false };
}
