/** Exact native Messages read-presence policy; no serializer, model call or loop. */
import { createHash } from 'node:crypto';
import { handleReadArguments, handleReadView, parseUniqueNativeArguments, stableHandleJson } from './bundle_handles.mjs';
const sha = value => createHash('sha256').update(value).digest('hex');
const fail = code => { const error = new Error(code); error.code = code; throw error; };
const stable = value => JSON.stringify(value && typeof value === 'object'
  ? Array.isArray(value) ? value.map(part => JSON.parse(stable(part)))
    : Object.fromEntries(Object.keys(value).sort().map(key => [key, JSON.parse(stable(value[key]))])) : value);
const textBlocks = content => typeof content === 'string' ? [content]
  : Array.isArray(content) ? content.filter(row => row?.type === 'text' && typeof row.text === 'string').map(row => row.text) : [];

export function verifyBundleNativePresence(body, state) {
  if (state?.handles !== undefined || state?.native_reads !== undefined) return verifyProjectedPresence(body, state);
  if (!state || !Array.isArray(state.reads) || !Array.isArray(state.tool_names)
      || typeof state.question_data !== 'string' || !state.question_data) fail('bundle_presence_invalid_state');
  if (typeof body?.system !== 'string' || !Array.isArray(body.messages) || !Array.isArray(body.tools)
      || body.system.split(state.question_data).length - 1 !== 1) fail('bundle_presence_original_question_missing');
  const names = body.tools.map(row => row?.name);
  if (new Set(names).size !== names.length || stable([...names].sort()) !== stable([...state.tool_names].sort())) fail('bundle_presence_tool_contract_changed');
  const calls = body.messages.flatMap(message => message.role === 'assistant' && Array.isArray(message.content)
    ? message.content.filter(row => row.type === 'tool_use' && row.name === 'read_source_bundles') : []);
  const results = body.messages.flatMap(message => message.role === 'user' && Array.isArray(message.content)
    ? message.content.filter(row => row.type === 'tool_result') : []);
  const allText = [body.system, ...body.messages.flatMap(message => textBlocks(message.content)),
    ...results.flatMap(result => textBlocks(result.content))];
  const seen = new Set(), receipts = [];
  for (const read of state.reads) {
    if (typeof read.tool_use_id !== 'string' || !read.tool_use_id || seen.has(read.tool_use_id)
        || read.result?.complete_original_slices !== true || !Array.isArray(read.result?.bundles)
        || !read.result.bundles.length) fail('bundle_presence_invalid_read_binding');
    seen.add(read.tool_use_id);
    const matchingCalls = calls.filter(row => row.id === read.tool_use_id);
    if (matchingCalls.length !== 1 || stable(matchingCalls[0].input) !== stable(read.arguments)) fail('bundle_presence_read_call_missing_or_changed');
    const matchingResults = results.filter(row => row.tool_use_id === read.tool_use_id);
    const rendered = JSON.stringify(read.result);
    if (matchingResults.length !== 1 || matchingResults[0].is_error === true
        || textBlocks(matchingResults[0].content).length !== 1
        || textBlocks(matchingResults[0].content)[0] !== rendered) fail('bundle_presence_complete_read_result_missing');
    if (allText.reduce((count, text) => count + (text.split(rendered).length - 1), 0) !== 1) fail('bundle_presence_read_result_duplicated');
    receipts.push({ tool_use_id: read.tool_use_id, native_read_result_sha256: sha(rendered),
      bundle_ids: read.result.bundles.map(bundle => bundle.bundle_id),
      source_bundle_set_sha256: read.result.source_bundle_set_sha256,
      complete_original_slices_present: true });
  }
  for (const call of calls) {
    const success = results.some(row => row.tool_use_id === call.id && row.is_error !== true);
    if (success && !seen.has(call.id)) fail('bundle_presence_untracked_successful_read');
  }
  return { format: 'dsh-source-bundle-native-presence-v1', read_receipts: receipts,
    native_messages_sha256: sha(JSON.stringify(body.messages)), source_bytes_present_in_current_request: true,
    semantic_verified: false, publishable: false };
}

function verifyProjectedPresence(body, state) {
  const handles = state?.handles;
  if (!handles || handles.manifest?.format !== 'source-bundle-native-handles-v1'
      || !/^[a-f0-9]{64}$/.test(handles.sha256 ?? '') || !Array.isArray(state.reads)
      || !Array.isArray(state.native_reads) || state.native_reads.length !== state.reads.length
      || !Array.isArray(state.tool_names) || typeof state.question_data !== 'string' || !state.question_data) fail('bundle_presence_invalid_projection_state');
  if (typeof body?.system !== 'string' || !Array.isArray(body.messages) || !Array.isArray(body.tools)
      || body.system.split(state.question_data).length - 1 !== 1) fail('bundle_presence_original_question_missing');
  const names = body.tools.map(row => row?.name);
  if (new Set(names).size !== names.length || stable([...names].sort()) !== stable([...state.tool_names].sort())) fail('bundle_presence_tool_contract_changed');
  const calls = body.messages.flatMap(message => message.role === 'assistant' && Array.isArray(message.content)
    ? message.content.filter(row => row.type === 'tool_use' && row.name === 'read_source_bundles') : []);
  const results = body.messages.flatMap(message => message.role === 'user' && Array.isArray(message.content)
    ? message.content.filter(row => row.type === 'tool_result') : []);
  const allText = [body.system, ...body.messages.flatMap(message => textBlocks(message.content)),
    ...results.flatMap(result => textBlocks(result.content))];
  const seen = new Set(), receipts = [];
  for (const read of state.reads) {
    if (typeof read.tool_use_id !== 'string' || !read.tool_use_id || seen.has(read.tool_use_id)
        || read.result?.complete_original_slices !== true || !Array.isArray(read.result?.bundles)
        || !read.result.bundles.length) fail('bundle_presence_invalid_read_binding');
    seen.add(read.tool_use_id);
    const projections = state.native_reads.filter(row => row?.operation === 'read' && row.tool_use_id === read.tool_use_id);
    if (projections.length !== 1) fail('bundle_presence_unique_read_projection_required');
    const projection = projections[0];
    let args, business, view;
    try {
      args = parseUniqueNativeArguments(projection.raw_native_arguments);
      business = handleReadArguments(handles.manifest, args);
      view = handleReadView(handles.manifest, read.result, args);
    } catch { fail('bundle_presence_read_projection_arguments_changed'); }
    const canonicalSha = sha(stableHandleJson(read.result)), rendered = stableHandleJson(view), nativeSha = sha(rendered);
    if (projection.handles_sha256 !== handles.sha256 || stable(args) !== stable(projection.native_arguments)
        || stable(business) !== stable(read.arguments) || stable(business) !== stable(projection.business_arguments)
        || stable(read.result) !== stable(projection.canonical_result) || projection.canonical_result_sha256 !== canonicalSha
        || stable(view) !== stable(projection.native_result) || projection.native_read_view_sha256 !== nativeSha) fail('bundle_presence_read_projection_source_changed');
    const matchingCalls = calls.filter(row => row.id === read.tool_use_id), matchingResults = results.filter(row => row.tool_use_id === read.tool_use_id);
    if (matchingCalls.length !== 1 || stable(matchingCalls[0].input) !== stable(args)) fail('bundle_presence_read_call_missing_or_changed');
    if (matchingResults.length !== 1 || matchingResults[0].is_error === true
        || textBlocks(matchingResults[0].content).length !== 1
        || textBlocks(matchingResults[0].content)[0] !== rendered) fail('bundle_presence_complete_read_result_missing');
    if (allText.reduce((count, text) => count + (text.split(rendered).length - 1), 0) !== 1) fail('bundle_presence_read_result_duplicated');
    receipts.push({ tool_use_id: read.tool_use_id, canonical_read_result_sha256: canonicalSha,
      native_read_view_sha256: nativeSha, handles_sha256: handles.sha256,
      bundle_ids: read.result.bundles.map(bundle => bundle.bundle_id),
      source_bundle_set_sha256: read.result.source_bundle_set_sha256, complete_original_slices_present: true });
  }
  for (const call of calls) {
    if (results.some(row => row.tool_use_id === call.id && row.is_error !== true) && !seen.has(call.id)) fail('bundle_presence_untracked_successful_read');
  }
  return { format: 'dsh-source-bundle-native-presence-v2', handles_sha256: handles.sha256, read_receipts: receipts,
    native_messages_sha256: sha(JSON.stringify(body.messages)), source_bytes_present_in_current_request: true,
    semantic_verified: false, publishable: false };
}

