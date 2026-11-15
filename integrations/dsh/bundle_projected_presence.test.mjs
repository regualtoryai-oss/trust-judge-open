import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { buildHandleManifest, handleReadArguments, handleReadView, stableHandleJson } from './bundle_handles.mjs';
import { verifyBundleNativePresence } from './bundle_presence.mjs';
const sha = value => createHash('sha256').update(value).digest('hex');

function fixture() {
  const first = '😀完整开头。', last = '限定尚未确认。', quote = `${first}|原始gap|${last}`;
  const secondStart = Array.from(`${first}|原始gap|`).length, end = Array.from(quote).length;
  const raw = { documents: [{ id: 'doc', content: quote, paragraph_manifest: [
    { id: 'opaque-a', start_cp: 0, end_cp: Array.from(first).length },
    { id: 'opaque-b', start_cp: secondStart, end_cp: end }] }], paragraphs: [
    { id: 'opaque-a', document_id: 'doc', document_start_cp: 0, document_end_cp: Array.from(first).length },
    { id: 'opaque-b', document_id: 'doc', document_start_cp: secondStart, document_end_cp: end }] };
  const bundle = { span_scope: 'document', paragraph_id: 'opaque-a', raw_doc_id: 'rawdoc', raw_para_id: 'raw-parent',
    start_cp: 0, end_cp: end, quote, source_paragraph_ids: ['opaque-a', 'opaque-b'], document_content_sha256: sha(quote),
    paragraph_manifest_sha256: 'a'.repeat(64), document_id: 'doc', bundle_id: 'synthetic-bound-range',
    bundle_purpose: 'read_context', anchor_paragraph_ids: ['opaque-a'] };
  const handles = buildHandleManifest(raw, { tool: 'search_source_bundles', results: [{ paragraph_id: 'opaque-a' }] },
    { tool: 'plan_source_bundles', status: 'planned', bundles: [bundle], source_binding: { input_sha256: 'b'.repeat(64) } },
    { case_sha256: 'c'.repeat(64), backend_navigation_sha256: 'd'.repeat(64) });
  const handlesSha = sha(`${stableHandleJson(handles)}\n`), args = { bundle_refs: { B1: true } };
  const result = { tool: 'read_source_bundles', status: 'completed', complete_original_slices: true, bundles: [bundle],
    source_bundle_set_sha256: sha(stableHandleJson([bundle])), delivery: { read_call_index: 2, complete_delivery_index: 1,
      new_original_paragraph_ids: ['opaque-a', 'opaque-b'], already_read_original_paragraph_ids: [], full_original_source_bytes_returned: true } };
  const business = handleReadArguments(handles, args), view = handleReadView(handles, result, args), rendered = stableHandleJson(view);
  const projection = { operation: 'read', tool_use_id: 'actual-read', raw_native_arguments: JSON.stringify(args), native_arguments: args,
    business_arguments: business, canonical_result: result, canonical_result_sha256: sha(stableHandleJson(result)),
    native_result: view, native_read_view_sha256: sha(rendered), handles_sha256: handlesSha };
  const state = { question_data: '{"original_question":"是否确认？"}', tool_names: ['submit_bundle_evidence'],
    handles: { manifest: handles, sha256: handlesSha }, reads: [{ tool_use_id: 'actual-read', arguments: business, result }], native_reads: [projection] };
  const body = { system: state.question_data, tools: [{ name: 'submit_bundle_evidence' }], messages: [
    { role: 'assistant', content: [{ type: 'tool_use', name: 'read_source_bundles', id: 'actual-read', input: structuredClone(args) }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'actual-read', content: [{ type: 'text', text: rendered }] }] }] };
  return { state, body };
}

test('short view proves untouched Unicode source including original gap, with separate backend/native hashes', () => {
  const { body, state } = fixture(), proof = verifyBundleNativePresence(body, state);
  assert.equal(proof.format, 'dsh-source-bundle-native-presence-v2');
  assert.equal(proof.handles_sha256, state.handles.sha256);
  assert.notEqual(proof.read_receipts[0].canonical_read_result_sha256, proof.read_receipts[0].native_read_view_sha256);
  assert.equal(state.native_reads[0].native_result.bundles[0].quote, state.reads[0].result.bundles[0].quote);
  assert.equal(proof.semantic_verified, false);
  assert.equal(proof.publishable, false);
});

for (const mutation of ['missing_result', 'truncated_text', 'noncanonical_text', 'split_text', 'wrong_call',
  'wrong_native_arguments', 'changed_business_result', 'changed_native_quote', 'changed_coordinates',
  'duplicate_raw_key', 'changed_handle_hash', 'duplicate_result', 'duplicate_elsewhere', 'untracked_read']) {
  test(`projected presence rejects ${mutation}`, () => {
    const { body, state } = fixture(), projection = state.native_reads[0], actual = body.messages[1].content[0];
    if (mutation === 'missing_result') body.messages.pop();
    if (mutation === 'truncated_text') actual.content[0].text = actual.content[0].text.replace('限定尚未确认。', '[cut]');
    if (mutation === 'noncanonical_text') actual.content[0].text = JSON.stringify(projection.native_result, null, 2);
    if (mutation === 'split_text') actual.content = [{ type: 'text', text: actual.content[0].text.slice(0, 30) }, { type: 'text', text: actual.content[0].text.slice(30) }];
    if (mutation === 'wrong_call') actual.tool_use_id = 'other';
    if (mutation === 'wrong_native_arguments') body.messages[0].content[0].input = { bundle_refs: { B999: true } };
    if (mutation === 'changed_business_result') projection.canonical_result = { ...projection.canonical_result, bundles: [] };
    if (mutation === 'changed_native_quote') projection.native_result.bundles[0].quote += '伪造';
    if (mutation === 'changed_coordinates') state.handles.manifest.paragraphs[1].start_cp++;
    if (mutation === 'duplicate_raw_key') projection.raw_native_arguments = '{"bundle_refs":{"B1":true,"B1":true}}';
    if (mutation === 'changed_handle_hash') projection.handles_sha256 = '0'.repeat(64);
    if (mutation === 'duplicate_result') body.messages[1].content.push(structuredClone(actual));
    if (mutation === 'duplicate_elsewhere') body.messages.push({ role: 'user', content: actual.content[0].text });
    if (mutation === 'untracked_read') {
      body.messages[0].content.push({ ...structuredClone(body.messages[0].content[0]), id: 'untracked' });
      body.messages[1].content.push({ ...structuredClone(actual), tool_use_id: 'untracked' });
    }
    assert.throws(() => verifyBundleNativePresence(body, state), /bundle_presence_/);
  });
}
