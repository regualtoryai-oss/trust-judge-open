import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyBundleNativePresence } from './bundle_presence.mjs';

function fixture() {
  const result = { complete_original_slices: true, bundles: [{ bundle_id: 'bound-1', quote: '😀全文前缀。中部限定尚未确认。完整尾部。',
    source_paragraph_ids: ['opaque-a', 'opaque-b'], start_cp: 8, end_cp: 39 }], source_bundle_set_sha256: 'a'.repeat(64) };
  const args = { anchor_paragraph_ids: ['opaque-a'], before: 0, after: 1 };
  const state = { question_data: '{"original_question":"是否确认？"}', tool_names: ['submit_bundle_evidence'],
    reads: [{ tool_use_id: 'native-read-1', arguments: args, result }] };
  const body = { system: `instructions\n${state.question_data}`, tools: [{ name: 'submit_bundle_evidence' }], messages: [
    { role: 'assistant', content: [{ type: 'tool_use', id: 'native-read-1', name: 'read_source_bundles', input: structuredClone(args) }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'native-read-1', content: [{ type: 'text', text: JSON.stringify(result) }] }] },
  ] };
  return { state, body };
}

test('actual native tool use and exactly one complete result bind to original read IDs', () => {
  const { state, body } = fixture();
  const proof = verifyBundleNativePresence(body, state);
  assert.equal(proof.read_receipts.length, 1);
  assert.equal(proof.read_receipts[0].tool_use_id, 'native-read-1');
  assert.equal(proof.read_receipts[0].complete_original_slices_present, true);
  assert.equal(proof.semantic_verified, false);
  assert.equal(proof.publishable, false);
});

for (const mutation of ['missing_result', 'pruned_middle', 'preview_only', 'wrong_call_id', 'changed_arguments', 'error_result',
  'duplicate_result', 'duplicate_text_elsewhere', 'untracked_read', 'wrong_tools', 'missing_question']) {
  test(`native presence rejects ${mutation}`, () => {
    const { state, body } = fixture();
    const result = body.messages[1].content[0];
    if (mutation === 'missing_result') body.messages.splice(1, 1);
    if (mutation === 'pruned_middle') result.content[0].text = result.content[0].text.replace('中部限定尚未确认。', '[...pruned...]');
    if (mutation === 'preview_only') result.content[0].text = JSON.stringify({ excerpt_text: '😀全文前缀。', navigation_only: true });
    if (mutation === 'wrong_call_id') result.tool_use_id = 'different-native-id';
    if (mutation === 'changed_arguments') body.messages[0].content[0].input.after = 0;
    if (mutation === 'error_result') result.is_error = true;
    if (mutation === 'duplicate_result') body.messages[1].content.push(structuredClone(result));
    if (mutation === 'duplicate_text_elsewhere') body.messages.push({ role: 'user', content: result.content[0].text });
    if (mutation === 'untracked_read') {
      body.messages[0].content.push({ ...structuredClone(body.messages[0].content[0]), id: 'untracked' });
      body.messages[1].content.push({ ...structuredClone(result), tool_use_id: 'untracked' });
    }
    if (mutation === 'wrong_tools') body.tools.push({ name: 'terminal' });
    if (mutation === 'missing_question') body.system = 'no original question';
    assert.throws(() => verifyBundleNativePresence(body, state), /bundle_presence_/);
  });
}

test('two native full-read bursts are each required and never inferred from local scope alone', () => {
  const { state, body } = fixture();
  const second = { tool_use_id: 'native-read-2', arguments: { anchor_paragraph_ids: ['opaque-c'], before: 0, after: 0 },
    result: { complete_original_slices: true, bundles: [{ bundle_id: 'bound-2', quote: '第二批完整原文', source_paragraph_ids: ['opaque-c'] }],
      source_bundle_set_sha256: 'b'.repeat(64) } };
  state.reads.push(second);
  assert.throws(() => verifyBundleNativePresence(body, state), /read_call_missing_or_changed/);
  body.messages.push({ role: 'assistant', content: [{ type: 'tool_use', id: second.tool_use_id, name: 'read_source_bundles', input: second.arguments }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: second.tool_use_id, content: [{ type: 'text', text: JSON.stringify(second.result) }] }] });
  assert.equal(verifyBundleNativePresence(body, state).read_receipts.length, 2);
});

