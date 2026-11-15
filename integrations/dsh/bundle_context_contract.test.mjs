/** Zero-network verification of the installed official public session API. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const sessionUrl = new URL('../../.runtime/dsh-official-r1/node_modules/@deepseek-ai/dsh-session/lib/index.js', import.meta.url);
const llmUrl = new URL('../../.runtime/dsh-official-r1/node_modules/@deepseek-ai/dsh-llm/lib/index.js', import.meta.url);
const installed = await access(fileURLToPath(sessionUrl)).then(() => true, () => false);

async function fixture() {
  const { Session } = await import(sessionUrl.href);
  const { createSystemMessage, createToolResultMessage } = await import(llmUrl.href);
  const session = Session.create('00000000-0000-4000-8000-000000000001');
  session.append('system/message', { turn: 1, step: 1, message: createSystemMessage('Explicit source-only fixture.') }, { surfaceOp: 'append' });
  const append = (callId, value) => session.append('tool/result', { turn: 1, step: 1,
    message: createToolResultMessage({ callId, content: [{ type: 'text', text: JSON.stringify(value) }], isError: false }) }, { surfaceOp: 'append' });
  const navigation = append('search-fixture', { tool: 'search_source_bundles', navigation_only: true, results: [['p1', 'preview']] });
  const read = append('read-fixture', { tool: 'read_source_bundles', complete_original_slices: true, bundles: [{ quote: 'Exact full original body.\nSecond original paragraph.' }] });
  const replacement = { ...navigation.data, message: { ...navigation.data.message,
    content: [{ type: 'text', text: JSON.stringify({ navigation_withdrawn: true, original_result_seq: navigation.seq }) }] } };
  const intent = { surfaceOp: { op: 'replace', startSeq: navigation.seq, endSeq: navigation.seq }, sourceEventSeqs: [navigation.seq] };
  return { session, navigation, read, replacement, intent };
}

test('official public surface replacement withdraws only navigation and leaves source events/read bytes unchanged', { skip: !installed }, async () => {
  const { session, navigation, read, replacement, intent } = await fixture();
  const original = JSON.stringify(navigation), readBytes = JSON.stringify(read.data.message);
  session.append('tool/result', replacement, intent);
  const visible = session.deriveMessages();
  assert.equal(JSON.stringify(navigation), original);
  assert.equal(visible.find(message => message.toolCallId === 'read-fixture') &&
    JSON.stringify(visible.find(message => message.toolCallId === 'read-fixture')), readBytes);
  assert.equal(visible.filter(message => message.toolCallId === 'search-fixture').length, 1);
  assert.deepEqual(JSON.parse(visible.find(message => message.toolCallId === 'search-fixture').content[0].text),
    { navigation_withdrawn: true, original_result_seq: navigation.seq });
});

test('official replacement refuses changed call identity', { skip: !installed }, async () => {
  const { session, replacement, intent } = await fixture();
  replacement.message.toolCallId = 'unrelated-call';
  assert.throws(() => session.append('tool/result', replacement, intent), /only content/);
});

test('official replacement refuses rewriting a range spanning the full read', { skip: !installed }, async () => {
  const { session, read, replacement, intent } = await fixture();
  intent.surfaceOp.endSeq = read.seq;
  intent.sourceEventSeqs.push(read.seq);
  assert.throws(() => session.append('tool/result', replacement, intent), /exactly one current node/);
});
