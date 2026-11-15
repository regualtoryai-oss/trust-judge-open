import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyStagedNativePresence, screenStagedRead } from './staged_presence.mjs';
const preview = JSON.stringify({ views: [['P1', '原段短窗'], ['P2', '未读预览']] });
const full = JSON.stringify({ paragraphs: [{ p: 'P1', blocks: [['S1', '完整原段短窗之外的全文。🦊']] }] });
const stage = () => ({ phase: 'final', preview_context: preview, post_read_context: '已读取P1，未读保持未知。', full_read_result_json: full, read_refs: ['P1'] });
const wire = () => ({ system: '规则\n已读取P1，未读保持未知。', tools: [{ name: 'submit_evidence' }], messages: [
  { role: 'user', content: '原题' }, { role: 'assistant', content: [{ type: 'tool_use', id: 'c1', name: 'read_selected_sources', input: { refs: ['P1'] } }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'c1', content: [{ type: 'text', text: full }] }] }] });
test('native final phase requires exact original full result bound to the one read call', () => {
  const verified = verifyStagedNativePresence(wire(), stage());
  assert.equal(verified.phase, 'final'); assert.equal(verified.semantic_verified, false);
  for (const mutate of [b => b.messages[2].content[0].content[0].text = full.slice(1),
    b => b.messages[2].content[0].tool_use_id = 'other', b => b.messages[2].content[0].is_error = true,
    b => b.messages[1].content[0].input.refs = ['P2'], b => b.system += preview,
    b => b.tools.push({ name: 'read_selected_sources' }), b => b.system += full,
    b => b.messages.push({ role: 'user', content: preview }),
    b => b.messages[2].content[0].content.push({ type: 'text', text: full })]) {
    const body = wire(); mutate(body); assert.throws(() => verifyStagedNativePresence(body, stage()));
  }
});
test('preview phase exposes only the exact literal candidate section and read tool', () => {
  const p = { ...stage(), phase: 'preview', full_read_result_json: null, read_refs: [] };
  const body = { system: preview, tools: [{ name: 'read_selected_sources' }], messages: [{ role: 'user', content: '原题' }] };
  assert.equal(verifyStagedNativePresence(body, p).phase, 'preview');
  assert.throws(() => verifyStagedNativePresence({ ...body, system: preview.slice(1) }, p));
  assert.throws(() => verifyStagedNativePresence({ ...body, tools: [{ name: 'submit_evidence' }] }, p));
});
test('local read screen counts settled known usage and never certifies future native fit', async () => {
  const snapshot = { halted: false, in_flight: false, unknown_usage_requests: 0, started_calls: 1, known_episode_tokens: 3000, whole_limit: 20000 };
  const run = state => screenStagedRead({ guard: { snapshot: () => state }, countTokens: async () => 4000,
    readRequest: { full_read_result: { original: '全文' } }, maxOutputTokens: 4096, safetyTokens: 512 });
  const result = await run(snapshot);
  assert.equal(result.preflight_total_tokens, 11608); assert.equal(result.admitted, true);
  assert.equal(result.future_request_fit_guaranteed, false); assert.equal(result.native_next_request_still_required, true);
  assert.equal((await run({ ...snapshot, known_episode_tokens: 15000 })).admitted, false);
  for (const change of [{ in_flight: true }, { halted: true }, { unknown_usage_requests: 1 }, { started_calls: 0 }]) {
    await assert.rejects(run({ ...snapshot, ...change }));
  }
});
