/** Explicit no-network provider fixture. No fallback to the host's real fetch. */
import { readFileSync, appendFileSync } from 'node:fs';
if (process.env.TRUST_DSH_BUNDLE_FIXTURE !== '1') throw new Error('bundle_fixture_explicit_environment_required');
const plan = JSON.parse(readFileSync(process.env.TRUST_DSH_BUNDLE_FIXTURE_PLAN, 'utf8'));
let index = 0;
globalThis[Symbol.for('trust-judge:bundle-fixture-transport')] = true;
globalThis.fetch = async (url, init) => {
  if (String(url) !== 'https://api.deepseek.com/anthropic/v1/messages' || !plan[index]) throw new Error('bundle_fixture_unexpected_request');
  const row = plan[index++], body = JSON.parse(init.body);
  appendFileSync(process.env.TRUST_DSH_BUNDLE_FIXTURE_WIRES, `${JSON.stringify(body)}\n`);
  if (row.delay_ms !== undefined) {
    if (!Number.isSafeInteger(row.delay_ms) || row.delay_ms < 0 || row.delay_ms > 10000) throw new Error('bundle_fixture_invalid_delay');
    await new Promise((resolve, reject) => {
      if (init.signal?.aborted) { reject(new Error('bundle_fixture_aborted')); return; }
      const timer = setTimeout(resolve, row.delay_ms);
      init.signal?.addEventListener('abort', () => { clearTimeout(timer); reject(new Error('bundle_fixture_aborted')); }, { once: true });
    });
  }
  const calls = row.tool_calls ?? [{ name: row.name, arguments: row.arguments }];
  const contentEvents = row.text_only === true ? [
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: row.text ?? 'Explicit no-submission fixture.' } },
    { type: 'content_block_stop', index: 0 },
  ] : calls.flatMap((call, position) => [
    { type: 'content_block_start', index: position, content_block: { type: 'tool_use',
      id: row.tool_calls ? `bundle_fixture_call_${index}_${position + 1}` : `bundle_fixture_call_${index}`, name: call.name, input: {} } },
    { type: 'content_block_delta', index: position, delta: { type: 'input_json_delta', partial_json: typeof call.raw_arguments === 'string'
      ? call.raw_arguments : JSON.stringify(call.arguments) } },
    { type: 'content_block_stop', index: position },
  ]);
  const events = [
    { type: 'message_start', message: { id: `bundle_fixture_${index}`, type: 'message', role: 'assistant', model: body.model,
      content: [], stop_reason: null, stop_sequence: null, ...(row.omit_usage ? {} : { usage: { input_tokens: 100, output_tokens: 0 } }) } },
    ...contentEvents,
    { type: 'message_delta', delta: { stop_reason: row.text_only ? 'end_turn' : 'tool_use', stop_sequence: null },
      ...(row.omit_usage ? {} : { usage: { output_tokens: 50 } }) },
    { type: 'message_stop' },
  ];
  return new Response(events.map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(''),
    { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
};

