/** No-network provider fixture, loaded only by an explicit --fixture SDK integration test. */
import { readFileSync, appendFileSync } from 'node:fs';
const responses = JSON.parse(readFileSync(process.env.TRUST_DSH_FIXTURE_PLAN, 'utf8'));
let index = 0;
globalThis.fetch = async (url, init) => {
  if (String(url) !== 'https://api.deepseek.com/anthropic/v1/messages' || !responses[index]) throw new Error('fixture_unexpected_request');
  const row = responses[index++], body = JSON.parse(init.body);
  appendFileSync(process.env.TRUST_DSH_FIXTURE_WIRES, `${JSON.stringify({ model: body.model, tools: body.tools?.map(tool => tool.name), system: body.system, messages: body.messages, max_tokens: body.max_tokens })}\n`);
  const events = [
    { type: 'message_start', message: { id: `fixture_${index}`, type: 'message', role: 'assistant', model: body.model,
      content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 100, output_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: `fixture_call_${index}`, name: row.name, input: {} } },
    { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: JSON.stringify(row.arguments) } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'tool_use', stop_sequence: null }, usage: { output_tokens: 50 } },
    { type: 'message_stop' },
  ];
  const text = events.map(value => `event: ${value.type}\ndata: ${JSON.stringify(value)}\n\n`).join('');
  return new Response(text, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
};
