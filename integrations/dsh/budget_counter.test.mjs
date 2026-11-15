import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { createOfficialTokenizerCounter } from './budget_counter.mjs';

const pythonExecutable = resolve('.runtime/v20-embedding-env/Scripts/python.exe');
const tokenizerPath = resolve('.runtime/v32-deepseek-tokenizer-r2/tokenizer.json');
const tokenizerSha256 = '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf';
const available = existsSync(pythonExecutable) && existsSync(tokenizerPath);

test('counter requires pinned absolute local assets', () => {
  assert.throws(() => createOfficialTokenizerCounter({ pythonExecutable: 'python', tokenizerPath, tokenizerSha256 }));
});
test('offline official tokenizer includes tools and preserves Unicode with no provider', { skip: !available }, async () => {
  const counter = createOfficialTokenizerCounter({ pythonExecutable, tokenizerPath, tokenizerSha256 });
  const wire = { model: 'deepseek-v4-pro', messages: [{ role: 'user', content: '原题😀 é' }], max_tokens: 512 };
  const plain = await counter({ serialized_body: JSON.stringify(wire) });
  const tools = await counter({ serialized_body: JSON.stringify({ ...wire,
    tools: [{ type: 'function', function: { name: 'read_sources', parameters: { type: 'object', properties: { refs: { type: 'array', items: { type: 'string' } } } } } }] }) });
  assert(Number.isSafeInteger(plain) && plain > 0); assert(tools > plain);
});
test('mismatched tokenizer and oversized input fail before Python work', { skip: !available }, async () => {
  const wrong = createOfficialTokenizerCounter({ pythonExecutable, tokenizerPath, tokenizerSha256: '0'.repeat(64) });
  await assert.rejects(wrong({ serialized_body: '{}' }), { code: 'tokenizer_binding_changed' });
  const bounded = createOfficialTokenizerCounter({ pythonExecutable, tokenizerPath, tokenizerSha256, maxWireBytes: 2 });
  await assert.rejects(bounded({ serialized_body: 'more' }), { code: 'unknown_input_cost' });
});
