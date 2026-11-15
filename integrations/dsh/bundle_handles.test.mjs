import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createBundleBridge, createBundleHandleBridge } from './bundle_bridge.mjs';
import { handleReadParameters, handleReadArguments, handleSubmitArguments, parseUniqueNativeArguments,
  stableHandleJson, handleSha256 } from './bundle_handles.mjs';
import { syntheticBundleCase } from './bundle_fixture.mjs';
import { assertSupportedJsonSchema, validateJsonSchemaValue } from '../../.runtime/dsh-official-r1/node_modules/@deepseek-ai/dsh-tools/lib/index.js';
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

async function setup() {
  const dir = await mkdtemp(resolve(ROOT, '.runtime/bundle-handles-unit-')), raw = syntheticBundleCase();
  const casePath = resolve(dir, 'synthetic-raw.json'); await writeFile(casePath, stableHandleJson(raw));
  const config = { casePath, caseSha256: handleSha256(await readFile(casePath)), pythonExecutable: resolve(ROOT, '.venv/Scripts/python.exe'),
    scriptSha256: handleSha256(await readFile(resolve(ROOT, 'scripts/source_bundle_tool.py'))) };
  const bridge = await createBundleBridge(config), navigation = await bridge.call('search', { query: raw.question.question, limit: 40, exclude_paragraph_ids: [] });
  const paths = { handlesPath: resolve(dir, 'handles.json'), nativeProjectionsPath: resolve(dir, 'native-projections.jsonl') };
  const handles = await createBundleHandleBridge(bridge, config, navigation, paths);
  return { raw, bridge, handles, paths, close: async () => { await handles.close(); await bridge.close(); } };
}

test('raw tool arguments reject repeated original anchors, escaped duplicates and nested duplicates', () => {
  for (const raw of ['{"selected":{"P1":{},"P1":{}}}', '{"selected":{"P1":{},"P\\u0031":{}}}',
    '{"bundle_refs":{"B1":true,"B1":true}}', '{"selected":{"P1":{"argument":{"内容":"甲","内容":"乙"}}}}']) {
    assert.throws(() => parseUniqueNativeArguments(raw), /duplicate_native_json_key/);
  }
  const value = { selected: { P1: { fact_des: 'literal { [ : " "' } } };
  assert.deepEqual(parseUniqueNativeArguments(JSON.stringify(value)), value);
});

test('program legal handles preserve exact CP, gaps and emoji; impossible old range and duplicate-row schemas', async () => {
  const { raw, bridge, handles, paths, close } = await setup();
  try {
    const manifest = handles.manifest;
    assert.equal(manifest.format, 'source-bundle-native-handles-v1');
    assert.equal(new Set(manifest.bundles.map(row => row.canonical_bundle.bundle_id)).size, manifest.bundles.length);
    assert.deepEqual(manifest.paragraphs.map(row => row.ref), ['P1', 'P2', 'P3', 'P4']);
    manifest.bundles.forEach(row => {
      assert.ok(row.paragraph_refs.length >= 1 && row.paragraph_refs.length <= 5);
      const source = row.canonical_bundle;
      assert.equal(source.quote, Array.from(raw.documents[0].content).slice(source.start_cp, source.end_cp).join(''));
    });
    const readSchema = handleReadParameters(manifest); assertSupportedJsonSchema(readSchema);
    assert.ok(validateJsonSchemaValue(readSchema, { anchor_paragraph_ids: ['opaque-source-0'], before: 1, after: 4 }).length);
    assert.ok(validateJsonSchemaValue(readSchema, { bundle_refs: { UNKNOWN: true } }).length);
    assert.throws(() => handleReadArguments(manifest, { bundle_refs: { UNKNOWN: true } }), /unknown_or_oversize/);
    assert.throws(() => handleReadArguments(manifest, { bundle_refs: { B1: true }, before: 1, after: 4 }), /read_shape/);
    const args = { bundle_refs: { B1: true } }, view = await handles.read(args,
      { callId: 'native-read-1', rawArguments: JSON.stringify(args) });
    assert.equal(view.bundles[0].quote, manifest.bundles[0].canonical_bundle.quote);
    assert.ok(view.bundles[0].quote.includes('\r\n<真实原始分隔😀>'));
    const quote = Array.from(view.bundles[0].quote);
    for (const [ref, start, end] of view.bundles[0].paragraph_spans) {
      const paragraph = manifest.paragraphs.find(row => row.ref === ref);
      const original = raw.paragraphs.find(row => row.id === paragraph.paragraph_id);
      assert.equal(quote.slice(start, end).join(''), original.text);
    }
    const schema = handles.submitParameters(); assertSupportedJsonSchema(schema);
    assert.ok(validateJsonSchemaValue(schema, { selected: [{ start_paragraph_id: 'same', end_paragraph_id: 'x' },
      { start_paragraph_id: 'same', end_paragraph_id: 'y' }] }).length);
    const value = { end_ref: 'P2', fact_des: '显式合成来源注释，未检验语义。',
      argument: Object.fromEntries(['时间', '主体', '地点', '内容', '影响', '数值口径'].map(key => [key, key === '内容' ? raw.paragraphs[0].text : ''])),
      entities: [], proposed_roles: ['L'] };
    const native = { selected: { P1: value } };
    assert.deepEqual(validateJsonSchemaValue(schema, native), []);
    assert.throws(() => handleSubmitArguments(manifest, bridge.snapshot().reads, { selected: { P999: value } }), /unread_or_cross_bundle/);
    const artifact = await handles.submit(native, { callId: 'native-submit', rawArguments: JSON.stringify(native) });
    assert.equal(artifact.evidence.length, 1);
    assert.equal(artifact.evidence[0].paragraph_id, raw.paragraphs[0].id);
    assert.equal(artifact.evidence[0].quote, Array.from(raw.documents[0].content)
      .slice(raw.paragraphs[0].document_start_cp, raw.paragraphs[1].document_end_cp).join(''));
    const archived = (await readFile(paths.nativeProjectionsPath, 'utf8')).trim().split('\n').map(JSON.parse);
    assert.deepEqual(archived.map(row => row.operation), ['read', 'submit']);
    assert.equal(archived[0].native_read_view_sha256, handleSha256(stableHandleJson(view)));
    assert.equal(archived[0].canonical_result_sha256, handleSha256(stableHandleJson(archived[0].canonical_result)));
    assert.equal(archived[0].handles_sha256, handleSha256(await readFile(paths.handlesPath)));
    assert.equal(archived[0].canonical_result.bundles[0].quote, view.bundles[0].quote);
    assert.notEqual(archived[0].canonical_result_sha256, archived[0].native_read_view_sha256);
  } finally { await close(); }
});
