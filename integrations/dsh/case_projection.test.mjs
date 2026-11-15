import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { projectV18RawCase } from './case_projection.mjs';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
function fixture() { return { question: { question_id: 'q-new', question: '申请是否获批？', id: 'internal' }, title: '合成材料',
  paragraphs: [{ raw_doc_id: 'doc-a', raw_para_id: 'p1', text: '😀 申请尚未获批。', title: '原始标题', start_cp: 4, end_cp: 14, id: 'paragraph-uuid', document_id: 'document-uuid', content_sha256: 'frozen-original-hash', raw_publish_time: '三日前', content_url: 'https://example.invalid/source' }],
  selection: { reference: 'EXPECTED_SENTINEL_NOT_RUNTIME', gold: ['never'] } }; }
function convert(value) { const bytes = Buffer.from(JSON.stringify(value)); return projectV18RawCase(bytes, sha(bytes)); }
test('only original question/title/paragraph source values enter projection, not selection/reference', () => {
  const value = fixture(); value.paragraphs[0].end_cp = 4 + Array.from(value.paragraphs[0].text).length;
  const projected = convert(value), wire = JSON.stringify(projected);
  assert.ok(!wire.includes('EXPECTED_SENTINEL')); assert.ok(!wire.includes('"gold"')); assert.ok(!wire.includes('"selection"'));
  assert.deepEqual(projected.question, { question_id: 'q-new', question: '申请是否获批？' });
  assert.equal(projected.catalog[0].text, value.paragraphs[0].text); assert.equal(projected.source_metadata.P1.original_frozen_content_sha256, 'frozen-original-hash');
  assert.equal(projected.source_metadata.P1.raw_publish_time, '三日前'); assert.ok(!wire.includes('source_credibility'));
  assert.equal(projected.source_metadata.P1.raw_source_coordinate_binding_verified, false);
});
test('raw file SHA is checked and original source IDs and half-open codepoint length are validated', () => {
  const value = fixture(); assert.throws(() => projectV18RawCase(Buffer.from('{}'), '0'.repeat(64)), /sha256_mismatch/);
  value.paragraphs[0].end_cp = 999; assert.throws(() => convert(value), /coordinates_invalid/);
  value.paragraphs[0].end_cp = 4 + Array.from(value.paragraphs[0].text).length; value.paragraphs.push({ ...value.paragraphs[0] });
  assert.throws(() => convert(value), /duplicate_original/);
});
