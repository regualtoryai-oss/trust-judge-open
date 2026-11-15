/** Whitelist-only projection of a CLOSED v18 input; selection/reference fields are never accessed. */
import { createHash } from 'node:crypto';
const sha = value => createHash('sha256').update(value).digest('hex');
const cpLength = value => Array.from(value).length;
function fail(code) { throw new Error(code); }
export function projectV18RawCase(bytes, expectedSha256) {
  if (!(bytes instanceof Uint8Array) || !/^[a-f0-9]{64}$/.test(expectedSha256) || sha(bytes) !== expectedSha256) fail('closed_v18_input_sha256_mismatch');
  const source = JSON.parse(Buffer.from(bytes).toString('utf8'));
  if (!source.question || typeof source.question.question !== 'string' || typeof source.question.question_id !== 'string' || typeof source.title !== 'string' || !Array.isArray(source.paragraphs)) fail('v18_raw_question_and_paragraphs_required');
  const catalog = [], source_metadata = {}, seen = new Set();
  for (const [i, row] of source.paragraphs.entries()) {
    if (!row || ['raw_doc_id', 'raw_para_id', 'text', 'title', 'id', 'document_id', 'content_sha256'].some(key => typeof row[key] !== 'string') || !Number.isInteger(row.start_cp) || !Number.isInteger(row.end_cp) || row.start_cp < 0 || row.end_cp - row.start_cp !== cpLength(row.text)) fail('v18_literal_parent_or_supplied_coordinates_invalid');
    const identity = JSON.stringify([row.raw_doc_id, row.raw_para_id]);
    if (seen.has(identity)) fail('duplicate_original_paragraph_identity');
    seen.add(identity); const p = `P${i + 1}`;
    catalog.push({ paragraph_ref: p, doc_id: row.raw_doc_id, para_id: row.raw_para_id, text: row.text, title: row.title, paragraph_complete: true, parent_complete: true });
    source_metadata[p] = { original_paragraph_id: row.id, original_document_id: row.document_id, doc_id: row.raw_doc_id, para_id: row.raw_para_id,
      original_frozen_content_sha256: row.content_sha256, source_text_sha256_utf8: sha(row.text),
      document_start_cp: row.start_cp, document_end_cp: row.end_cp,
      original_coordinates_supplied: true, raw_source_coordinate_binding_verified: false,
      ...(Object.hasOwn(row, 'content_url') ? { content_url: structuredClone(row.content_url) } : {}),
      ...(Object.hasOwn(row, 'raw_publish_time') ? { raw_publish_time: structuredClone(row.raw_publish_time) } : {}), semantic_verified: false };
  }
  return { format: 'dsh-whitelisted-v18-raw-case-v1', question: { question_id: source.question.question_id, question: source.question.question }, topic_title: source.title,
    catalog, source_metadata, binding: { original_frozen_file_sha256: expectedSha256, projection: 'question/title/paragraphs-only', reference_fields_accessed: false },
    unprovided_context_state: 'unknown_not_irrelevant', publishable: false };
}
