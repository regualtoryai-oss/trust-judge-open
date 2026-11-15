import test from 'node:test';
import assert from 'node:assert/strict';
import { compactBundleSearchResult, withdrawnBundleNavigation } from './bundle_evidence.mjs';

function result() {
  const base = { navigation_score: .123456789, navigation_only: true, eligible_for_citation: false,
    unread_context_state: 'unknown_not_irrelevant', semantic_verified: false };
  return { tool: 'search_source_bundles', status: 'completed', original_question: '原题保留时间和口径。', query: '检索词',
    corpus_paragraph_count: 80, matching_parent_count: 40, more_matching_sources_available: true,
    no_match_proves_irrelevance: false, semantic_verified: false, results: [
      { ...base, paragraph_id: 'opaque-uuid-1', raw_doc_id: 'doc-1', raw_para_id: '原始段甲', title: '完整标题😀',
        excerpt_text: '😀原文限定尚未确认。', excerpt_start_cp: 5, excerpt_end_cp: 5 + Array.from('😀原文限定尚未确认。').length, parent_codepoints: 100, omitted_before: true, omitted_after: true },
      { ...base, paragraph_id: 'opaque-uuid-2', raw_doc_id: 'doc-1', raw_para_id: '原始段乙', title: '完整标题😀',
        excerpt_text: '短段。', excerpt_start_cp: 0, excerpt_end_cp: 3, parent_codepoints: 3, omitted_before: false, omitted_after: false },
      { ...base, paragraph_id: 'opaque-uuid-3', raw_doc_id: 'doc-2', raw_para_id: '原始段丙', title: '另一完整标题',
        excerpt_text: '原文尾部。', excerpt_start_cp: 8, excerpt_end_cp: 13, parent_codepoints: 13, omitted_before: true, omitted_after: false },
    ] };
}

test('compact navigation preserves every original ID/title/excerpt/CP/order without duplicate flags or scores', () => {
  const raw = result(), before = structuredClone(raw), compact = compactBundleSearchResult(raw);
  assert.deepEqual(raw, before);
  assert.equal(compact.results.length, raw.results.length);
  assert.equal(compact.documents.length, 2);
  const documents = new Map(compact.documents.map(([alias, raw_doc_id, title]) => [alias, { raw_doc_id, title }]));
  compact.results.forEach((row, index) => {
    const [paragraph_id, alias, raw_para_id, excerpt_text, excerpt_start_cp, excerpt_end_cp, parent_codepoints] = row;
    assert.deepEqual({ paragraph_id, ...documents.get(alias), raw_para_id, excerpt_text, excerpt_start_cp, excerpt_end_cp, parent_codepoints },
      Object.fromEntries(['paragraph_id', 'raw_doc_id', 'title', 'raw_para_id', 'excerpt_text', 'excerpt_start_cp', 'excerpt_end_cp', 'parent_codepoints']
        .map(key => [key, raw.results[index][key]])));
  });
  assert.equal(compact.navigation_only, true);
  assert.equal(compact.eligible_for_citation, false);
  assert.equal(compact.unread_context_state, 'unknown_not_irrelevant');
  assert.equal(compact.more_matching_sources_available, true);
  assert.equal(compact.semantic_verified, false);
  assert.ok(!JSON.stringify(compact).includes('navigation_score'));
  assert.ok(JSON.stringify(compact).length < JSON.stringify(raw).length);
});

for (const mutation of ['changed_title', 'duplicate_id', 'wrong_CP', 'boolean_CP', 'wrong_omission', 'verified_claim']) {
  test(`compact navigation fails closed on ${mutation}`, () => {
    const raw = result();
    if (mutation === 'changed_title') raw.results[1].title = '不一致标题';
    if (mutation === 'duplicate_id') raw.results[1].paragraph_id = raw.results[0].paragraph_id;
    if (mutation === 'wrong_CP') raw.results[0].excerpt_end_cp++;
    if (mutation === 'boolean_CP') raw.results[0].excerpt_start_cp = true;
    if (mutation === 'wrong_omission') raw.results[0].omitted_after = false;
    if (mutation === 'verified_claim') raw.results[0].semantic_verified = true;
    assert.throws(() => compactBundleSearchResult(raw), /bundle_search_projection_/);
  });
}

test('withdrawn program navigation keeps every candidate address/title but no preview text', () => {
  const raw = result(), compact = compactBundleSearchResult(raw), receipt = withdrawnBundleNavigation(raw);
  assert.deepEqual(receipt.documents, compact.documents);
  assert.deepEqual(receipt.results, compact.results.map(row => row.slice(0, 3)));
  assert.equal(receipt.candidate_count, raw.results.length);
  assert.equal(receipt.preview_text_withdrawn, true);
  assert.equal(receipt.unread_context_state, 'unknown_not_irrelevant');
  assert.match(receipt.original_navigation_sha256, /^[0-9a-f]{64}$/);
  raw.results.forEach(row => assert.ok(!JSON.stringify(receipt).includes(row.excerpt_text)));
});

