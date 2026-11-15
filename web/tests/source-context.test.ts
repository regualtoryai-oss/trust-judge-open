import test from "node:test";
import assert from "node:assert/strict";
import { sourceContextExcerpt, sourceContextWindow } from "../src/source-context.ts";
import type { Evidence, SourceContextBundle } from "../src/types.ts";

function context(): SourceContextBundle {
  return {
    span_scope: "document", paragraph_id: "p1", raw_doc_id: "doc_1", raw_para_id: "doc_1#p1",
    start_cp: 10, end_cp: 23, quote: "前文😀\n\n引文🦊结束\n后文",
    source_paragraph_ids: ["p1", "p2", "p3"], document_id: "d1",
    document_content_sha256: "doc-sha", paragraph_manifest_sha256: "paragraph-sha", bundle_id: "bundle",
    bundle_purpose: "read_context", anchor_paragraph_ids: ["p2"], anchor_paragraph_id: "p2",
    anchor_raw_para_id: "doc_1#p2", anchor_start_cp: 15, anchor_end_cp: 20,
    paragraph_count: 3, members: [
      { paragraph_id: "p1", raw_para_id: "doc_1#p1", start_cp: 10, end_cp: 13 },
      { paragraph_id: "p2", raw_para_id: "doc_1#p2", start_cp: 15, end_cp: 20 },
      { paragraph_id: "p3", raw_para_id: "doc_1#p3", start_cp: 21, end_cp: 23 },
    ],
  };
}
const oldEvidence = (): Evidence => ({ paragraph_id: "p2", raw_doc_id: "doc_1", quote: "引文🦊", start_cp: 0, end_cp: 3 });

test("document citations request all five possible original parents after their starting anchor", () => {
  assert.deepEqual(sourceContextWindow({ span_scope: "document" }), { before: 0, after: 4 });
  for (const scope of [undefined, null, "paragraph"] as const) {
    assert.deepEqual(sourceContextWindow({ span_scope: scope }), { before: 1, after: 3 });
  }
});

test("legacy paragraph-local coordinates add only the anchor offset, preserving emoji and paragraph gaps", () => {
  const bundle = context();
  assert.deepEqual(sourceContextExcerpt(bundle, oldEvidence()), {
    before: "前文😀\n\n", quote: "引文🦊", after: "结束\n后文",
  });
  assert.equal(oldEvidence().span_scope, undefined);
  assert.deepEqual(sourceContextExcerpt(bundle, { ...oldEvidence(), span_scope: null }), sourceContextExcerpt(bundle, oldEvidence()));
});

test("document-scope evidence keeps absolute coordinates and may span contiguous paragraphs and their real gap", () => {
  const bundle = context();
  const evidence: Evidence = { ...oldEvidence(), span_scope: "document", start_cp: 15, end_cp: 23, quote: "引文🦊结束\n后文" };
  assert.deepEqual(sourceContextExcerpt(bundle, evidence), { before: "前文😀\n\n", quote: evidence.quote, after: "" });
  assert.deepEqual(sourceContextExcerpt(bundle, { ...oldEvidence(), span_scope: "document", start_cp: 15, end_cp: 18 }), {
    before: "前文😀\n\n", quote: "引文🦊", after: "结束\n后文",
  });
});

test("mismatching quotes and invalid coordinates do not fall back to searching or repairing", () => {
  const bundle = context();
  for (const patch of [
    { quote: "改写" }, { start_cp: -1 }, { end_cp: 2.5 }, { end_cp: 0 }, { end_cp: 4 }, { end_cp: 9 },
    { span_scope: "document" as const, start_cp: 14, end_cp: 17 },
    { span_scope: "document" as const, start_cp: 25, end_cp: 28 },
  ]) assert.equal(sourceContextExcerpt(bundle, { ...oldEvidence(), ...patch }), null);
  assert.equal(oldEvidence().quote, "引文🦊");
});

test("range start is distinct from anchor, and stale or inconsistent source metadata is rejected", () => {
  const bundle = context();
  assert.notEqual(bundle.paragraph_id, oldEvidence().paragraph_id);
  for (const patch of [
    { anchor_paragraph_id: "p1" }, { anchor_start_cp: 16 }, { raw_doc_id: "other" },
    { end_cp: 22 }, { source_paragraph_ids: ["p2", "p1", "p3"] }, { paragraph_count: 2 },
  ]) assert.equal(sourceContextExcerpt({ ...bundle, ...patch }, oldEvidence()), null);
});

test("old data does not borrow paragraph document offsets or apply anchor offset twice", () => {
  const bundle = context();
  const evidence = { ...oldEvidence(), start_cp: 2, end_cp: 3, quote: "🦊" };
  assert.deepEqual(sourceContextExcerpt(bundle, evidence), { before: "前文😀\n\n引文", quote: "🦊", after: "结束\n后文" });
  assert.equal(sourceContextExcerpt(bundle, { ...evidence, start_cp: 17, end_cp: 18 }), null);
});

test("original overlapping, nested and empty paragraph members retain the full document envelope", () => {
  const bundle: SourceContextBundle = { ...context(), start_cp: 10, end_cp: 14, quote: "甲😀乙丙",
    anchor_start_cp: 11, anchor_end_cp: 12, members: [
      { paragraph_id: "p1", raw_para_id: "doc_1#p1", start_cp: 10, end_cp: 14 },
      { paragraph_id: "p2", raw_para_id: "doc_1#p2", start_cp: 11, end_cp: 12 },
      { paragraph_id: "p3", raw_para_id: "doc_1#p3", start_cp: 12, end_cp: 12 },
    ] };
  assert.deepEqual(sourceContextExcerpt(bundle, { ...oldEvidence(), quote: "😀", end_cp: 1 }), {
    before: "甲", quote: "😀", after: "乙丙",
  });
  assert.deepEqual(sourceContextExcerpt(bundle, { ...oldEvidence(), span_scope: "document", quote: "😀乙丙", start_cp: 11, end_cp: 14 }), {
    before: "甲", quote: "😀乙丙", after: "",
  });
});
