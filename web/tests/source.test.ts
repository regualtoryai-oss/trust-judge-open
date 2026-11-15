import test from "node:test";
import assert from "node:assert/strict";
import { codepointExcerpt } from "../src/source.ts";

test("emoji before a repeated quote uses the specified codepoint span", () => {
  assert.deepEqual(codepointExcerpt("🦊原文。原文。", "原文", 4, 6), {
    before: "🦊原文。",
    quote: "原文",
    after: "。",
  });
});

test("mismatched, UTF-16 and out-of-bounds coordinates are never highlighted", () => {
  assert.equal(codepointExcerpt("🦊原文。原文。", "原文", 5, 7), null);
  assert.equal(codepointExcerpt("原文", "改写", 0, 2), null);
  assert.equal(codepointExcerpt("原文", "原文", -1, 2), null);
  assert.equal(codepointExcerpt("原文", "", 1, 1), null);
  assert.equal(codepointExcerpt("原文", "原文", 0, 3), null);
});
