import test from "node:test";
import assert from "node:assert/strict";
import { decodeImportEntry, importTemplate, IMPORT_MAX_FILE_BYTES, IMPORT_MAX_TOTAL_BYTES, validateImportFileSelection } from "../src/import.ts";

test("import file boundaries reject empty, duplicate, incompatible and oversized inputs", () => {
  assert.throws(() => validateImportFileSelection([]), /先选择/);
  assert.throws(() => validateImportFileSelection([{name:"input.zip",size:10}]), /仅支持 JSON/);
  assert.throws(() => validateImportFileSelection([{name:"input.json",size:0}]), /文件为空/);
  assert.throws(() => validateImportFileSelection([{name:"input.json",size:1},{name:"INPUT.JSON",size:1}]), /同名文件/);
  assert.throws(() => validateImportFileSelection([{name:"input.json",size:IMPORT_MAX_FILE_BYTES+1}]), /6 MiB/);
  assert.throws(() => validateImportFileSelection([{name:"documents.json",size:IMPORT_MAX_FILE_BYTES},{name:"questions.json",size:IMPORT_MAX_TOTAL_BYTES-IMPORT_MAX_FILE_BYTES+1}]), /8 MiB/);
  assert.throws(() => validateImportFileSelection(["a","b","c"].map(name => ({name:`${name}.json`,size:1}))), /最多选择 2/);
  assert.doesNotThrow(() => validateImportFileSelection([{name:"documents.json",size:IMPORT_MAX_FILE_BYTES},{name:"questions.json",size:IMPORT_MAX_TOTAL_BYTES-IMPORT_MAX_FILE_BYTES}]));
});

test("decode refuses invalid UTF-8 and malformed JSON without disclosing input excerpts", () => {
  assert.throws(() => decodeImportEntry("input.json", new Uint8Array([0xff]).buffer), /UTF-8/);
  assert.throws(() => decodeImportEntry("input.json", new TextEncoder().encode('{"sensitive":"private"').buffer), (error: Error) => error.message.includes("JSON 格式错误") && !error.message.includes("private"));
});

test("decode preserves original whitespace, source Unicode and duplicate keys for server validation", () => {
  const text = ' { "title": "🦊原文", "title":"不得静默覆盖" }\n';
  assert.deepEqual(decodeImportEntry("input.json", new TextEncoder().encode(text).buffer), {name:"input.json",content:text});
});

test("download template contains only sources and projected questions with codepoint spans", () => {
  const payload = JSON.parse(importTemplate());
  assert.deepEqual(Object.keys(payload).sort(), ["documents","questions","title"]);
  assert.match(payload.documents[0].content, /模拟资料/);
  assert.equal(payload.documents[0].source_credibility, null);
  assert.deepEqual(Object.keys(payload.questions[0]).sort(),["question","question_id"]);
  for (const document of payload.documents) for (const paragraph of document.paragraphs) {
    assert.equal(paragraph.char_start,0);
    assert.equal(paragraph.char_end,Array.from(document.content).length);
  }
});
