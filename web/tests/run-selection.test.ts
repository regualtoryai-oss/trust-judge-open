import test from "node:test";
import assert from "node:assert/strict";
import { selectRunQuestions } from "../src/run-selection.ts";

test("all question request preserves the complete selection, deduplicates IDs and never mutates loaded questions", () => {
  const source = ["q_002", "q_001", "q_002", "q_003"];
  assert.deepEqual(selectRunQuestions(source, "all", 3),["q_002", "q_001", "q_003"]);
  assert.deepEqual(source,["q_002", "q_001", "q_002", "q_003"]);
  assert.deepEqual(selectRunQuestions(source, ["q_001"], 3),["q_001"]);
});

test("over-limit batches and stale cross-topic question IDs are blocked instead of silently truncated", () => {
  const source = Array.from({length:21},(_,index) => `q_${index}`);
  assert.equal(selectRunQuestions(source.slice(0,20), "all", 20).length,20);
  assert.throws(() => selectRunQuestions(source,"all",20), /21 个问题.*20 题上限/);
  assert.throws(() => selectRunQuestions(source,["other_topic"],20), /已不在当前专题/);
  assert.throws(() => selectRunQuestions([],"all",20), /没有可研判/);
});
