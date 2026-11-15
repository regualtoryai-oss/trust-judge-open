import test from "node:test";
import assert from "node:assert/strict";
import { exportFailure, exportMatches, exportProgress, exportReady, readExportDownload } from "../src/export.ts";
import type { ExportJob } from "../src/types.ts";

const job: ExportJob = { export_id: "e1", topic_id: "t1", revision_id: "r1", status: "succeeded" };

test("a download job must belong to both the displayed topic and immutable revision", () => {
  assert.equal(exportMatches(job, "t1", "r1"), true);
  assert.equal(exportMatches(job, "t2", "r1"), false);
  assert.equal(exportMatches(job, "t1", "r2"), false);
  assert.equal(exportMatches({ export_id: "legacy", status: "succeeded" } as ExportJob, "t1", "r1"), false);
});

test("only the backend's actual succeeded export status permits a ZIP download", () => {
  assert.equal(exportReady(job), true);
  for (const status of ["queued", "running", "failed", "ready", "completed", "unknown"]) {
    assert.equal(exportReady({ ...job, status }), false);
  }
  assert.equal(exportReady(null), false);
});

test("queue, packing and failures are distinct and errors do not stringify arbitrary fields", () => {
  assert.equal(exportProgress("queued"), "等待导出任务执行");
  assert.equal(exportProgress("running"), "校验并打包中");
  assert.equal(exportProgress("failed"), "提交包生成失败");
  assert.equal(exportFailure({ message: "缺少 q_003", private_debug: "hidden" }), "缺少 q_003");
  assert.doesNotMatch(exportFailure({ private_debug: "hidden" }), /hidden/);
});

test("valid ZIP and non-formal JSON bytes are retained before the download is triggered", async () => {
  const archive = new Uint8Array([80, 75, 3, 4]);
  const zip = await readExportDownload(new Response(archive, { headers: { "content-type": "application/zip" } }), "application/zip");
  assert.deepEqual(new Uint8Array(await zip.arrayBuffer()), archive);
  const payload = '{"formal_submission":false,"files":{"evidence.json":[]}}';
  const preview = await readExportDownload(new Response(payload, { headers: { "content-type": "application/json; charset=utf-8" } }), "application/json");
  assert.equal(await preview.text(), payload);
});

test("HTTP download errors remain readable instead of being saved as fake ZIP files", async () => {
  const response = new Response(JSON.stringify({ detail: { code: "artifact_missing", message: "导出文件缺失，请重新创建。" } }), { status: 404, headers: { "content-type": "application/json" } });
  await assert.rejects(() => readExportDownload(response, "application/zip"), /导出文件缺失/);
});

test("plain-text transport failures expose the HTTP status and do not echo response bodies", async () => {
  const response = new Response("private upstream body", { status: 502 });
  await assert.rejects(() => readExportDownload(response, "application/zip"), (reason: unknown) => reason instanceof Error && /HTTP 502/.test(reason.message) && !reason.message.includes("private"));
});

test("an HTML error page or JSON body with success status cannot masquerade as an archive", async () => {
  for (const contentType of ["text/html", "application/json", "application/octet-stream"]) {
    await assert.rejects(() => readExportDownload(new Response("unexpected", { headers: { "content-type": contentType } }), "application/zip"), /响应类型不正确/);
  }
});

test("empty downloads fail visibly instead of producing a zero-byte submission", async () => {
  await assert.rejects(() => readExportDownload(new Response("", { headers: { "content-type": "application/zip" } }), "application/zip"), /文件为空/);
});
