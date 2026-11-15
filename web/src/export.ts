import type { ExportJob } from "./types";

export interface ExportReadiness {
  topic_id: string;
  revision_id: string;
  total_questions: number;
  completed_questions: number;
  missing_questions: { id: string; raw_question_id: string; question: string }[];
  can_export: boolean;
  blockers: { code: string; message: string }[];
  published_revision_id: string | null;
  existing_export: ExportJob | null;
  preview_url: string;
}

export function exportMatches(job: ExportJob, topicId: string, revisionId: string): boolean {
  return job.topic_id === topicId && job.revision_id === revisionId;
}

export function exportReady(job: ExportJob | null): boolean {
  return job?.status === "succeeded";
}

export function exportProgress(status: string): string {
  return ({ queued: "等待导出任务执行", running: "校验并打包中", succeeded: "提交包已就绪", failed: "提交包生成失败" } as Record<string, string>)[status] || "正在读取导出状态";
}

export function exportFailure(error: unknown): string {
  if (error && typeof error === "object" && typeof (error as Record<string, unknown>).message === "string") {
    return String((error as Record<string, unknown>).message);
  }
  return "导出校验未通过，请重新检查导出资格后再试。";
}

/** Read the response before triggering a browser download so HTTP failures stay visible. */
export async function readExportDownload(
  response: Response,
  expectedType: "application/zip" | "application/json",
): Promise<Blob> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail ?? body?.error ?? body;
    const message = typeof detail === "string" ? detail : typeof detail?.message === "string" ? detail.message : `下载失败（HTTP ${response.status}），请检查导出状态后重试。`;
    throw new Error(message);
  }
  const contentType = response.headers.get("content-type")?.split(";")[0].trim();
  if (contentType !== expectedType) throw new Error("下载响应类型不正确，文件未保存，请刷新导出状态后重试。");
  const blob = await response.blob();
  if (!blob.size) throw new Error("下载文件为空，文件未保存，请重新生成。");
  return blob;
}
