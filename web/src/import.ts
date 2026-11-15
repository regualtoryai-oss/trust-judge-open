/** File checks improve feedback; the server remains the authority for input integrity. */
export const IMPORT_MAX_FILE_BYTES = 6 * 1024 * 1024;
export const IMPORT_MAX_TOTAL_BYTES = 8 * 1024 * 1024;

export interface ImportEntry { name: string; content: string }
export interface ImportCounts { documents: number; paragraphs: number; questions: number }
export interface ImportPreview {
  preview_id: string;
  fingerprint: string;
  title: string;
  counts: ImportCounts;
  excluded_fields: { path: string; fields: string[]; count: number }[];
  warnings: string[];
  existing_topic_id: string | null;
  expires_at: string;
  questions: { question_id: string; question: string }[];
  truncated_questions_count: number;
}
export interface ImportResult {
  topic_id: string;
  idempotent: boolean;
  counts: ImportCounts;
  title: string;
  fingerprint: string;
}

export function validateImportFileSelection(files: readonly { name: string; size: number }[]): void {
  if (!files.length) throw new Error("请先选择原始数据 JSON 文件。");
  if (files.length > 2) throw new Error("最多选择 2 个文件：一个 input.json，或原始文档与问题文件各一个。");
  const names = new Set<string>();
  let total = 0;
  for (const file of files) {
    if (!/\.json$/i.test(file.name)) throw new Error(`${file.name}：仅支持 JSON 文件，请先解压或转换后导入。`);
    const name = file.name.toLowerCase();
    if (names.has(name)) throw new Error(`${file.name}：重复选择了同名文件，请重新选择。`);
    names.add(name);
    if (file.size === 0) throw new Error(`${file.name}：文件为空。`);
    if (file.size > IMPORT_MAX_FILE_BYTES) throw new Error(`${file.name}：超过单文件 6 MiB 限制。`);
    total += file.size;
  }
  if (total > IMPORT_MAX_TOTAL_BYTES) throw new Error("文件总量超过 8 MiB，请拆分为独立专题后导入。");
}

export function decodeImportEntry(name: string, bytes: ArrayBuffer): ImportEntry {
  let content: string;
  try { content = new TextDecoder("utf-8", { fatal: true }).decode(bytes); }
  catch { throw new Error(`${name}：不是有效的 UTF-8 文本，请按 UTF-8 编码保存。`); }
  try { JSON.parse(content); }
  catch { throw new Error(`${name}：JSON 格式错误，请检查引号、逗号和括号。`); }
  // Do not reserialize, project, or normalize source data. The server checks duplicate keys and spans.
  return { name, content };
}

export function importTemplate(): string {
  const content = "【模拟资料】10月1日09时，临河市发布通报：受道路积水影响，2条公交线路临时停运。";
  return JSON.stringify({
    title: "新赛题专题（请替换模拟内容）",
    documents: [{
      doc_id: "doc_001", title: "临河市公交运行通报（模拟）", content,
      publish_time: "2026-10-01 09:00", content_url: "", source_credibility: null,
      paragraphs: [{ para_id: "p_001", char_start: 0, char_end: Array.from(content).length }],
    }],
    questions: [{ question_id: "q_001", question: "截至10月1日09时，有多少条公交线路临时停运？" }],
  }, null, 2);
}

export function downloadImportTemplate(): void {
  const url = URL.createObjectURL(new Blob([importTemplate()], { type: "application/json;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "input.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
