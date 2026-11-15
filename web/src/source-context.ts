import { codepointExcerpt } from "./source.ts";
import type { Evidence, SourceContextBundle } from "./types.ts";

const coordinate = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

/** A document citation starts at its first parent and may cover all five. */
export function sourceContextWindow(evidence: Pick<Evidence, "span_scope">) {
  return evidence.span_scope === "document" ? { before: 0, after: 4 } : { before: 1, after: 3 };
}

/** Resolve stored evidence coordinates without searching, repairing or changing its quote. */
export function sourceContextExcerpt(context: SourceContextBundle, evidence: Evidence) {
  if (!context || context.span_scope !== "document" || context.bundle_purpose !== "read_context" ||
      typeof context.quote !== "string" || typeof evidence.quote !== "string" ||
      !coordinate(context.start_cp) || !coordinate(context.end_cp) || context.start_cp >= context.end_cp ||
      Array.from(context.quote).length !== context.end_cp - context.start_cp ||
      !coordinate(context.anchor_start_cp) || !coordinate(context.anchor_end_cp) ||
      context.anchor_start_cp < context.start_cp || context.anchor_end_cp > context.end_cp ||
      context.anchor_start_cp > context.anchor_end_cp ||
      context.anchor_paragraph_id !== evidence.paragraph_id ||
      !Array.isArray(context.anchor_paragraph_ids) || !context.anchor_paragraph_ids.includes(evidence.paragraph_id) ||
      (evidence.raw_doc_id !== undefined && evidence.raw_doc_id !== context.raw_doc_id) ||
      (evidence.raw_para_id !== undefined && evidence.raw_para_id !== context.anchor_raw_para_id) ||
      !coordinate(evidence.start_cp) || !coordinate(evidence.end_cp) || evidence.start_cp >= evidence.end_cp ||
      !Array.isArray(context.members) || !context.members.length ||
      !Array.isArray(context.source_paragraph_ids) ||
      context.paragraph_count !== context.members.length ||
      context.source_paragraph_ids.length !== context.members.length) return null;

  let previousStart = context.start_cp;
  let maximumEnd = context.start_cp;
  const seen = new Set<string>();
  for (const [index, member] of context.members.entries()) {
    if (!member || typeof member.paragraph_id !== "string" || !member.paragraph_id ||
        typeof member.raw_para_id !== "string" || !member.raw_para_id || seen.has(member.paragraph_id) ||
        context.source_paragraph_ids[index] !== member.paragraph_id ||
        !coordinate(member.start_cp) || !coordinate(member.end_cp) ||
        member.start_cp < previousStart || member.start_cp > member.end_cp || member.end_cp > context.end_cp) return null;
    seen.add(member.paragraph_id);
    previousStart = member.start_cp;
    maximumEnd = Math.max(maximumEnd, member.end_cp);
  }
  const first = context.members[0];
  const anchor = context.members.find(member => member.paragraph_id === evidence.paragraph_id);
  if (first.paragraph_id !== context.paragraph_id || first.raw_para_id !== context.raw_para_id ||
      first.start_cp !== context.start_cp || maximumEnd !== context.end_cp || !anchor ||
      anchor.raw_para_id !== context.anchor_raw_para_id || anchor.start_cp !== context.anchor_start_cp ||
      anchor.end_cp !== context.anchor_end_cp) return null;

  let absoluteStart: number;
  let absoluteEnd: number;
  if (evidence.span_scope === "document") {
    absoluteStart = evidence.start_cp;
    absoluteEnd = evidence.end_cp;
  } else if (evidence.span_scope == null || evidence.span_scope === "paragraph") {
    if (evidence.end_cp > context.anchor_end_cp - context.anchor_start_cp) return null;
    absoluteStart = context.anchor_start_cp + evidence.start_cp;
    absoluteEnd = context.anchor_start_cp + evidence.end_cp;
  } else return null;
  if (!Number.isSafeInteger(absoluteStart) || !Number.isSafeInteger(absoluteEnd)) return null;
  return codepointExcerpt(context.quote, evidence.quote,
    absoluteStart - context.start_cp, absoluteEnd - context.start_cp);
}
