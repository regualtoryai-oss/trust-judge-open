/** Local annotation isolation only: no model calls, reference answers or orchestration. */
import { createEvidenceBackend } from './evidence_tools.mjs';
import { normalizeSubmissionShape } from './submission_shape.mjs';

const ROLES = new Set(['D', 'C', 'F', 'P', 'L']);
const clone = value => structuredClone(value);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const numbers = value => new Set(value.match(/\d+(?:[.,]\d+)*/g) ?? []);

function identityFailure(originalArgs) {
  const error = new Error('unknown_or_duplicate_source_ref');
  error.code = 'unknown_or_duplicate_source_ref';
  error.original_args = clone(originalArgs);
  throw error;
}

function resolveRelations(args, acceptedIds) {
  if (!Array.isArray(args?.supporting) || !Array.isArray(args?.conflicting)) return false;
  const supported = args.supporting;
  const conflicted = args.conflicting.map(row => row?.p);
  return [supported, conflicted].every(refs => new Set(refs).size === refs.length && refs.every(p => acceptedIds.has(p)));
}

/**
 * Preserve the submitted proposal and isolate mechanically invalid annotation rows.
 * Exact positional rows may first be adapted without changing field contents.
 * The sole field-content transformation is a non-literal time becoming explicitly unknown.
 * No relation, conclusion, source identity, quote, role or other annotation is repaired.
 *
 * Unknown/duplicate selected source identities throw, before any transformation.
 * Other strict-backend failures are returned as validation_failed, never as success.
 * This function validates local structure and literals, not truth or semantic quality.
 */
export function isolateAnnotations(originalArgs, rawCase) {
  const original = clone(originalArgs);
  // Also validates immutable full-parent/source identities and any bound text hashes.
  const backend = createEvidenceBackend(rawCase, { contextMode: 'full-candidate-context' });
  const byRef = new Map(rawCase.catalog.map(row => [row.paragraph_ref, row]));
  let shape;
  try { shape = normalizeSubmissionShape(original); }
  catch (error) {
    if (error?.code !== 'invalid_submission_shape') throw error;
    return { format: 'annotation-isolation-v1', original_args: original, normalized_args: null, format_transformations: [], accepted_args: null,
      quarantine: [], transformations: [], original_selected_ids: [], accepted_ids: [], relations_resolvable: false,
      validation: { status: 'validation_failed', code: error.code, path: error.path }, submission: null,
      quality: { semantic_verified: false, support_conflict_verified: false, publishable: false } };
  }
  const normalized = shape.normalized_args, proposed = clone(normalized);
  const originalSelectedIds = Array.isArray(normalized?.selected) ? normalized.selected.map(row => row?.p) : [];
  const seen = new Set();
  for (const p of originalSelectedIds) {
    if (!byRef.has(p) || seen.has(p)) identityFailure(original);
    seen.add(p);
  }

  const quarantine = [], transformations = [];
  if (Array.isArray(proposed?.selected)) {
    proposed.selected = proposed.selected.filter((row, index) => {
      const source = byRef.get(row.p);
      // A non-literal time is unknown. Its original content remains in the audit.
      if (object(row.argument) && typeof row.argument.时间 === 'string' && row.argument.时间 && !source.text.includes(row.argument.时间)) {
        transformations.push({ selected_index: index, p: row.p, path: `selected[${index}].argument.时间`,
          original_value: row.argument.时间, accepted_value: '', reason: 'time_must_be_literal_or_empty' });
        row.argument.时间 = '';
      }
      const reasons = [];
      if (!Array.isArray(row.roles) || !row.roles.length || new Set(row.roles).size !== row.roles.length || row.roles.some(role => !ROLES.has(role))) reasons.push('invalid_proposed_roles');
      if (typeof row.fact_des !== 'string' || !row.fact_des.trim()) reasons.push('empty_or_invalid_fact_des');
      const available = numbers(source.text);
      const unsupported = [...numbers(JSON.stringify([row.fact_des, row.argument, row.entities]))].filter(value => !available.has(value));
      if (unsupported.length) reasons.push('annotation_number_not_in_source');
      if (!reasons.length) return true;
      quarantine.push({ selected_index: index, p: row.p, original_row: clone(original.selected[index]),
        reasons, ...(unsupported.length ? { unsupported_numeric_literals: unsupported } : {}) });
      return false;
    });
  }

  const acceptedIds = Array.isArray(proposed?.selected) ? proposed.selected.map(row => row?.p) : [];
  const resolvable = resolveRelations(proposed, new Set(acceptedIds));
  const referencedQuarantine = quarantine.filter(row => (Array.isArray(normalized?.supporting) && normalized.supporting.includes(row.p))
    || (Array.isArray(normalized?.conflicting) && normalized.conflicting.some(ref => ref?.p === row.p)));
  const result = {
    format: 'annotation-isolation-v1', original_args: original, normalized_args: normalized,
    format_transformations: shape.format_transformations, accepted_args: null,
    quarantine, transformations, original_selected_ids: originalSelectedIds, accepted_ids: acceptedIds,
    relations_resolvable: resolvable, validation: null, submission: null,
    quality: { semantic_verified: false, support_conflict_verified: false, publishable: false },
  };
  if (referencedQuarantine.length) {
    result.validation = { status: 'blocked_by_quarantined_relation', code: 'referenced_annotation_quarantined',
      refs: referencedQuarantine.map(row => row.p) };
    return result;
  }
  try {
    backend.submit(proposed);
    result.accepted_args = clone(proposed);
    result.validation = { status: 'validated', code: null };
    result.submission = backend.snapshot().submission;
  } catch (error) {
    result.validation = { status: 'validation_failed', code: typeof error?.code === 'string' ? error.code : 'local_tool_failed' };
  }
  return result;
}
