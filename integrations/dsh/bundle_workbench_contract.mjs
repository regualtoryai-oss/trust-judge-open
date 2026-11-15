/** Immutable, single-question workbench launch authorization. No credential access. */
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { isAbsolute } from 'node:path';
export const digest = value => createHash('sha256').update(value).digest('hex');
export const canonical = value => JSON.stringify(value && typeof value === 'object'
  ? Array.isArray(value) ? value.map(item => JSON.parse(canonical(item)))
    : Object.fromEntries(Object.keys(value).sort().map(key => [key, JSON.parse(canonical(value[key]))])) : value);
export function fail(code) { const error = new Error(code); error.code = code; throw error; }
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
export function descriptor(value) {
  if (!object(value) || Object.keys(value).sort().join(',') !== 'file,sha256' || !isAbsolute(value.file ?? '')
      || !/^[0-9a-f]{64}$/.test(value.sha256 ?? '')) fail('bundle_workbench_bound_descriptor_required');
  return value;
}
export async function readBound(value) {
  descriptor(value);
  const bytes = await readFile(value.file);
  if (digest(bytes) !== value.sha256) fail('bundle_workbench_source_binding_changed');
  return bytes;
}
export function knownUsage(records) {
  if (!Array.isArray(records) || records.some(row => !object(row) || row.usage_source !== 'provider'
      || !['input_tokens', 'output_tokens', 'total_tokens'].every(key => Number.isSafeInteger(row[key]) && row[key] >= 0)
      || row.input_tokens + row.output_tokens !== row.total_tokens)) fail('bundle_workbench_previous_usage_unknown');
  return records.reduce((sum, row) => sum + row.total_tokens, 0);
}
export async function loadWorkbenchManifest(file, sha256) {
  const bytes = await readBound({ file, sha256 }), manifest = JSON.parse(bytes);
  const required = ['format', 'authorization', 'execution_mode', 'raw_case', 'documents', 'paragraphs', 'previous_usage',
    'source_archives', 'excluded_spans', 'output_directory', 'model', 'whole_token_stop', 'front_token_stop', 'tail_reserved_tokens',
    'max_total_requests', 'max_native_requests', 'required_tail_requests', 'output_capacity', 'safety_tokens', 'timeout_seconds'];
  if (!object(manifest) || required.some(key => !Object.hasOwn(manifest, key))
      || Object.keys(manifest).some(key => !required.includes(key) && !['fixture_plan', 'compact_handles'].includes(key))
      || (manifest.compact_handles !== undefined && typeof manifest.compact_handles !== 'boolean')
      || manifest.format !== 'dsh-source-bundle-workbench-launch-v1' || !['workbench', 'fixture'].includes(manifest.execution_mode)
      || manifest.model !== 'deepseek-v4-pro' || !isAbsolute(manifest.output_directory ?? '')
      || manifest.required_tail_requests !== 3 || !Array.isArray(manifest.excluded_spans)
      || !object(manifest.output_capacity) || Object.keys(manifest.output_capacity).sort().join(',') !== 'navigation,submission') fail('bundle_workbench_explicit_launch_required');
  for (const key of ['whole_token_stop', 'front_token_stop', 'tail_reserved_tokens', 'max_total_requests', 'max_native_requests', 'required_tail_requests', 'safety_tokens', 'timeout_seconds']) {
    if (!Number.isSafeInteger(manifest[key]) || manifest[key] < 1) fail('bundle_workbench_positive_budget_required');
  }
  if (Object.values(manifest.output_capacity).some(value => !Number.isSafeInteger(value) || value < 1)
      || manifest.safety_tokens < 512 || manifest.tail_reserved_tokens >= manifest.whole_token_stop) fail('bundle_workbench_tail_reserve_required');
  const authorization = manifest.authorization;
  if (!object(authorization) || Object.keys(authorization).sort().join(',') !== 'case_sha256,episode_id,intent,question_id,question_sha256'
      || authorization.intent !== 'workbench-source-bundle-selection' || typeof authorization.episode_id !== 'string'
      || !authorization.episode_id || typeof authorization.question_id !== 'string' || !authorization.question_id) fail('bundle_workbench_single_question_authorization_required');
  const raw = JSON.parse(await readBound(manifest.raw_case));
  if (raw.format !== 'frozen-source-bundle-case-v1' || raw.question?.question_id !== authorization.question_id
      || authorization.question_sha256 !== digest(canonical(raw.question)) || authorization.case_sha256 !== manifest.raw_case.sha256) fail('bundle_workbench_original_question_binding_changed');
  const documents = JSON.parse(await readBound(manifest.documents)), paragraphs = JSON.parse(await readBound(manifest.paragraphs));
  if (canonical(documents) !== canonical(raw.documents) || canonical(paragraphs) !== canonical(raw.paragraphs)) fail('bundle_workbench_complete_source_files_mismatch');
  const byId = new Map(paragraphs.map(row => [row.id, row]));
  for (const span of manifest.excluded_spans) {
    const row = Array.isArray(span) ? byId.get(span[0]) : null;
    if (!row || span.length !== 4 || !Number.isSafeInteger(span[1]) || !Number.isSafeInteger(span[2])
        || span[1] < 0 || span[2] <= span[1] || span[2] > Array.from(row.text).length
        || Array.from(row.text).slice(span[1], span[2]).join('') !== span[3]) fail('bundle_workbench_original_exclusion_span_required');
  }
  if (!object(manifest.source_archives) || !Object.keys(manifest.source_archives).length) fail('bundle_workbench_source_archive_required');
  for (const [relative, row] of Object.entries(manifest.source_archives)) {
    if (!/^(?:app|scripts|integrations)\//.test(relative) || relative.split('/').some(part => ['', '.', '..'].includes(part))) fail('bundle_workbench_relative_source_archive_required');
    await readBound(row);
  }
  const previous = JSON.parse(await readBound(manifest.previous_usage));
  if (!object(previous) || Object.keys(previous).join(',') !== 'records') fail('bundle_workbench_previous_usage_records_required');
  const parentTokens = knownUsage(previous.records);
  const effectiveFrontStop = Math.min(manifest.front_token_stop, manifest.whole_token_stop - manifest.tail_reserved_tokens);
  const maxNativeRequests = Math.min(manifest.max_native_requests, manifest.max_total_requests - previous.records.length - manifest.required_tail_requests);
  if (parentTokens >= effectiveFrontStop || maxNativeRequests < 1) fail('bundle_workbench_no_front_budget_after_parent_and_tail');
  if (manifest.execution_mode === 'fixture') { descriptor(manifest.fixture_plan); await readBound(manifest.fixture_plan); }
  else if (manifest.fixture_plan !== undefined) fail('bundle_workbench_fixture_plan_forbidden_in_production');
  return { manifest, raw, previous: previous.records, parentTokens, effectiveFrontStop, maxNativeRequests };
}

