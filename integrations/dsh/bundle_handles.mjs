/** Pure source-handle projection. No model, loop, I/O, or semantic selection. */
import { createHash } from 'node:crypto';
const fail = code => { const error = new Error(code); error.code = code; throw error; };
const plain = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const clone = value => structuredClone(value);
export const stableHandleJson = value => JSON.stringify(value, function (_key, child) {
  return plain(child) ? Object.fromEntries(Object.keys(child).sort().map(key => [key, child[key]])) : child;
});
export const handleSha256 = value => createHash('sha256').update(value).digest('hex');
const core = ['span_scope', 'paragraph_id', 'raw_doc_id', 'raw_para_id', 'start_cp', 'end_cp', 'quote',
  'source_paragraph_ids', 'document_content_sha256', 'paragraph_manifest_sha256', 'document_id', 'bundle_id'];
const fields = (value, keys) => Object.fromEntries(keys.map(key => [key, value[key]]));
const closed = properties => ({ type: 'object', properties, required: Object.keys(properties), additionalProperties: false });
const text = { type: 'string' };
const argumentKeys = ['时间', '主体', '地点', '内容', '影响', '数值口径'];

/** Check raw logged JSON before using the runtime's already-parsed arguments. */
export function parseUniqueNativeArguments(raw) {
  if (typeof raw !== 'string' || raw.length > 1000000) fail('bundle_handles_raw_arguments_required');
  let parsed;
  try { parsed = JSON.parse(raw); } catch { fail('bundle_handles_invalid_native_json'); }
  const stack = [], tokens = /"(?:\\[\s\S]|[^"\\])*"|[{}\[\]]/g;
  for (const match of raw.matchAll(tokens)) {
    const token = match[0];
    if (token === '{') stack.push(new Set());
    else if (token === '[') stack.push(null);
    else if (token === '}' || token === ']') stack.pop();
    else if (raw.slice(match.index + token.length).trimStart().startsWith(':')) {
      const keys = stack.at(-1), key = JSON.parse(token);
      if (!keys || keys.has(key)) fail('bundle_handles_duplicate_native_json_key');
      keys.add(key);
    }
  }
  return parsed;
}

export function buildHandleManifest(raw, navigation, plan, binding) {
  if (!plain(raw) || !Array.isArray(raw.documents) || !Array.isArray(raw.paragraphs)
      || navigation?.tool !== 'search_source_bundles' || !Array.isArray(navigation.results)
      || navigation.results.length > 40 || plan?.tool !== 'plan_source_bundles' || plan.status !== 'planned'
      || !Array.isArray(plan.bundles) || !plain(binding)
      || !/^[a-f0-9]{64}$/.test(binding.case_sha256 ?? '') || !/^[a-f0-9]{64}$/.test(binding.backend_navigation_sha256 ?? '')) fail('bundle_handles_bound_plan_required');
  const paragraphs = [], seen = new Set(), byId = new Map(raw.paragraphs.map(row => [row.id, row]));
  for (const doc of raw.documents) for (const item of doc.paragraph_manifest) {
    const row = byId.get(item.id);
    if (!row || seen.has(item.id) || row.document_id !== doc.id || row.document_start_cp !== item.start_cp
        || row.document_end_cp !== item.end_cp) fail('bundle_handles_original_manifest_mismatch');
    paragraphs.push({ ref: `P${paragraphs.length + 1}`, paragraph_id: item.id, document_id: doc.id,
      start_cp: item.start_cp, end_cp: item.end_cp }); seen.add(item.id);
  }
  if (seen.size !== raw.paragraphs.length) fail('bundle_handles_complete_paragraph_manifest_required');
  const refs = new Map(paragraphs.map(row => [row.paragraph_id, row.ref]));
  const navIds = navigation.results.map(row => row.paragraph_id);
  if (new Set(navIds).size !== navIds.length) fail('bundle_handles_duplicate_navigation_anchor');
  const bundles = [], bundleIds = new Set(), covered = new Set();
  for (const bundle of plan.bundles) {
    const doc = raw.documents.find(row => row.id === bundle.document_id);
    const anchors = bundle.anchor_paragraph_ids;
    if (!doc || !Array.isArray(anchors) || !anchors.length || anchors.some(id => !navIds.includes(id))
        || bundleIds.has(bundle.bundle_id) || !Array.isArray(bundle.source_paragraph_ids)
        || !bundle.source_paragraph_ids.length || bundle.source_paragraph_ids.length > 5
        || Array.from(doc.content).slice(bundle.start_cp, bundle.end_cp).join('') !== bundle.quote) fail('bundle_handles_illegal_source_bundle');
    const paragraphRefs = bundle.source_paragraph_ids.map(id => refs.get(id));
    if (paragraphRefs.some(ref => !ref)) fail('bundle_handles_unknown_original_paragraph');
    anchors.forEach(id => covered.add(id)); bundleIds.add(bundle.bundle_id);
    bundles.push({ bundle_ref: `B${bundles.length + 1}`, anchor_paragraph_id: anchors[0],
      start_ref: paragraphRefs[0], end_ref: paragraphRefs.at(-1), paragraph_refs: paragraphRefs, canonical_bundle: clone(bundle) });
  }
  if (bundles.length > 40 || covered.size !== navIds.length || navIds.some(id => !covered.has(id))) fail('bundle_handles_incomplete_navigation_plan');
  return { format: 'source-bundle-native-handles-v1', case_sha256: binding.case_sha256,
    source_binding: clone(plan.source_binding), backend_navigation_sha256: binding.backend_navigation_sha256,
    window_policy: { before: 1, after: 3, max_paragraphs: 5 }, paragraphs, bundles };
}

function tables(handles) {
  if (handles?.format !== 'source-bundle-native-handles-v1' || !Array.isArray(handles.paragraphs) || !Array.isArray(handles.bundles)) fail('bundle_handles_manifest_required');
  return { paragraphs: new Map(handles.paragraphs.map(row => [row.ref, row])),
    byId: new Map(handles.paragraphs.map(row => [row.paragraph_id, row])),
    bundles: new Map(handles.bundles.map(row => [row.bundle_ref, row])) };
}

export function handleReadParameters(handles) {
  return closed({ bundle_refs: { type: 'object', properties: Object.fromEntries(handles.bundles.map(row => [row.bundle_ref,
    { type: 'boolean', const: true }])), required: [], additionalProperties: false } });
}

export function handleReadArguments(handles, args) {
  const { bundles } = tables(handles);
  if (!plain(args) || Object.keys(args).length !== 1 || !plain(args.bundle_refs)) fail('bundle_handles_read_shape_required');
  const entries = Object.entries(args.bundle_refs);
  if (!entries.length || entries.length > 12 || entries.some(([ref, value]) => !bundles.has(ref) || value !== true)) fail('bundle_handles_unknown_or_oversize_read');
  return { anchor_paragraph_ids: entries.map(([ref]) => bundles.get(ref).anchor_paragraph_id), before: 1, after: 3 };
}

export function handleReadView(handles, result, nativeArguments) {
  handleReadArguments(handles, nativeArguments);
  const { bundles, byId } = tables(handles), selected = Object.keys(nativeArguments.bundle_refs).map(ref => bundles.get(ref));
  if (result?.tool !== 'read_source_bundles' || result.status !== 'completed' || result.complete_original_slices !== true
      || !Array.isArray(result.bundles) || result.bundles.length !== selected.length || !plain(result.delivery)) fail('bundle_handles_complete_canonical_read_required');
  const views = result.bundles.map((bundle, index) => {
    const planned = selected[index];
    if (stableHandleJson(fields(bundle, core)) !== stableHandleJson(fields(planned.canonical_bundle, core))) fail('bundle_handles_read_source_binding_changed');
    return { bundle_ref: planned.bundle_ref, start_ref: planned.start_ref, end_ref: planned.end_ref,
      paragraph_refs: [...planned.paragraph_refs], paragraph_spans: bundle.source_paragraph_ids.map(id => {
        const row = byId.get(id); return [row.ref, row.start_cp - bundle.start_cp, row.end_cp - bundle.start_cp];
      }), quote: bundle.quote };
  });
  const refs = ids => ids.map(id => { const p = byId.get(id); if (!p) fail('bundle_handles_unknown_delivery_paragraph'); return p.ref; });
  return { format: 'source-bundle-native-read-v1', tool: 'read_source_bundles', status: 'completed',
    complete_original_slices: true, bundles: views, delivery: { read_call_index: result.delivery.read_call_index,
      complete_delivery_index: result.delivery.complete_delivery_index,
      new_paragraph_refs: refs(result.delivery.new_original_paragraph_ids),
      already_read_paragraph_refs: refs(result.delivery.already_read_original_paragraph_ids), full_original_source_bytes_returned: true },
    semantic_verified: false, native_model_delivery_verified: false };
}

export function handleNavigationView(handles, navigation, withdrawn = false) {
  const { byId } = tables(handles), documents = [], docs = new Map();
  const results = navigation.results.map(row => {
    if (!docs.has(row.raw_doc_id)) { docs.set(row.raw_doc_id, `D${documents.length + 1}`); documents.push([docs.get(row.raw_doc_id), row.raw_doc_id, row.title]); }
    const bundle = handles.bundles.find(b => b.canonical_bundle.anchor_paragraph_ids.includes(row.paragraph_id));
    if (!bundle) fail('bundle_handles_navigation_source_missing');
    const prefix = [bundle.bundle_ref, byId.get(row.paragraph_id).ref, docs.get(row.raw_doc_id), row.raw_para_id];
    return withdrawn ? prefix : [...prefix, row.excerpt_text, row.excerpt_start_cp, row.excerpt_end_cp, row.parent_codepoints];
  });
  return { format: 'source-bundle-handle-navigation-v1', navigation_only: true, eligible_for_citation: false,
    unread_context_state: 'unknown_not_irrelevant', preview_text_withdrawn: withdrawn, candidate_count: results.length,
    document_columns: ['document', 'raw_doc_id', 'title'], documents,
    result_columns: withdrawn ? ['bundle_ref', 'anchor_ref', 'document', 'raw_para_id']
      : ['bundle_ref', 'anchor_ref', 'document', 'raw_para_id', 'excerpt_text', 'excerpt_start_cp', 'excerpt_end_cp', 'parent_codepoints'], results,
    legal_bundle_ranges: handles.bundles.map(row => [row.bundle_ref, row.start_ref, row.end_ref, row.paragraph_refs]), semantic_verified: false };
}

export function allowedHandleEnds(handles, reads) {
  const { byId } = tables(handles), allowed = new Map();
  for (const read of reads) for (const bundle of read.result.bundles) {
    const refs = bundle.source_paragraph_ids.map(id => byId.get(id)?.ref);
    if (refs.some(ref => !ref)) fail('bundle_handles_unknown_read_paragraph');
    refs.forEach((start, index) => { if (!allowed.has(start)) allowed.set(start, new Set()); refs.slice(index).forEach(end => allowed.get(start).add(end)); });
  }
  return allowed;
}

export function handleSubmitParameters(handles, reads) {
  const allowed = allowedHandleEnds(handles, reads);
  const properties = Object.fromEntries([...allowed].map(([start, ends]) => [start, closed({ end_ref: { type: 'string', enum: [...ends] },
    fact_des: text, argument: closed(Object.fromEntries(argumentKeys.map(key => [key, text]))),
    entities: { type: 'array', items: text }, proposed_roles: { type: 'array', items: { type: 'string', enum: ['S', 'C', 'L'] } } })]));
  return closed({ selected: { type: 'object', properties, required: [], additionalProperties: false } });
}

export function handleSubmitArguments(handles, reads, args) {
  const { paragraphs } = tables(handles), allowed = allowedHandleEnds(handles, reads);
  if (!plain(args) || Object.keys(args).length !== 1 || !plain(args.selected) || Object.keys(args.selected).length > 30) fail('bundle_handles_selected_object_required');
  const selected = Object.entries(args.selected).map(([start, row]) => {
    const names = ['end_ref', 'fact_des', 'argument', 'entities', 'proposed_roles'];
    if (!plain(row) || Object.keys(row).length !== names.length || names.some(key => !Object.hasOwn(row, key))
        || !allowed.get(start)?.has(row.end_ref) || !paragraphs.has(row.end_ref)) fail('bundle_handles_unread_or_cross_bundle_selected_range');
    if (typeof row.fact_des !== 'string' || !row.fact_des.trim() || !plain(row.argument)
        || Object.keys(row.argument).length !== argumentKeys.length || argumentKeys.some(key => typeof row.argument[key] !== 'string')
        || !row.argument.内容.trim() || !Array.isArray(row.entities) || row.entities.some(v => typeof v !== 'string' || !v.trim())
        || new Set(row.entities).size !== row.entities.length || !Array.isArray(row.proposed_roles) || !row.proposed_roles.length
        || row.proposed_roles.some(role => !['S', 'C', 'L'].includes(role)) || new Set(row.proposed_roles).size !== row.proposed_roles.length) fail('bundle_handles_annotation_shape_required');
    return { start_paragraph_id: paragraphs.get(start).paragraph_id, end_paragraph_id: paragraphs.get(row.end_ref).paragraph_id,
      ...clone(fields(row, names.slice(1))) };
  });
  return { selected };
}
