/** Read-only verified carry of an already CLOSED question's actual provider usage. */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const fail = code => { throw new Error(code); };
async function bound(descriptor) {
  if (!descriptor || typeof descriptor.file !== 'string' || !/^[a-f0-9]{64}$/.test(descriptor.sha256 ?? '')) fail('bound_prior_receipt_required');
  const bytes = await readFile(descriptor.file);
  if (sha(bytes) !== descriptor.sha256) fail('prior_episode_binding_changed');
  return { value: JSON.parse(bytes), binding: { file: resolve(descriptor.file), sha256: sha(bytes) } };
}

export async function verifyPriorEpisode(descriptor, { rawCase, root, expectedKnownTokens, depth = 0 }) {
  if (depth > 12) fail('prior_episode_chain_too_deep');
  const closed = await bound(descriptor), receipt = closed.value;
  if (receipt.closed !== true || receipt.fixture !== false || receipt.initializeOnly !== false
      || receipt.unknown_usage_requests !== 0 || !Number.isSafeInteger(receipt.known_tokens)
      || receipt.known_tokens !== expectedKnownTokens) fail('known_closed_real_prior_required');
  const directory = dirname(closed.binding.file);
  const launch = JSON.parse(await readFile(resolve(directory, 'launch-start.json'), 'utf8'));
  if (launch.manifest_sha256 !== receipt.manifest_sha256 || launch.fixture !== false) fail('prior_launch_binding_changed');
  // The descriptor includes the exact manifest; never infer a sibling filename.
  const priorManifest = await bound(descriptor.manifest);
  if (priorManifest.binding.sha256 !== receipt.manifest_sha256) fail('prior_manifest_binding_changed');
  const manifest = priorManifest.value;
  const originalCase = await bound({ file: resolve(root, manifest.casePath), sha256: receipt.case_sha256 });
  if (['question_id', 'question'].some(key => originalCase.value.question?.[key] !== rawCase.question?.[key])
      || originalCase.value.topic_title !== rawCase.topic_title) fail('prior_question_identity_changed');
  const journalBytes = await readFile(resolve(directory, 'budget.jsonl'));
  if (sha(journalBytes) !== receipt.budget_audit_sha256) fail('prior_usage_binding_changed');
  const rows = journalBytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
  const pending = new Map(); let total = manifest.priorKnownTokens, starts = 0;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    if (row.episode_id !== manifest.episodeId || row.sequence !== i + 1) fail('prior_usage_order_changed');
    if (row.status === 'request_started') {
      if (pending.has(row.request_sequence)) fail('prior_duplicate_request');
      pending.set(row.request_sequence, row.wire_sha256); starts++;
    } else if (row.status === 'usage_unknown') fail('unknown_prior_usage');
    else if (row.status === 'usage_settled') {
      const u = row.provider_usage;
      if (!pending.has(row.request_sequence) || pending.get(row.request_sequence) !== row.wire_sha256
          || !u || !['input_tokens', 'output_tokens', 'total_tokens'].every(k => Number.isSafeInteger(u[k]) && u[k] >= 0)
          || u.input_tokens + u.output_tokens !== u.total_tokens || row.unknown_usage_requests !== 0) fail('invalid_prior_settlement');
      pending.delete(row.request_sequence); total += u.total_tokens;
      if (row.known_episode_tokens_after !== total) fail('prior_usage_total_changed');
    }
  }
  if (pending.size || starts !== receipt.model_requests || total !== expectedKnownTokens) fail('prior_pending_or_total_changed');
  const bindings = [closed.binding, priorManifest.binding, originalCase.binding,
    { file: resolve(directory, 'launch-start.json'), sha256: sha(await readFile(resolve(directory, 'launch-start.json'))) },
    { file: resolve(directory, 'budget.jsonl'), sha256: sha(journalBytes) }];
  if (manifest.priorKnownTokens > 0) {
    const previous = await verifyPriorEpisode(manifest.priorClosedReceipt, { rawCase, root, expectedKnownTokens: manifest.priorKnownTokens, depth: depth + 1 });
    bindings.push(...previous.bindings);
  } else if (manifest.priorKnownTokens !== 0) fail('explicit_prior_usage_required');
  return { known_tokens: total, bindings, prior_status: receipt.status };
}
