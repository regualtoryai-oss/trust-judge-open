/** Local replay of a CLOSED failed tool proposal. Never imports a provider or opens a socket. */
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isolateAnnotations } from './annotation_isolation.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const encoded = value => `${JSON.stringify(value, null, 2)}\n`;
async function bound(file, expected) {
  const bytes = await readFile(file);
  if (expected && sha(bytes) !== expected) throw new Error('recovery_bound_bytes_changed');
  return { value: JSON.parse(bytes), descriptor: { file: resolve(file), sha256: sha(bytes) } };
}
async function writeNew(file, value) {
  const bytes = encoded(value); await writeFile(file, bytes, { flag: 'wx' });
  return { file: resolve(file), sha256: sha(bytes) };
}

export async function recoverFailedSubmission({ receiptPath, receiptSha256, manifestPath, outputDirectory }) {
  if (!/^[a-f0-9]{64}$/.test(receiptSha256 ?? '')) throw new Error('explicit_closed_receipt_sha_required');
  const original = await bound(receiptPath, receiptSha256), receipt = original.value;
  if (!receipt.closed || receipt.fixture !== false || receipt.initializeOnly !== false
      || receipt.submission_sha256 !== null || receipt.unknown_usage_requests !== 0
      || !['failed_model_turn', 'closed_without_submission'].includes(receipt.status)) throw new Error('closed_known_failed_real_turn_required');
  const manifest = await bound(manifestPath, receipt.manifest_sha256);
  if (manifest.value.contextMode !== 'full-candidate-context') throw new Error('full_context_replay_required');
  const raw = await bound(resolve(ROOT, manifest.value.casePath), receipt.case_sha256);
  const sdk = await bound(resolve(dirname(receiptPath), 'sdk-summary.json'));
  const calls = sdk.value.trace.filter(row => row.event_type === 'tool/call' && row.tool === 'submit_evidence');
  if (calls.length !== 1 || typeof calls[0].arguments !== 'string') throw new Error('single_literal_submit_proposal_required');
  const sessionFiles = [];
  async function findSession(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) await findSession(path);
      else if (entry.name === 'session.v4.jsonl' && dirname(path).endsWith(sdk.value.sessionId)) sessionFiles.push(path);
    }
  }
  await findSession(resolve(dirname(receiptPath), 'sessions'));
  if (sessionFiles.length !== 1) throw new Error('one_official_session_journal_required');
  const sessionBytes = await readFile(sessionFiles[0]);
  const sessionEvents = sessionBytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
  const officialCalls = sessionEvents.filter(row => row.type === 'tool/call' && row.data?.name === 'submit_evidence');
  if (officialCalls.length !== 1 || officialCalls[0].data.arguments !== calls[0].arguments
      || !sessionEvents.some(row => row.type === 'tool/result' && row.data?.message?.toolCallId === officialCalls[0].data.callId
        && row.data.message.isError === true)) throw new Error('official_failed_tool_journal_agreement_required');
  const args = JSON.parse(calls[0].arguments);
  const journal = await readFile(resolve(dirname(receiptPath), 'budget.jsonl'));
  if (sha(journal) !== receipt.budget_audit_sha256) throw new Error('recovery_usage_binding_changed');
  const rows = journal.toString('utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
  const starts = rows.filter(row => row.status === 'request_started');
  const settled = rows.filter(row => row.status === 'usage_settled');
  if (starts.length !== receipt.model_requests || settled.length !== starts.length
      || rows.some(row => row.status === 'usage_unknown')
      || settled.some(row => row.unknown_usage_requests !== 0 || !row.response_completed)
      || manifest.value.priorKnownTokens + settled.reduce((sum, row) => sum + row.provider_usage.total_tokens, 0) !== receipt.known_tokens) throw new Error('settled_original_usage_required');
  const output = resolve(outputDirectory), runtime = resolve(ROOT, '.runtime');
  if (!output.startsWith(`${runtime}/`) && !output.startsWith(`${runtime}\\`)) throw new Error('isolated_recovery_output_required');
  await mkdir(output, { recursive: true });
  const result = isolateAnnotations(args, raw.value);
  const rawArgs = await writeNew(resolve(output, 'raw-tool-arguments.json'), args);
  const audit = await writeNew(resolve(output, 'annotation-audit.json'), result);
  if (!result.accepted_args || !result.submission) throw new Error('local_recovery_has_no_accepted_proposal');
  const acceptedArgs = await writeNew(resolve(output, 'accepted-tool-arguments.json'), result.accepted_args);
  const submission = structuredClone(result.submission);
  submission.source_kind = 'offline_recovered_from_failed_real_tool';
  submission.quality.offline_recovered = true;
  submission.quality.end_to_end_real_success = false;
  submission.quality.annotation_quarantine_count = result.quarantine.length;
  const recovered = await writeNew(resolve(output, 'submission.json'), submission);
  const implementationFiles = [];
  const implementationRoot = resolve(output, 'implementation'); await mkdir(implementationRoot);
  for (const name of ['recover_failed_submission.mjs', 'annotation_isolation.mjs', 'evidence_tools.mjs']) {
    const sourceFile = resolve(ROOT, 'integrations/dsh', name), bytes = await readFile(sourceFile), file = resolve(implementationRoot, name);
    await writeFile(file, bytes, { flag: 'wx' });
    implementationFiles.push({ file, sha256: sha(bytes), original_source_file: sourceFile });
  }
  const report = {
    format: 'offline-recovered-real-tool-v1', closed: true,
    source_kind: 'offline_recovered_from_failed_real_tool',
    original_closed_receipt: original.descriptor, manifest: manifest.descriptor, raw_case: raw.descriptor,
    sdk_summary: sdk.descriptor, original_tool_arguments_sha256: sha(calls[0].arguments),
    official_session_journal: { file: sessionFiles[0], sha256: sha(sessionBytes) },
    tool_trace_association: { official_call_id: officialCalls[0].data.callId, official_event_sequence: officialCalls[0].seq,
      arguments_equal_in_both_logs: true, original_closed_receipt_bound_session_journal: false,
      cross_log_agreement_is_provider_signed_proof: false },
    raw_tool_arguments: rawArgs, accepted_tool_arguments: acceptedArgs, quarantine: audit,
    recovered_submission: recovered, recovery_implementation: implementationFiles[0], implementation_files: implementationFiles,
    transformation: { program_only: true, model_calls: 0,
      selection_refs_changed: JSON.stringify(result.original_selected_ids) !== JSON.stringify(result.accepted_ids),
      original_selected_refs: result.original_selected_ids, accepted_refs: result.accepted_ids,
      annotations_changed: result.transformations, quarantine_count: result.quarantine.length,
      supporting_conflicting_conclusion_basis_unchanged: true },
    original_real_status: receipt.status, end_to_end_real_success: false,
    known_episode_tokens: receipt.known_tokens, new_model_calls: 0, new_tokens: 0,
    semantic_verified: false, published: false, references_read_during_recovery: false,
  };
  const desc = await writeNew(resolve(output, 'recovery-receipt.json'), report);
  return { ...desc, accepted_count: submission.evidence.length, quarantined_count: result.quarantine.length, new_model_calls: 0, new_tokens: 0 };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [receiptPath, receiptSha256, manifestPath, outputDirectory] = process.argv.slice(2);
  recoverFailedSubmission({ receiptPath: resolve(receiptPath), receiptSha256, manifestPath: resolve(manifestPath), outputDirectory: resolve(outputDirectory) })
    .then(result => process.stdout.write(encoded(result)))
    .catch(error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
}
