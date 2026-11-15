import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { recoverFailedSubmission } from './recover_failed_submission.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function syntheticFixture() {
  const dir = resolve('.runtime/dsh-recovery-tests', randomUUID()); await mkdir(dir, { recursive: true });
  const save = async (name, value) => { const bytes = JSON.stringify(value); const file = resolve(dir, name); await writeFile(file, bytes); return { file, sha256: sha(bytes) }; };
  const raw = await save('raw.json', { question: { question_id: 'synthetic', question: 'ZX482项目是否已经批准？' }, catalog: [
    { paragraph_ref: 'P1', doc_id: 'synthetic_doc', para_id: 'synthetic_p1', title: '合成登记', text: '登记表：申请已受理，尚未批准。', paragraph_complete: true, parent_complete: true },
  ] });
  const args = { conclusion: 'ZX482项目尚未确认批准。', verdict_label: 'undetermined',
    selected: [{ p: 'P1', roles: ['L'], fact_des: '申请已受理但尚未批准。',
      argument: { 时间: '9月31日', 主体: '', 地点: '', 内容: '', 影响: '', 数值口径: '' }, entities: [] }],
    supporting: ['P1'], conflicting: [], basis: '仅确认程序状态。' };
  const literalArgs = JSON.stringify(args), sessionId = 'synthetic-session';
  const sessionDir = resolve(dir, 'sessions', sessionId); await mkdir(sessionDir, { recursive: true });
  const session = [{ type: 'tool/call', seq: 1, data: { callId: 'synthetic-call', name: 'submit_evidence', arguments: literalArgs } },
    { type: 'tool/result', seq: 2, data: { message: { toolCallId: 'synthetic-call', isError: true } } }];
  await writeFile(resolve(sessionDir, 'session.v4.jsonl'), session.map(JSON.stringify).join('\n'));
  await save('sdk-summary.json', { sessionId, trace: [{ event_type: 'tool/call', tool: 'submit_evidence', arguments: literalArgs }] });
  const manifest = await save('manifest.json', { contextMode: 'full-candidate-context', casePath: raw.file, priorKnownTokens: 0 });
  const journal = [{ status: 'request_started' }, { status: 'usage_settled', response_completed: true, unknown_usage_requests: 0, provider_usage: { total_tokens: 123 } }].map(JSON.stringify).join('\n');
  await writeFile(resolve(dir, 'budget.jsonl'), journal);
  const receiptValue = { closed: true, fixture: false, initializeOnly: false, status: 'failed_model_turn',
    submission_sha256: null, unknown_usage_requests: 0, model_requests: 1, known_tokens: 123,
    manifest_sha256: manifest.sha256, case_sha256: raw.sha256, budget_audit_sha256: sha(journal), synthetic_test_only: true };
  const receipt = await save('closed-receipt.json', receiptValue);
  return { dir, save, receiptValue, receiptPath: receipt.file, receiptSha256: receipt.sha256, manifestPath: manifest.file, outputDirectory: resolve(dir, 'recovered') };
}

test('local replay preserves failed real status, original proposal and immutable implementation snapshots', async () => {
  const input = await syntheticFixture(), result = await recoverFailedSubmission(input);
  assert.equal(result.new_model_calls, 0); assert.equal(result.new_tokens, 0);
  const receipt = JSON.parse(await readFile(result.file, 'utf8'));
  assert.equal(receipt.end_to_end_real_success, false); assert.equal(receipt.known_episode_tokens, 123);
  assert.equal(receipt.tool_trace_association.arguments_equal_in_both_logs, true);
  for (const descriptor of receipt.implementation_files) {
    assert.ok(descriptor.file.startsWith(resolve(input.outputDirectory, 'implementation')));
    assert.equal(sha(await readFile(descriptor.file)), descriptor.sha256);
  }
  const original = JSON.parse(await readFile(receipt.original_closed_receipt.file, 'utf8'));
  assert.equal(original.status, 'failed_model_turn'); assert.equal(original.submission_sha256, null);
  const accepted = JSON.parse(await readFile(receipt.recovered_submission.file, 'utf8'));
  assert.equal(accepted.evidence[0].argument.时间, ''); assert.equal(accepted.quality.end_to_end_real_success, false);
  assert.equal(accepted.quality.publishable, false);
});

test('replay rejects missing or changed bindings, mock runs and unknown usage', async () => {
  const input = await syntheticFixture();
  await assert.rejects(recoverFailedSubmission({ ...input, receiptSha256: undefined }), /explicit_closed_receipt_sha/);
  await assert.rejects(recoverFailedSubmission({ ...input, receiptSha256: '0'.repeat(64) }), /bound_bytes_changed/);
  for (const mutation of [{ fixture: true }, { unknown_usage_requests: 1 }]) {
    const next = await input.save(`bad-${randomUUID()}.json`, { ...input.receiptValue, ...mutation });
    await assert.rejects(recoverFailedSubmission({ ...input, receiptPath: next.file, receiptSha256: next.sha256 }), /closed_known_failed_real_turn_required/);
  }
});

test('a mismatched official tool journal cannot be silently accepted from an SDK summary', async () => {
  const input = await syntheticFixture();
  await input.save('sdk-summary.json', { sessionId: 'synthetic-session', trace: [{ event_type: 'tool/call', tool: 'submit_evidence', arguments: '{}' }] });
  await assert.rejects(recoverFailedSubmission(input), /official_failed_tool_journal_agreement_required/);
});
