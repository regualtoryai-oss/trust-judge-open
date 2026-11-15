import test from 'node:test';
import assert from 'node:assert/strict';
import { safeToolErrorCodes } from './run_comparison.mjs';
test('safe cause codes preserve the original validation failure without publishing arbitrary response text', () => {
  const error = text => ({ event_type: 'tool/result', is_error: true, content: [{ type: 'text', text }] });
  assert.deepEqual(safeToolErrorCodes([error('Error: invalid_submission_shape'), error('Error: invalid_submission_shape'),
    error('Error: unknown_or_duplicate_source_ref'), error('Error: external response with credentials'),
    { event_type: 'turn/end', reason: { error: { message: 'episode_call_limit' } } }]),
  ['invalid_submission_shape', 'unknown_or_duplicate_source_ref']);
});
