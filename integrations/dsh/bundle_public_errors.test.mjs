import test from 'node:test';
import assert from 'node:assert/strict';
import { publicBundleToolFailure } from './bundle_public_errors.mjs';

test('only exact known business failures become public diagnostics', () => {
  const known = 'bundle_bridge_duplicate_selected_start_anchor';
  assert.deepEqual(publicBundleToolFailure([{ type: 'text', text: `Error: ${known}` }]), { failure_code: known });
  for (const text of [`Error: ${known}\nprivate`, `Error: ${known} private`, 'Error: private_provider_payload', 'Bearer private', 'Error: source_bundle_tools_invalid_source_range\n']) {
    assert.deepEqual(publicBundleToolFailure([{ type: 'text', text }]), {});
  }
  assert.deepEqual(publicBundleToolFailure([{ type: 'text', text: `Error: ${known}` }, { type: 'text', text: 'private' }]), {});
});
