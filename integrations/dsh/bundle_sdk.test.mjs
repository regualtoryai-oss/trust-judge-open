import test from 'node:test';
import assert from 'node:assert/strict';
import { access, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { runBundleFixture } from './bundle_fixture.mjs';
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const runtimeAvailable = await access(resolve(ROOT, '.runtime/dsh-official-r1/node_modules/@deepseek-ai/dsh-sdk-client/lib/index.js')).then(() => true, () => false);

test('official DSH loop performs search, two complete bundle reads and terminal proposal through actual native messages',
  { skip: !runtimeAvailable, timeout: 120000 }, async () => {
    const parent = resolve(ROOT, '.runtime/dsh-bundle-sdk-tests');
    await mkdir(parent, { recursive: true });
    const receipt = await runBundleFixture(resolve(parent, randomUUID()));
    assert.equal(receipt.closed, true);
    assert.equal(receipt.real_api_requests, 0);
    assert.equal(receipt.paid_tokens, 0);
    assert.equal(receipt.simulated_provider_requests, 4);
    assert.deepEqual(receipt.actual_native_read_counts, [0, 0, 1, 2]);
    assert.equal(receipt.quality_metrics, null);
    assert.equal(receipt.publishable, false);
    assert.equal(receipt.repeated_full_history_input_risk, true);
  });

