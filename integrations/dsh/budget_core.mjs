/** Whole-question accounting around an existing DSH model stream; owns no agent loop. */
import { createHash } from 'node:crypto';

export class BudgetError extends Error {
  constructor(code) { super(code); this.name = 'BudgetError'; this.code = code; }
}
const fail = code => { throw new BudgetError(code); };
const integer = value => Number.isSafeInteger(value) && value >= 0;
const hash = text => createHash('sha256').update(text, 'utf8').digest('hex');
const sum = values => {
  if (!values.every(integer)) fail('invalid_token_count');
  const total = values.reduce((a, b) => a + b, 0);
  if (!integer(total)) fail('invalid_token_count');
  return total;
};

/** DSH usage buckets are disjoint; reasoning is already inside outputTokens. */
export function nativeUsage(usage) {
  if (!usage || typeof usage !== 'object') fail('unknown_provider_usage');
  const input = sum([usage.inputTokens, usage.cacheReadTokens ?? 0, usage.cacheWriteTokens ?? 0]);
  const output = sum([usage.outputTokens]);
  if (usage.reasoningTokens !== undefined && (!integer(usage.reasoningTokens) || usage.reasoningTokens > output)) {
    fail('invalid_provider_usage');
  }
  const total = sum([input, output]);
  if (usage.totalTokens !== undefined && usage.totalTokens !== total) fail('invalid_provider_usage');
  return Object.freeze({ input_tokens: input, output_tokens: output, total_tokens: total });
}

/**
 * countInputTokens receives the actual serialized native request, including tools.
 * It must use the pinned official tokenizer, never a character heuristic.
 * Its result is a preflight estimate, not a provider billing guarantee.
 * record must durably persist safe metadata before returning; it receives no body or key.
 */
export class EpisodeBudgetGuard {
  constructor({ episodeId, priorKnownTokens, wholeLimit = 20000, safetyTokens = 512,
    maxCalls, maxOutputTokens, countInputTokens, record, sourceCheck }) {
    if (typeof episodeId !== 'string' || !episodeId || !integer(priorKnownTokens)
      || !integer(wholeLimit) || wholeLimit === 0 || !integer(safetyTokens)
      || !integer(maxCalls) || maxCalls === 0 || !integer(maxOutputTokens) || maxOutputTokens === 0
      || typeof countInputTokens !== 'function' || typeof record !== 'function' || typeof sourceCheck !== 'function') {
      fail('invalid_budget_configuration');
    }
    this.config = Object.freeze({ episodeId, wholeLimit, safetyTokens, maxCalls, maxOutputTokens });
    this.known = priorKnownTokens;
    this.calls = 0;
    this.unknown = 0;
    this.inFlight = false;
    this.halted = false;
    this.countInputTokens = countInputTokens;
    this.record = record;
    this.sourceCheck = sourceCheck;
    this.events = [];
  }

  snapshot() {
    return Object.freeze({ episode_id: this.config.episodeId, known_episode_tokens: this.known,
      started_calls: this.calls, unknown_usage_requests: this.unknown, in_flight: this.inFlight,
      halted: this.halted, whole_limit: this.config.wholeLimit });
  }

  async save(event) {
    const safe = Object.freeze({ episode_id: this.config.episodeId, sequence: this.events.length + 1, ...event });
    await this.record(safe);
    this.events.push(safe);
  }

  /**
   * prepareWire must use the pinned DSH serializer and return the exact request JSON string.
   * invoke must retain its existing DSH transport. Recomputed bytes are checked immediately
   * before delegation; the host must also enforce this hash at its native HTTP prehook.
   */
  async * protectStream({ prepareWire, invoke }) {
    if (typeof prepareWire !== 'function' || typeof invoke !== 'function') fail('invalid_stream_adapter');
    yield* this.protectNativeStream({ invoke: async function* (beforeWire) {
      const serialized = await prepareWire();
      if (await prepareWire() !== serialized) fail('native_request_changed');
      const receipt = await beforeWire(serialized);
      yield* invoke(receipt);
    } });
  }

  /** Native HTTP prehook supplies final bytes; original DSH stream supplies usage. */
  async * protectNativeStream({ invoke }) {
    if (this.halted) fail('episode_halted');
    if (this.inFlight) fail('concurrent_request_forbidden');
    if (this.calls >= this.config.maxCalls) fail('episode_call_limit');
    if (typeof invoke !== 'function') fail('invalid_stream_adapter');
    this.inFlight = true;
    let started = false;
    let usage;
    let partialUsage;
    let usageUncertain = false;
    let complete = false;
    let failed = false;
    let receipt;
    const beforeWire = async serialized => {
      if (started) fail('multiple_native_requests_forbidden');
      if (await this.sourceCheck() !== true) fail('source_binding_changed');
      if (typeof serialized !== 'string') fail('unknown_native_request');
      let body;
      try { body = JSON.parse(serialized); } catch { fail('invalid_native_request'); }
      if (!body || !Array.isArray(body.messages) || !body.messages.length
        || (body.tools !== undefined && !Array.isArray(body.tools))) fail('invalid_native_request');
      if (!integer(body.max_tokens) || body.max_tokens === 0 || body.max_tokens > this.config.maxOutputTokens) {
        fail('invalid_output_reservation');
      }
      if (body.thinking?.type !== 'disabled' || body.reasoning_effort !== undefined
        || body.output_config?.effort !== undefined) fail('thinking_policy_violation');
      const input = await this.countInputTokens({ serialized_body: serialized, body });
      if (!integer(input)) fail('unknown_input_cost');
      const reserved = sum([this.known, input, body.max_tokens, this.config.safetyTokens]);
      receipt = { request_sequence: this.calls + 1, wire_sha256: hash(serialized),
        estimated_input_tokens: input, known_episode_tokens_before: this.known,
        reserved_output_tokens: body.max_tokens, safety_tokens: this.config.safetyTokens,
        preflight_total_tokens: reserved, cost_basis: 'official_tokenizer_native_request_estimate' };
      if (reserved > this.config.wholeLimit) {
        await this.save({ ...receipt, status: 'blocked', code: 'whole_episode_budget_exceeded' });
        fail('whole_episode_budget_exceeded');
      }
      await this.save({ ...receipt, status: 'admitted' });
      if (await this.sourceCheck() !== true) fail('source_binding_changed');
      await this.save({ ...receipt, status: 'request_started' });
      started = true;
      this.calls++;
      return Object.freeze({ ...receipt });
    };
    try {
      const observeProviderUsage = sample => { partialUsage = nativeUsage(sample); };
      for await (const chunk of invoke(beforeWire, observeProviderUsage)) {
        if (chunk?.type === 'usage') {
          if (!started) fail('unbound_provider_usage');
          try { usage = nativeUsage(chunk.usage); }
          catch (error) { usageUncertain = true; throw error; }
        }
        if (chunk?.type === 'finish') {
          complete = true;
          failed = !['stop', 'tool-calls'].includes(chunk.reason?.kind);
        }
        yield chunk;
      }
      if (!started && complete && !failed) fail('missing_native_request');
    } catch (error) {
      failed = true;
      throw error instanceof BudgetError ? error : new BudgetError('model_stream_failed');
    } finally {
      if (started) {
        const retained = usage ?? partialUsage;
        if (retained) this.known = sum([this.known, retained.total_tokens]);
        if (!usage || usageUncertain) this.unknown++;
        if (!usage || usageUncertain || !complete || failed || this.known > this.config.wholeLimit
          || usage.output_tokens > receipt.reserved_output_tokens) this.halted = true;
        try {
          await this.save({ ...receipt, status: usage && !usageUncertain ? 'usage_settled' : 'usage_unknown',
            provider_usage: usage ?? null, response_completed: complete && !failed,
            partial_provider_usage: usage ? null : partialUsage ?? null,
            known_episode_tokens_after: this.known, unknown_usage_requests: this.unknown,
            halted: this.halted });
        } catch { this.halted = true; }
      }
      this.inFlight = false;
    }
  }
}
