/** Fixture-only JSON-lines bridge to the existing DSH business adapter. No model client. */
import { readFile } from 'node:fs/promises';
import { resolve, isAbsolute } from 'node:path';
import { createHash } from 'node:crypto';
import { createInterface } from 'node:readline';
import { apply, bindStagedAdmission, getStagedPresence } from '../integrations/dsh/staged_evidence.mjs';
import { createOfficialTokenizerCounter } from '../integrations/dsh/budget_counter.mjs';

const root = resolve(import.meta.dirname, '..');
const sha = value => createHash('sha256').update(value).digest('hex');
const fail = code => { throw new Error(code); };
const [manifestPath, artifactPath] = process.argv.slice(2);
if (process.argv.length !== 4 || !isAbsolute(manifestPath ?? '') || !isAbsolute(artifactPath ?? '')
  || !resolve(artifactPath).startsWith(resolve(root, '.runtime') + '\\')) fail('fixture_runtime_paths_required');
const manifest = JSON.parse(await readFile(manifestPath));
if (manifest.contextMode !== 'staged-short-window' || manifest.selectionPolicy !== 'source-local-v2'
  || manifest.previewCodepoints !== 160 || manifest.maxCalls !== 2 || manifest.maxOutputTokens !== 6144) fail('fixed_fixture_contract_required');
const rawBytes = await readFile(manifest.casePath);
if (sha(rawBytes) !== manifest.caseSha256) fail('raw_case_sha_changed');
const raw = JSON.parse(rawBytes);
if (raw.catalog.length !== 40 || raw.question.question_id !== 'q_002') fail('fixed_raw_scope_required');
const plan = JSON.parse(await readFile(manifest.fixturePlan));
if (plan.length !== 2 || plan[0].name !== 'read_selected_sources' || plan[1].name !== 'submit_evidence'
  || JSON.stringify(plan[0].arguments.refs) !== JSON.stringify(raw.catalog.slice(0, 6).map(p => p.paragraph_ref))) fail('fixed_synthetic_plan_required');
const sections = [], tools = new Map();
const episodeId = 'langgraph-fixture-bridge';
const dispose = bindStagedAdmission(episodeId, async () => ({ admitted: true, whole_limit: 20000,
  preflight_total_tokens: 0, native_next_request_still_required: true, fixture: true }));
const ctx = { trustJudgeStagedAdmission: { episodeId }, systemPrompt: { section: value => sections.push(value) },
  tools: { register: tool => { tools.set(tool.name, tool); return () => tools.delete(tool.name); } }, effect: () => {} };
await apply(ctx, { casePath: manifest.casePath, caseSha256: manifest.caseSha256, submissionPath: artifactPath,
  episodeId, selectionPolicy: 'source-local-v2', previewCodepoints: 160 });
let ended = false, nextStep = 0;
const signal = new AbortController().signal;
const system = () => ['You evaluate disputed claims using original sources and explicit uncertainty.',
  ...sections.toSorted((a, b) => a.order - b.order).map(s => typeof s.text === 'function' ? s.text() : s.text)].join('\n\n');
const visible = () => ({ system: system(), tools: [...tools.values()].map(t => ({ type: 'function', function: {
  name: t.name, description: t.description, parameters: t.parameters } })), presence: getStagedPresence(episodeId), ended });
const counter = createOfficialTokenizerCounter({ pythonExecutable: resolve(root, '.runtime/v20-embedding-env/Scripts/python.exe'),
  tokenizerPath: resolve(root, '.runtime/v32-deepseek-tokenizer-r2/tokenizer.json'),
  tokenizerSha256: '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf' });
for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  try {
    const command = JSON.parse(line);
    let result;
    if (command.op === 'state') result = visible();
    else if (command.op === 'execute') {
      const step = plan[nextStep];
      if (!step || command.name !== step.name || JSON.stringify(command.arguments) !== JSON.stringify(step.arguments)) fail('synthetic_tool_plan_changed');
      const tool = tools.get(command.name); if (!tool) fail('unexpected_tool');
      const value = await tool.execute(command.arguments, { signal, concludeTurn() { ended = true; } });
      nextStep++; result = { value, ...visible() };
    } else if (command.op === 'count') {
      if (typeof command.serialized_body !== 'string') fail('native_serialized_body_required');
      result = { input_tokens: await counter({ serialized_body: command.serialized_body }), wire_sha256: sha(command.serialized_body) };
    } else fail('fixture_operation_required');
    process.stdout.write(JSON.stringify({ ok: true, result }) + '\n');
  } catch (error) { process.stdout.write(JSON.stringify({ ok: false, code: error.message }) + '\n'); }
}
dispose();
