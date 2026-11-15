/** Offline official-tokenizer count of the complete native JSON request, never an API call. */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { isAbsolute } from 'node:path';
import { spawn } from 'node:child_process';
import { BudgetError } from './budget_core.mjs';

const digest = value => createHash('sha256').update(value).digest('hex');
const PROGRAM = [
  'import hashlib,json,sys',
  'from tokenizers import Tokenizer',
  'raw=sys.stdin.buffer.read()',
  'path,expected=sys.argv[1:3]',
  'model=open(path,"rb").read()',
  'assert hashlib.sha256(model).hexdigest()==expected',
  'tok=Tokenizer.from_str(model.decode("utf-8"))',
  'text=raw.decode("utf-8")',
  'print(json.dumps({"wire_sha256":hashlib.sha256(raw).hexdigest(),"input_tokens":len(tok.encode(text,add_special_tokens=False).ids)}))',
].join('\n');

/** Exact file hash is mandatory; complete JSON token count remains an estimate of server input. */
export function createOfficialTokenizerCounter({ pythonExecutable, tokenizerPath, tokenizerSha256,
  timeoutMs = 10000, maxWireBytes = 2097152 }) {
  if (!isAbsolute(pythonExecutable ?? '') || !isAbsolute(tokenizerPath ?? '')
    || !/^[a-f0-9]{64}$/.test(tokenizerSha256 ?? '') || !Number.isSafeInteger(timeoutMs) || timeoutMs <= 0
    || !Number.isSafeInteger(maxWireBytes) || maxWireBytes <= 0) throw new BudgetError('invalid_counter_configuration');
  return async ({ serialized_body: text }) => {
    if (typeof text !== 'string' || Buffer.byteLength(text, 'utf8') > maxWireBytes) throw new BudgetError('unknown_input_cost');
    if (digest(await readFile(tokenizerPath)) !== tokenizerSha256) throw new BudgetError('tokenizer_binding_changed');
    // Deliberately exclude API keys, proxy values, Python injection variables and credentials.
    const env = Object.fromEntries(['SystemRoot', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP']
      .filter(key => process.env[key] !== undefined).map(key => [key, process.env[key]]));
    const output = await new Promise((resolve, reject) => {
      const child = spawn(pythonExecutable, ['-I', '-c', PROGRAM, tokenizerPath, tokenizerSha256], {
        env, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
      });
      let stdout = ''; let settled = false;
      const done = (error, result) => {
        if (settled) return;
        settled = true; clearTimeout(timer);
        if (error) { child.kill(); reject(new BudgetError('unknown_input_cost')); }
        else resolve(result);
      };
      const timer = setTimeout(() => done(true), timeoutMs);
      child.on('error', () => done(true));
      child.stdout.on('data', chunk => {
        stdout += chunk.toString('utf8');
        if (stdout.length > 4096) done(true);
      });
      // Drain without printing Python paths, payload fragments or arbitrary exception text.
      child.stderr.on('data', () => {});
      child.stdin.on('error', () => done(true));
      child.on('close', code => done(code !== 0, stdout));
      child.stdin.end(text, 'utf8');
    });
    let parsed;
    try { parsed = JSON.parse(output); } catch { throw new BudgetError('unknown_input_cost'); }
    if (parsed.wire_sha256 !== digest(Buffer.from(text, 'utf8'))
      || !Number.isSafeInteger(parsed.input_tokens) || parsed.input_tokens < 0) throw new BudgetError('unknown_input_cost');
    return parsed.input_tokens;
  };
}
