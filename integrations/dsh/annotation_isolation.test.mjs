import test from 'node:test';
import assert from 'node:assert/strict';
import { isolateAnnotations } from './annotation_isolation.mjs';

function raw() {
  return { question: { question_id: 'q_synthetic', question: '该项目完成了吗？' }, catalog: [
    { paragraph_ref: 'P1', doc_id: 'd1', para_id: 'p1', title: '项目记录', text: '10月2日，项目已有40人到场，仍未完成。😀', paragraph_complete: true, parent_complete: true },
    { paragraph_ref: 'P2', doc_id: 'd2', para_id: 'p2', title: '说明', text: '有关部门表示仍在核查。', paragraph_complete: true, parent_complete: true },
  ] };
}
function row(p = 'P1') {
  return { p, roles: ['D', 'L'], fact_des: '项目仍未完成。',
    argument: { 时间: '', 主体: '项目', 地点: '', 内容: '仍未完成', 影响: '', 数值口径: '' }, entities: ['项目'] };
}
function args() {
  return { conclusion: '项目仍未完成。', verdict_label: 'partial_support', selected: [row()], supporting: ['P1'], conflicting: [], basis: '原文限定。' };
}

test('valid annotations preserve args, identities, program quotes and codepoint coordinates', () => {
  const source = raw(), input = args(), original = structuredClone(input);
  const result = isolateAnnotations(input, source);
  assert.deepEqual(input, original);
  assert.deepEqual(result.original_args, original);
  assert.deepEqual(result.normalized_args, original);
  assert.deepEqual(result.format_transformations, []);
  assert.deepEqual(result.accepted_args, original);
  assert.equal(result.validation.status, 'validated');
  assert.equal(result.relations_resolvable, true);
  assert.deepEqual(result.quarantine, []);
  assert.equal(result.submission.evidence[0].quote, source.catalog[0].text);
  assert.equal(result.submission.evidence[0].parent_end_cp, Array.from(source.catalog[0].text).length);
  assert.equal(result.submission.evidence[0].doc_id, 'd1');
  assert.equal(result.submission.quality.publishable, false);
});

test('non-literal time is explicitly unknown with original retained and no other repair', () => {
  const input = args(); input.selected[0].argument.时间 = '2026年10月2日';
  const result = isolateAnnotations(input, raw());
  assert.equal(result.validation.status, 'validated');
  assert.equal(result.original_args.selected[0].argument.时间, '2026年10月2日');
  assert.equal(result.accepted_args.selected[0].argument.时间, '');
  assert.deepEqual(result.transformations, [{ selected_index: 0, p: 'P1', path: 'selected[0].argument.时间',
    original_value: '2026年10月2日', accepted_value: '', reason: 'time_must_be_literal_or_empty' }]);
  assert.deepEqual(result.quarantine, []);
  assert.equal(input.selected[0].argument.时间, '2026年10月2日');
});

test('literal time and supported numeric literals remain unchanged', () => {
  const input = args(); input.selected[0].argument.时间 = '10月2日'; input.selected[0].fact_des = '已有40人到场。';
  const result = isolateAnnotations(input, raw());
  assert.equal(result.validation.status, 'validated');
  assert.deepEqual(result.accepted_args, input);
  assert.deepEqual(result.transformations, []);
});

for (const [label, change, reason] of [
  ['unsupported annotation number', row => { row.entities = ['型号171']; }, 'annotation_number_not_in_source'],
  ['empty fact description', row => { row.fact_des = '  '; }, 'empty_or_invalid_fact_des'],
  ['invalid role', row => { row.roles = ['invented']; }, 'invalid_proposed_roles'],
  ['duplicate role', row => { row.roles = ['D', 'D']; }, 'invalid_proposed_roles'],
]) {
  test(`${label} isolates only an unreferenced row and preserves relations/conclusion/basis`, () => {
    const input = args(); input.selected.push(row('P2')); change(input.selected[1]);
    const original = structuredClone(input), result = isolateAnnotations(input, raw());
    assert.equal(result.validation.status, 'validated');
    assert.deepEqual(result.original_args, original);
    assert.deepEqual(input, original);
    assert.deepEqual(result.original_selected_ids, ['P1', 'P2']);
    assert.deepEqual(result.accepted_ids, ['P1']);
    assert.equal(result.quarantine.length, 1);
    assert.ok(result.quarantine[0].reasons.includes(reason));
    assert.deepEqual(result.quarantine[0].original_row, original.selected[1]);
    assert.deepEqual(result.accepted_args, { ...original, selected: [original.selected[0]] });
  });
}

for (const relation of ['supporting', 'conflicting']) {
  test(`quarantined ${relation} row blocks acceptance without deleting any relation`, () => {
    const input = args(); input.selected.push(row('P2')); input.selected[1].roles = [];
    if (relation === 'supporting') input.supporting.push('P2');
    else input.conflicting.push({ p: 'P2', resolution_type: 'Other' });
    const result = isolateAnnotations(input, raw());
    assert.equal(result.accepted_args, null);
    assert.equal(result.submission, null);
    assert.equal(result.relations_resolvable, false);
    assert.equal(result.validation.status, 'blocked_by_quarantined_relation');
    assert.deepEqual(result.original_args, input);
    assert.deepEqual(result.validation.refs, ['P2']);
  });
}

test('unknown and duplicate selected references hard fail before time transformation', () => {
  for (const unknown of [true, false]) {
    const input = args(); input.selected[0].argument.时间 = '2026年';
    if (unknown) input.selected[0].p = 'P99'; else input.selected.push(row());
    assert.throws(() => isolateAnnotations(input, raw()), error => {
      assert.equal(error.code, 'unknown_or_duplicate_source_ref');
      assert.deepEqual(error.original_args, input);
      return true;
    });
    assert.equal(input.selected[0].argument.时间, '2026年');
  }
});

test('strict backend errors remain explicit failures, not quarantine-based silent repairs', () => {
  for (const [change, code] of [
    [input => { input.selected[0].quote = 'invented'; }, 'invalid_tool_arguments'],
    [input => { input.selected[0].argument.地点 = { unknown: true }; }, 'invalid_string'],
    [input => { input.conclusion = '99人已完成项目。'; }, 'conclusion_number_not_in_cited_sources'],
    [input => { input.conflicting = [{ p: 'P1', resolution_type: 'newest' }]; }, 'invalid_resolution_type'],
    [input => { input.supporting = ['P2']; }, 'unknown_or_duplicate_source_ref'],
  ]) {
    const input = args(); change(input);
    const result = isolateAnnotations(input, raw());
    assert.equal(result.validation.status, 'validation_failed');
    assert.equal(result.validation.code, code);
    assert.equal(result.accepted_args, null);
    assert.equal(result.submission, null);
    assert.deepEqual(result.original_args, input);
  }
});

test('no support is fabricated for valid undetermined output', () => {
  const input = { conclusion: '公开信息不足。', verdict_label: 'undetermined', selected: [], supporting: [], conflicting: [], basis: '' };
  const result = isolateAnnotations(input, raw());
  assert.equal(result.validation.status, 'validated');
  assert.deepEqual(result.accepted_ids, []);
  assert.deepEqual(result.submission.conclusion.supporting_facts, []);
});

test('raw source identity and text-binding validation still hard fail', () => {
  const source = raw(); source.catalog[1].doc_id = 'd1'; source.catalog[1].para_id = 'p1';
  assert.throws(() => isolateAnnotations(args(), source), /unique_original_source_required/);
});

const tuple = value => [value.p, value.roles, value.fact_des, Object.values(value.argument), value.entities];

test('exact tuple output passes strict validation after independent lossless format adaptation', () => {
  const input = args(); input.selected = input.selected.map(tuple); input.conflicting = [['P1', 'Other']];
  const original = structuredClone(input), result = isolateAnnotations(input, raw());
  assert.deepEqual(input, original); assert.deepEqual(result.original_args, original);
  assert.equal(result.normalized_args.selected[0].p, 'P1');
  assert.deepEqual(result.normalized_args.conflicting, [{ p: 'P1', resolution_type: 'Other' }]);
  assert.deepEqual(result.accepted_args, result.normalized_args); assert.equal(result.validation.status, 'validated');
  assert.equal(result.format_transformations.length, 3); assert.deepEqual(result.transformations, []);
  assert.equal(result.submission.evidence[0].quote, raw().catalog[0].text); assert.equal(result.submission.evidence[0].para_id, 'p1');
  assert.equal(result.submission.quality.publishable, false);
});

test('tuple time keeps original content in both original and normalized audits before optional-time clearing', () => {
  const input = args(); input.selected[0].argument.时间 = '2026年10月2日'; input.selected = input.selected.map(tuple);
  const result = isolateAnnotations(input, raw());
  assert.equal(result.validation.status, 'validated');
  assert.equal(result.original_args.selected[0][3][0], '2026年10月2日');
  assert.equal(result.normalized_args.selected[0].argument.时间, '2026年10月2日');
  assert.equal(result.accepted_args.selected[0].argument.时间, '');
  assert.equal(result.transformations[0].original_value, '2026年10月2日');
  assert.equal(result.format_transformations[0].original_value[0], '2026年10月2日');
});

test('conflicting tuple pointing at a quarantined tuple row still blocks with unchanged relation meaning', () => {
  const input = args(); input.selected.push(row('P2')); input.selected[1].fact_des = '';
  input.selected = input.selected.map(tuple); input.conflicting = [['P2', 'Other']];
  const result = isolateAnnotations(input, raw());
  assert.equal(result.validation.code, 'referenced_annotation_quarantined'); assert.equal(result.accepted_args, null);
  assert.deepEqual(result.original_args.conflicting, [['P2', 'Other']]);
  assert.deepEqual(result.normalized_args.conflicting, [{ p: 'P2', resolution_type: 'Other' }]);
  assert.deepEqual(result.quarantine[0].original_row, input.selected[1]); assert.deepEqual(result.validation.refs, ['P2']);
});

test('malformed positional rows return explicit audit failures without guessing or dropping a row', () => {
  for (const change of [
    input => input.selected[0].pop(), input => input.selected[0][3].pop(), input => input.conflicting = [['P1', 'Other', 'extra']],
  ]) {
    const input = args(); input.selected = input.selected.map(tuple); change(input);
    const result = isolateAnnotations(input, raw());
    assert.deepEqual(result.original_args, input); assert.equal(result.normalized_args, null);
    assert.equal(result.validation.code, 'invalid_submission_shape'); assert.equal(result.validation.status, 'validation_failed');
    assert.equal(result.accepted_args, null); assert.equal(result.submission, null); assert.deepEqual(result.quarantine, []);
  }
});

test('unknown and duplicate tuple identities hard fail before optional-time or quarantine processing', () => {
  for (const duplicate of [false, true]) {
    const input = args(); input.selected[0].argument.时间 = '2026年'; input.selected = input.selected.map(tuple);
    if (duplicate) input.selected.push(structuredClone(input.selected[0])); else input.selected[0][0] = 'P99';
    assert.throws(() => isolateAnnotations(input, raw()), error => {
      assert.equal(error.code, 'unknown_or_duplicate_source_ref'); assert.deepEqual(error.original_args, input); return true;
    });
  }
});
