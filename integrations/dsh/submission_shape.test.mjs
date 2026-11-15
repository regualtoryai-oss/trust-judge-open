import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeSubmissionShape } from './submission_shape.mjs';

const argumentsTuple = () => ['昨日', '项目', '', '尚未完成', '', ''];
const selectedTuple = () => ['P1', ['D', 'L'], '项目尚未完成。', argumentsTuple(), ['项目']];
const input = () => ({ conclusion: '尚未完成。', verdict_label: 'partial_support', selected: [selectedTuple()], supporting: ['P1'],
  conflicting: [['P1', 'Other']], basis: '保留来源限定。' });

test('exact positional rows normalize losslessly with separate complete format audit', () => {
  const value = input(), original = structuredClone(value), result = normalizeSubmissionShape(value);
  assert.deepEqual(value, original);
  assert.deepEqual(result.normalized_args.selected[0], { p: 'P1', roles: ['D', 'L'], fact_des: '项目尚未完成。',
    argument: { 时间: '昨日', 主体: '项目', 地点: '', 内容: '尚未完成', 影响: '', 数值口径: '' }, entities: ['项目'] });
  assert.deepEqual(result.normalized_args.conflicting, [{ p: 'P1', resolution_type: 'Other' }]);
  assert.deepEqual(result.normalized_args.supporting, original.supporting);
  for (const key of ['conclusion', 'verdict_label', 'basis']) assert.equal(result.normalized_args[key], original[key]);
  assert.deepEqual(result.format_transformations.map(row => row.reason), ['exact_argument_tuple_to_object', 'exact_selected_tuple_to_object', 'exact_conflicting_tuple_to_object']);
  assert.deepEqual(result.format_transformations[1].original_value, original.selected[0]);
  result.normalized_args.selected[0].roles.push('C'); assert.deepEqual(value.selected[0][1], ['D', 'L']);
});

test('normal object rows and unknown keys remain unchanged for the strict validator', () => {
  const value = input();
  value.selected = [{ p: 'P1', roles: ['D'], fact_des: '项目未完。', argument: { 时间: '', 主体: '项目', 地点: '', 内容: '', 影响: '', 数值口径: '' }, entities: [], extra: 'strict must reject' }];
  value.conflicting = [{ p: 'P1', resolution_type: 'Other' }];
  const result = normalizeSubmissionShape(value);
  assert.deepEqual(result.normalized_args, value); assert.deepEqual(result.format_transformations, []);
});

test('argument exact tuple also normalizes inside a selected object', () => {
  const value = input(); value.selected = [{ p: 'P1', roles: ['D'], fact_des: '项目未完。', argument: argumentsTuple(), entities: [] }];
  const result = normalizeSubmissionShape(value);
  assert.equal(result.normalized_args.selected[0].argument.时间, '昨日');
  assert.deepEqual(result.format_transformations.map(row => row.path), ['selected[0].argument', 'conflicting[0]']);
});

test('inexact lengths, primitive types, sparse tuples and reordered guesses are never adapted', () => {
  for (const change of [
    value => value.selected[0].pop(), value => value.selected[0].push('extra'), value => value.selected[0][0] = 1,
    value => value.selected[0][1] = 'D', value => value.selected[0][2] = {}, value => value.selected[0][3] = 'a',
    value => value.selected[0][4] = ['entity', 2], value => value.selected[0][3].pop(), value => value.selected[0][3].push('extra'),
    value => value.selected[0][3][0] = 1, value => delete value.selected[0][3][0],
    value => value.conflicting[0].pop(), value => value.conflicting[0].push('extra'), value => value.conflicting[0][1] = null,
  ]) {
    const value = input(); change(value); const original = structuredClone(value);
    assert.throws(() => normalizeSubmissionShape(value), error => error.code === 'invalid_submission_shape');
    assert.deepEqual(value, original);
  }
  for (const value of [null, [], '', 1]) assert.throws(() => normalizeSubmissionShape(value), /invalid_submission_shape/);
});

test('source identities, role labels and resolution strings are never corrected', () => {
  const value = input(); value.selected[0][0] = 'P99'; value.selected[0][1] = ['invented']; value.conflicting[0] = ['P99', 'newest'];
  const result = normalizeSubmissionShape(value);
  assert.equal(result.normalized_args.selected[0].p, 'P99'); assert.deepEqual(result.normalized_args.selected[0].roles, ['invented']);
  assert.deepEqual(result.normalized_args.conflicting, [{ p: 'P99', resolution_type: 'newest' }]);
});

test('source addresses alone are incomplete evidence rows, not unknown source identities', () => {
  for (const row of ['P1', null, 1, true]) {
    const value = input(); value.selected = [row];
    assert.throws(() => normalizeSubmissionShape(value), error => error.code === 'invalid_submission_shape' && error.path === 'selected[0]');
    assert.equal(value.selected[0], row);
  }
});
