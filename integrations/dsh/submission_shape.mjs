/** Lossless adaptation of explicitly defined positional tool rows. No semantic or source repair. */
const clone = value => structuredClone(value);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const ARGUMENT_KEYS = ['时间', '主体', '地点', '内容', '影响', '数值口径'];
const strings = value => Array.isArray(value) && Array.from(value).every(item => typeof item === 'string');

function invalid(path) {
  const error = new Error('invalid_submission_shape');
  error.code = 'invalid_submission_shape'; error.path = path;
  throw error;
}

/** Only exact selected(5), conflicting(2), and argument(6 strings) tuples are adapted. */
export function normalizeSubmissionShape(originalArgs) {
  if (!object(originalArgs)) invalid('');
  const normalized = clone(originalArgs), transformations = [];
  function record(path, original_value, normalized_value, reason) {
    transformations.push({ path, original_value: clone(original_value), normalized_value: clone(normalized_value), reason });
  }
  function argument(value, path) {
    if (!Array.isArray(value)) return value;
    if (value.length !== ARGUMENT_KEYS.length || !strings(value)) invalid(path);
    const result = Object.fromEntries(ARGUMENT_KEYS.map((key, index) => [key, value[index]]));
    record(path, value, result, 'exact_argument_tuple_to_object'); return result;
  }
  if (Array.isArray(normalized.selected)) normalized.selected = normalized.selected.map((value, index) => {
    const path = `selected[${index}]`;
    if (Array.isArray(value)) {
      if (value.length !== 5 || typeof value[0] !== 'string' || !strings(value[1]) || typeof value[2] !== 'string'
        || (!object(value[3]) && !Array.isArray(value[3])) || !strings(value[4])) invalid(path);
      const result = { p: value[0], roles: value[1], fact_des: value[2], argument: argument(value[3], `${path}.argument`), entities: value[4] };
      record(path, value, result, 'exact_selected_tuple_to_object'); return result;
    }
    if (!object(value)) invalid(path);
    if (Object.hasOwn(value, 'argument')) value.argument = argument(value.argument, `${path}.argument`);
    return value;
  });
  if (Array.isArray(normalized.conflicting)) normalized.conflicting = normalized.conflicting.map((value, index) => {
    if (!Array.isArray(value)) return value;
    const path = `conflicting[${index}]`;
    if (value.length !== 2 || !strings(value)) invalid(path);
    const result = { p: value[0], resolution_type: value[1] };
    record(path, value, result, 'exact_conflicting_tuple_to_object'); return result;
  });
  return { normalized_args: normalized, format_transformations: transformations };
}
