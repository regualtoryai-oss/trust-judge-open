/** Fixed business diagnostics only. Never expose arbitrary tool exception text. */
const KNOWN = new Set([
  'source_bundle_tools_invalid_source_range',
  'bundle_bridge_duplicate_selected_start_anchor',
]);

export function publicBundleToolFailure(content) {
  if (!Array.isArray(content) || content.length !== 1 || content[0]?.type !== 'text') return {};
  const text = content[0].text;
  if (typeof text !== 'string') return {};
  const match = /^Error: ([a-z][a-z0-9_]+)$/.exec(text);
  return match && match[0] === text && KNOWN.has(match[1]) ? { failure_code: match[1] } : {};
}
