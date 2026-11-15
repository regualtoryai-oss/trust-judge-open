export function codepointExcerpt(
  text: string,
  quote: string,
  start: number,
  end: number,
) {
  const chars = Array.from(text);
  if (
    !Number.isInteger(start) ||
    !Number.isInteger(end) ||
    start < 0 ||
    end > chars.length ||
    start >= end
  )
    return null;
  if (chars.slice(start, end).join("") !== quote) return null;
  return {
    before: chars.slice(0, start).join(""),
    quote: chars.slice(start, end).join(""),
    after: chars.slice(end).join(""),
  };
}
