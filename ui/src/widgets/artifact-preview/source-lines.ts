/** Same CRLF/CR/LF coordinates as Runtime core.lines; preserve literal endings. */
export function sourceLines(source: string, maximum = 4000) {
  const lines: { text: string; ending: string }[] = [];
  const breaks = /\r\n|\r|\n/g;
  let start = 0;
  for (let match = breaks.exec(source); match; match = breaks.exec(source)) {
    if (lines.length === maximum) return undefined;
    lines.push({ text: source.slice(start, match.index), ending: match[0] });
    start = match.index + match[0].length;
  }
  if (start < source.length) {
    if (lines.length === maximum) return undefined;
    lines.push({ text: source.slice(start), ending: "" });
  }
  return lines;
}
