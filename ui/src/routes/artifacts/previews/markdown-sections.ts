/**
 * Splits a Markdown body into sections that render separately exactly as they
 * render together (UI-R202). Parsing is superlinear in the size of one
 * document (a 1 MiB report takes seconds, one 1 MiB table minutes), so
 * parsing sections keeps every step short.
 *
 * A section starts only before an unindented line that follows a blank line
 * and does not start a list item: that line closes every list, block quote,
 * indented code block and paragraph before it. Top-level fenced code and
 * HTML blocks can run past such a line, so they are followed line by line;
 * where a line cannot be classified without the full parser, or a link
 * reference or footnote definition could be used from another section, the
 * whole body stays one section.
 */

// About 16 Ki characters of Markdown parse and render in a few tens of
// milliseconds, short enough to keep the page responsive between sections.
const SECTION_CHARACTERS = 16 * 1024;

// LIM-UI-16: a longer section is shown as text instead of being parsed.
export const MAXIMUM_RENDERED_SECTION_CHARACTERS = 256 * 1024;

const BLANK = /^[ \t]*$/;
const LIST_ITEM = /^(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)/;
const FENCE = /^(`{3,}|~{3,})(.*)$/;
const CLOSING_FENCE = /^(`{3,}|~{3,})[ \t]*$/;
// HTML blocks of CommonMark kinds 1 to 5 may contain blank lines and end only
// at their own end marker.
const HTML_BLOCKS: readonly (readonly [start: RegExp, end: RegExp])[] = [
  [
    /^<(?:script|pre|style|textarea)(?:[ \t>]|$)/i,
    /<\/(?:script|pre|style|textarea)>/i,
  ],
  [/^<!--/, /-->/],
  [/^<\?/, /\?>/],
  [/^<![A-Za-z]/, />/],
  [/^<!\[CDATA\[/, /\]\]>/],
];
// Kinds 6 and 7 end at the next blank line, but a line like this may also be
// paragraph text, in which a fence would open where the HTML block has none.
const POSSIBLE_HTML = /^<\/?[A-Za-z]/;
// Any line that may hold a link reference or footnote definition, even inside
// a container or a code block: a definition is visible from every section.
const DEFINITION =
  /^[ \t>]*(?:(?:[-+*]|\d{1,9}[.)])[ \t]+)*\[(?:[^\]\\[]|\\[\s\S]){0,999}\]:/m;

type OpenBlock =
  | { kind: "fence"; marker: string; size: number; indent: number }
  | { kind: "html"; end: RegExp; indent: number }
  | { kind: "possible-html" };

function indentation(line: string): { columns: number; offset: number } {
  let columns = 0;
  let offset = 0;
  for (; offset < line.length; offset++) {
    if (line[offset] === " ") {
      columns++;
    } else if (line[offset] === "\t") {
      columns += 4 - (columns % 4);
    } else {
      break;
    }
  }
  return { columns, offset };
}

/** Offsets where a section may start, or undefined when none is safe. */
function sectionStarts(body: string): number[] | undefined {
  if (DEFINITION.test(body)) {
    return undefined;
  }
  const starts: number[] = [];
  let open: OpenBlock | undefined;
  let afterBlank = false;
  for (let start = 0; start < body.length;) {
    const newline = body.indexOf("\n", start);
    const end = newline < 0 ? body.length : newline;
    const line = body.slice(start, body[end - 1] === "\r" ? end - 1 : end);
    const blank = BLANK.test(line);
    const { columns, offset } = indentation(line);
    const content = line.slice(offset);

    if (open?.kind === "fence" || open?.kind === "html") {
      // A fence or HTML block opened with indentation may belong to a list
      // item, which a less indented line closes; the top-level block would
      // continue. Only the full parser can tell.
      if (!blank && columns < open.indent) {
        return undefined;
      }
      if (open.kind === "fence") {
        const closing = columns <= 3 ? CLOSING_FENCE.exec(content) : null;
        if (
          closing?.[1] !== undefined &&
          closing[1][0] === open.marker &&
          closing[1].length >= open.size
        ) {
          open = undefined;
        }
      } else if (open.end.test(line)) {
        open = undefined;
      }
    } else if (blank) {
      open = undefined;
    } else {
      if (
        afterBlank &&
        columns === 0 &&
        start > 0 &&
        !LIST_ITEM.test(content)
      ) {
        starts.push(start);
      }
      if (columns <= 3) {
        const fence = FENCE.exec(content);
        const html = HTML_BLOCKS.find(([opening]) => opening.test(content));
        if (
          fence?.[1] !== undefined &&
          !(fence[1][0] === "`" && (fence[2] ?? "").includes("`"))
        ) {
          if (open !== undefined) {
            return undefined;
          }
          open = {
            kind: "fence",
            marker: fence[1][0] ?? "`",
            size: fence[1].length,
            indent: columns,
          };
        } else if (html !== undefined) {
          if (open !== undefined) {
            return undefined;
          }
          const [opening, closing] = html;
          const rest = content.slice(opening.exec(content)?.[0].length ?? 0);
          open = closing.test(rest)
            ? undefined
            : { kind: "html", end: closing, indent: columns };
        } else if (POSSIBLE_HTML.test(content)) {
          open = { kind: "possible-html" };
        }
      }
    }
    afterBlank = blank;
    start = end + 1;
  }
  return starts;
}

/**
 * Sections of about SECTION_CHARACTERS each, in order; joined they are the
 * body. A run without a safe start longer than that is a section of its own.
 */
export function splitMarkdownSections(body: string): string[] {
  const starts = sectionStarts(body);
  if (starts === undefined) {
    return [body];
  }
  const sections: string[] = [];
  let sectionStart = 0;
  let previous = 0;
  for (const next of [...starts, body.length]) {
    if (previous > sectionStart && next - sectionStart > SECTION_CHARACTERS) {
      sections.push(body.slice(sectionStart, previous));
      sectionStart = previous;
    }
    previous = next;
  }
  sections.push(body.slice(sectionStart));
  return sections;
}
