/**
 * Reading a finding proposal for display. Everything here selects or
 * reformats the Worker's own text; nothing adds words to it.
 */
import type { AuditFinding } from "../../api/audits";
import type { components } from "../../api/generated/public";

type ProposalDocument = components["schemas"]["FindingProposalDocument"];
type StandardReference = components["schemas"]["FindingStandardReference"];

const FENCE = /^\s{0,3}(`{3,}|~{3,})/;
const HEADING = /^\s{0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t#]*$/;
const IMPACT_HEADING = /^(?:(?:potential|business|security)\s+)?impact\b/i;
const IMPACT_LABEL =
  /^\s*(?:\*\*|__)\s*(?:(?:potential|business|security)\s+)?impact\s*:?\s*(?:\*\*|__)\s*:?\s*/i;

interface Block {
  /** Source lines of the block, without the blank lines around it. */
  lines: string[];
  /** ATX heading level when the block is a heading line, else undefined. */
  heading?: { level: number; text: string } | undefined;
}

/**
 * Splits Markdown into blocks: paragraphs separated by blank lines, heading
 * lines on their own, and fenced code kept whole even across blank lines.
 */
function blocks(markdown: string): Block[] {
  const result: Block[] = [];
  let current: string[] = [];
  let fence: string | undefined;
  const flush = () => {
    if (current.length > 0) result.push({ lines: current });
    current = [];
  };
  for (const line of markdown.replaceAll("\r\n", "\n").split("\n")) {
    if (fence !== undefined) {
      current.push(line);
      if (line.trim().startsWith(fence)) {
        fence = undefined;
        flush();
      }
      continue;
    }
    const opening = FENCE.exec(line);
    if (opening !== null) {
      flush();
      fence = opening[1]!.slice(0, 3);
      current.push(line);
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading !== null) {
      flush();
      result.push({
        lines: [line],
        heading: { level: heading[1]!.length, text: heading[2]?.trim() ?? "" },
      });
      continue;
    }
    if (line.trim() === "") flush();
    else current.push(line);
  }
  flush();
  return result;
}

function join(parts: readonly Block[]): string {
  return parts.map((block) => block.lines.join("\n")).join("\n\n");
}

/** The AI's description split into what it found and its stated impact. */
export interface DescriptionParts {
  found: string;
  /** Present when the description has an "Impact" heading or label. */
  impact?: string | undefined;
}

/**
 * Moves an "Impact" section of the description into its own part: a heading
 * named Impact (with what follows up to the next heading of the same or a
 * higher level), or a paragraph that starts with a bold "Impact:" label.
 */
export function splitImpact(markdown: string): DescriptionParts {
  const parts = blocks(markdown);
  const headingAt = parts.findIndex(
    (block) =>
      block.heading !== undefined && IMPACT_HEADING.test(block.heading.text),
  );
  if (headingAt >= 0) {
    const level = parts[headingAt]!.heading!.level;
    let end = headingAt + 1;
    while (
      end < parts.length &&
      (parts[end]!.heading === undefined || parts[end]!.heading!.level > level)
    )
      end += 1;
    const impact = join(parts.slice(headingAt + 1, end)).trim();
    if (impact !== "")
      return {
        found: join([...parts.slice(0, headingAt), ...parts.slice(end)]).trim(),
        impact,
      };
  }
  const labelAt = parts.findIndex(
    (block) =>
      block.heading === undefined &&
      !FENCE.test(block.lines[0] ?? "") &&
      IMPACT_LABEL.test(block.lines[0] ?? ""),
  );
  if (labelAt >= 0) {
    const block = parts[labelAt]!;
    const impact = [
      (block.lines[0] ?? "").replace(IMPACT_LABEL, ""),
      ...block.lines.slice(1),
    ]
      .join("\n")
      .trim();
    if (impact !== "")
      return {
        found: join([
          ...parts.slice(0, labelAt),
          ...parts.slice(labelAt + 1),
        ]).trim(),
        impact,
      };
  }
  return { found: markdown.trim() };
}

/**
 * The first paragraph of Markdown for a short preview: headings are skipped,
 * a fenced code block stays whole.
 */
export function firstParagraph(markdown: string): string {
  const first = blocks(markdown).find((block) => block.heading === undefined);
  return first === undefined ? "" : first.lines.join("\n").trim();
}

/** Markdown reduced to its words, for a summary in a plain text field. */
export function plainText(markdown: string): string {
  return markdown
    .replace(/^\s{0,3}(`{3,}|~{3,}).*$/gm, "")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, "")
    .replace(/^\s{0,3}#{1,6}\s*/gm, "")
    .replace(/^\s{0,3}>\s?/gm, "")
    .replace(/^\s*(?:[-*+]|\d+[.)])\s+/gm, "")
    .replace(/(\*\*|__)(.+?)\1/g, "$2")
    .replace(/(?<![\w*])([*_])(?!\s)(.+?)(?<!\s)\1(?![\w*])/g, "$2")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

const SENTENCE_END = /^(.+?[.!?])(?=\s|$)/;
const SUMMARY_LIMIT = 300;

function shorten(text: string, limit: number): string {
  if (text.length <= limit) return text;
  const cut = text.slice(0, limit - 1);
  const space = cut.lastIndexOf(" ");
  return `${(space > limit / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

/**
 * A short summary of the AI's conclusion for the decision reason: the
 * proposal's title and the first sentence of what it found. Only inserted
 * when the user asks for it.
 */
export function aiSummary(finding: AuditFinding): string {
  const document = finding.firstProposal.document;
  const title = document.title.trim();
  const paragraph = plainText(
    firstParagraph(splitImpact(document.description).found),
  );
  const sentence = (SENTENCE_END.exec(paragraph)?.[1] ?? paragraph).trim();
  if (sentence === "" || sentence.toLowerCase() === title.toLowerCase())
    return shorten(title, SUMMARY_LIMIT);
  if (title === "") return shorten(sentence, SUMMARY_LIMIT);
  const lead = /[.!?…]$/.test(title) ? title : `${title}.`;
  return shorten(`${lead} ${sentence}`, SUMMARY_LIMIT);
}

const HTTP_METHODS =
  "GET|HEAD|POST|PUT|PATCH|DELETE|OPTIONS|TRACE|CONNECT|QUERY";
const HTTP_OPERATION = new RegExp(
  `^(${HTTP_METHODS})\\s+((?:/|https?://)\\S*)$`,
  "i",
);

/** "GET /orders/{id}" as an HTTP method and path, else undefined. */
export function httpOperation(
  key: string,
): { method: string; path: string } | undefined {
  const match = HTTP_OPERATION.exec(key.trim());
  return match === null
    ? undefined
    : { method: match[1]!.toUpperCase(), path: match[2]! };
}

function isWeakness(reference: StandardReference): boolean {
  return reference.scheme.trim().toUpperCase() === "CWE";
}

/** CWE references of the proposal ("CWE-639"). */
export function weaknessReferences(
  document: Pick<ProposalDocument, "standard_refs">,
): StandardReference[] {
  return document.standard_refs.filter(isWeakness);
}

/** Every other standard reference (OWASP Top 10, ASVS, …). */
export function standardReferences(
  document: Pick<ProposalDocument, "standard_refs">,
): StandardReference[] {
  return document.standard_refs.filter((reference) => !isWeakness(reference));
}
