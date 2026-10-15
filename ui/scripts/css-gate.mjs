// The CSS half of the UI dead-code gate (V385-001, AGENTS.md 3.7): every
// class a stylesheet under src/ styles must be named by production code, and
// every @media width must be one of the documented breakpoints
// (src/ui/README.md, "Style scale").
//
// A class counts as referenced when a production source file (TS/TSX outside
// tests, index.html, server/*.mjs) contains it as a whole word inside a
// string or template literal. A class built at run time (`diff-line-${kind}`)
// or rendered by a third-party component is listed in css-allowlist.json:
// "prefixes" for dynamic families, "classes" for single names, each with the
// source that produces it. Both lists are explicit, so a new dynamic family
// is a reviewed decision rather than a guess, and an entry that no longer
// matches a stylesheet class or is referenced directly is reported, so the
// list only shrinks.

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

import ts from "typescript";

import { classify } from "./source-kinds.mjs";

/** The only max-width / min-width values @media rules may use, in px. */
export const BREAKPOINTS = Object.freeze([480, 640, 820, 1100]);

const CLASS_TOKEN = /^-?[_a-zA-Z][_a-zA-Z0-9-]*$/;

/** Files under `root` (recursively) whose name passes `accept`. */
export function walk(root, accept) {
  const found = [];
  const visit = (directory) => {
    for (const name of readdirSync(directory)) {
      if (name === "node_modules" || name === "generated") continue;
      const path = join(directory, name);
      if (statSync(path).isDirectory()) visit(path);
      else if (accept(path)) found.push(path);
    }
  };
  if (existsSync(root)) visit(root);
  return found.sort();
}

/** Production sources of the UI package rooted at `ui`. */
export function productionSources(ui) {
  return [
    ...walk(join(ui, "src"), (path) => /\.(ts|tsx)$/.test(path)),
    ...walk(join(ui, "server"), (path) => path.endsWith(".mjs")),
    ...(existsSync(join(ui, "index.html")) ? [join(ui, "index.html")] : []),
  ].filter((path) => classify(ui, path) === "prod");
}

/** Every stylesheet under src/. */
export function stylesheets(ui) {
  return walk(join(ui, "src"), (path) => path.endsWith(".css"));
}

function literalTexts(path, text) {
  if (!/\.(ts|tsx|mjs|js)$/.test(path)) return [text];
  const kind = path.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const source = ts.createSourceFile(
    path,
    text,
    ts.ScriptTarget.Latest,
    false,
    kind,
  );
  const texts = [];
  const visit = (node) => {
    if (
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      ts.isTemplateHead(node) ||
      ts.isTemplateMiddle(node) ||
      ts.isTemplateTail(node)
    ) {
      texts.push(node.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return texts;
}

/** Whole words of every string and template literal in the given files. */
export function referencedTokens(files) {
  const tokens = new Set();
  for (const path of files) {
    for (const text of literalTexts(path, readFileSync(path, "utf8"))) {
      for (const word of text.split(/[^A-Za-z0-9_-]+/)) {
        if (word !== "") tokens.add(word);
      }
    }
  }
  return tokens;
}

function stripComments(css) {
  return css.replace(/\/\*[\s\S]*?\*\//g, (comment) =>
    comment.replace(/[^\n]/g, " "),
  );
}

/**
 * The class selectors and @media rules of one stylesheet, with line numbers.
 * Selector text inside strings and attribute selectors is ignored.
 */
export function parseStylesheet(css) {
  const text = stripComments(css);
  const classes = [];
  const media = [];
  let preludeStart = 0;
  const skipDepths = [];
  let depth = 0;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (char === '"' || char === "'") {
      const end = text.indexOf(char, index + 1);
      index = end < 0 ? text.length : end;
      continue;
    }
    if (char === "{") {
      const prelude = text.slice(preludeStart, index);
      const line = text.slice(0, preludeStart).split("\n").length;
      const trimmed = prelude.trim();
      const offset = prelude.length - prelude.trimStart().length;
      const startLine = line + prelude.slice(0, offset).split("\n").length - 1;
      if (trimmed.startsWith("@")) {
        if (/^@media\b/.test(trimmed))
          media.push({ query: trimmed, line: startLine });
        if (/^@(-webkit-)?keyframes\b|^@font-face\b|^@page\b/.test(trimmed))
          skipDepths.push(depth + 1);
      } else if (skipDepths.at(-1) !== depth && trimmed !== "") {
        const selector = trimmed
          .replace(/\[[^\]]*\]/g, "")
          .replace(/url\([^)]*\)/g, "");
        for (const match of selector.matchAll(
          /\.(-?[_a-zA-Z][_a-zA-Z0-9-]*)/g,
        )) {
          classes.push({ name: match[1], line: startLine });
        }
      }
      depth += 1;
      preludeStart = index + 1;
    } else if (char === "}") {
      if (skipDepths.at(-1) === depth) skipDepths.pop();
      depth -= 1;
      preludeStart = index + 1;
    } else if (char === ";") {
      preludeStart = index + 1;
    }
  }
  return { classes, media };
}

/** Widths in px that one @media query tests, e.g. [820] for (max-width: 820px). */
export function mediaWidths(query) {
  const widths = [];
  for (const match of query.matchAll(
    /\b(?:max|min)-width\s*:\s*([0-9.]+)(px|rem|em)\b|\bwidth\s*[<>]=?\s*([0-9.]+)(px|rem|em)\b/g,
  )) {
    const value = Number(match[1] ?? match[3]);
    const unit = match[2] ?? match[4];
    widths.push(unit === "px" ? value : value * 16);
  }
  return widths;
}

/**
 * The gate itself. Returns one message per finding: an unreferenced class
 * (first rule that styles it), an @media width outside BREAKPOINTS, and a
 * stale allowlist entry. `min-width` rules may use a breakpoint plus one
 * pixel (821px).
 */
export function checkStyles({ ui, allowlist }) {
  const tokens = referencedTokens(productionSources(ui));
  const prefixes = (allowlist.prefixes ?? []).map((entry) => entry.prefix);
  const named = new Set((allowlist.classes ?? []).map((entry) => entry.name));
  const findings = [];
  const reported = new Set();
  const styled = new Set();
  for (const path of stylesheets(ui)) {
    const file = relative(ui, path).split("\\").join("/");
    const { classes, media } = parseStylesheet(readFileSync(path, "utf8"));
    for (const { name, line } of classes) {
      if (!CLASS_TOKEN.test(name)) continue;
      styled.add(name);
      if (
        tokens.has(name) ||
        named.has(name) ||
        prefixes.some((prefix) => name.startsWith(prefix)) ||
        reported.has(name)
      )
        continue;
      reported.add(name);
      findings.push(
        `${file}:${line}: class .${name} is not referenced by production code`,
      );
    }
    for (const { query, line } of media) {
      for (const width of mediaWidths(query)) {
        const allowed =
          BREAKPOINTS.includes(width) ||
          (/min-width|width\s*>/.test(query) &&
            BREAKPOINTS.includes(width - 1));
        if (!allowed)
          findings.push(
            `${file}:${line}: @media width ${width}px is not a breakpoint (${BREAKPOINTS.join(", ")} px)`,
          );
      }
    }
  }
  for (const prefix of prefixes) {
    const family = [...styled].filter((name) => name.startsWith(prefix));
    if (family.length === 0 || family.every((name) => tokens.has(name)))
      findings.push(
        `scripts/css-allowlist.json: prefix "${prefix}" is stale: no stylesheet class needs it`,
      );
  }
  for (const name of named) {
    if (!styled.has(name) || tokens.has(name))
      findings.push(
        `scripts/css-allowlist.json: class "${name}" is stale: ${styled.has(name) ? "production code names it" : "no stylesheet styles it"}`,
      );
  }
  return findings;
}
