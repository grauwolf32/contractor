// Guards the one-definition rule for contract grammars (AGENTS.md 3.1): the
// ResourceId and Revision grammar comes from the generated contract
// (src/api/contract.ts). A copy elsewhere drifts from the Server.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import ts from "typescript";
import { describe, expect, it } from "vitest";

const ROOT = join(__dirname, "..", "..");
const SOURCE = join(ROOT, "src");

/** The modules that own these rules; every other module imports them. */
const SHARED_DEFINITIONS = new Set(["src/api/contract.ts"]);

function* modules(directory: string): Generator<string> {
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      if (
        path !== join(SOURCE, "api", "generated") &&
        path !== join(SOURCE, "test")
      ) {
        yield* modules(path);
      }
    } else if (/\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name)) {
      yield path;
    }
  }
}

/** The single characters and ranges of a regular-expression class. */
function classMembers(body: string): Set<string> {
  const members = new Set<string>();
  const text = body.replace(/\\(.)/g, "$1");
  for (let index = 0; index < text.length; index++) {
    if (text[index + 1] === "-" && index + 2 < text.length) {
      members.add(text.slice(index, index + 3));
      index += 2;
    } else {
      members.add(text[index]!);
    }
  }
  return members;
}

function sameMembers(left: Set<string>, right: readonly string[]): boolean {
  return left.size === right.length && right.every((item) => left.has(item));
}

/** The upper bound a quantifier allows, or Infinity. */
function upperBound(quantifier: string | undefined): number {
  if (quantifier === undefined) return 1;
  if (quantifier === "*" || quantifier === "+") return Infinity;
  const bounds = /^\{(\d*)(,(\d*))?\}$/.exec(quantifier);
  if (bounds === null) return Infinity;
  if (bounds[2] === undefined) return Number(bounds[1]);
  return bounds[3] === "" ? Infinity : Number(bounds[3]);
}

/**
 * Grammar copies in one pattern or string literal. A class with another
 * bound is another rule (HTTP capture identifiers, Idempotency-Keys).
 */
function grammarCopies(text: string): string[] {
  const found: string[] = [];
  for (const match of text.matchAll(
    /\[((?:\\.|[^\]\\])+)\](\{[0-9,]*\}|\*|\+)?/g,
  )) {
    const members = classMembers(match[1]!);
    const bound = upperBound(match[2]);
    if (
      sameMembers(members, ["A-Z", "a-z", "0-9", "_", ".", ":", "-"]) &&
      (bound === Infinity || bound >= 255)
    ) {
      found.push("ResourceId or Revision");
    }
  }
  return found;
}

function patternTexts(path: string, text: string): string[] {
  const file = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true);
  const texts: string[] = [];
  const visit = (node: ts.Node) => {
    if (
      ts.isRegularExpressionLiteral(node) ||
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node)
    ) {
      texts.push(node.text);
    } else if (ts.isTemplateExpression(node)) {
      texts.push(node.getText(file));
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return texts;
}

describe("contract rules have one definition", () => {
  it("recognizes the guarded grammar in any spelling", () => {
    expect(grammarCopies("^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")).toEqual([
      "ResourceId or Revision",
    ]);
    expect(grammarCopies("^[A-Za-z0-9][A-Za-z0-9._:-]*$")).toEqual([
      "ResourceId or Revision",
    ]);
    expect(grammarCopies("^[a-zA-Z0-9_.:\\-]{1,256}$")).toEqual([
      "ResourceId or Revision",
    ]);
    expect(grammarCopies("^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")).toEqual([]);
    expect(grammarCopies("^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")).toEqual([]);
  });

  it("finds no copy outside the shared definitions", () => {
    const copies: string[] = [];
    for (const path of modules(SOURCE)) {
      const name = relative(ROOT, path);
      if (SHARED_DEFINITIONS.has(name)) continue;
      const text = readFileSync(path, "utf8");
      for (const literal of patternTexts(path, text)) {
        for (const copy of grammarCopies(literal)) {
          copies.push(`${name}: ${copy}`);
        }
      }
    }
    expect(copies).toEqual([]);
  });
});
