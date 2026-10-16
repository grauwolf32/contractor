// Guards the one-definition rule for contract grammars and state lists
// (AGENTS.md 3.1): identifier, digest and Runtime identity grammars come
// from the generated contract (src/api/contract.ts), and Audit and Run state
// lists from src/api/states.ts. A copy elsewhere drifts from the Server.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import ts from "typescript";
import { describe, expect, it } from "vitest";

import { schemaEnum } from "./contract";

const ROOT = join(__dirname, "..", "..");
const SOURCE = join(ROOT, "src");

/**
 * The modules that own these rules; every other module imports them. The
 * Run state vocabulary (words and tones per state) maps states in a Record,
 * which this guard does not flag; if it ever lists states it belongs here.
 */
const SHARED_DEFINITIONS = new Set([
  "src/api/contract.ts",
  "src/api/states.ts",
]);

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
    const before = text.slice(0, match.index);
    if (
      sameMembers(members, ["A-Z", "a-z", "0-9", "_", ".", ":", "-"]) &&
      (bound === Infinity || bound >= 255)
    ) {
      found.push("ResourceId or Revision");
    } else if (
      sameMembers(members, ["a-z", "0-9", "_", "-"]) &&
      before.endsWith("[a-z]")
    ) {
      found.push("Runtime or configuration identifier");
    } else if (
      sameMembers(members, ["A-Z", "a-z", "0-9", ".", "_", "+", "-"]) &&
      before.endsWith("[A-Za-z0-9]")
    ) {
      found.push("Runtime or configuration version");
    } else if (sameMembers(members, ["0-9", "a-f"]) && match[2] === "{64}") {
      found.push(before.endsWith("sha256:") ? "digest" : "Runtime Agent ID");
    }
  }
  return found;
}

const AUDIT_STATES = new Set<string>(schemaEnum("AuditState"));
const RUN_STATES = new Set<string>(schemaEnum("WorkflowRunState"));
// A list counts as a state list when its declaration names or types states.
const STATE_CONTEXT =
  /AuditState|WorkflowRunState|RunState|QueueState|\["state"\]|\bstates?\b|STATES?\b/;

/** The name and type text of the declaration that holds an array literal. */
function declarationContext(node: ts.Node, file: ts.SourceFile): string {
  for (let current = node.parent; current !== undefined;) {
    if (
      ts.isVariableDeclaration(current) ||
      ts.isPropertyAssignment(current) ||
      ts.isPropertyDeclaration(current) ||
      ts.isParameter(current)
    ) {
      const type =
        "type" in current && current.type !== undefined
          ? current.type.getText(file)
          : "";
      const initializer =
        "initializer" in current && current.initializer !== undefined
          ? current.initializer.getText(file)
          : "";
      const satisfies = /\bsatisfies\b[^;]*$/.exec(initializer)?.[0] ?? "";
      return `${current.name.getText(file)} ${type} ${satisfies}`;
    }
    if (ts.isStatement(current)) return "";
    current = current.parent;
  }
  return "";
}

/** Audit or Run state lists written out in one module. */
function stateListCopies(path: string, text: string): string[] {
  const file = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true);
  const found: string[] = [];
  const visit = (node: ts.Node) => {
    if (ts.isArrayLiteralExpression(node) && node.elements.length >= 2) {
      const values = node.elements.map((element) =>
        ts.isStringLiteral(element) ? element.text : undefined,
      );
      const strings = values.filter(
        (value): value is string => value !== undefined,
      );
      if (
        strings.length === values.length &&
        (strings.every((value) => AUDIT_STATES.has(value)) ||
          strings.every((value) => RUN_STATES.has(value))) &&
        STATE_CONTEXT.test(declarationContext(node, file))
      ) {
        found.push(`state list [${strings.join(", ")}]`);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return found;
}

/**
 * The patterns a module compiles: regular-expression literals and the
 * string arguments of RegExp. Messages that quote a grammar are not rules.
 */
function patternTexts(path: string, text: string): string[] {
  const file = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true);
  const texts: string[] = [];
  const visit = (node: ts.Node) => {
    if (ts.isRegularExpressionLiteral(node)) {
      texts.push(node.text);
    } else if (
      (ts.isNewExpression(node) || ts.isCallExpression(node)) &&
      node.expression.getText(file) === "RegExp"
    ) {
      const source = node.arguments?.[0];
      if (source !== undefined) texts.push(source.getText(file));
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return texts;
}

describe("contract rules have one definition", () => {
  it("recognizes each guarded grammar and state list", () => {
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
    expect(grammarCopies("^sha256:[a-f0-9]{64}$")).toEqual(["digest"]);
    expect(grammarCopies("^[0-9a-f]{64}$")).toEqual(["Runtime Agent ID"]);
    expect(grammarCopies("^[a-z][a-z0-9_-]{0,62}$")).toEqual([
      "Runtime or configuration identifier",
    ]);
    expect(grammarCopies("^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")).toEqual([
      "Runtime or configuration version",
    ]);
    expect(grammarCopies("^[a-z0-9][a-z0-9._-]*$")).toEqual([]);
    expect(
      stateListCopies(
        "probe.ts",
        'const DONE: readonly AuditState[] = ["completed", "failed"];\n' +
          'const states = new Set(["running", "cancelled"]);\n' +
          'const outcomes = ["succeeded", "failed"];\n',
      ),
    ).toEqual([
      "state list [completed, failed]",
      "state list [running, cancelled]",
    ]);
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
      for (const copy of stateListCopies(path, text)) {
        copies.push(`${name}: ${copy}`);
      }
    }
    expect(copies).toEqual([]);
  });
});
