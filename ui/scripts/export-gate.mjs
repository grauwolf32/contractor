// The TypeScript half of the UI dead-code gate (V385-001, AGENTS.md 3.7),
// modelled on tools/testonlyexports: every export of a production module
// under src/ needs a production importer in another file.
//
// An export is reported as
//   - "unused" when no other file imports it (an export used only inside its
//     own module should not be exported), and
//   - "test-only" when only tests import it (*.test.ts(x), src/test/, e2e/).
// Production code is every other TS/TSX file under src/; src/api/generated
// is generated and not checked. src/test is the shared test helper
// directory: its exports are not checked, and production code must not
// import it.
//
// Imports are resolved through re-exports, so `export { X } from "./x"` in a
// barrel counts as a use of x.X, and the barrel's own entry is reported when
// nobody imports X from the barrel. A dynamic import(...) uses the module's
// default export and every export whose name appears as an identifier or
// string in the importing file (lazyRoute(load, "RunsRoute")).

import { dirname, join, relative, sep } from "node:path";

import ts from "typescript";

import { classify } from "./source-kinds.mjs";

function loadConfig(ui, name) {
  const path = join(ui, name);
  const read = ts.readConfigFile(path, ts.sys.readFile);
  if (read.error !== undefined)
    throw new Error(
      ts.flattenDiagnosticMessageText(read.error.messageText, "\n"),
    );
  return ts.parseJsonConfigFileContent(read.config, ts.sys, dirname(path));
}

function moduleSymbol(checker, source) {
  return checker.getSymbolAtLocation(source);
}

/**
 * Analyses the UI package at `ui` and returns the findings as sorted
 * "<file>:<export> unused|test-only" keys, plus hard errors (production code
 * importing src/test).
 */
export function analyzeExports(ui, { extraRoots = [] } = {}) {
  const main = loadConfig(ui, "tsconfig.json");
  const e2e = loadConfig(ui, "tsconfig.e2e.json");
  const roots = [
    ...new Set([...main.fileNames, ...e2e.fileNames, ...extraRoots]),
  ];
  const program = ts.createProgram(roots, { ...main.options, noEmit: true });
  const checker = program.getTypeChecker();
  const uses = new Map(); // symbol -> Set<"prod" | "test">
  const errors = [];

  const record = (symbol, kind, fromFile) => {
    let current = symbol;
    const seen = new Set();
    while (current !== undefined && !seen.has(current)) {
      seen.add(current);
      const declaredIn = current.declarations?.[0]?.getSourceFile().fileName;
      if (declaredIn !== fromFile) {
        let set = uses.get(current);
        if (set === undefined) uses.set(current, (set = new Set()));
        set.add(kind);
      }
      if ((current.flags & ts.SymbolFlags.Alias) === 0) break;
      current = checker.getImmediateAliasedSymbol(current);
    }
  };

  const resolveModule = (specifier, containing) =>
    ts.resolveModuleName(specifier, containing, main.options, ts.sys)
      .resolvedModule?.resolvedFileName;

  for (const source of program.getSourceFiles()) {
    const kind = classify(ui, source.fileName);
    if (kind === "skip") continue;
    const fileName = source.fileName;
    const words = new Set();
    const collectWords = (node) => {
      if (ts.isIdentifier(node)) words.add(node.text);
      else if (ts.isStringLiteralLike(node)) words.add(node.text);
      ts.forEachChild(node, collectWords);
    };
    let wordsCollected = false;

    const visit = (node) => {
      if (
        (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
        node.moduleSpecifier !== undefined &&
        ts.isStringLiteral(node.moduleSpecifier)
      ) {
        const target = resolveModule(node.moduleSpecifier.text, fileName);
        if (
          kind === "prod" &&
          target !== undefined &&
          classify(ui, target) === "test"
        ) {
          errors.push(
            `${relative(ui, fileName)}: production code imports test code ${relative(ui, target)}`,
          );
        }
        const clause = ts.isImportDeclaration(node)
          ? node.importClause
          : undefined;
        if (
          clause?.namedBindings !== undefined &&
          ts.isNamespaceImport(clause.namedBindings)
        ) {
          const targetSource =
            target === undefined ? undefined : program.getSourceFile(target);
          const symbol =
            targetSource === undefined
              ? undefined
              : moduleSymbol(checker, targetSource);
          if (symbol !== undefined)
            for (const exported of checker.getExportsOfModule(symbol))
              record(exported, kind, fileName);
        }
        if (ts.isExportDeclaration(node) && node.exportClause === undefined) {
          // export * from: the star target's exports are reported where they
          // are declared; importers resolve through the star.
        }
      }
      if (
        ts.isImportSpecifier(node) ||
        ts.isExportSpecifier(node) ||
        ts.isImportClause(node)
      ) {
        const name = ts.isImportClause(node) ? node.name : node.name;
        if (name !== undefined) {
          const symbol = checker.getSymbolAtLocation(name);
          if (symbol !== undefined) record(symbol, kind, fileName);
        }
      }
      if (
        ts.isCallExpression(node) &&
        node.expression.kind === ts.SyntaxKind.ImportKeyword &&
        node.arguments.length === 1 &&
        ts.isStringLiteral(node.arguments[0])
      ) {
        const target = resolveModule(node.arguments[0].text, fileName);
        const targetSource =
          target === undefined ? undefined : program.getSourceFile(target);
        const symbol =
          targetSource === undefined
            ? undefined
            : moduleSymbol(checker, targetSource);
        if (symbol !== undefined) {
          if (!wordsCollected) {
            collectWords(source);
            wordsCollected = true;
          }
          for (const exported of checker.getExportsOfModule(symbol)) {
            if (exported.escapedName === "default" || words.has(exported.name))
              record(exported, kind, fileName);
          }
        }
      }
      ts.forEachChild(node, visit);
    };
    visit(source);
  }

  const findings = [];
  for (const source of program.getSourceFiles()) {
    if (classify(ui, source.fileName) !== "prod") continue;
    const symbol = moduleSymbol(checker, source);
    if (symbol === undefined) continue;
    const file = relative(ui, source.fileName).split(sep).join("/");
    for (const exported of checker.getExportsOfModule(symbol)) {
      const declaration = exported.declarations?.[0];
      // Star re-exports are reported in the module that declares them.
      if (declaration === undefined || declaration.getSourceFile() !== source)
        continue;
      const kinds = uses.get(exported);
      if (kinds?.has("prod")) continue;
      findings.push(
        `${file}:${exported.name} ${kinds?.has("test") ? "test-only" : "unused"}`,
      );
    }
  }
  return { findings: findings.sort(), errors };
}
