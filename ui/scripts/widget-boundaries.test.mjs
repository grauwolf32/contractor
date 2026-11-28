import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";
import ts from "typescript";

import { productionSources } from "./css-gate.mjs";

const ui = fileURLToPath(new URL("..", import.meta.url));

function routeDependencies(file, text) {
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
  const imports = [];
  function inspect(node) {
    let specifier;
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node))
      specifier = node.moduleSpecifier;
    else if (
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword
    )
      specifier = node.arguments[0];
    else if (
      ts.isNewExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "URL"
    )
      specifier = node.arguments?.[0];
    if (specifier && ts.isStringLiteralLike(specifier)) {
      const target = relative(
        resolve(ui, "src"),
        resolve(dirname(file), specifier.text),
      );
      if (target === "routes" || target.startsWith(`routes${sep}`))
        imports.push(specifier.text);
    }
    ts.forEachChild(node, inspect);
  }
  inspect(source);
  return imports;
}

describe("widget route boundary", () => {
  it("rejects static, re-export, CSS, lazy and Worker dependencies on routes", () => {
    const file = resolve(ui, "src/widgets/example/view.tsx");
    for (const text of [
      'import View from "../../routes/reports/view";',
      'export { View } from "../../routes/reports/view";',
      'import "../../routes/reports/style.css";',
      'const view = () => import("../../routes/reports/view");',
      'new Worker(new URL("../../routes/reports/worker.ts", import.meta.url));',
    ])
      assert.equal(routeDependencies(file, text).length, 1, text);
    assert.deepEqual(
      routeDependencies(file, 'import { Button } from "../../ui";'),
      [],
    );
  });

  it("keeps every production widget independent of route modules", () => {
    const files = productionSources(ui).filter((path) =>
      path.startsWith(resolve(ui, "src/widgets") + sep),
    );
    assert.ok(files.length > 0, "production widgets must be scanned");
    const violations = files.flatMap((file) =>
      routeDependencies(file, readFileSync(file, "utf8")).map(
        (dependency) => `${relative(ui, file)} imports ${dependency}`,
      ),
    );
    assert.deepEqual(violations, []);
  });
});
