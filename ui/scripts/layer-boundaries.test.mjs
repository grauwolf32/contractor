import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";
import ts from "typescript";

import { productionSources } from "./css-gate.mjs";

const ui = fileURLToPath(new URL("..", import.meta.url));

function blockedDependencies(file, text, layers, packages = []) {
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
      if (
        layers.some(
          (layer) => target === layer || target.startsWith(`${layer}${sep}`),
        ) ||
        packages.some(
          (name) =>
            specifier.text === name || specifier.text.startsWith(`${name}/`),
        )
      )
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
      assert.equal(blockedDependencies(file, text, ["routes"]).length, 1, text);
    assert.deepEqual(
      blockedDependencies(file, 'import { Button } from "../../ui";', [
        "routes",
      ]),
      [],
    );
  });

  it("keeps every production widget independent of route modules", () => {
    const files = productionSources(ui).filter((path) =>
      path.startsWith(resolve(ui, "src/widgets") + sep),
    );
    assert.ok(files.length > 0, "production widgets must be scanned");
    const violations = files.flatMap((file) =>
      blockedDependencies(file, readFileSync(file, "utf8"), ["routes"]).map(
        (dependency) => `${relative(ui, file)} imports ${dependency}`,
      ),
    );
    assert.deepEqual(violations, []);
  });
});

describe("shared module boundaries", () => {
  const forbidden = ["routes", "widgets", "ui", "app/pagination"];
  const router = ["react-router"];
  it("rejects view and router dependencies without rejecting API and shared helpers", () => {
    const file = resolve(ui, "src/modules/checks/example.ts");
    for (const text of [
      'import { View } from "../../routes/checks";',
      'const load = () => import("../../widgets/audit-report/view");',
      'export { Button } from "../../ui";',
      'import { useSearchParams } from "react-router";',
      'import { useNavigate } from "react-router/dom";',
      'import { useURLCursorStack } from "../../app/pagination";',
    ])
      assert.equal(
        blockedDependencies(file, text, forbidden, router).length,
        1,
        text,
      );
    assert.deepEqual(
      blockedDependencies(
        file,
        'import { pollInterval } from "../../app/polling"; import { getAudit } from "../../api/audits";',
        forbidden,
        router,
      ),
      [],
    );
  });
  it("scans every production shared module and leaves route state outside it", () => {
    const files = productionSources(ui).filter((file) =>
      file.startsWith(resolve(ui, "src/modules") + sep),
    );
    assert.ok(files.length > 0, "production modules must be scanned");
    const violations = files.flatMap((file) =>
      blockedDependencies(
        file,
        readFileSync(file, "utf8"),
        forbidden,
        router,
      ).map((name) => `${relative(ui, file)} imports ${name}`),
    );
    assert.deepEqual(violations, []);
  });
});
