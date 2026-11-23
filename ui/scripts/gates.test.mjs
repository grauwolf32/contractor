// Probes for the UI dead-code and building-block gates (V385-001 A8, A9):
// each check must fail on a new unused export, an unused class, a stray
// breakpoint and a hand-written button, alert dialog or SVG icon.
import assert from "node:assert/strict";
import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { RuleTester } from "eslint";
import tseslint from "typescript-eslint";

import { checkStyles, mediaWidths } from "./css-gate.mjs";
import buildingBlocks from "./eslint-building-blocks.mjs";
import { analyzeExports } from "./export-gate.mjs";

const ui = join(dirname(fileURLToPath(import.meta.url)), "..");
const roots = [];

/** A throwaway UI package with the given files. */
function fixture(files) {
  const root = mkdtempSync(join(tmpdir(), "contractor-ui-gate-"));
  roots.push(root);
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(dirname(join(root, path)), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

after(() => {
  for (const root of roots) rmSync(root, { recursive: true, force: true });
});

describe("CSS gate", () => {
  const files = {
    "index.html": "<div id=root></div>",
    "src/page.tsx":
      'export function Page() { return <div className="used ui-row" data-kind={`state-${"x"}`} />; }\n',
    "src/page.css": [
      ".used { color: red; }",
      ".unused, .used:hover { color: blue; }",
      '.used[data-tone="state-unknown"] { margin: 0; }',
      "@keyframes spin { from { opacity: 0; } 50.5% { opacity: 1; } }",
      "@media (max-width: 820px) { .used { gap: 0; } }",
      "@media (min-width: 821px) { .used { gap: 1px; } }",
      "@media (max-width: 700px) { .used { gap: 2px; } }",
      ".state-done { color: green; }",
    ].join("\n"),
  };

  it("passes a stylesheet whose classes are all named and allowlisted", () => {
    const root = fixture({
      ...files,
      "src/page.css": ".used { color: red; } .state-done { color: green; }",
    });
    assert.deepEqual(
      checkStyles({
        ui: root,
        allowlist: { prefixes: [{ prefix: "state-" }], classes: [] },
      }),
      [],
    );
  });

  it("fails on an unused class, a stray breakpoint and stale allowlist entries", () => {
    const root = fixture(files);
    const findings = checkStyles({
      ui: root,
      allowlist: {
        prefixes: [{ prefix: "state-" }, { prefix: "gone-" }],
        classes: [{ name: "used" }, { name: "missing" }],
      },
    });
    assert.deepEqual(findings, [
      "src/page.css:2: class .unused is not referenced by production code",
      "src/page.css:7: @media width 700px is not a breakpoint (480, 640, 820, 1100 px)",
      'scripts/css-allowlist.json: prefix "gone-" is stale: no stylesheet class needs it',
      'scripts/css-allowlist.json: class "used" is stale: production code names it',
      'scripts/css-allowlist.json: class "missing" is stale: no stylesheet styles it',
    ]);
  });

  it("reads widths in px, rem and range syntax", () => {
    assert.deepEqual(mediaWidths("(max-width: 820px)"), [820]);
    assert.deepEqual(mediaWidths("(max-width: 40rem)"), [640]);
    assert.deepEqual(mediaWidths("(width >= 1100px)"), [1100]);
    assert.deepEqual(mediaWidths("(prefers-reduced-motion: reduce)"), []);
  });
});

describe("export gate", () => {
  const tsconfig = JSON.stringify({
    compilerOptions: {
      target: "ES2023",
      module: "ESNext",
      moduleResolution: "Bundler",
      jsx: "react-jsx",
      strict: true,
      noEmit: true,
      skipLibCheck: true,
      types: [],
    },
    include: ["src"],
  });
  const tsconfigE2E = JSON.stringify({
    extends: "./tsconfig.json",
    include: ["e2e"],
  });

  function exportsOf(files) {
    const root = fixture({
      "tsconfig.json": tsconfig,
      "tsconfig.e2e.json": tsconfigE2E,
      ...files,
    });
    // The fixture resolves its own imports only.
    symlinkSync(join(ui, "node_modules"), join(root, "node_modules"), "dir");
    return analyzeExports(root);
  }

  it("reports unused, own-module-only and test-only exports", () => {
    const { findings, errors } = exportsOf({
      "src/model.ts": [
        "export const used = 1;",
        "export const unused = 2;",
        "export const onlyHere = 3;",
        "export const onlyTests = 4;",
        "export const viaBarrel = 5;",
        "export default function lazyPage() { return onlyHere; }",
        "export const RouteName = 6;",
      ].join("\n"),
      "src/index.ts":
        'export { viaBarrel, onlyHere as renamed } from "./model";\n',
      "src/page.ts": [
        'import { used } from "./model";',
        'import { viaBarrel } from "./index";',
        'const load = () => import("./model");',
        'export const name = "RouteName";',
        "void [used, viaBarrel, load];",
      ].join("\n"),
      "src/model.test.ts":
        'import { onlyTests } from "./model";\nvoid onlyTests;\n',
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(findings, [
      "src/index.ts:renamed unused",
      "src/model.ts:onlyTests test-only",
      "src/model.ts:unused unused",
      "src/page.ts:name unused",
    ]);
  });

  it("rejects production code that imports test helpers", () => {
    const { errors } = exportsOf({
      "src/test/fake-server.ts": "export const server = 1;\n",
      "src/page.ts":
        'import { server } from "./test/fake-server";\nvoid server;\n',
    });
    assert.deepEqual(errors, [
      "src/page.ts: production code imports test code src/test/fake-server.ts",
    ]);
  });
});

describe("building-blocks ESLint rule", () => {
  RuleTester.describe = describe;
  RuleTester.it = it;
  const tester = new RuleTester({
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  });
  const options = [
    {
      root: "/ui",
      baseline: {
        drawings: { "src/routes/chart.tsx": "a chart" },
        files: { "src/routes/legacy.tsx": { button: 1 } },
      },
    },
  ];
  tester.run("use-shared", buildingBlocks.rules["use-shared"], {
    valid: [
      {
        filename: "/ui/src/routes/page.tsx",
        code: 'const a = <Button variant="primary">Save</Button>;',
        options,
      },
      {
        filename: "/ui/src/routes/legacy.tsx",
        code: 'const a = <button className="ui-btn">Old</button>;',
        options,
      },
      {
        filename: "/ui/src/routes/chart.tsx",
        code: "const a = <svg><path d='M0 0' /></svg>;",
        options,
      },
      {
        filename: "/ui/src/routes/page.tsx",
        code: 'const a = <button className="tab">Tab</button>;',
        options,
      },
    ],
    invalid: [
      {
        filename: "/ui/src/routes/page.tsx",
        code: 'const a = <button className={`ui-btn ${x}`} data-size="sm">New</button>;',
        options,
        errors: [{ messageId: "found" }],
      },
      {
        filename: "/ui/src/routes/page.tsx",
        code: 'const a = <section role="alertdialog" />;',
        options,
        errors: [{ messageId: "found" }],
      },
      {
        filename: "/ui/src/routes/page.tsx",
        code: "const a = <svg viewBox='0 0 24 24'><path d='M0 0' /></svg>;",
        options,
        errors: [{ messageId: "found" }],
      },
      {
        filename: "/ui/src/routes/legacy.tsx",
        code: 'const a = <button className="ui-btn">One</button>; const b = <button className="ui-btn">Two</button>;',
        options,
        errors: [{ messageId: "found" }],
      },
      {
        filename: "/ui/src/routes/legacy.tsx",
        code: "const a = <Button>Adopted</Button>;",
        options,
        errors: [{ messageId: "stale" }],
      },
    ],
  });
});
