// Which UI sources are production code and which are test code, for the
// dead-code gates (css-gate.mjs, export-gate.mjs). Test code: unit tests
// (*.test.ts, *.test.tsx), the shared helpers in src/test/, area test helper
// files (test-support.tsx, test-fixtures.ts, check-test-support.tsx),
// testdata/ directories and the Playwright specs under e2e/. Generated code
// and type declarations are neither.

import { relative, sep } from "node:path";

/** "prod", "test" or "skip" for a path inside the UI package rooted at `ui`. */
export function classify(ui, fileName) {
  const path = relative(ui, fileName).split(sep).join("/");
  if (path.startsWith("e2e/") || path === "playwright.config.ts") return "test";
  if (path.startsWith("server/"))
    return path.endsWith(".test.mjs") ? "test" : "prod";
  if (path === "index.html") return "prod";
  if (!path.startsWith("src/")) return "skip";
  if (path.startsWith("src/api/generated/") || path.endsWith(".d.ts"))
    return "skip";
  if (
    path.startsWith("src/test/") ||
    path.includes("/testdata/") ||
    /\.test\.tsx?$/.test(path) ||
    /(^|[/-])test-(support|fixtures)\.tsx?$/.test(path)
  )
    return "test";
  return "prod";
}
