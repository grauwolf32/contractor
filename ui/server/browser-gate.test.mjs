import assert from "node:assert/strict";
import { test } from "node:test";

import { validateBrowserReport } from "./browser-gate.mjs";

function report() {
  return {
    suites: [
      {
        specs: [
          {
            title: "nested browser journey",
            file: "ui/e2e/catalog.spec.ts",
            ok: true,
            tests: [
              {
                expectedStatus: "passed",
                status: "expected",
                results: [{ status: "passed", errors: [] }],
              },
            ],
          },
        ],
      },
    ],
    errors: [],
    stats: { expected: 1, skipped: 0, unexpected: 0, flaky: 0 },
  };
}

test("browser gate requires passing attempts in every selected file", () => {
  assert.equal(validateBrowserReport(report(), ["e2e/catalog.spec.ts"]), 1);
  assert.throws(
    () =>
      validateBrowserReport(report(), [
        "e2e/catalog.spec.ts",
        "e2e/dialogs.spec.ts",
      ]),
    /dialogs.spec.ts has no executed tests/,
  );

  const skipped = report();
  skipped.stats.skipped = 1;
  assert.throws(
    () => validateBrowserReport(skipped, ["e2e/catalog.spec.ts"]),
    /skipped=1/,
  );

  const failedAttempt = report();
  failedAttempt.suites[0].specs[0].tests[0].results.unshift({
    status: "failed",
    errors: [{ message: "first attempt failed" }],
  });
  assert.throws(
    () => validateBrowserReport(failedAttempt, ["e2e/catalog.spec.ts"]),
    /non-passing attempt/,
  );
});
