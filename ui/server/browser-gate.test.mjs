import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  checkBrowserSpecs,
  directFulfilments,
  validateBrowserReport,
} from "./browser-gate.mjs";

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

test("browser specs answer requests only through the contract-checked mock", async () => {
  const e2eDir = await mkdtemp(join(tmpdir(), "contractor-e2e-probe-"));
  try {
    await mkdir(join(e2eDir, "support"));
    await writeFile(
      join(e2eDir, "support", "mock-api.ts"),
      "await route.fulfill({ status: 204 });\n",
    );
    await writeFile(
      join(e2eDir, "probe.spec.ts"),
      "test();\nawait route.fulfill({ json: {} });\n",
    );
    assert.deepEqual(await directFulfilments(e2eDir), ["probe.spec.ts:2"]);
    await assert.rejects(
      checkBrowserSpecs(e2eDir, ["e2e/probe.spec.ts"]),
      /without the contract check .*probe.spec.ts:2/,
    );
    await assert.rejects(
      checkBrowserSpecs(e2eDir, []),
      /unlisted=e2e\/probe.spec.ts/,
    );
  } finally {
    await rm(e2eDir, { recursive: true, force: true });
  }
});

test("the browser specs serve no unchecked answer", async () => {
  const e2eDir = fileURLToPath(new URL("../e2e", import.meta.url));
  assert.deepEqual(await directFulfilments(e2eDir), []);
});
