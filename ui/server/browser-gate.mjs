import { spawn, execFileSync } from "node:child_process";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

import { createStaticServer } from "./static-server.mjs";
import { UI_VERSION } from "./runtime-config.mjs";

const uiRoot = fileURLToPath(new URL("..", import.meta.url));

// A zero exit status can still mean that Playwright skipped every selected
// case. Require actual passing attempts from every explicitly selected file.
export function validateBrowserReport(report, requiredFiles) {
  if (report.errors?.length || !report.stats || report.stats.expected <= 0) {
    throw new Error(
      "browser report has runner errors or no expected executions",
    );
  }
  const { expected, skipped, unexpected, flaky } = report.stats;
  if (skipped !== 0 || unexpected !== 0 || flaky !== 0) {
    throw new Error(
      `browser report has skipped=${skipped} unexpected=${unexpected} flaky=${flaky}`,
    );
  }
  const counts = new Map();
  for (const file of requiredFiles) {
    const name = basename(file);
    if (counts.has(name)) throw new Error(`duplicate browser spec ${name}`);
    counts.set(name, 0);
  }
  let total = 0;
  function visit(suites) {
    for (const suite of suites ?? []) {
      for (const spec of suite.specs ?? []) {
        if (!spec.ok || !spec.file || !spec.tests?.length) {
          throw new Error(
            `browser spec ${spec.title} has no successful executions`,
          );
        }
        for (const test of spec.tests) {
          if (
            test.expectedStatus !== "passed" ||
            test.status !== "expected" ||
            !test.results?.length ||
            test.results.some(
              (result) => result.status !== "passed" || result.errors?.length,
            )
          ) {
            throw new Error(
              `browser spec ${spec.title} has a non-passing attempt`,
            );
          }
          total++;
          const name = basename(spec.file);
          if (counts.has(name)) counts.set(name, counts.get(name) + 1);
        }
      }
      visit(suite.suites);
    }
  }
  visit(report.suites);
  if (total !== expected) {
    throw new Error(
      `browser report expected ${expected} tests but executed ${total}`,
    );
  }
  for (const [file, count] of counts) {
    if (count === 0)
      throw new Error(`browser spec ${file} has no executed tests`);
  }
  return total;
}

export async function runBrowserGate(label, specs) {
  // Build and serve an isolated production UI on an OS-selected port. No
  // database or independently started development server is required.
  const work = await mkdtemp(join(tmpdir(), "contractor-ui-browser-"));
  let server;
  try {
    const distDir = join(work, "dist");
    execFileSync(
      "corepack",
      ["pnpm", "exec", "vite", "build", "--outDir", distDir],
      {
        cwd: uiRoot,
        stdio: "inherit",
      },
    );
    server = await createStaticServer({
      distDir,
      runtimeConfig: {
        uiVersion: UI_VERSION,
        supportedApiVersions: ["contractor.public.v1"],
        apiBaseUrl: "http://127.0.0.1:1",
      },
    });
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", resolve);
    });
    const reportPath = join(work, "browser-report.json");
    const baseURL = `http://127.0.0.1:${server.address().port}`;
    const child = spawn(
      "corepack",
      ["pnpm", "exec", "playwright", "test", "--reporter=line,json", ...specs],
      {
        cwd: uiRoot,
        stdio: "inherit",
        env: {
          ...process.env,
          CONTRACTOR_UI_E2E_BASE_URL: baseURL,
          CONTRACTOR_UI_E2E_API_URL: baseURL,
          CONTRACTOR_UI_E2E_OUTPUT_DIR: join(work, "results"),
          PLAYWRIGHT_JSON_OUTPUT_FILE: reportPath,
        },
      },
    );
    await new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("exit", (code, signal) => {
        if (code === 0) resolve();
        else reject(new Error(`${label} gate exited: ${signal ?? code}`));
      });
    });
    const count = validateBrowserReport(
      JSON.parse(await readFile(reportPath, "utf8")),
      specs,
    );
    console.log(
      `${label} gate verified ${count} passing browser tests across ${specs.length} specs`,
    );
  } finally {
    if (server) {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    }
    await rm(work, { recursive: true, force: true });
  }
}
