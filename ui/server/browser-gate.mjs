import { spawn, execFileSync } from "node:child_process";
import { readFile, readdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

import { createStaticServer } from "./static-server.mjs";
import { UI_VERSION } from "./runtime-config.mjs";

const uiRoot = fileURLToPath(new URL("..", import.meta.url));
const SUPPORT_DIRECTORY = "support";

/**
 * The places in the browser specs under `e2eDir` that answer a routed
 * request themselves. Every answer goes through e2e/support/mock-api.ts,
 * which checks it against the public contract before serving it (UI-R196);
 * a direct route.fulfill would serve a body the contract never checked.
 */
export async function directFulfilments(e2eDir) {
  const found = [];
  for (const entry of await readdir(e2eDir, {
    recursive: true,
    withFileTypes: true,
  })) {
    if (!entry.isFile() || !entry.name.endsWith(".ts")) continue;
    const path = join(entry.parentPath, entry.name);
    const relative = path.slice(e2eDir.length + 1);
    if (relative.split(/[\\/]/)[0] === SUPPORT_DIRECTORY) continue;
    const lines = (await readFile(path, "utf8")).split("\n");
    lines.forEach((line, index) => {
      if (/\.fulfill\s*\(/.test(line)) found.push(`${relative}:${index + 1}`);
    });
  }
  return found.sort();
}

/**
 * Refuses a spec inventory that differs from the gate lists, and specs that
 * bypass the contract-checked mock answers.
 */
export async function checkBrowserSpecs(e2eDir, listedSpecs) {
  const listed = new Set(listedSpecs);
  const actual = new Set(
    (await readdir(e2eDir))
      .filter((name) => name.endsWith(".spec.ts"))
      .map((name) => `e2e/${name}`),
  );
  const unlisted = [...actual].filter((name) => !listed.has(name));
  const missing = [...listed].filter((name) => !actual.has(name));
  if (unlisted.length !== 0 || missing.length !== 0) {
    throw new Error(
      `browser spec inventory differs: unlisted=${unlisted.join(", ")} missing=${missing.join(", ")}`,
    );
  }
  const direct = await directFulfilments(e2eDir);
  if (direct.length !== 0) {
    throw new Error(
      `browser specs answer requests without the contract check of e2e/support/mock-api.ts: ${direct.join(", ")}`,
    );
  }
}

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

/**
 * Runs one gate attempt that writes its Playwright screenshots and traces to
 * `directory`. Results of an earlier run are removed first and a passing run
 * removes its own, so whatever remains after a run belongs to a run that
 * failed. A second run of the same gate started concurrently in the same
 * checkout removes the first one's results, which makes that run fail rather
 * than pass; run one gate at a time per checkout.
 */
export async function withGateResults(directory, run) {
  await rm(directory, { recursive: true, force: true });
  try {
    await run(directory);
  } catch (error) {
    console.error(
      `browser gate failed; its screenshots and traces, if any, are in ${directory}`,
    );
    throw error;
  }
  await rm(directory, { recursive: true, force: true });
}

const GATE_NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/**
 * Builds and serves an isolated production UI on an OS-selected port and runs
 * `specs` against it. No database or independently started development
 * server is required. The build stays in a temporary directory; Playwright
 * output goes to ui/test-results/<name>, which CI uploads with the stage's
 * reports (.github/workflows/ci.yml), and is kept only when the gate fails.
 */
export async function runBrowserGate(name, label, specs) {
  if (!GATE_NAME.test(name)) throw new Error(`invalid gate name ${name}`);
  const work = await mkdtemp(join(tmpdir(), "contractor-ui-browser-"));
  let server;
  try {
    await withGateResults(
      join(uiRoot, "test-results", name),
      async (output) => {
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
          [
            "pnpm",
            "exec",
            "playwright",
            "test",
            "--reporter=line,json",
            ...specs,
          ],
          {
            cwd: uiRoot,
            stdio: "inherit",
            env: {
              ...process.env,
              CONTRACTOR_UI_E2E_BASE_URL: baseURL,
              CONTRACTOR_UI_E2E_API_URL: baseURL,
              CONTRACTOR_UI_E2E_OUTPUT_DIR: output,
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
      },
    );
  } finally {
    if (server) {
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    }
    await rm(work, { recursive: true, force: true });
  }
}
