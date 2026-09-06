import { spawn, execFileSync } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { createStaticServer } from "./static-server.mjs";

// Test the production static router, including direct /settings navigation.
// Isolate build/output directories and bind an OS-selected port so this gate
// does not replace a running development server or another build's dist.
const uiRoot = fileURLToPath(new URL("..", import.meta.url));
const work = await mkdtemp(join(tmpdir(), "contractor-git-ui-"));
let server;
try {
  const distDir = join(work, "dist");
  execFileSync(
    "corepack",
    ["pnpm", "exec", "vite", "build", "--outDir", distDir],
    { cwd: uiRoot, stdio: "inherit" },
  );
  server = await createStaticServer({
    distDir,
    runtimeConfig: {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:1",
    },
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const child = spawn(
    "corepack",
    ["pnpm", "exec", "playwright", "test", "e2e/git-artifacts.spec.ts"],
    {
      cwd: uiRoot,
      stdio: "inherit",
      env: {
        ...process.env,
        CONTRACTOR_UI_E2E_BASE_URL: `http://127.0.0.1:${server.address().port}`,
        CONTRACTOR_UI_E2E_OUTPUT_DIR: join(work, "results"),
      },
    },
  );
  await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`Git browser gate exited: ${signal ?? code}`));
    });
  });
} finally {
  if (server) {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
  await rm(work, { recursive: true, force: true });
}
