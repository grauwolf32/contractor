import { defineConfig, devices } from "@playwright/test";

const baseURL =
  process.env.CONTRACTOR_UI_E2E_BASE_URL ?? "http://127.0.0.2:4173";
const outputDir =
  process.env.CONTRACTOR_UI_E2E_OUTPUT_DIR ?? "test-results/ui-stack";
const browserHostnames = [
  new URL(baseURL).hostname,
  new URL(process.env.CONTRACTOR_UI_E2E_API_URL ?? baseURL).hostname,
].filter((hostname, index, values) => {
  return (
    values.indexOf(hostname) === index && hostname.endsWith(".contractor.test")
  );
});
const hostResolverRules = browserHostnames
  .map((hostname) => `MAP ${hostname} 127.0.0.1`)
  .join(",");

export default defineConfig({
  testDir: "./e2e",
  outputDir,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 240_000,
  expect: { timeout: 20_000 },
  reporter: [["line"]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
    ignoreHTTPSErrors: true,
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    screenshot: "only-on-failure",
    launchOptions: {
      args:
        hostResolverRules === ""
          ? []
          : [`--host-resolver-rules=${hostResolverRules}`],
    },
    trace: {
      mode: "on",
      screenshots: false,
      snapshots: false,
      sources: false,
      attachments: false,
    },
    video: "off",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
