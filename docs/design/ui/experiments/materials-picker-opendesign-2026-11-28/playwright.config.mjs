import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
const require = createRequire(
  new URL("../../../../../ui/package.json", import.meta.url),
);
const { defineConfig } = require("@playwright/test");
export default defineConfig({
  testDir: ".",
  testMatch: "review.spec.mjs",
  workers: 1,
  retries: 0,
  timeout: 20000,
  expect: { timeout: 5000 },
  reporter: [["line"]],
  outputDir: fileURLToPath(
    new URL(
      "../../../../../ui/test-results/materials-picker-design",
      import.meta.url,
    ),
  ),
  use: {
    baseURL: process.env.MATERIALS_PREVIEW_URL || "http://127.0.0.1:17574",
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
  },
});
