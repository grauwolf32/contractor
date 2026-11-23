import { defineConfig } from "@playwright/test";
import base from "./playwright.config";

export default defineConfig({
  ...base,
  testMatch: "**/audit-composition.production.ts",
  timeout: 10 * 60 * 1000,
  use: { ...base.use, trace: "off" },
});
