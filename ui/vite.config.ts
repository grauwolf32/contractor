import { readFileSync } from "node:fs";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

interface PackageMetadata {
  version: string;
}

const packageMetadata = JSON.parse(
  readFileSync(new URL("./package.json", import.meta.url), "utf8"),
) as PackageMetadata;

export default defineConfig({
  plugins: [react()],
  define: {
    __CONTRACTOR_UI_VERSION__: JSON.stringify(packageMetadata.version),
    __CONTRACTOR_SUPPORTED_API_VERSIONS__: JSON.stringify([
      "contractor.public.v1",
    ]),
  },
  build: {
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              // Stable vendor chunk: framework code that every route shares and
              // that changes only on dependency bumps, so it caches across
              // application releases.
              name: "vendor",
              test: (id: string) =>
                /[\\/]node_modules[\\/](react|react-dom|react-router|scheduler|@tanstack[\\/](?:query-core|react-query))[\\/]/.test(
                  id,
                ) &&
                // react-dom's server renderers are only reached from the lazy
                // LikeC4 preview; keep them out of the eager payload.
                !/react-dom[\\/](server|cjs[\\/]react-dom-server)/.test(id),
            },
          ],
        },
      },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
    clearMocks: true,
  },
});
