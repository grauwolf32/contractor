import eslint from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

import buildingBlocks, {
  readBaseline,
} from "./scripts/eslint-building-blocks.mjs";

export default tseslint.config(
  {
    ignores: [
      "dist/",
      "node_modules/",
      "coverage/",
      "playwright-report/",
      "test-results/",
      "src/api/generated/",
    ],
  },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  reactHooks.configs.flat.recommended,
  {
    files: ["src/**/*.{ts,tsx}", "vite.config.ts"],
    languageOptions: {
      globals: globals.browser,
    },
    plugins: {
      "react-refresh": reactRefresh,
    },
    rules: {
      "react-refresh/only-export-components": [
        "warn",
        {
          allowConstantExport: true,
          allowExportNames: ["usePublicAPI", "useSession"],
        },
      ],
    },
  },
  {
    // Pages use the shared Button, AlertDialog and Icon (src/ui/README.md);
    // scripts/building-blocks-baseline.json lists what is left to adopt.
    files: ["src/**/*.tsx"],
    ignores: [
      "src/ui/**",
      "src/app/**",
      "src/test/**",
      "src/**/*.test.tsx",
      "src/**/*test-support.tsx",
    ],
    plugins: { "building-blocks": buildingBlocks },
    rules: {
      "building-blocks/use-shared": [
        "error",
        {
          root: import.meta.dirname,
          baseline: readBaseline(
            new URL("./scripts/building-blocks-baseline.json", import.meta.url),
          ),
        },
      ],
    },
  },
  {
    files: ["server/**/*.mjs", "scripts/**/*.mjs", "eslint.config.js"],
    languageOptions: {
      globals: globals.node,
    },
  },
  {
    files: ["e2e/**/*.ts", "playwright.config.ts"],
    languageOptions: {
      globals: globals.node,
    },
  },
);
