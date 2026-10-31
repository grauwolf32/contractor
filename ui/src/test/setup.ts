import "@testing-library/jest-dom/vitest";
import { webcrypto } from "node:crypto";
import { cleanup, configure } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Browser startup verifies the same SHA-256 policy digest as the Runtime.
vi.stubGlobal("crypto", webcrypto);

import { discardSessionRunDrafts } from "../run-drafts/session-stores";

// Route components load lazily; chunk resolution under a full parallel run
// can exceed the default one-second findBy*/waitFor budget.
configure({ asyncUtilTimeout: 4000 });

afterEach(() => {
  cleanup();
  // Session drafts are retained per owner for the page lifetime; each test
  // starts a fresh page.
  discardSessionRunDrafts();
});
