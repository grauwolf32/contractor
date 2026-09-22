import "@testing-library/jest-dom/vitest";
import { cleanup, configure } from "@testing-library/react";
import { afterEach } from "vitest";

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
