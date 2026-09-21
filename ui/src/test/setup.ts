import "@testing-library/jest-dom/vitest";
import { cleanup, configure } from "@testing-library/react";
import { afterEach } from "vitest";

// Route components load lazily; chunk resolution under a full parallel run
// can exceed the default one-second findBy*/waitFor budget.
configure({ asyncUtilTimeout: 4000 });

afterEach(() => {
  cleanup();
});
