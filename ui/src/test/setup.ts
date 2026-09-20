import "@testing-library/jest-dom/vitest";
import { webcrypto } from "node:crypto";

// jsdom does not expose SubtleCrypto; production uses the browser Web Crypto API.
if (globalThis.crypto.subtle === undefined)
  Object.defineProperty(globalThis.crypto, "subtle", {
    value: webcrypto.subtle,
  });
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});
