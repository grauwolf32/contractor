import { describe, expect, it } from "vitest";

import { PublicAPI } from "./client";
import { mutationHeaders } from "./evals";

function api(): PublicAPI {
  const result = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    async () => new Response(null, { status: 204 }),
  );
  result.csrf.replace("a".repeat(43));
  return result;
}

describe("Eval mutation headers", () => {
  it("sends If-Match only with a revision", () => {
    expect(mutationHeaders(api(), "eval-key")).toEqual({
      "Idempotency-Key": "eval-key",
    });
    expect(mutationHeaders(api(), "eval-key", 3)).toEqual({
      "Idempotency-Key": "eval-key",
      "If-Match": '"3"',
    });
  });

  it("rejects header values that fail validation", () => {
    expect(() => mutationHeaders(api(), "eval key")).toThrow(TypeError);
  });
});
