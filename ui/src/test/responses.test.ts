import { describe, expect, it } from "vitest";

import { conflict, errorRetryable, failure } from "./responses";

describe("registered test error responses", () => {
  it("uses the Server's retryability for default failures and conflicts", async () => {
    expect(await failure(500).json()).toMatchObject({
      code: "internal_error",
      retryable: true,
    });
    expect(await conflict().json()).toMatchObject({
      code: "conflict",
      retryable: true,
    });
    expect(await failure(404).json()).toMatchObject({
      code: "not_found",
      retryable: false,
    });
    expect(errorRetryable("storage_transaction_conflict")).toBe(true);
    expect(errorRetryable("eval_invalid")).toBe(false);
  });

  it("refuses codes missing from the Server registry", () => {
    expect(() => errorRetryable("unregistered_test_error")).toThrow(
      "Unregistered public API error code",
    );
    expect(() => errorRetryable("constructor")).toThrow(
      "Unregistered public API error code",
    );
  });
});
