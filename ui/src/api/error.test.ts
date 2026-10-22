import { describe, expect, it } from "vitest";

import {
  invalidAPIResponse,
  InvalidResponseError,
  PublicAPIError,
  requireData,
} from "./error";

describe("response helpers", () => {
  it("carries Server Retry-After seconds into finite read retries", () => {
    expect(() =>
      requireData({
        response: new Response(null, {
          status: 503,
          headers: { "Retry-After": "7" },
        }),
        error: {
          code: "storage_transaction_conflict",
          message: "Retry",
          retryable: true,
        },
      }),
    ).toThrow(expect.objectContaining({ retryAfterMs: 7000, retryable: true }));
  });
  it("returns data and throws the carried error envelope", () => {
    const response = new Response(null, { status: 409 });

    expect(requireData({ data: { id: "a" }, response })).toEqual({ id: "a" });
    expect(() =>
      requireData({
        error: { code: "conflict", message: "Changed", retryable: false },
        response,
      }),
    ).toThrow(
      expect.objectContaining({
        status: 409,
        code: "conflict",
        message: "Changed",
      }),
    );
  });

  it("reports contract violations as invalid API responses", () => {
    const error = invalidAPIResponse(200, "Server returned an invalid page");

    expect(error).toBeInstanceOf(InvalidResponseError);
    expect(error).toBeInstanceOf(PublicAPIError);
    expect(error).toMatchObject({
      status: 200,
      code: "invalid_api_response",
      detail: "Server returned an invalid page",
    });
  });
});
