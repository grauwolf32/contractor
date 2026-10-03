import { describe, expect, it } from "vitest";

import { invalidAPIResponse, PublicAPIError, requireData } from "./error";

describe("response helpers", () => {
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

    expect(error).toBeInstanceOf(PublicAPIError);
    expect(error).toMatchObject({
      status: 200,
      code: "invalid_api_response",
      message: "Server returned an invalid page",
    });
  });
});
