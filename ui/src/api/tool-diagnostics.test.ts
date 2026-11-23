import { describe, expect, it, vi } from "vitest";

import fixture from "../../../api/testdata/public/tool-diagnostic-values.json";
import { PublicAPI } from "./client";
import { getToolCallDiagnostics } from "./tool-diagnostics";

const config = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};
function response(item: Record<string, unknown>) {
  return {
    status: "available",
    participant: "planner",
    complete: true,
    detailTruncated: false,
    items: [
      {
        callReference: "a".repeat(64),
        tool: "probe",
        toolTruncated: false,
        outcome: "succeeded",
        argumentsTruncated: false,
        argumentsJson: fixture.argumentsJson,
        durationMs: null,
        resultSizeBytes: null,
        ...item,
      },
    ],
    page: { hasMore: false },
  };
}
function apiFor(value: unknown) {
  return new PublicAPI(
    config,
    vi.fn(
      async () =>
        new Response(JSON.stringify(value), {
          status: 200,
          headers: {
            "content-type": "application/json",
            "X-Contractor-API-Version": "contractor.public.v1",
          },
        }),
    ),
  );
}

describe("retained tool-call diagnostic values", () => {
  it.each(fixture.cases)(
    "reads optional decimal $value losslessly",
    async ({ value, accepted }) => {
      const api = apiFor(
        response({ durationMs: value, resultSizeBytes: value }),
      );
      if (accepted) {
        const page = await getToolCallDiagnostics(
          api,
          "run-1",
          "stage-1",
          "planner",
          undefined,
        );
        expect(page.items[0]?.durationMs).toBe(value);
        expect(page.items[0]?.resultSizeBytes).toBe(value);
        expect(page.items[0]?.argumentsJson).toBe(fixture.argumentsJson);
      } else {
        await expect(
          getToolCallDiagnostics(api, "run-1", "stage-1", "planner", undefined),
        ).rejects.toThrow();
      }
    },
  );
  it.each([
    { tool: "я".repeat(65) },
    { argumentsJson: JSON.stringify({ text: "я".repeat(2048) }) },
    {
      outcome: "failed",
      error: {
        code: "tool_call_failed",
        message: "я".repeat(2049),
        messageTruncated: false,
      },
    },
    { outcome: "invented" },
    {
      outcome: "failed",
      error: {
        code: "tool_call_failed",
        message: "failed",
        messageTruncated: false,
        providerBody: "forbidden",
      },
    },
  ])("rejects malformed or oversized public fields", async (item) => {
    await expect(
      getToolCallDiagnostics(
        apiFor(response(item)),
        "run-1",
        "stage-1",
        "planner",
        undefined,
      ),
    ).rejects.toThrow();
  });
  it("passes selected participant, cursor and abort signal without inventing unknown metrics", async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const request = new Request(input);
      const url = new URL(request.url);
      expect(url.searchParams.get("participant")).toBe("worker");
      expect(url.searchParams.get("logicalAgent")).toBe("Scanner.v2");
      expect(url.searchParams.get("cursor")).toBe("sealed");
      return new Response(
        JSON.stringify({
          ...response({}),
          participant: "worker",
          logicalAgent: "Scanner.v2",
        }),
        {
          status: 200,
          headers: {
            "content-type": "application/json",
            "X-Contractor-API-Version": "contractor.public.v1",
          },
        },
      );
    });
    const page = await getToolCallDiagnostics(
      new PublicAPI(config, fetcher),
      "run-1",
      "stage-1",
      "worker",
      "Scanner.v2",
      "sealed",
      new AbortController().signal,
    );
    expect(page.items[0]?.durationMs).toBeNull();
    expect(page.items[0]?.resultSizeBytes).toBeNull();
  });
});
