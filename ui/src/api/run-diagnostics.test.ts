import { describe, expect, it, vi } from "vitest";

import decimals from "../../../api/testdata/public/runtime-diagnostic-decimals.json";
import { PublicAPI } from "./client";
import { getRuntimeDiagnostics } from "./run-diagnostics";

const config = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

describe("retained Runtime diagnostics", () => {
  it.each([
    { count: 2048, accepted: true },
    { count: 2049, accepted: false },
  ])(
    "checks the UTF-8 stop-reason bound at $count multibyte characters",
    async ({ count, accepted }) => {
      const api = new PublicAPI(
        config,
        vi.fn(
          async () =>
            new Response(
              JSON.stringify({
                status: "available",
                items: [
                  {
                    logicalAgent: "reviewer",
                    status: "available",
                    complete: true,
                    stopReason: "я".repeat(count),
                    stopReasonTruncated: false,
                    adapters: [],
                  },
                ],
                page: { hasMore: false },
              }),
              {
                status: 200,
                headers: {
                  "content-type": "application/json",
                  "X-Contractor-API-Version": "contractor.public.v1",
                },
              },
            ),
        ),
      );
      if (accepted) {
        expect(
          (await getRuntimeDiagnostics(api, "run-1", "stage-1")).items[0]
            ?.stopReason,
        ).toHaveLength(count);
      } else {
        await expect(
          getRuntimeDiagnostics(api, "run-1", "stage-1"),
        ).rejects.toThrow();
      }
    },
  );
  it.each(decimals.cases)(
    "reads counter $value without browser rounding",
    async ({ value, accepted }) => {
      const api = new PublicAPI(
        config,
        vi.fn(
          async () =>
            new Response(
              JSON.stringify({
                status: "available",
                items: [
                  {
                    logicalAgent: "reviewer",
                    status: "available",
                    complete: true,
                    stopReasonTruncated: false,
                    adapters: [
                      {
                        ref: "otlp-http@1",
                        operations: value,
                        failedOperations: "0",
                      },
                    ],
                  },
                ],
                page: { hasMore: false },
              }),
              {
                status: 200,
                headers: {
                  "content-type": "application/json",
                  "X-Contractor-API-Version": "contractor.public.v1",
                },
              },
            ),
        ),
      );
      if (accepted) {
        const page = await getRuntimeDiagnostics(api, "run-1", "stage-1");
        expect(page.items[0]?.adapters[0]?.operations).toBe(value);
      } else {
        await expect(
          getRuntimeDiagnostics(api, "run-1", "stage-1"),
        ).rejects.toThrow();
      }
    },
  );

  it("passes the page cursor and cancellation signal, preserving unknown facts", async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const request = new Request(input);
      expect(new URL(request.url).searchParams.get("cursor")).toBe(
        "sealed-cursor",
      );
      return new Response(
        JSON.stringify({
          status: "pending",
          reason: "report_missing",
          items: [],
          page: { hasMore: false },
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
    const api = new PublicAPI(config, fetcher);
    const page = await getRuntimeDiagnostics(
      api,
      "run-1",
      "stage-1",
      "sealed-cursor",
      new AbortController().signal,
    );
    expect(page.status).toBe("pending");
    expect(page.items).toEqual([]);
  });
});
