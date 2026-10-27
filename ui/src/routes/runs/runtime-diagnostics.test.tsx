import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import type { StageAttempt } from "../../api/runs";
import { RuntimeDiagnostics } from "./runtime-diagnostics";
import { useRuntimeDiagnosticRefresh } from "./runtime-diagnostics-refresh";

describe("Runtime diagnostics disclosure", () => {
  it("keeps facts across live remounts and failed attempt-transition refreshes", async () => {
    const headers = {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    };
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            status: "available",
            items: [
              {
                logicalAgent: "Scanner.v2",
                status: "available",
                complete: true,
                stopReasonTruncated: false,
                adapters: [],
              },
            ],
            page: { hasMore: false },
          }),
          { headers },
        ),
      )
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            code: "conflict",
            message: "Read failed",
            retryable: false,
          }),
          { status: 409, headers },
        ),
      );
    const api = new PublicAPI(
      {
        uiVersion: "0.1.0",
        supportedApiVersions: ["contractor.public.v1"],
        apiBaseUrl: "http://127.0.0.1:8080",
      },
      fetcher,
    );
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    function Body({
      state,
      snapshot,
    }: {
      state: StageAttempt["state"];
      snapshot: number;
    }) {
      const attempt = { stageExecutionId: "stage-1", state };
      useRuntimeDiagnosticRefresh("run-1", [attempt]);
      return (
        <RuntimeDiagnostics
          key={snapshot}
          runId="run-1"
          attempt={attempt}
          open
        />
      );
    }
    const view = (state: StageAttempt["state"], snapshot: number) => (
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <Body state={state} snapshot={snapshot} />
        </PublicAPIProvider>
      </QueryClientProvider>
    );
    const result = render(view("running", 1));
    expect(await screen.findByText("Logical Worker Scanner.v2")).toBeVisible();
    await client.invalidateQueries({ queryKey: queryKeys.runs.all });
    result.rerender(view("running", 2));
    expect(fetcher).toHaveBeenCalledTimes(1);
    result.rerender(view("succeeded", 3));
    expect(
      await screen.findByText(
        "Refresh failed. Previously loaded facts are shown.",
      ),
    ).toBeVisible();
    expect(screen.getByText("Logical Worker Scanner.v2")).toBeVisible();
    expect(fetcher).toHaveBeenCalledTimes(2);
    result.rerender(view("succeeded", 4));
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("loads only when opened, shows exact delivery failures and follows every page", async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const request = new Request(input);
      const next = new URL(request.url).searchParams.has("cursor");
      return new Response(
        JSON.stringify({
          status: "available",
          items: [
            {
              logicalAgent: next ? "second" : "reviewer",
              status: "partial",
              complete: false,
              stopReasonTruncated: false,
              adapters: [
                {
                  ref: "otlp-http@1",
                  operations: "18446744073709551615",
                  failedOperations: "1",
                  flushAttempted: true,
                  flushSucceeded: false,
                  droppedSpans: {
                    queueOverflow: "0",
                    encodingFailed: "0",
                    nonRetryable: "0",
                    retryExhausted: "0",
                    collectorRejected: "0",
                    deadlineExceeded: "0",
                    cancelled: "0",
                    shutdown: "9007199254740993",
                  },
                },
              ],
            },
          ],
          page: next
            ? { hasMore: false }
            : { hasMore: true, nextCursor: "next-page" },
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
    const api = new PublicAPI(
      {
        uiVersion: "0.1.0",
        supportedApiVersions: ["contractor.public.v1"],
        apiBaseUrl: "http://127.0.0.1:8080",
      },
      fetcher,
    );
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const view = (open: boolean) => (
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <RuntimeDiagnostics
            runId="run-1"
            attempt={{ stageExecutionId: "stage-1", state: "succeeded" }}
            open={open}
          />
        </PublicAPIProvider>
      </QueryClientProvider>
    );
    const result = render(view(false));
    expect(fetcher).not.toHaveBeenCalled();
    result.rerender(view(true));
    expect(await screen.findByText("18446744073709551615")).toBeVisible();
    expect(screen.getByText("9007199254740993")).toBeVisible();
    expect(screen.getByText("Flush succeeded")).toBeVisible();
    fireEvent.click(
      screen.getByRole("button", { name: "Load more Runtime diagnostics" }),
    );
    expect(await screen.findByText("Logical Worker second")).toBeVisible();
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Load more Runtime diagnostics" }),
      ).toBeNull(),
    );
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
