import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import type { StageAttempt } from "../../api/runs";
import { RuntimeDiagnostics } from "./runtime-diagnostics";
import { useRuntimeDiagnosticRefresh } from "./runtime-diagnostics-refresh";
import { ToolCallDiagnostics } from "./tool-diagnostics";

const headers = {
  "content-type": "application/json",
  "X-Contractor-API-Version": "contractor.public.v1",
};
const attempt = {
  stageExecutionId: "stage-1",
  executionConfig: { variant: "base" as const, agents: { "Scanner.v2": {} } },
};

function page(kind: "runtime" | "tool", value: string) {
  return kind === "runtime"
    ? {
        status: "available",
        items: [
          {
            logicalAgent: "Scanner.v2",
            status: "partial",
            complete: false,
            durationMs: value,
            stopReasonTruncated: false,
            adapters: [],
          },
        ],
        page: { hasMore: false },
      }
    : {
        status: "available",
        participant: "planner",
        complete: false,
        detailTruncated: false,
        items: [
          {
            callReference: "a".repeat(64),
            tool: "probe",
            toolTruncated: false,
            outcome: "succeeded",
            argumentsTruncated: false,
            durationMs: value,
            resultSizeBytes: null,
          },
        ],
        page: { hasMore: false },
      };
}

function fixture(kind: "runtime" | "tool", fetcher: typeof fetch) {
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
    open,
  }: {
    state: StageAttempt["state"];
    snapshot: number;
    open: boolean;
  }) {
    const current = { ...attempt, state };
    useRuntimeDiagnosticRefresh("run-1", [current]);
    return kind === "runtime" ? (
      <RuntimeDiagnostics
        key={snapshot}
        runId="run-1"
        attempt={current}
        open={open}
      />
    ) : (
      <ToolCallDiagnostics
        key={snapshot}
        runId="run-1"
        attempt={current}
        open={open}
      />
    );
  }
  const view = ({
    entered = true,
    state = "running",
    entry = 0,
    snapshot = 0,
    open = true,
  }: {
    entered?: boolean;
    state?: StageAttempt["state"];
    entry?: number;
    snapshot?: number;
    open?: boolean;
  } = {}) => (
    <QueryClientProvider client={client}>
      <PublicAPIProvider api={api}>
        {entered ? (
          <Body key={entry} state={state} snapshot={snapshot} open={open} />
        ) : null}
      </PublicAPIProvider>
    </QueryClientProvider>
  );
  const key =
    kind === "runtime"
      ? queryKeys.runs.runtimeDiagnostics("run-1", "stage-1")
      : [...queryKeys.runs.toolCalls("run-1", "stage-1"), "planner", null];
  return { client, view, key };
}

describe.each(["runtime", "tool"] as const)(
  "%s diagnostic page entry",
  (kind) => {
    it.each(["success", "permanent", "retryable"] as const)(
      "restarts a navigation-cancelled terminal read and handles %s",
      async (outcome) => {
        let count = 0;
        let aborted = 0;
        const fetcher = vi.fn(
          async (input: RequestInfo | URL): Promise<Response> => {
            const request = new Request(input);
            count++;
            if (count === 1)
              return new Response(JSON.stringify(page(kind, "1")), { headers });
            if (count === 2)
              return new Promise((_resolve, reject) => {
                request.signal.addEventListener(
                  "abort",
                  () => {
                    aborted++;
                    reject(
                      new DOMException(
                        "Navigation cancelled read",
                        "AbortError",
                      ),
                    );
                  },
                  { once: true },
                );
              });
            if (outcome === "success")
              return new Response(JSON.stringify(page(kind, "2")), { headers });
            return new Response(
              JSON.stringify({
                code:
                  outcome === "retryable"
                    ? "storage_transaction_conflict"
                    : "conflict",
                message: "Retained read failed",
                retryable: outcome === "retryable",
                requestId: "request-1",
              }),
              {
                status: outcome === "retryable" ? 503 : 409,
                headers: { ...headers, "Retry-After": "0" },
              },
            );
          },
        );
        const { view } = fixture(kind, fetcher);
        const rendered = render(view());
        expect(
          await screen.findByText(kind === "tool" ? "1 ms" : "1", {
            exact: true,
          }),
        ).toBeVisible();
        expect(fetcher).toHaveBeenCalledTimes(1);
        rendered.rerender(view({ state: "failed", snapshot: 1 }));
        await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
        rendered.rerender(view({ entered: false, state: "failed" }));
        await waitFor(() => expect(aborted).toBe(1));
        rendered.rerender(view({ state: "failed", entry: 1 }));
        if (outcome === "success")
          expect(
            await screen.findByText(kind === "tool" ? "2 ms" : "2", {
              exact: true,
            }),
          ).toBeVisible();
        else {
          expect(await screen.findByText("Retained read failed")).toBeVisible();
          expect(screen.getByText(/Previously loaded/i)).toBeVisible();
          expect(
            screen.getByText(kind === "tool" ? "1 ms" : "1", { exact: true }),
          ).toBeVisible();
        }
        expect(fetcher).toHaveBeenCalledTimes(outcome === "retryable" ? 5 : 3);
      },
    );

    it("marks only reopened Run caches stale while closed and ignores same-page remounts", async () => {
      let count = 0;
      const fetcher = vi.fn(
        async () =>
          new Response(JSON.stringify(page(kind, String(++count))), {
            headers,
          }),
      );
      const { view, client, key } = fixture(kind, fetcher);
      const rendered = render(view());
      expect(
        await screen.findByText(kind === "tool" ? "1 ms" : "1", {
          exact: true,
        }),
      ).toBeVisible();
      rendered.rerender(view({ snapshot: 1 }));
      await act(() =>
        client.invalidateQueries({ queryKey: queryKeys.runs.all }),
      );
      expect(fetcher).toHaveBeenCalledTimes(1);
      const other = queryKeys.runs.runtimeDiagnostics("other-run", "stage-1");
      client.setQueryData(other, {
        pages: [page("runtime", "99")],
        pageParams: [undefined],
      });
      const inactive = [
        ...queryKeys.runs.toolCalls("run-1", "stage-1"),
        "worker",
        "Scanner.v2",
      ];
      client.setQueryData(inactive, {
        pages: [
          {
            ...page("tool", "9"),
            participant: "worker",
            logicalAgent: "Scanner.v2",
          },
        ],
        pageParams: [undefined],
      });
      rendered.rerender(view({ entered: false }));
      rendered.rerender(view({ entry: 1, open: false }));
      await waitFor(() =>
        expect(client.getQueryState(key)?.isInvalidated).toBe(true),
      );
      expect(client.getQueryState(inactive)?.isInvalidated).toBe(true);
      expect(client.getQueryState(other)?.isInvalidated).toBe(false);
      expect(fetcher).toHaveBeenCalledTimes(1);
      rendered.rerender(view({ entry: 1, open: true }));
      expect(
        await screen.findByText(kind === "tool" ? "2 ms" : "2", {
          exact: true,
        }),
      ).toBeVisible();
      expect(fetcher).toHaveBeenCalledTimes(2);
    });
  },
);
