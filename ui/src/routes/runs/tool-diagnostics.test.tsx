import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import fixture from "../../../../api/testdata/public/tool-diagnostic-values.json";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import type { StageAttempt } from "../../api/runs";
import { useRuntimeDiagnosticRefresh } from "./runtime-diagnostics-refresh";
import { ToolCallDiagnostics } from "./tool-diagnostics";

describe("retained Tool calls panel", () => {
  it("selects participants, follows pages, preserves choice and facts across remounts and failed state refresh", async () => {
    let fail = false;
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const request = new Request(input);
      const url = new URL(request.url);
      if (fail) {
        return new Response(
          JSON.stringify({
            code: "conflict",
            message: "Diagnostic data changed",
            retryable: false,
            requestId: "request-1",
          }),
          {
            status: 409,
            headers: {
              "content-type": "application/json",
              "X-Contractor-API-Version": "contractor.public.v1",
            },
          },
        );
      }
      const worker = url.searchParams.get("participant") === "worker";
      const next = url.searchParams.has("cursor");
      return new Response(
        JSON.stringify({
          status: "available",
          participant: worker ? "worker" : "planner",
          ...(worker ? { logicalAgent: "Scanner.v2" } : {}),
          complete: !worker,
          detailTruncated: worker,
          items: [
            {
              callReference: (next ? "b" : worker ? "c" : "a").repeat(64),
              tool: worker
                ? "worker_probe"
                : next
                  ? "planner_second"
                  : "planner_probe",
              toolTruncated: false,
              outcome: "succeeded",
              argumentsJson: fixture.argumentsJson,
              argumentsTruncated: worker,
              durationMs: "9223372036854775807",
              resultSizeBytes: "9007199254740993",
            },
          ],
          page:
            !worker && !next
              ? { hasMore: true, nextCursor: "next-page" }
              : { hasMore: false },
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
    function Stable({
      state,
      revision,
      open,
    }: {
      state: StageAttempt["state"];
      revision: number;
      open: boolean;
    }) {
      const attempt = {
        stageExecutionId: "stage-1",
        state,
        executionConfig: {
          variant: "base" as const,
          agents: { "Scanner.v2": {} },
        },
      };
      useRuntimeDiagnosticRefresh("run-1", [attempt]);
      return (
        <ToolCallDiagnostics
          key={revision}
          runId="run-1"
          attempt={attempt}
          open={open}
        />
      );
    }
    const view = (
      state: StageAttempt["state"],
      revision: number,
      open = true,
    ) => (
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <Stable state={state} revision={revision} open={open} />
        </PublicAPIProvider>
      </QueryClientProvider>
    );
    const rendered = render(view("running", 0, false));
    expect(fetcher).not.toHaveBeenCalled();
    rendered.rerender(view("running", 0));
    expect(await screen.findByText("planner_probe")).toBeVisible();
    fireEvent.click(
      screen.getByRole("button", { name: "Load more tool calls" }),
    );
    expect(await screen.findByText("planner_second")).toBeVisible();
    fireEvent.change(screen.getByLabelText("Tool-call participant"), {
      target: { value: "worker:Scanner.v2" },
    });
    expect(await screen.findByText("worker_probe")).toBeVisible();
    expect(
      screen.getByText("The selected report is incomplete."),
    ).toBeVisible();
    fireEvent.click(screen.getByText("Sanitized arguments"));
    expect(screen.getByText(fixture.argumentsJson)).toBeVisible();
    expect(screen.getByText("9007199254740993", { exact: true })).toBeVisible();
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(3));
    rendered.rerender(view("running", 1));
    expect(screen.getByLabelText("Tool-call participant")).toHaveValue(
      "worker:Scanner.v2",
    );
    await act(() => client.invalidateQueries({ queryKey: queryKeys.runs.all }));
    expect(fetcher).toHaveBeenCalledTimes(3);
    fail = true;
    rendered.rerender(view("succeeded", 2));
    expect(
      await screen.findByText(
        "Previously loaded call detail remains visible; refresh failed.",
      ),
    ).toBeVisible();
    expect(screen.getByText("worker_probe")).toBeVisible();
    expect(fetcher).toHaveBeenCalledTimes(4);
  });
});
