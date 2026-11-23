import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type { StageAttempt } from "../../api/runs";
import { RuntimeDiagnostics } from "./runtime-diagnostics";
import { useRuntimeDiagnosticRefresh } from "./runtime-diagnostics-refresh";

const headers = {
  "content-type": "application/json",
  "X-Contractor-API-Version": "contractor.public.v1",
  "Retry-After": "0",
};

describe("Runtime diagnostic polling", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => vi.useRealTimers());

  it("backs off completed reads, recovers, and stops on refusal, closure and settlement", async () => {
    const reads: number[] = [];
    let status = 503;
    const api = new PublicAPI(
      {
        uiVersion: "0.1.0",
        supportedApiVersions: ["contractor.public.v1"],
        apiBaseUrl: "http://127.0.0.1:8080",
      },
      async () => {
        reads.push(Date.now());
        return new Response(
          JSON.stringify(
            status === 200
              ? { status: "available", items: [], page: { hasMore: false } }
              : {
                  code:
                    status === 503
                      ? "storage_transaction_conflict"
                      : "not_found",
                  message: "Diagnostic read failed",
                  retryable: status === 503,
                },
          ),
          { status, headers },
        );
      },
    );
    const client = new QueryClient();
    function Body({
      state,
      open,
    }: {
      state: StageAttempt["state"];
      open: boolean;
    }) {
      const attempt = { stageExecutionId: "stage-1", state };
      useRuntimeDiagnosticRefresh("run-1", [attempt]);
      return <RuntimeDiagnostics runId="run-1" attempt={attempt} open={open} />;
    }
    const view = (state: StageAttempt["state"] = "running", open = true) => (
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <Body state={state} open={open} />
        </PublicAPIProvider>
      </QueryClientProvider>
    );
    const rendered = render(view());
    const advance = async (ms: number) => {
      await act(() => vi.advanceTimersByTimeAsync(ms));
    };
    // Three requests belong to each failed read because the existing query
    // retries twice. The outer backoff must count completed reads separately.
    await advance(72_100);
    expect(reads).toHaveLength(15);
    expect(
      [0, 1, 2, 3].map(
        (index) => reads[(index + 1) * 3]! - reads[index * 3 + 2]!,
      ),
    ).toEqual([6000, 12000, 24000, 30000]);
    status = 200;
    await advance(33_000);
    expect(reads).toHaveLength(17);
    expect(reads.at(-1)! - reads.at(-2)!).toBe(3000);
    status = 404;
    await advance(33_000);
    expect(reads).toHaveLength(18);
    const stopped = reads.length;
    rendered.rerender(view("running", false));
    await advance(60_000);
    expect(reads).toHaveLength(stopped);
    status = 200;
    rendered.rerender(view());
    await advance(100);
    const reopened = reads.length;
    expect(reopened).toBe(stopped + 1);
    rendered.rerender(view("succeeded"));
    await advance(60_000);
    expect(reads).toHaveLength(reopened + 1);
    rendered.unmount();
    client.clear();
  });
});
