import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PublicAPIProvider } from "../../api/context";
import { EVAL_POLL_MS } from "../../api/evals";
import { createApplicationQueryClient } from "../../app/query-client";
import { createEvalFixture } from "../../test/evals-fixture";
import { testAPI } from "../../test/render";
import { EvalListRoute } from "./list";
import { evalExperimentPollInterval, evalListPollInterval } from "./polling";
import { useEvalExperiment } from "./queries";
import { evalServer } from "./test-support";

afterEach(() => vi.useRealTimers());

/** A query that has read `data` once, as refetchInterval sees it. */
function read<T>(data: T) {
  return {
    state: {
      data,
      error: null,
      status: "success" as const,
      dataUpdateCount: 1,
      errorUpdateCount: 0,
    },
  };
}

function DetailProbe() {
  const experiment = useEvalExperiment("experiment-1");
  return <p>{experiment.data?.freshness ?? "loading"}</p>;
}

function reads(fixture: ReturnType<typeof createEvalFixture>, path: string) {
  return fixture.state.requests.filter(
    (request) => request.method === "GET" && request.path === path,
  ).length;
}

describe("Eval polling", () => {
  it("keeps active, stale and deletion-pending detail views on short polling", () => {
    expect(evalExperimentPollInterval(read({ state: "draft" }))).toBe(false);
    for (const state of [
      "preparing",
      "running",
      "settling",
      "cancelling",
    ] as const) {
      expect(
        evalExperimentPollInterval(read({ state, freshness: "current" })),
      ).toBe(EVAL_POLL_MS);
    }
    for (const freshness of ["stale", "pending"] as const) {
      expect(
        evalExperimentPollInterval(read({ state: "finished", freshness })),
      ).toBe(EVAL_POLL_MS);
    }
    expect(
      evalExperimentPollInterval(
        read({
          state: "cancelled",
          freshness: "current",
          deletionRequestedAt: "2026-09-01T00:00:00Z",
        }),
      ),
    ).toBe(EVAL_POLL_MS);
    expect(
      evalExperimentPollInterval(
        read({ state: "finished", freshness: "current" }),
      ),
    ).toBeGreaterThan(EVAL_POLL_MS);
  });

  it("backs off list page one only while all rows are terminal and current", () => {
    const settled = [
      { state: "finished" as const, freshness: "current" as const },
      { state: "cancelled" as const, freshness: "current" as const },
    ];
    expect(evalListPollInterval(read({ items: settled }))).toBeGreaterThan(
      EVAL_POLL_MS,
    );
    expect(
      evalListPollInterval(
        read({ items: [...settled, { state: "running" as const }] }),
      ),
    ).toBe(EVAL_POLL_MS);
    expect(
      evalListPollInterval(
        read({
          items: [
            ...settled,
            { state: "finished" as const, freshness: "stale" as const },
          ],
        }),
      ),
    ).toBe(EVAL_POLL_MS);
  });

  it("waits longer after each failed read of a running experiment", () => {
    const failed = {
      state: {
        data: { state: "running" as const },
        error: new Error("unavailable"),
        status: "error" as const,
        dataUpdateCount: 1,
        errorUpdateCount: 1,
      },
    };
    expect(evalExperimentPollInterval(failed)).toBe(2 * EVAL_POLL_MS);
  });

  it("refetches a settled detail after invalidation and polls until its view is current", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.experiment.state = "finished";
    fixture.state.experiment.freshness = "current";
    const client = createApplicationQueryClient();
    const view = render(
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={testAPI(evalServer(fixture)).api}>
          <DetailProbe />
        </PublicAPIProvider>
      </QueryClientProvider>,
    );
    const path = "/v1/eval-experiments/experiment-1";
    try {
      await screen.findByText("current");
      expect(reads(fixture, path)).toBe(1);
      await act(async () => vi.advanceTimersByTimeAsync(10_000));
      expect(reads(fixture, path)).toBe(1);

      fixture.state.experiment.freshness = "stale";
      await act(async () => client.invalidateQueries({ queryKey: ["evals"] }));
      await screen.findByText("stale");
      expect(reads(fixture, path)).toBe(2);
      fixture.state.experiment.freshness = "current";
      await act(async () => vi.advanceTimersByTimeAsync(EVAL_POLL_MS));
      await screen.findByText("current");
      expect(reads(fixture, path)).toBe(3);
    } finally {
      view.unmount();
      client.clear();
    }
  });

  it("backs off a settled list and resumes short polling when an active row appears", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.experiment.state = "finished";
    fixture.state.experiment.freshness = "current";
    const client = createApplicationQueryClient();
    const view = render(
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={testAPI(evalServer(fixture)).api}>
          <MemoryRouter initialEntries={["/evals"]}>
            <EvalListRoute />
          </MemoryRouter>
        </PublicAPIProvider>
      </QueryClientProvider>,
    );
    const path = "/v1/eval-experiments";
    try {
      await screen.findByText("Trace instructions");
      expect(reads(fixture, path)).toBe(1);
      await act(async () => vi.advanceTimersByTimeAsync(10_000));
      expect(reads(fixture, path)).toBe(1);

      fixture.state.experiment.state = "running";
      await act(async () =>
        client.invalidateQueries({ queryKey: ["evals", "list"] }),
      );
      await waitFor(() => expect(reads(fixture, path)).toBe(2));
      await act(async () => vi.advanceTimersByTimeAsync(EVAL_POLL_MS));
      expect(reads(fixture, path)).toBe(3);
    } finally {
      view.unmount();
      client.clear();
    }
  });
});
