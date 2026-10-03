import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { EVAL_POLL_MS } from "../../api/evals";
import { createApplicationQueryClient } from "../../app/query-client";
import {
  createEvalFixture,
  EVAL_FIXTURE_ORIGIN,
} from "../../test/evals-fixture";
import { EvalListRoute } from "./list";
import {
  EVAL_SETTLED_POLL_MS,
  evalExperimentPollInterval,
  evalListPollInterval,
} from "./polling";
import { useEvalExperiment } from "./queries";

afterEach(() => vi.useRealTimers());

function transport(fixture: ReturnType<typeof createEvalFixture>) {
  return new PublicAPI(
    {
      uiVersion: "0.4.0",
      apiBaseUrl: EVAL_FIXTURE_ORIGIN,
      supportedApiVersions: ["contractor.public.v1"],
    },
    async (input) =>
      fixture.fetch(input instanceof Request ? input : new Request(input)),
  );
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
    expect(evalExperimentPollInterval({ state: "draft" })).toBe(false);
    for (const state of [
      "preparing",
      "running",
      "settling",
      "cancelling",
    ] as const) {
      expect(evalExperimentPollInterval({ state, freshness: "current" })).toBe(
        EVAL_POLL_MS,
      );
    }
    for (const freshness of ["stale", "pending"] as const) {
      expect(evalExperimentPollInterval({ state: "finished", freshness })).toBe(
        EVAL_POLL_MS,
      );
    }
    expect(
      evalExperimentPollInterval({
        state: "cancelled",
        freshness: "current",
        deletionRequestedAt: "2026-09-01T00:00:00Z",
      }),
    ).toBe(EVAL_POLL_MS);
    expect(
      evalExperimentPollInterval({ state: "finished", freshness: "current" }),
    ).toBe(EVAL_SETTLED_POLL_MS);
  });

  it("backs off list page one only while all rows are terminal and current", () => {
    const settled = [
      { state: "finished" as const, freshness: "current" as const },
      { state: "cancelled" as const, freshness: "current" as const },
    ];
    expect(evalListPollInterval(settled, false)).toBe(EVAL_SETTLED_POLL_MS);
    expect(
      evalListPollInterval([...settled, { state: "running" }], false),
    ).toBe(EVAL_POLL_MS);
    expect(
      evalListPollInterval(
        [...settled, { state: "finished", freshness: "stale" }],
        false,
      ),
    ).toBe(EVAL_POLL_MS);
    expect(evalListPollInterval(settled, true)).toBe(false);
  });

  it("refetches a settled detail after invalidation and polls until its view is current", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.experiment.state = "finished";
    fixture.state.experiment.freshness = "current";
    const client = createApplicationQueryClient();
    const view = render(
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={transport(fixture)}>
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
        <PublicAPIProvider api={transport(fixture)}>
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
