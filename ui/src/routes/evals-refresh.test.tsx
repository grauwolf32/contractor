import { webcrypto } from "node:crypto";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import { createMemoryRouter } from "react-router";
import { beforeEach, expect, it, vi } from "vitest";

import { PublicAPI } from "../api/client";
import { Application } from "../app/application";
import * as queryClients from "../app/query-client";
import { applicationRoutes } from "../app/router";
import {
  createEvalFixture,
  EVAL_API_VERSION,
  EVAL_FIXTURE_ORIGIN,
} from "../test/evals-fixture";
import { MemoryStorage } from "../test/storage";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
  vi.stubGlobal("crypto", webcrypto);
});

function refreshFixture() {
  const fixture = createEvalFixture({ prepared: true });
  const original = fixture.fetch;
  const reads: URL[] = [];
  const pending: (() => void)[] = [];
  let holdMetadata = false;
  let failMembers = false;
  fixture.fetch = async (request) => {
    const url = new URL(request.url);
    reads.push(url);
    if (failMembers && url.pathname.endsWith("/members")) {
      throw new TypeError("offline");
    }
    const response = await original(request);
    if (
      holdMetadata &&
      url.pathname === "/v1/eval-experiments/experiment-1" &&
      request.method === "GET"
    ) {
      return new Promise<Response>((resolve) => {
        pending.push(() => resolve(response));
      });
    }
    return response;
  };
  return {
    fixture,
    reads,
    pending,
    hold() {
      holdMetadata = true;
    },
    failMembers(value: boolean) {
      failMembers = value;
    },
  };
}

function start(fixture: ReturnType<typeof createEvalFixture>, path: string) {
  const cache = queryClients.createApplicationQueryClient();
  vi.spyOn(queryClients, "createApplicationQueryClient").mockReturnValue(cache);
  const api = new PublicAPI(
    {
      uiVersion: "0.4.0",
      apiBaseUrl: EVAL_FIXTURE_ORIGIN,
      supportedApiVersions: [EVAL_API_VERSION],
    },
    async (input) =>
      fixture.fetch(input instanceof Request ? input : new Request(input)),
  );
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  render(
    <StrictMode>
      <Application api={api} publicAPI={api} router={router} />
    </StrictMode>,
  );
  return { cache, router };
}

async function release(pending: (() => void)[], index = 0) {
  await act(async () => pending[index]!());
}

it("preserves comparison filters selected while refreshing to a newer snapshot", async () => {
  const context = refreshFixture();
  const user = userEvent.setup();
  const { router } = start(
    context.fixture,
    "/evals/experiments/experiment-1/comparison?filter=all&viewSnapshot=view-7&chartCursor=old-chart&binFilter=old-bin",
  );
  await screen.findByLabelText("Pair filter");
  context.fixture.state.experiment.viewSnapshot = "view-8";
  context.hold();
  await user.click(screen.getByRole("button", { name: "Refresh" }));
  await waitFor(() => expect(context.pending).toHaveLength(1));
  await user.selectOptions(
    screen.getByLabelText("Comparison metric"),
    "duration",
  );
  await user.selectOptions(screen.getByLabelText("Pair filter"), "unresolved");
  await release(context.pending);

  await waitFor(() =>
    expect(
      new URLSearchParams(router.state.location.search).get("viewSnapshot"),
    ).toBe("view-8"),
  );
  expect(screen.getByLabelText("Pair filter")).toHaveValue("unresolved");
  expect(screen.getByLabelText("Comparison metric")).toHaveValue("duration");
  const params = new URLSearchParams(router.state.location.search);
  expect(params.has("cursor")).toBe(false);
  expect(params.has("chartCursor")).toBe(false);
  expect(params.has("binFilter")).toBe(false);
  await waitFor(() =>
    expect(
      context.reads.some(
        (url) =>
          url.pathname.endsWith("/pairs") &&
          url.searchParams.get("viewSnapshot") === "view-8" &&
          url.searchParams.get("filter") === "unresolved",
      ),
    ).toBe(true),
  );
});

it("preserves an attempt filter selected while retrying a failed page", async () => {
  const context = refreshFixture();
  context.failMembers(true);
  const user = userEvent.setup();
  const { router } = start(
    context.fixture,
    "/evals/experiments/experiment-1/attempts?filter=all&viewSnapshot=view-7&cursor=old-page",
  );
  await screen.findByText("Public API is unavailable");
  context.fixture.state.experiment.viewSnapshot = "view-8";
  context.hold();
  await user.click(screen.getByRole("button", { name: "Try again" }));
  await waitFor(() => expect(context.pending).toHaveLength(1));
  await user.selectOptions(screen.getByLabelText("Attempt filter"), "failed");
  context.failMembers(false);
  await release(context.pending);

  await waitFor(() =>
    expect(
      context.reads.some(
        (url) =>
          url.pathname.endsWith("/members") &&
          url.searchParams.get("viewSnapshot") === "view-8" &&
          url.searchParams.get("filter") === "failed",
      ),
    ).toBe(true),
  );
  expect(screen.getByLabelText("Attempt filter")).toHaveValue("failed");
  const params = new URLSearchParams(router.state.location.search);
  expect(params.get("filter")).toBe("failed");
  expect(params.has("viewSnapshot")).toBe(false);
  expect(params.has("cursor")).toBe(false);
});

it.each(["section", "experiment"] as const)(
  "does not change navigation to another %s when a refresh completes",
  async (destination) => {
    const context = refreshFixture();
    const user = userEvent.setup();
    const { cache, router } = start(
      context.fixture,
      "/evals/experiments/experiment-1/comparison?filter=all&viewSnapshot=view-7",
    );
    await screen.findByLabelText("Pair filter");
    context.hold();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(context.pending).toHaveLength(1));

    let target = "/evals/experiments/experiment-1/overview";
    if (destination === "experiment") {
      // Cached metadata keeps EvalComparison mounted while its identity changes.
      const other = {
        ...structuredClone(context.fixture.state.experiment),
        experimentId: "experiment-2",
        name: "Another experiment",
        viewSnapshot: "other-view",
      };
      cache.setQueryData(["evals", "experiment", other.experimentId], other);
      const original = context.fixture.fetch;
      context.fixture.fetch = async (request) =>
        new URL(request.url).pathname === "/v1/eval-experiments/experiment-2"
          ? new Response(JSON.stringify(other), {
              headers: { "content-type": "application/json" },
            })
          : original(request);
      target =
        "/evals/experiments/experiment-2/comparison?filter=unresolved&viewSnapshot=other-view";
    }
    await act(async () => {
      await router.navigate(target);
    });
    expect(router.state.location.pathname + router.state.location.search).toBe(
      target,
    );
    const location = router.state.location;
    await release(context.pending);
    expect(router.state.location).toEqual(location);
  },
);

it("lets only the latest overlapping refresh apply its snapshot", async () => {
  const context = refreshFixture();
  const user = userEvent.setup();
  const { router } = start(
    context.fixture,
    "/evals/experiments/experiment-1/comparison?filter=all&viewSnapshot=view-7",
  );
  await screen.findByLabelText("Pair filter");
  context.hold();
  context.fixture.state.experiment.viewSnapshot = "view-8";
  await user.click(screen.getByRole("button", { name: "Refresh" }));
  await waitFor(() => expect(context.pending).toHaveLength(1));
  await user.selectOptions(screen.getByLabelText("Pair filter"), "unresolved");
  context.fixture.state.experiment.viewSnapshot = "view-9";
  await user.click(screen.getByRole("button", { name: "Refresh" }));
  await waitFor(() => expect(context.pending).toHaveLength(2));
  expect(screen.getByLabelText("Pair filter")).toHaveValue("unresolved");
  expect(
    new URLSearchParams(router.state.location.search).get("viewSnapshot"),
  ).toBe("view-7");
  await release(context.pending, 1);
  await waitFor(() =>
    expect(
      new URLSearchParams(router.state.location.search).get("viewSnapshot"),
    ).toBe("view-9"),
  );
  await release(context.pending, 0);
  expect(
    new URLSearchParams(router.state.location.search).get("viewSnapshot"),
  ).toBe("view-9");
  expect(screen.getByLabelText("Pair filter")).toHaveValue("unresolved");
});
