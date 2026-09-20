import { webcrypto } from "node:crypto";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, expect, it, vi } from "vitest";
import { PublicAPI } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import { MemoryStorage } from "../test/storage";
import {
  createEvalFixture,
  EVAL_API_VERSION,
  EVAL_FIXTURE_ORIGIN,
} from "../test/evals-fixture";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
  vi.stubGlobal("crypto", webcrypto);
});

function finishedFixture() {
  const fixture = createEvalFixture({ prepared: true });
  const experiment = fixture.state.experiment;
  experiment.state = "finished";
  experiment.allowedCommands = ["duplicate"];
  experiment.expectedMembers = 2;
  for (const counts of Object.values(experiment.summary!.counts)) {
    Object.assign(
      counts,
      Object.fromEntries(Object.keys(counts).map((key) => [key, 0])),
    );
    counts.expected = 1;
  }
  return fixture;
}

function start(fixture: ReturnType<typeof createEvalFixture>) {
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
    initialEntries: ["/evals/experiments/experiment-1/overview"],
  });
  render(<Application api={api} publicAPI={api} router={router} />);
  return router;
}

it("explains zero executions and lets the user prepare a fresh copy", async () => {
  const fixture = finishedFixture();
  for (const counts of Object.values(fixture.state.experiment.summary!.counts))
    counts.unsupported = 1;
  const original = fixture.fetch;
  fixture.fetch = async (request) => {
    const response = await original(request);
    if (!new URL(request.url).pathname.endsWith("/members")) return response;
    const document = await response.json();
    for (const item of document.items)
      item.member.reason = "A required output role is unavailable.";
    return new Response(JSON.stringify(document), {
      headers: response.headers,
    });
  };
  const router = start(fixture),
    user = userEvent.setup();
  expect(
    await screen.findByRole("heading", { name: "No executions started" }),
  ).toBeVisible();
  expect(
    await screen.findByText("A required output role is unavailable."),
  ).toBeVisible();
  expect(
    screen.getByText(/Preparation: 2 unsupported, 0 blocked/),
  ).toBeVisible();
  expect(
    screen.queryByRole("button", { name: "Resume" }),
  ).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Duplicate" }));
  await waitFor(() =>
    expect(router.state.location.pathname).toBe(
      "/evals/experiments/experiment-duplicate/setup",
    ),
  );
  expect(await screen.findByRole("button", { name: "Prepare" })).toBeEnabled();
  expect(
    fixture.state.requests.filter(
      (r) => r.method === "POST" && r.path.endsWith("/commands"),
    ),
  ).toHaveLength(1);
});

it("links completed unscored executions to review without implying execution failure", async () => {
  const fixture = finishedFixture();
  for (const counts of Object.values(
    fixture.state.experiment.summary!.counts,
  )) {
    Object.assign(counts, {
      eligible: 1,
      submitted: 1,
      terminal: 1,
      executionSucceeded: 1,
      collectionComplete: 1,
    });
  }
  start(fixture);
  expect(
    await screen.findByRole("heading", {
      name: "Execution finished; assessment is incomplete",
    }),
  ).toBeVisible();
  expect(
    screen.getByRole("link", { name: "Review results in Comparison" }),
  ).toHaveAttribute(
    "href",
    "/evals/experiments/experiment-1/comparison?filter=unresolved",
  );
  expect(
    screen.getByText(
      /Completing their assessment does not require another execution/,
    ),
  ).toBeVisible();
});

it("shows failed execution coverage instead of just a finished experiment badge", async () => {
  const fixture = finishedFixture();
  for (const counts of Object.values(
    fixture.state.experiment.summary!.counts,
  )) {
    Object.assign(counts, { eligible: 1, submitted: 1, terminal: 1 });
  }
  start(fixture);
  expect(
    await screen.findByRole("heading", {
      name: "Some executions ended without success",
    }),
  ).toBeVisible();
  expect(screen.getByText(/2 executions did not succeed/)).toBeVisible();
  expect(
    screen.getByRole("link", {
      name: "Inspect all attempts and execution evidence",
    }),
  ).toHaveAttribute(
    "href",
    "/evals/experiments/experiment-1/attempts?filter=all",
  );
});

it("explains when a paused experiment can resume", async () => {
  const fixture = finishedFixture();
  fixture.state.experiment.state = "paused";
  fixture.state.experiment.allowedCommands = ["resume", "cancel", "duplicate"];
  start(fixture);
  expect(
    await screen.findByRole("heading", { name: "Dispatch paused" }),
  ).toBeVisible();
  expect(screen.getByRole("button", { name: "Resume" })).toBeEnabled();
  expect(screen.queryByText(/This attempt is closed/)).not.toBeInTheDocument();
});
