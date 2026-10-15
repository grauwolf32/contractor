import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it } from "vitest";
import { renderApplication } from "../test/application";
import { createEvalFixture } from "../test/evals-fixture";
import { json } from "../test/responses";
import { evalServer, stubEvalBrowser } from "./evals/test-support";

beforeEach(stubEvalBrowser);

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
  return renderApplication("/evals/experiments/experiment-1/overview", {
    server: evalServer(fixture),
  }).router;
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
    return json(document);
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

it.each([
  [
    "a cancelled experiment with unsubmitted members",
    "partial",
    "cancelled",
    1,
  ],
  ["a finished experiment with unsupported members", "partial", "finished", 1],
  ["a finished experiment with every member scored", "done", "finished", 2],
] as const)(
  "marks %s as %s in its status glyph",
  async (_, tone, state, armsSubmitted) => {
    const fixture = finishedFixture();
    fixture.state.experiment.state = state;
    Object.values(fixture.state.experiment.summary!.counts).forEach(
      (counts, index) => {
        if (index >= armsSubmitted) {
          counts.unsupported = state === "finished" ? 1 : 0;
          return;
        }
        Object.assign(counts, {
          eligible: 1,
          submitted: 1,
          terminal: 1,
          executionSucceeded: 1,
          collectionComplete: 1,
          scored: 1,
        });
      },
    );
    start(fixture);
    expect(
      await screen.findByRole("heading", { name: "Execution finished" }),
    ).toBeVisible();
    expect(
      screen.getByRole("region", { name: "Execution status" }),
    ).toHaveAttribute("data-tone", tone);
  },
);

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
