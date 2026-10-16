import type { QueryClient } from "@tanstack/react-query";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { queryKeys } from "../../api/query-keys";
import { createApplicationQueryClient } from "../../app/query-client";
import { renderApplication } from "../../test/application";
import {
  createEvalFixture,
  EVAL_FIXTURE_ORIGIN,
} from "../../test/evals-fixture";
import { readCommand, writeCommand } from "./recovery";
import { evalServer, stubEvalBrowser } from "./test-support";

beforeEach(stubEvalBrowser);
afterEach(() => {
  vi.restoreAllMocks();
});

function start(
  fixture: ReturnType<typeof createEvalFixture>,
  path: string,
  prepare?: (cache: QueryClient) => void,
) {
  const cache = createApplicationQueryClient();
  prepare?.(cache);
  const invalidated = vi.spyOn(cache, "invalidateQueries");
  renderApplication(path, { server: evalServer(fixture), queryClient: cache });
  return invalidated;
}

function crossProjectRefreshes(invalidated: ReturnType<typeof start>) {
  return invalidated.mock.calls.filter(
    ([filters]) =>
      JSON.stringify(filters?.queryKey) ===
      JSON.stringify(queryKeys.crossProject.all),
  ).length;
}

describe("Experiment controls", () => {
  it("settles a command whose receipt another instance of the controls fetched", async () => {
    // The detail view replaces the controls while a command runs. The
    // replacement recovers the stored command and can observe the finished
    // receipt from the poll of the instance it replaced, without fetching.
    const fixture = createEvalFixture({ prepared: true });
    const owner = `${EVAL_FIXTURE_ORIGIN}/user_eval`;
    const experiment = fixture.state.experiment;
    writeCommand(owner, experiment.experimentId, {
      key: `eval-ui-${"0".repeat(32)}`,
      revision: experiment.revision,
      body: { kind: "prepare" },
      commandId: "command-earlier",
    });
    const key = queryKeys.evals.command(
      experiment.experimentId,
      "command-earlier",
    );
    start(fixture, "/evals/experiments/experiment-1/setup", (cache) => {
      cache.setQueryDefaults(key, { staleTime: Infinity });
      cache.setQueryData(key, {
        commandId: "command-earlier",
        experimentId: experiment.experimentId,
        kind: "prepare",
        state: "completed",
        experimentRevision: experiment.revision,
        planSha256: experiment.planSha256,
        diagnostics: [],
      });
    });
    const controls = await screen.findByRole("region", {
      name: "Experiment controls",
    });
    await within(controls).findByText("Prepare: completed.");
    expect(
      within(controls).getByRole("button", { name: "Start" }),
    ).toBeEnabled();
    expect(
      within(controls).queryByText(/Waiting for the server to confirm/),
    ).toBeNull();
    expect(readCommand(owner, experiment.experimentId)).toBeNull();
  });

  it("confirms Start in a dialog and refreshes the check lists for a check experiment", async () => {
    const fixture = createEvalFixture({ prepared: true, audit: true });
    const user = userEvent.setup();
    const invalidated = start(fixture, "/evals/experiments/experiment-1/setup");
    const controls = await screen.findByRole("region", {
      name: "Experiment controls",
    });
    await user.click(within(controls).getByRole("button", { name: "Start" }));
    const dialog = screen.getByRole("alertdialog", {
      name: "Start experiment?",
    });
    expect(within(dialog).getByText(/8 expected members/)).toBeVisible();
    await user.click(
      within(dialog).getByRole("button", { name: "Confirm start" }),
    );
    await screen.findByText("Start: completed.");
    await waitFor(() =>
      expect(crossProjectRefreshes(invalidated)).toBeGreaterThan(0),
    );
    expect(fixture.state.experiment.state).toBe("running");
  });

  it("leaves the check lists alone for a Workflow experiment", async () => {
    const fixture = createEvalFixture({ prepared: true });
    const user = userEvent.setup();
    const invalidated = start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Start" }));
    await user.click(screen.getByRole("button", { name: "Confirm start" }));
    await screen.findByText("Start: completed.");
    expect(crossProjectRefreshes(invalidated)).toBe(0);
  });
});
