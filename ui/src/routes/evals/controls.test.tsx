import { webcrypto } from "node:crypto";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { queryKeys } from "../../api/query-keys";
import { Application } from "../../app/application";
import * as queryClients from "../../app/query-client";
import { applicationRoutes } from "../../app/router";
import {
  createEvalFixture,
  EVAL_API_VERSION,
  EVAL_FIXTURE_ORIGIN,
} from "../../test/evals-fixture";
import { MemoryStorage } from "../../test/storage";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
  vi.stubGlobal("crypto", webcrypto);
});
afterEach(() => {
  vi.restoreAllMocks();
});

function start(fixture: ReturnType<typeof createEvalFixture>, path: string) {
  const cache = queryClients.createApplicationQueryClient();
  vi.spyOn(queryClients, "createApplicationQueryClient").mockReturnValue(cache);
  const invalidated = vi.spyOn(cache, "invalidateQueries");
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
  render(<Application api={api} publicAPI={api} router={router} />);
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
  it("confirms Start in a dialog and refreshes the check lists for a check experiment", async () => {
    const fixture = createEvalFixture({ prepared: true, audit: true });
    const user = userEvent.setup();
    const invalidated = start(fixture, "/evals/experiments/experiment-1/setup");
    const controls = await screen.findByRole("region", {
      name: "Experiment controls",
    });
    await user.click(within(controls).getByRole("button", { name: "Start" }));
    const dialog = screen.getByRole("dialog", { name: "Start experiment?" });
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
