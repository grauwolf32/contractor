import { webcrypto } from "node:crypto";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { Application } from "../../app/application";
import * as queryClientFactory from "../../app/query-client";
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

function start(fixture: ReturnType<typeof createEvalFixture>, path: string) {
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
  return router;
}

describe("Experiment setup vocabulary", () => {
  it("offers a check as the execution kind and names its check type family", async () => {
    const fixture = createEvalFixture();
    const user = userEvent.setup();
    start(fixture, "/evals/new?project=evaluation-1");
    const kind = await screen.findByLabelText("Execution kind");
    expect(within(kind).getByRole("option", { name: "Check" })).toHaveValue(
      "audit",
    );
    expect(
      within(screen.getByLabelText("Comparison purpose")).getByRole("option", {
        name: "Workflow or check configuration",
      }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("A Workflow family")).toBeInTheDocument();
    await user.selectOptions(kind, "audit");
    expect(screen.getByLabelText("A Check type family")).toBeInTheDocument();
    expect(screen.getByLabelText("B Check type family")).toBeInTheDocument();
    await user.click(
      screen.getAllByText("Execution settings", { exact: true })[0]!,
    );
    expect(
      screen.getAllByText(
        "Execution settings come from the pinned check type.",
      ),
    ).toHaveLength(2);
    expect(screen.getByLabelText("Check execution")).toHaveValue("different");
  });

  it("adds, names and removes criteria", async () => {
    const fixture = createEvalFixture();
    const user = userEvent.setup();
    start(fixture, "/evals/new?project=evaluation-1");
    await user.click(
      await screen.findByRole("button", {
        name: "3. Assessment and repetitions",
      }),
    );
    expect(
      screen.getByText("No criteria yet. Add at least one required criterion."),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Add criterion" }));
    const criterion = screen.getByRole("group", { name: "Criterion 1" });
    expect(within(criterion).getByLabelText("Criterion ID")).toHaveValue(
      "criterion-1",
    );
    await user.click(
      within(criterion).getByRole("button", { name: "Remove criterion" }),
    );
    expect(screen.queryByRole("group", { name: "Criterion 1" })).toBeNull();
  });

  it("lists the pinned criteria on the readiness step", async () => {
    const fixture = createEvalFixture();
    const user = userEvent.setup();
    start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(
      await screen.findByRole("button", { name: "4. Readiness" }),
    );
    expect(
      screen.getByRole("heading", { name: "Criteria", level: 3 }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Human review · required · rubric r1/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Prepare creates no Runs or checks/)).toBeVisible();
  });

  it("names a private rubric by the criterion it reviews", async () => {
    const fixture = createEvalFixture();
    const user = userEvent.setup();
    start(fixture, "/evals/datasets?project=evaluation-1");
    await user.click(
      await screen.findByRole("button", { name: "Create or import dataset" }),
    );
    await user.click(screen.getByText("Private human review rubrics"));
    await user.click(screen.getByRole("button", { name: "Add human rubric" }));
    expect(screen.getByLabelText("Review criterion ID")).toBeInTheDocument();
    expect(screen.getByLabelText("Rubric revision")).toBeInTheDocument();
  });
});

// AGENTS.md 3.8, V385-001 A7: a late answer never moves a user who left.
describe("Experiment setup navigation", () => {
  it(
    "leaves a user who moved on where they are when saving a draft answers late",
    { timeout: 15_000 },
    async () => {
      const fixture = createEvalFixture();
      let release: () => void = () => undefined;
      const released = new Promise<void>((resolve) => {
        release = resolve;
      });
      const api = new PublicAPI(
        {
          uiVersion: "0.4.0",
          apiBaseUrl: EVAL_FIXTURE_ORIGIN,
          supportedApiVersions: [EVAL_API_VERSION],
        },
        async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          if (
            request.method === "POST" &&
            new URL(request.url).pathname.endsWith("/eval-experiments")
          )
            await released;
          return fixture.fetch(request);
        },
      );
      const router = createMemoryRouter(applicationRoutes(), {
        initialEntries: ["/evals/new"],
      });
      const queryClient = queryClientFactory.createApplicationQueryClient();
      const factory = vi
        .spyOn(queryClientFactory, "createApplicationQueryClient")
        .mockReturnValueOnce(queryClient);
      render(<Application api={api} publicAPI={api} router={router} />);
      factory.mockRestore();
      const user = userEvent.setup();
      await user.type(
        await screen.findByLabelText("Experiment name"),
        "Browser experiment",
      );
      await screen.findByRole("option", { name: "Evaluation workspace" });
      await user.selectOptions(
        screen.getByLabelText("Evaluation workspace"),
        "evaluation-1",
      );
      await screen.findAllByRole("option", { name: "trace-a" });
      await user.selectOptions(
        screen.getByLabelText("A Workflow family"),
        "trace-a",
      );
      await user.selectOptions(
        screen.getByLabelText("B Workflow family"),
        "trace-b",
      );
      await user.click(screen.getByRole("button", { name: "Next step" }));
      await screen.findByRole("option", { name: /Trace examples/ });
      await user.selectOptions(screen.getByLabelText("Dataset revision"), "r1");
      await user.click(
        await screen.findByRole("button", { name: "Select all 2 cases" }),
      );
      await user.click(screen.getByRole("button", { name: "Next step" }));
      await user.click(screen.getByRole("button", { name: "Add criterion" }));
      await user.click(screen.getByRole("button", { name: "Next step" }));
      await user.click(screen.getByRole("button", { name: "Save draft" }));
      await act(() => router.navigate("/settings"));
      expect(queryClient.isMutating()).toBe(1);
      release();
      await waitFor(() => expect(queryClient.isMutating()).toBe(0));
      await act(async () => undefined);
      expect(
        queryClient
          .getMutationCache()
          .getAll()
          .map((m) => m.state.status),
      ).toEqual(["success"]);
      expect(router.state.navigation.state).toBe("idle");
      expect(router.state.location.pathname).toBe("/settings");
    },
  );
});
