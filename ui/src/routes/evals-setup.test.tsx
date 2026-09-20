import { MemoryStorage } from "../test/storage";
import { webcrypto } from "node:crypto";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import {
  createEvalFixture,
  EVAL_FIXTURE_ORIGIN,
  EVAL_API_VERSION,
} from "../test/evals-fixture";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
  localStorage.clear();
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
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

describe("Managed Evals setup", () => {
  it(
    "saves an exact A/B matrix, restores it, and creates no execution before separate Prepare and Start",
    { timeout: 15_000 },
    async () => {
      const fixture = createEvalFixture(),
        user = userEvent.setup();
      const view = start(fixture, "/evals/new");
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
      expect(screen.getByLabelText("A exact version")).toHaveValue("trace-a@2");
      await user.click(screen.getByRole("button", { name: "Next step" }));
      await screen.findByRole("option", { name: /Trace examples/ });
      await user.selectOptions(screen.getByLabelText("Dataset revision"), "r1");
      await user.click(
        await screen.findByRole("button", { name: "Select all 2 cases" }),
      );
      await user.click(screen.getByRole("button", { name: "Next step" }));
      await user.click(
        screen.getByRole("button", { name: "Add assessment check" }),
      );
      await user.clear(screen.getByLabelText("Repetitions"));
      await user.type(screen.getByLabelText("Repetitions"), "2");
      await user.click(screen.getByRole("button", { name: "Next step" }));
      expect(screen.getAllByText(/8 expected members/).length).toBeGreaterThan(
        0,
      );
      await user.click(screen.getByRole("button", { name: "Save draft" }));
      await waitFor(() =>
        expect(view.router.state.location.pathname).toContain(
          "experiment-1/setup",
        ),
      );
      await screen.findByText(/Saved on server/);
      const created = fixture.state.requests.find(
        (r) => r.path.endsWith("/eval-experiments") && r.method === "POST",
      );
      expect(created?.body).toMatchObject({
        controlMode: "server",
        draft: {
          repetitions: 2,
          caseIds: ["unsafe-query", "safe-query"],
          variants: [{ selector: "trace-a@2" }, { selector: "trace-b@2" }],
        },
      });
      expect(
        fixture.state.requests.some((r) => r.path.endsWith("/commands")),
      ).toBe(false);
      view.unmount();
      start(fixture, "/evals/experiments/experiment-1/setup");
      expect(await screen.findByLabelText("Experiment name")).toHaveValue(
        "Browser experiment",
      );
      await user.click(await screen.findByRole("button", { name: "Prepare" }));
      await screen.findByText("Verified preparation");
      expect(
        fixture.state.requests
          .filter((r) => r.path.endsWith("/commands"))
          .every((r) => (r.body as { kind: string }).kind === "prepare"),
      ).toBe(true);
      await user.click(await screen.findByRole("button", { name: "Start" }));
      expect(await screen.findByRole("dialog")).toBeVisible();
      expect(fixture.state.experiment.state).toBe("ready");
      await user.click(screen.getByRole("button", { name: "Confirm start" }));
      await waitFor(() =>
        expect(fixture.state.experiment.state).toBe("running"),
      );
    },
  );

  it("replays a lost Start with its original key/body/revision after a browser remount", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.lostCommand = true;
    const user = userEvent.setup(),
      view = start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Start" }));
    await user.click(screen.getByRole("button", { name: "Confirm start" }));
    await screen.findByText("Public API is unavailable");
    const first = fixture.state.requests.find((r) =>
      r.path.endsWith("/commands"),
    )!;
    expect(fixture.state.experiment.state).toBe("running");
    view.unmount();
    start(fixture, "/evals/experiments/experiment-1/setup");
    await screen.findByText("Start: completed.");
    const retries = fixture.state.requests.filter((r) =>
      r.path.endsWith("/commands"),
    );
    expect(retries).toHaveLength(2);
    expect(retries[1]).toMatchObject({
      key: first.key,
      etag: first.etag,
      body: first.body,
    });
    expect(fixture.state.experiment.revision).toBe(2);
  });

  it("keeps the reviewed Start revision when a competing update arrives before confirmation", async () => {
    const fixture = createEvalFixture({ prepared: true }),
      user = userEvent.setup();
    start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Start" }));
    fixture.state.experiment.revision = 2;
    await user.click(screen.getByRole("button", { name: "Confirm start" }));
    await screen.findByText("eval_revision_mismatch");
    expect(fixture.state.experiment.state).toBe("ready");
    expect(
      fixture.state.requests.find((r) => r.path.endsWith("/commands"))?.etag,
    ).toBe('"1"');
  });

  it("blocks Prepare for unsaved changes and restores focus after dismissing cancellation", async () => {
    const fixture = createEvalFixture(),
      user = userEvent.setup();
    const view = start(fixture, "/evals/experiments/experiment-1/setup");
    await user.type(
      await screen.findByLabelText("Experiment name"),
      " changed",
    );
    expect(screen.getByRole("button", { name: "Prepare" })).toBeDisabled();
    view.unmount();
    fixture.state.experiment = {
      ...fixture.state.experiment,
      state: "running",
      allowedCommands: ["cancel"],
    };
    delete fixture.state.experiment.draft;
    start(fixture, "/evals/experiments/experiment-1/setup");
    const cancel = await screen.findByRole("button", {
      name: "Cancel",
    });
    await user.click(cancel);
    await user.keyboard("{Escape}");
    await waitFor(() => expect(cancel).toHaveFocus());
    expect(
      fixture.state.requests.filter((r) => r.path.endsWith("/commands")),
    ).toHaveLength(0);
  });

  it("keeps external dispatch controls absent", async () => {
    const fixture = createEvalFixture({ prepared: true, external: true });
    start(fixture, "/evals/experiments/experiment-1/setup");
    await screen.findByText(/Externally controlled/);
    for (const label of [
      "Prepare",
      "Start",
      "Pause",
      "Resume",
      "Cancel",
      "Duplicate",
    ])
      expect(
        screen.queryByRole("button", { name: label }),
      ).not.toBeInTheDocument();
  });
});
