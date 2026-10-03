import { MemoryStorage, storedValues } from "../test/storage";
import { webcrypto } from "node:crypto";
import { onlineManager } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
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
      expect(screen.getByLabelText("A version")).toHaveValue("trace-a@2");
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

  it("does not resend a finished command when queries refetch", async () => {
    const fixture = createEvalFixture({ prepared: true }),
      user = userEvent.setup();
    start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Start" }));
    await user.click(screen.getByRole("button", { name: "Confirm start" }));
    await screen.findByText("Start: completed.");
    expect(storedValues(localStorage)).not.toContain("eval-recovery");
    act(() => {
      onlineManager.setOnline(false);
      onlineManager.setOnline(true);
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(
      fixture.state.requests.filter((r) => r.path.endsWith("/commands")),
    ).toHaveLength(1);
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
    const command = fixture.state.requests.find((r) =>
      r.path.endsWith("/commands"),
    );
    expect(command?.etag).toBe('"1"');
    expect(command?.body).toMatchObject({
      kind: "start",
      planSha256: fixture.state.experiment.planSha256,
    });
  });

  it("confirms Cancel using the revision reached while its dialog was open", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.experiment.state = "running";
    fixture.state.experiment.allowedCommands = ["pause", "cancel"];
    const user = userEvent.setup();
    start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Cancel" }));
    fixture.state.experiment.revision += 3;
    await user.click(screen.getByRole("button", { name: "Confirm cancel" }));
    await screen.findByText("Cancel: completed.");
    expect(fixture.state.experiment.state).toBe("cancelled");
    const commands = fixture.state.requests.filter((request) =>
      request.path.endsWith("/commands"),
    );
    expect(commands).toHaveLength(1);
    expect(commands[0]?.etag).toBe('"4"');
    expect(commands[0]?.body).toMatchObject({
      kind: "cancel",
      planSha256: fixture.state.experiment.planSha256,
    });
  });

  it("retries Pause once with a new key when progress races the command", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.experiment.state = "running";
    fixture.state.experiment.allowedCommands = ["pause", "cancel"];
    fixture.state.commandRaceOnce = true;
    const user = userEvent.setup();
    start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Pause" }));
    await screen.findByText("Pause: completed.");
    expect(fixture.state.experiment.state).toBe("paused");
    const commands = fixture.state.requests.filter((request) =>
      request.path.endsWith("/commands"),
    );
    expect(commands).toHaveLength(2);
    expect(commands.map((request) => request.etag)).toEqual(['"1"', '"2"']);
    expect(commands[1]?.key).not.toBe(commands[0]?.key);
    expect(commands[1]?.body).toEqual(commands[0]?.body);
  });

  it("replays an uncertain Pause with its original key and revision", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.experiment.state = "running";
    fixture.state.experiment.allowedCommands = ["pause", "cancel"];
    fixture.state.lostCommand = true;
    const user = userEvent.setup();
    const view = start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Pause" }));
    await screen.findByText("Public API is unavailable");
    const first = fixture.state.requests.find((request) =>
      request.path.endsWith("/commands"),
    )!;
    view.unmount();
    start(fixture, "/evals/experiments/experiment-1/setup");
    await screen.findByText("Pause: completed.");
    const commands = fixture.state.requests.filter((request) =>
      request.path.endsWith("/commands"),
    );
    expect(commands).toHaveLength(2);
    expect(commands[1]).toMatchObject({
      key: first.key,
      etag: first.etag,
      body: first.body,
    });
  });

  it("drops a Pause the coordinator ruled out instead of replaying it", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.experiment.state = "running";
    fixture.state.experiment.allowedCommands = ["pause", "cancel"];
    fixture.state.commandSettleOnce = true;
    const user = userEvent.setup();
    const view = start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Pause" }));
    await screen.findByText(/no longer available/);
    expect(storedValues(localStorage)).not.toContain("eval-recovery");
    expect(
      screen.queryByRole("button", { name: "Try again" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Pause" }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Duplicate" })).toBeEnabled();
    view.unmount();
    start(fixture, "/evals/experiments/experiment-1/setup");
    expect(
      await screen.findByRole("button", { name: "Duplicate" }),
    ).toBeEnabled();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(
      fixture.state.requests.filter((request) =>
        request.path.endsWith("/commands"),
      ),
    ).toHaveLength(1);
    expect(fixture.state.experiment.state).toBe("finished");
  });

  it("keeps a Pause that met a server outage for resume after reload", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.experiment.state = "running";
    fixture.state.experiment.allowedCommands = ["pause", "cancel"];
    fixture.state.commandUnavailableOnce = true;
    const user = userEvent.setup();
    const view = start(fixture, "/evals/experiments/experiment-1/setup");
    await user.click(await screen.findByRole("button", { name: "Pause" }));
    expect(
      await screen.findByRole("button", { name: "Try again" }),
    ).toBeVisible();
    expect(storedValues(localStorage)).toContain("eval-recovery");
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    const first = fixture.state.requests.find((request) =>
      request.path.endsWith("/commands"),
    )!;
    view.unmount();
    start(fixture, "/evals/experiments/experiment-1/setup");
    await screen.findByText("Pause: completed.");
    expect(fixture.state.experiment.state).toBe("paused");
    const commands = fixture.state.requests.filter((request) =>
      request.path.endsWith("/commands"),
    );
    expect(commands).toHaveLength(2);
    expect(commands[1]).toMatchObject({
      key: first.key,
      etag: first.etag,
      body: first.body,
    });
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
