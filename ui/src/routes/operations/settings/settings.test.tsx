import {
  act,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { FakeEventSocket, renderApplication } from "../../../test/application";
import { operationsSnapshot } from "../../../test/builders-execution";
import { conforming } from "../../../test/builders";
import {
  json,
  preconditionFailed,
  type Routes,
} from "../../../test/fake-server";

function schedulerSettings(maxConcurrentRuns: number, revision: string) {
  return json(
    conforming("SchedulerSettings", {
      maxConcurrentRuns,
      revision,
      updatedAt: `2026-09-06T01:00:0${revision}Z`,
    }),
    200,
    { ETag: `"${revision}"`, "Cache-Control": "no-store" },
  );
}

/** The snapshot and the personal Git key every Settings page reads. */
const common: Routes = {
  "GET /v1/operations/snapshot": () => json(operationsSnapshot()),
  "GET /v1/settings/git-key": () => json({ configured: false }),
};

describe("Operations Scheduler settings", () => {
  it("keeps personal repository settings available without Operations capability", async () => {
    renderApplication("/operations/settings", {
      capabilities: ["user"],
      server: { "GET /v1/settings/git-key": () => json({ configured: false }) },
    });

    expect(
      await screen.findByRole("heading", { name: "Git SSH key" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /Repository access/ }),
    ).toHaveAttribute("href", "#repository-access");
    expect(
      screen.queryByRole("heading", { name: "Workflow scheduling" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("Operations capability required"),
    ).not.toBeInTheDocument();
    // A page of its own, reached from the account menu: no Operations tabs.
    expect(
      screen.getByRole("heading", { level: 1, name: "Settings" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Git SSH key" }),
    ).toBeInTheDocument();
    // Headings do not skip a level below the section title.
    expect(
      screen.getByRole("heading", { level: 3, name: "Private repository key" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Theme" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("navigation", { name: "Operations sections" }),
    ).not.toBeInTheDocument();
    // The title is set in an effect, which may run after the page shows.
    await waitFor(() => expect(document.title).toBe("Settings · Contractor"));
  });

  it("points sessions without the capability from other Operations pages to Settings", async () => {
    const { requests } = renderApplication("/operations/runtime-agents", {
      capabilities: ["user"],
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "not authorized to observe or manage Operations",
    );
    const main = screen.getByRole("main");
    expect(
      within(main).getByRole("heading", { level: 1, name: "Operations" }),
    ).toBeInTheDocument();
    expect(
      within(main).getByRole("link", { name: "Settings" }),
    ).toHaveAttribute("href", "/operations/settings");
    expect(
      screen.queryByRole("navigation", { name: "Operations sections" }),
    ).not.toBeInTheDocument();
    expect(
      requests
        .map((request) => new URL(request.url).pathname)
        .filter(
          (path) =>
            ![
              "/v1/projects",
              "/v1/audits",
              "/v1/findings",
              "/v1/reviews",
            ].includes(path),
        ),
    ).toEqual(["/v1/auth/session"]);
  });

  it("shows Settings as an Operations section to operators", async () => {
    renderApplication("/operations/settings", {
      server: {
        ...common,
        "GET /v1/operations/settings/scheduler": () =>
          schedulerSettings(2, "3"),
      },
    });
    expect(
      await screen.findByRole("heading", {
        level: 3,
        name: "Workflow scheduling",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 2, name: "Settings" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 3, name: "Git SSH key" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 4, name: "Private repository key" }),
    ).toBeInTheDocument();
    const sections = screen.getByRole("navigation", {
      name: "Operations sections",
    });
    expect(
      within(sections).getByRole("link", { name: "Settings" }),
    ).toHaveAttribute("aria-current", "page");
    expect(
      screen.getByRole("link", { name: /Workflow scheduling/ }),
    ).toHaveAttribute("href", "#workflow-scheduling");
    // The title is set in an effect, which may run after the page shows.
    await waitFor(() =>
      expect(document.title).toBe("Settings · Operations · Contractor"),
    );
  });

  it("validates, CAS-saves once and settles on the returned value", async () => {
    let finishPut: ((value: Response) => void) | undefined;
    const pendingPut = new Promise<Response>((resolve) => {
      finishPut = resolve;
    });
    const { sent, user } = renderApplication("/operations/settings", {
      server: {
        ...common,
        "GET /v1/operations/settings/scheduler": () =>
          schedulerSettings(1, "1"),
        "PUT /v1/operations/settings/scheduler": () => pendingPut,
      },
    });
    const puts = () => sent("PUT", "/v1/operations/settings/scheduler");
    const input = await screen.findByLabelText(
      "Maximum concurrent Workflow Runs",
    );
    expect(input).toHaveValue(1);
    expect(screen.getByText("Admission, not capacity.")).toBeInTheDocument();
    expect(
      screen.getByText(/one Workflow Run may require several agents/),
    ).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "" } });
    expect(
      screen.getByText("Enter a maximum from 1 through 32."),
    ).toBeInTheDocument();
    expect(puts()).toHaveLength(0);
    fireEvent.change(input, { target: { value: "1.5" } });
    expect(screen.getByText(/must be a whole number/)).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "33" } });
    expect(screen.getByText(/must be from 1 through 32/)).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "2" } });
    await user.click(
      screen.getByRole("button", { name: "Save scheduling limit" }),
    );
    expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled();
    await waitFor(() => expect(puts()).toHaveLength(1));
    const put = puts()[0];
    expect(put?.headers.get("If-Match")).toBe('"1"');
    await expect(put?.json()).resolves.toEqual({ maxConcurrentRuns: 2 });

    finishPut?.(schedulerSettings(2, "2"));
    expect(
      await screen.findByText("Saved maximum 2 at revision 2."),
    ).toBeInTheDocument();
    expect(input).toHaveValue(2);
    expect(
      screen.getByRole("button", { name: "Save scheduling limit" }),
    ).toBeDisabled();
  });

  it("reloads after a 412 and preserves the unsaved draft without overwriting", async () => {
    let saved = 1;
    let revision = "1";
    let puts = 0;
    const { user } = renderApplication("/operations/settings", {
      server: {
        ...common,
        "GET /v1/operations/settings/scheduler": () =>
          schedulerSettings(saved, revision),
        "PUT /v1/operations/settings/scheduler": () => {
          puts += 1;
          saved = 3;
          revision = "2";
          return preconditionFailed("resource revision precondition failed");
        },
      },
    });
    const input = await screen.findByLabelText(
      "Maximum concurrent Workflow Runs",
    );
    fireEvent.change(input, { target: { value: "2" } });
    await user.click(
      screen.getByRole("button", { name: "Save scheduling limit" }),
    );

    expect(
      await screen.findByText(/changed on the Server|Another operator/),
    ).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("3")).toBeInTheDocument());
    expect(input).toHaveValue(2);
    expect(puts).toBe(1);
    expect(
      screen.getByRole("button", { name: "Save scheduling limit" }),
    ).toBeDisabled();

    await user.click(
      screen.getByRole("button", { name: "Reset to saved value" }),
    );
    expect(input).toHaveValue(3);
    expect(
      screen.queryByText(/changed on the Server|Another operator/),
    ).not.toBeInTheDocument();
  });

  it("refetches the singleton on typed Operations invalidation and live reconnect", async () => {
    let saved = 1;
    let revision = "1";
    let settingsReads = 0;
    renderApplication("/operations/settings", {
      server: {
        ...common,
        "GET /v1/operations/settings/scheduler": () => {
          settingsReads += 1;
          return schedulerSettings(saved, revision);
        },
      },
    });
    expect(
      await screen.findByRole("heading", { name: "Workflow scheduling" }),
    ).toBeInTheDocument();
    await waitFor(() => expect(FakeEventSocket.instances).toHaveLength(1));
    const socket = FakeEventSocket.instances[0];
    act(() => socket?.open());
    const subscription = JSON.parse(socket?.sent[0] ?? "{}");
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "subscribed",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "operations" },
        cursor: subscription.after,
      }),
    );
    await waitFor(() => expect(settingsReads).toBeGreaterThanOrEqual(2));

    saved = 4;
    revision = "2";
    act(() =>
      socket?.message({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: subscription.subscriptionId,
        stream: { kind: "operations" },
        cursor: {
          generation: "operations-generation-1",
          sequence: "8",
        },
        kind: "operations.changed",
        occurredAt: "2026-09-06T01:00:02Z",
        data: { resource: "schedulerSettings", revision: "8" },
      }),
    );
    await waitFor(() =>
      expect(
        screen.getByLabelText("Maximum concurrent Workflow Runs"),
      ).toHaveValue(4),
    );
    expect(screen.getByText("4")).toBeInTheDocument();
  });
});
