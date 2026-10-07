import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  audit,
  fakeServer,
  project,
  renderRoutes,
  report,
  type ServerState,
} from "./test-support";

/**
 * Three projects: one report waiting for acceptance, two ready ones, and
 * checks whose reports are not listed (pending, unavailable, running).
 */
function state(): ServerState {
  return {
    projects: [
      project("project_payment", "Payment service"),
      project("project_crapi", "crapi-workshop"),
      project("project_empty", "Empty project"),
    ],
    audits: [
      audit("project_payment", "audit_proposed", "waiting_review", {
        updatedAt: "2026-10-05T09:00:00Z",
      }),
      audit("project_crapi", "audit_trace", "completed", {
        profile: "openapi-operation-trace",
        updatedAt: "2026-10-04T09:00:00Z",
      }),
      audit("project_payment", "audit_ready", "completed", {
        profile: "owasp-asvs-5-0-l1-source-review",
        updatedAt: "2026-10-03T09:00:00Z",
      }),
      audit("project_payment", "audit_finishing", "finalizing"),
      audit("project_crapi", "audit_failed", "failed"),
      audit("project_crapi", "audit_running", "active"),
    ],
    reports: {
      audit_proposed: report("audit_proposed", "proposed"),
      audit_trace: report("audit_trace", "ready"),
      audit_ready: report("audit_ready", "ready"),
      audit_finishing: report("audit_finishing", "pending"),
      audit_failed: report("audit_failed", "unavailable"),
    },
  };
}

/** The report rows in display order, by their title link. */
function rowTitles() {
  return within(screen.getByRole("region", { name: "Reports" }))
    .getAllByRole("listitem")
    .map((row) => within(row).getAllByRole("link")[0]?.textContent);
}

async function settled() {
  // The All chip carries its count once every read has settled.
  await screen.findByRole("button", { name: "All 3" });
}

describe("Reports list", () => {
  it("lists reports of every project, waiting for acceptance first", async () => {
    renderRoutes("/reports", fakeServer(state()));
    await settled();

    expect(
      screen.getByRole("heading", { level: 1, name: "Reports" }),
    ).toBeVisible();
    expect(
      screen.getByText("1 waiting for acceptance, 2 ready."),
    ).toBeVisible();
    const waiting = screen.getByRole("region", {
      name: "Waiting for acceptance",
    });
    const ready = screen.getByRole("region", { name: "Ready" });
    expect(within(waiting).getAllByRole("listitem")).toHaveLength(1);
    expect(within(ready).getAllByRole("listitem")).toHaveLength(2);
    // Check type · project, the status chip and the update time.
    expect(rowTitles()).toEqual([
      "OWASP Top 10 · Source risks · Payment service",
      "OpenAPI · Operation trace · crapi-workshop",
      "ASVS 5.0 · Level 1 source review · Payment service",
    ]);
    const first = within(waiting).getByRole("listitem");
    expect(within(first).getByText("Waiting for acceptance")).toBeVisible();
    expect(
      within(first).getByText(
        (_, element) =>
          element?.getAttribute("datetime") === "2026-10-05T09:00:00Z",
      ),
    ).toBeInTheDocument();
    // Pending and unavailable reports are not reports yet.
    expect(screen.queryByText(/Not ready yet|Not available/)).toBeNull();
    // Nothing is selected: the detail asks for a choice.
    expect(screen.getByText("Choose a report")).toBeVisible();
  });

  it("keeps the status filter in the URL", async () => {
    const user = userEvent.setup();
    const { location } = renderRoutes("/reports", fakeServer(state()));
    await settled();
    const filters = screen.getByRole("group", { name: "Filter by status" });
    expect(
      within(filters).getByRole("button", { name: "All 3" }),
    ).toHaveAttribute("aria-pressed", "true");

    await user.click(
      within(filters).getByRole("button", { name: "Waiting for acceptance 1" }),
    );
    expect(location()).toBe("/reports?status=proposed");
    expect(rowTitles()).toEqual([
      "OWASP Top 10 · Source risks · Payment service",
    ]);

    await user.click(within(filters).getByRole("button", { name: "Ready 2" }));
    expect(location()).toBe("/reports?status=ready");
    expect(rowTitles()).toHaveLength(2);
    expect(
      screen.queryByRole("region", { name: "Waiting for acceptance" }),
    ).toBeNull();

    await user.click(within(filters).getByRole("button", { name: "All 3" }));
    expect(location()).toBe("/reports");
    expect(rowTitles()).toHaveLength(3);
  });

  it("opens with the filters a link carries and keeps them on selection", async () => {
    const user = userEvent.setup();
    const { location } = renderRoutes(
      "/reports?status=ready&project=project_payment",
      fakeServer(state()),
    );
    await screen.findByRole("button", { name: "Ready 1" });
    expect(screen.getByRole("button", { name: "Ready 1" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue(
      "project_payment",
    );
    expect(rowTitles()).toEqual([
      "ASVS 5.0 · Level 1 source review · Payment service",
    ]);

    await user.click(
      screen.getByRole("link", {
        name: "ASVS 5.0 · Level 1 source review · Payment service",
      }),
    );
    expect(location()).toBe(
      "/reports/audit_ready?status=ready&project=project_payment",
    );
    expect(
      screen.getByRole("link", {
        name: "ASVS 5.0 · Level 1 source review · Payment service",
      }),
    ).toHaveAttribute("aria-current", "true");
    // One-pane screens go back to the list as it was filtered.
    expect(
      screen.getByRole("link", { name: "Back to reports" }),
    ).toHaveAttribute("href", "/reports?status=ready&project=project_payment");
    // Changing a filter keeps the selected report, and Back follows it.
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Project" }),
      "",
    );
    expect(location()).toBe("/reports/audit_ready?status=ready");
    expect(
      screen.getByRole("link", { name: "Back to reports" }),
    ).toHaveAttribute("href", "/reports?status=ready");
  });

  it("filters by project in the URL", async () => {
    const user = userEvent.setup();
    const { location } = renderRoutes("/reports", fakeServer(state()));
    await settled();
    const select = screen.getByRole("combobox", { name: "Project" });
    expect(
      within(select)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual([
      "All projects",
      "crapi-workshop",
      "Empty project",
      "Payment service",
    ]);

    await user.selectOptions(select, "project_crapi");
    expect(location()).toBe("/reports?project=project_crapi");
    expect(rowTitles()).toEqual(["OpenAPI · Operation trace · crapi-workshop"]);
    expect(screen.getByRole("button", { name: "All 1" })).toBeVisible();

    await user.selectOptions(select, "project_empty");
    expect(location()).toBe("/reports?project=project_empty");
    expect(
      await screen.findByText("No reports in this project yet"),
    ).toBeVisible();
  });

  it("moves the selection with J and K", async () => {
    const user = userEvent.setup();
    const { location } = renderRoutes(
      "/reports?status=ready",
      fakeServer(state()),
    );
    await screen.findByRole("button", { name: "Ready 2" });
    await user.keyboard("j");
    expect(location()).toBe("/reports/audit_trace?status=ready");
    await user.keyboard("j");
    expect(location()).toBe("/reports/audit_ready?status=ready");
    await user.keyboard("k");
    expect(location()).toBe("/reports/audit_trace?status=ready");
  });

  it("says when nothing is listed", async () => {
    renderRoutes(
      "/reports",
      fakeServer({
        projects: [project("project_payment", "Payment service")],
        audits: [audit("project_payment", "audit_running", "active")],
        reports: {},
      }),
    );
    expect(await screen.findByText("No reports yet")).toBeVisible();
    expect(screen.getByRole("link", { name: "Go to checks" })).toHaveAttribute(
      "href",
      "/checks",
    );
    expect(screen.getByText("No reports yet.")).toBeVisible();
  });

  it("says which filter is empty", async () => {
    const user = userEvent.setup();
    const server = fakeServer(state());
    server.state.reports = {
      audit_trace: report("audit_trace", "ready"),
    };
    renderRoutes("/reports", server);
    await screen.findByRole("button", { name: "All 1" });
    await user.click(
      screen.getByRole("button", { name: "Waiting for acceptance 0" }),
    );
    expect(screen.getByText("Nothing is waiting for acceptance")).toBeVisible();
  });

  it("keeps the readable reports when some reads fail", async () => {
    const user = userEvent.setup();
    const server = fakeServer({
      ...state(),
      failingProjects: ["project_crapi"],
    });
    renderRoutes("/reports", server);
    await screen.findByRole("button", { name: "All 2" });
    expect(
      screen.getByText(
        "Some reports could not be loaded, so this list may be incomplete.",
      ),
    ).toBeVisible();
    expect(rowTitles()).toEqual([
      "OWASP Top 10 · Source risks · Payment service",
      "ASVS 5.0 · Level 1 source review · Payment service",
    ]);

    server.state.failingProjects = [];
    await user.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(rowTitles()).toHaveLength(3));
    expect(
      screen.queryByText(
        "Some reports could not be loaded, so this list may be incomplete.",
      ),
    ).toBeNull();
  });

  it("explains a list that cannot be read", async () => {
    const user = userEvent.setup();
    const server = fakeServer({ ...state(), indexFails: true });
    renderRoutes("/reports", server);
    expect(
      await screen.findByText("Reports could not be loaded"),
    ).toBeVisible();
    expect(screen.getByText("Index failed")).toBeVisible();
    // Unknown is not zero: no counts, and nothing says there are no reports.
    const filters = screen.getByRole("group", { name: "Filter by status" });
    expect(
      within(filters)
        .getAllByRole("button")
        .map((chip) => chip.textContent),
    ).toEqual(["Waiting for acceptance", "Ready", "All"]);
    expect(screen.queryByRole("button", { name: "All 0" })).toBeNull();
    expect(screen.queryByText("No reports yet.")).toBeNull();

    server.state.indexFails = false;
    await user.click(screen.getByRole("button", { name: "Try again" }));
    await settled();
    expect(rowTitles()).toHaveLength(3);
  });
});
