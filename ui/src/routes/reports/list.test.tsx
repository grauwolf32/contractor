import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  audit,
  reportsServer,
  project,
  renderRoutes,
  report,
  type ServerState,
} from "./test-support";

/**
 * Three projects: two ready reports and
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
  await screen.findByRole("button", { name: "All 2" });
}

describe("Reports list", () => {
  it("pages availability reads and loads complete content only into the selected detail", async () => {
    const checks = Array.from({ length: 25 }, (_, index) =>
      audit("project_reports", `audit_${index}`, "completed"),
    );
    const server = reportsServer({
      projects: [project("project_reports", "Reports project")],
      audits: checks,
      reports: Object.fromEntries(
        checks.map((check) => [
          check.auditId,
          report(check.auditId, "ready", {
            summary: `Saved report ${check.auditId}`,
          }),
        ]),
      ),
    });
    const { user, queryClient, location } = renderRoutes(
      "/reports?project=project_reports",
      server,
    );
    await screen.findByRole("button", { name: "All 20" });
    expect(
      server.requests.filter((request) =>
        new URL(request.url).pathname.endsWith("/report"),
      ),
    ).toHaveLength(20);
    expect(rowTitles()).toHaveLength(20);
    expect(
      screen.getByText("Counts describe this page.", { exact: false }),
    ).toBeVisible();
    const paging = screen.getByRole("navigation", { name: "Report pages" });
    await user.click(within(paging).getByRole("button", { name: "Next" }));
    await screen.findByRole("button", { name: "All 5" });
    expect(location()).toBe("/reports?project=project_reports&page=1");
    expect(
      server.requests.filter((request) =>
        new URL(request.url).pathname.endsWith("/report"),
      ),
    ).toHaveLength(25);
    const selected = screen
      .getAllByRole("link")
      .find(
        (link) =>
          link.getAttribute("href") ===
          "/reports/audit_20?project=project_reports&page=1",
      )!;
    await user.click(selected);
    expect(await screen.findByText("Saved report audit_20")).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Back to reports" }),
    ).toHaveAttribute("href", "/reports?project=project_reports&page=1");
    const listCache = queryClient
      .getQueryCache()
      .findAll({ queryKey: ["cross-project", "report"] });
    expect(listCache).toHaveLength(25);
    expect(
      listCache.every(
        (query) =>
          JSON.stringify(query.state.data) ===
          JSON.stringify({ status: "ready" }),
      ),
    ).toBe(true);
    expect(
      queryClient.getQueryData<{ summary: string }>([
        "audits",
        "detail",
        "audit_20",
        "report",
      ])?.summary,
    ).toBe("Saved report audit_20");
  });

  it("keeps later report pages reachable when this check page has no ready reports", async () => {
    const checks = Array.from({ length: 21 }, (_, index) =>
      audit("project_reports", `audit_${index}`, "completed"),
    );
    const server = reportsServer({
      projects: [project("project_reports", "Reports project")],
      audits: checks,
      reports: Object.fromEntries(
        checks.map((check, index) => [
          check.auditId,
          report(check.auditId, index < 20 ? "unavailable" : "ready"),
        ]),
      ),
    });
    const { user } = renderRoutes("/reports", server);
    expect(
      await screen.findByText("No ready reports in these checks"),
    ).toBeVisible();
    expect(screen.queryByText("No reports yet")).toBeNull();
    await user.click(
      within(
        screen.getByRole("navigation", { name: "Report pages" }),
      ).getByRole("button", { name: "Next" }),
    );
    await screen.findByRole("button", { name: "All 1" });
    expect(rowTitles()).toHaveLength(1);
  });

  it("lists ready reports across projects without acceptance actions", async () => {
    renderRoutes("/reports", reportsServer(state()));
    await settled();
    expect(rowTitles()).toEqual([
      "OpenAPI · Operation trace · crapi-workshop",
      "ASVS 5.0 · Level 1 source review · Payment service",
    ]);
    expect(screen.getByRole("region", { name: "Ready" })).toBeVisible();
    expect(screen.queryByText("Waiting for acceptance")).toBeNull();
    expect(screen.getByText("Choose a report")).toBeVisible();
  });

  it("keeps the status filter in the URL", async () => {
    const user = userEvent.setup();
    const { location } = renderRoutes("/reports", reportsServer(state()));
    await settled();
    await user.click(screen.getByRole("button", { name: "Ready 2" }));
    expect(location()).toBe("/reports?status=ready");
    expect(rowTitles()).toHaveLength(2);
    await user.click(screen.getByRole("button", { name: "All 2" }));
    expect(location()).toBe("/reports");
  });

  it("opens with the filters a link carries and keeps them on selection", async () => {
    const user = userEvent.setup();
    const { location } = renderRoutes(
      "/reports?status=ready&project=project_payment",
      reportsServer(state()),
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
    const { location } = renderRoutes("/reports", reportsServer(state()));
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
      reportsServer(state()),
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
      reportsServer({
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

  it("keeps the readable reports when some reads fail", async () => {
    const user = userEvent.setup();
    const server = reportsServer({
      ...state(),
      failingProjects: ["project_crapi"],
    });
    renderRoutes("/reports", server);
    await screen.findByRole("button", { name: "All 1" });
    expect(
      screen.getByText(
        "Some reports could not be loaded, so this list may be incomplete.",
      ),
    ).toBeVisible();
    expect(rowTitles()).toEqual([
      "ASVS 5.0 · Level 1 source review · Payment service",
    ]);

    server.state.failingProjects = [];
    await user.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(rowTitles()).toHaveLength(2));
    expect(
      screen.queryByText(
        "Some reports could not be loaded, so this list may be incomplete.",
      ),
    ).toBeNull();
  });

  it("explains a list that cannot be read", async () => {
    const user = userEvent.setup();
    const server = reportsServer({ ...state(), indexFails: true });
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
    ).toEqual(["Ready", "All"]);
    expect(screen.queryByRole("button", { name: "All 0" })).toBeNull();
    expect(screen.queryByText("No reports yet.")).toBeNull();

    server.state.indexFails = false;
    await user.click(screen.getByRole("button", { name: "Try again" }));
    await settled();
    expect(rowTitles()).toHaveLength(2);
  });
  it("lists ready reports and leaves completed checks without a summary out of the list", async () => {
    const server = reportsServer({
      projects: [project("project_reports", "Reports project")],
      audits: [
        audit("project_reports", "audit_ready", "completed"),
        audit("project_reports", "audit_results", "completed"),
      ],
      reports: {
        audit_ready: report("audit_ready", "ready"),
        audit_results: report("audit_results", "unavailable"),
      },
    });
    renderRoutes("/reports", server);
    await waitFor(() =>
      expect(
        screen
          .getAllByRole("link")
          .some((link) =>
            link.getAttribute("href")?.startsWith("/reports/audit_ready"),
          ),
      ).toBe(true),
    );
    expect(
      screen
        .getAllByRole("link")
        .some((link) =>
          link.getAttribute("href")?.startsWith("/reports/audit_results"),
        ),
    ).toBe(false);
    expect(screen.queryByText(/waiting for acceptance/i)).toBeNull();
    expect(
      server.requests.some((request) =>
        new URL(request.url).pathname.includes("/reviews"),
      ),
    ).toBe(false);
  });
});
