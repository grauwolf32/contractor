import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { queryKeys } from "../../api/query-keys";

import { failure, json } from "../../test/responses";
import {
  audit,
  project,
  renderRoutes,
  report,
  reportsServer,
  type ServerState,
} from "./test-support";

const CERTIFICATION = /not a security or compliance certification/;

function state(): ServerState {
  return {
    projects: [project("project_payment", "Payment service")],
    audits: [
      audit("project_payment", "audit_proposed", "completed"),
      audit("project_payment", "audit_ready", "completed", {
        profile: "owasp-asvs-5-0-l1-source-review",
      }),
      audit("project_payment", "audit_finishing", "finalizing"),
      audit("project_payment", "audit_failed", "failed"),
    ],
    reports: {
      audit_proposed: report("audit_proposed", "ready", {
        summary:
          "# Payment review\n\nThe summary records the retained evidence.",
      }),
      audit_ready: report("audit_ready", "ready", {
        summary:
          "# Coverage summary\n\n| Status | Count |\n| --- | ---: |\n| satisfied | 2 |\n",
      }),
      audit_finishing: report("audit_finishing", "pending"),
      audit_failed: report("audit_failed", "unavailable"),
    },
  };
}

/** The detail pane of /reports/:auditId. */
function detail() {
  return screen.getByRole("region", { name: "Report" });
}

afterEach(() => {
  document.title = "";
});

describe("Report detail", () => {
  it("shows a ready report with its header, links and files", async () => {
    const user = userEvent.setup();
    renderRoutes("/reports/audit_ready", reportsServer(state()));
    const pane = await waitFor(() => {
      const region = detail();
      within(region).getByText("Ready");
      return region;
    });
    expect(
      within(pane).getByRole("heading", {
        name: "ASVS 5.0 · Level 1 source review",
      }),
    ).toBeVisible();
    expect(document.title).toBe(
      "ASVS 5.0 · Level 1 source review report · Contractor",
    );
    expect(
      await within(pane).findByRole("link", { name: "Payment service" }),
    ).toHaveAttribute("href", "/projects/project_payment");
    expect(
      within(pane).getByRole("button", { name: "Copy check ID" }),
    ).toBeVisible();
    expect(
      within(pane).getByRole("link", { name: "Open check" }),
    ).toHaveAttribute("href", "/projects/project_payment/audits/audit_ready");
    expect(
      within(pane).getByRole("link", { name: "Review coverage gaps →" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_payment/audits/audit_ready/work?work=gap",
    );
    expect(
      await within(pane).findByRole("heading", { name: "Coverage summary" }),
    ).toBeVisible();
    expect(within(pane).getByRole("table")).toHaveTextContent("satisfied");
    expect(
      within(pane).queryByRole("region", { name: "Your decision" }),
    ).toBeNull();
    // Exact files stay behind Technical details.
    const technical = within(pane).getByText("Technical details");
    expect(
      within(pane).getByText("audit-audit_ready/report.json@r1"),
    ).not.toBeVisible();
    await user.click(technical);
    expect(
      within(pane).getByText("audit-audit_ready/report.json@r1"),
    ).toBeVisible();
    expect(
      within(pane).getByText("audit-audit_ready/report.md@r1"),
    ).toBeVisible();
  });

  it("downloads the JSON and the summary built from the report", async () => {
    const user = userEvent.setup();
    renderRoutes("/reports/audit_ready", reportsServer(state()));
    const json = await screen.findByRole("button", { name: "Download JSON" });
    const saved: { name: string; blob: Blob }[] = [];
    let latest: Blob | undefined;
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn((blob: Blob) => {
      latest = blob;
      return "blob:report-test";
    });
    URL.revokeObjectURL = vi.fn();
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        if (latest !== undefined)
          saved.push({ name: this.download, blob: latest });
      });
    try {
      await user.click(json);
      await user.click(
        screen.getByRole("button", { name: "Download summary" }),
      );
    } finally {
      click.mockRestore();
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
    }
    expect(saved.map(({ name, blob }) => [name, blob.type])).toEqual([
      ["audit_ready-report.json", "application/json"],
      ["audit_ready-report.md", "text/markdown"],
    ]);
    expect(JSON.parse(await saved[0]!.blob.text())).toEqual({
      conclusion: "completed-with-gaps",
      certification: false,
    });
    expect(await saved[1]!.blob.text()).toContain("# Coverage summary");
  });

  it("says when the check has not reached report generation", async () => {
    renderRoutes("/reports/audit_finishing", reportsServer(state()));
    expect(
      await screen.findByText("The check has not reached report generation."),
    ).toBeVisible();
    const pane = detail();
    expect(
      within(pane).getByText(
        "A report that is not ready yet is not a successful assessment.",
      ),
    ).toBeVisible();
    expect(within(pane).getByText("Not ready yet")).toBeVisible();
    expect(
      within(pane).queryByRole("button", { name: "Download summary" }),
    ).toBeNull();
    expect(
      within(pane).getByRole("link", { name: "Review coverage gaps →" }),
    ).toBeVisible();
  });

  it("says when no report is available", async () => {
    renderRoutes("/reports/audit_failed", reportsServer(state()));
    expect(await screen.findByText("No report is available.")).toBeVisible();
    expect(within(detail()).getByText("Not available")).toBeVisible();
    expect(
      within(detail()).queryByRole("heading", { name: "Summary" }),
    ).toBeNull();
  });

  it("keeps the last report when a refresh fails", async () => {
    const user = userEvent.setup();
    const server = reportsServer(state());
    let fail = false;
    const ready = server.state.reports.audit_ready;
    server.state.reports.audit_ready = () =>
      fail
        ? failure(
            503,
            "storage_transaction_conflict",
            "Report store unavailable",
          )
        : json(ready);
    const { queryClient } = renderRoutes("/reports/audit_ready", server);
    await screen.findByRole("heading", { name: "Coverage summary" });

    fail = true;
    await act(() =>
      queryClient.refetchQueries({
        queryKey: queryKeys.audits.report("audit_ready"),
      }),
    );
    const pane = detail();
    expect(
      await within(pane).findByText(
        "Could not refresh; showing the last loaded data.",
      ),
    ).toBeVisible();
    expect(within(pane).getByText("Report store unavailable")).toBeVisible();
    expect(
      within(pane).getByRole("heading", { name: "Coverage summary" }),
    ).toBeVisible();

    fail = false;
    await user.click(within(pane).getByRole("button", { name: "Try again" }));
    await waitFor(() =>
      expect(
        within(pane).queryByText(
          "Could not refresh; showing the last loaded data.",
        ),
      ).toBeNull(),
    );
  });

  it("explains a report that cannot be loaded and tries again", async () => {
    const user = userEvent.setup();
    const server = reportsServer(state());
    let fail = true;
    const ready = server.state.reports.audit_ready;
    server.state.reports.audit_ready = () =>
      fail ? failure(500, "internal_error", "Report read failed") : json(ready);
    renderRoutes("/reports/audit_ready", server);
    expect(await screen.findByText("Could not load the report")).toBeVisible();
    expect(screen.getByText("Report read failed")).toBeVisible();
    fail = false;
    await user.click(
      within(detail()).getByRole("button", { name: "Try again" }),
    );
    expect(
      await screen.findByRole("heading", { name: "Coverage summary" }),
    ).toBeVisible();
  });

  it("says once that a report is not a certification", async () => {
    const server = reportsServer(state());
    const { router } = renderRoutes("/reports/audit_ready", server);
    await screen.findByRole("heading", { name: "Coverage summary" });
    // The summary does not say it: the page does.
    expect(within(detail()).getAllByText(CERTIFICATION)).toHaveLength(1);
    expect(
      within(detail()).getByText(
        "A report describes this bounded check. It is not a security or compliance certification.",
      ),
    ).toBeVisible();

    // The Server's summaries say it themselves: not twice.
    server.state.reports.audit_proposed = report("audit_proposed", "ready", {
      summary:
        "# Audit audit_proposed\n\nCompletion describes the bounded Audit process; it is not a security or\ncompliance certification.\n",
    });
    await act(() => router.navigate("/reports/audit_proposed"));
    await within(detail()).findByRole("heading", {
      name: "Audit audit_proposed",
    });
    expect(within(detail()).getAllByText(CERTIFICATION)).toHaveLength(1);
    expect(
      within(detail()).queryByText(
        "A report describes this bounded check. It is not a security or compliance certification.",
      ),
    ).toBeNull();
  });

  it("explains a check that is not available or a broken link", async () => {
    const { router } = renderRoutes(
      "/reports/audit_gone",
      reportsServer(state()),
    );
    expect(
      await screen.findByText("This check is not available"),
    ).toBeVisible();
    await act(() => router.navigate("/reports/not%20an%20id"));
    expect(
      await screen.findByText("This report link is not valid"),
    ).toBeVisible();
  });
  it("shows normal completion without a report and links to saved task results", async () => {
    const fixture = state();
    fixture.reports.audit_ready = report("audit_ready", "unavailable");
    const server = reportsServer(fixture);
    renderRoutes("/reports/audit_ready", server);
    expect(
      await screen.findByText("This check completed without a summary report."),
    ).toBeVisible();
    expect(
      screen.getByRole("link", { name: "View task results →" }),
    ).toHaveAttribute("href", expect.stringContaining("audit_ready"));
    expect(screen.queryByRole("alert")).toBeNull();
  });
  it("keeps failed report reads distinct from normal unavailable reports", async () => {
    const fixture = state();
    fixture.reports.audit_ready = () =>
      failure(
        500,
        "internal_error",
        "Retained report failed integrity validation",
      );
    const server = reportsServer(fixture);
    renderRoutes("/reports/audit_ready", server);
    await waitFor(() =>
      expect(screen.getByText("Could not load the report")).toBeVisible(),
    );
    expect(
      screen.queryByText("This check completed without a summary report."),
    ).toBeNull();
  });
});
