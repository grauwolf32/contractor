import { useQuery } from "@tanstack/react-query";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useParams } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import { getAudit } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { AuditReportView } from "../projects/audits/report";
import {
  acceptanceReview,
  audit,
  failure,
  fakeServer,
  project,
  renderRoutes,
  report,
  type ServerState,
} from "./test-support";

const CERTIFICATION = /not a security or compliance certification/;

function state(): ServerState {
  return {
    projects: [project("project_payment", "Payment service")],
    audits: [
      audit("project_payment", "audit_proposed", "waiting_review"),
      audit("project_payment", "audit_ready", "completed", {
        profile: "owasp-asvs-5-0-l1-source-review",
      }),
      audit("project_payment", "audit_finishing", "finalizing"),
      audit("project_payment", "audit_failed", "failed"),
    ],
    reports: {
      audit_proposed: report("audit_proposed", "proposed", {
        summary: "# Payment review\n\nThe owner must accept this summary.",
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
  it("shows a proposed report as waiting for acceptance, never as accepted", async () => {
    renderRoutes("/reports/audit_proposed", fakeServer(state()));
    expect(
      await screen.findByText("This report is awaiting owner acceptance."),
    ).toBeVisible();
    const pane = detail();
    expect(
      within(pane).getByRole("heading", {
        level: 2,
        name: "OWASP Top 10 · Source risks",
      }),
    ).toBeVisible();
    expect(within(pane).getByText("Waiting for acceptance")).toBeVisible();
    expect(
      within(pane).getByText(
        "Review the contents below before making a decision.",
      ),
    ).toBeVisible();
    expect(
      await within(pane).findByRole("heading", { name: "Payment review" }),
    ).toBeVisible();
    const decision = within(pane).getByRole("region", {
      name: "Your decision",
    });
    expect(
      within(decision)
        .getAllByRole("button", { pressed: false })
        .map((button) => button.textContent),
    ).toEqual(["Approve", "Reject"]);
    expect(
      within(decision).getByRole("button", { name: "Record decision" }),
    ).toBeDisabled();
    // Nothing reads as accepted before the Server records an approval.
    expect(within(pane).queryByText("Ready")).toBeNull();
    expect(within(pane).queryByText("Approved")).toBeNull();
  });

  it("records the acceptance and keeps the decision on the page", async () => {
    const user = userEvent.setup();
    const server = fakeServer(state());
    renderRoutes("/reports/audit_proposed?status=proposed", server);
    const decision = await screen.findByRole("region", {
      name: "Your decision",
    });
    await user.click(within(decision).getByRole("button", { name: "Approve" }));
    await user.type(
      within(decision).getByRole("textbox", { name: "Why" }),
      "Coverage and gaps are stated plainly.",
    );
    await user.click(
      within(decision).getByRole("button", { name: "Record decision" }),
    );

    expect(
      await screen.findByText("Decision recorded: Approved."),
    ).toBeVisible();
    const [post] = server.sent(
      "POST",
      "/audits/audit_proposed/reviews/review_audit_proposed/decisions",
    );
    expect(post?.headers.get("If-Match")).toBe('"1"');
    expect(await post?.json()).toEqual({
      action: "approve",
      rationale: "Coverage and gaps are stated plainly.",
    });
    const pane = detail();
    // The report is ready now, and the decision stays with its reason.
    await waitFor(() =>
      expect(
        within(pane).queryByText("This report is awaiting owner acceptance."),
      ).toBeNull(),
    );
    const record = within(pane).getByRole("group", {
      name: "Decision on report acceptance",
    });
    expect(within(record).getByText("Approved")).toBeVisible();
    expect(
      await within(record).findByText("Coverage and gaps are stated plainly."),
    ).toBeVisible();
    expect(record).toHaveFocus();
    expect(within(pane).getByText("Ready")).toBeVisible();
    expect(
      within(pane).queryByRole("region", { name: "Your decision" }),
    ).toBeNull();
    // The list follows: nothing waits for acceptance any more.
    expect(
      await screen.findByRole("button", { name: "Waiting for acceptance 0" }),
    ).toBeVisible();
  });

  it("shows a decision recorded elsewhere once the report moves on", async () => {
    const server = fakeServer(state());
    const { queryClient } = renderRoutes("/reports/audit_proposed", server);
    await screen.findByRole("region", { name: "Your decision" });
    const review = acceptanceReview("audit_proposed");
    // Another session approves the report; the next read shows it ready.
    server.reviews.set(review.requestId, {
      ...review,
      state: "decided",
      revision: 2,
      decision: {
        decisionId: "decision_elsewhere",
        requestId: review.requestId,
        auditId: review.auditId,
        action: "approve",
        actorId: "user_other",
        rationale: "Accepted in the review meeting.",
        subjectRevision: review.subjectRevision,
        subjectDigest: review.subjectDigest,
        createdAt: review.createdAt,
      },
    });
    server.state.reports.audit_proposed = report("audit_proposed", "ready");
    await act(() =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.report("audit_proposed"),
      }),
    );
    const pane = detail();
    const record = await within(pane).findByRole("group", {
      name: "Decision on report acceptance",
    });
    expect(await within(record).findByText("user_other")).toBeVisible();
    expect(within(record).getByText("Approved")).toBeVisible();
    expect(within(pane).getByText("Ready")).toBeVisible();
    expect(
      within(pane).queryByRole("region", { name: "Your decision" }),
    ).toBeNull();
    expect(server.sent("POST", "/decisions")).toEqual([]);
  });

  it("ends without an accepted report when the owner rejects it", async () => {
    const user = userEvent.setup();
    renderRoutes("/reports/audit_proposed", fakeServer(state()));
    const decision = await screen.findByRole("region", {
      name: "Your decision",
    });
    await user.click(within(decision).getByRole("button", { name: "Reject" }));
    await user.type(
      within(decision).getByRole("textbox", { name: "Why" }),
      "The summary misses the export endpoint.",
    );
    await user.keyboard("{Control>}{Enter}{/Control}");

    expect(
      await screen.findByText("Decision recorded: Rejected."),
    ).toBeVisible();
    const pane = detail();
    expect(
      await within(pane).findByText("No accepted report is available."),
    ).toBeVisible();
    expect(within(pane).getByText("Not available")).toBeVisible();
    expect(
      within(
        within(pane).getByRole("group", {
          name: "Decision on report acceptance",
        }),
      ).getByText("Rejected"),
    ).toBeVisible();
  });

  it("shows a ready report with its header, links and files", async () => {
    const user = userEvent.setup();
    renderRoutes("/reports/audit_ready", fakeServer(state()));
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
      "/projects/project_payment/audits/audit_ready/coverage?result=uncertain",
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
    renderRoutes("/reports/audit_ready", fakeServer(state()));
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
    renderRoutes("/reports/audit_finishing", fakeServer(state()));
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

  it("says when no accepted report is available", async () => {
    renderRoutes("/reports/audit_failed", fakeServer(state()));
    expect(
      await screen.findByText("No accepted report is available."),
    ).toBeVisible();
    expect(within(detail()).getByText("Not available")).toBeVisible();
    expect(
      within(detail()).queryByRole("heading", { name: "Summary" }),
    ).toBeNull();
  });

  it("keeps the last report when a refresh fails", async () => {
    const user = userEvent.setup();
    const server = fakeServer(state());
    let fail = false;
    const ready = server.state.reports.audit_ready;
    server.state.reports.audit_ready = () =>
      fail
        ? failure(503, "Report store unavailable")
        : new Response(JSON.stringify(ready), {
            headers: {
              "Content-Type": "application/json",
              "X-Contractor-API-Version": "contractor.public.v1",
            },
          });
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
    const server = fakeServer(state());
    let fail = true;
    const ready = server.state.reports.audit_ready;
    server.state.reports.audit_ready = () =>
      fail
        ? failure(500, "Report read failed")
        : new Response(JSON.stringify(ready), {
            headers: {
              "Content-Type": "application/json",
              "X-Contractor-API-Version": "contractor.public.v1",
            },
          });
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
    const server = fakeServer(state());
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
    server.state.reports.audit_proposed = report("audit_proposed", "proposed", {
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

  it("does not decide a review that the report does not carry", async () => {
    const server = fakeServer(state());
    renderRoutes("/reports/audit_proposed?review=review_removed", server);
    expect(
      await screen.findByText(
        "The requested report review is unavailable or no longer current.",
      ),
    ).toBeVisible();
    expect(
      screen.getByText("This report cannot be used to decide that review."),
    ).toBeVisible();
    expect(screen.queryByRole("region", { name: "Your decision" })).toBeNull();
    expect(server.sent("POST", "/decisions")).toEqual([]);
  });

  it("offers the review a link names when the report carries it", async () => {
    renderRoutes(
      `/reports/audit_proposed?review=${acceptanceReview("audit_proposed").requestId}`,
      fakeServer(state()),
    );
    expect(
      await screen.findByRole("region", { name: "Your decision" }),
    ).toBeVisible();
    expect(
      screen.queryByText(
        "The requested report review is unavailable or no longer current.",
      ),
    ).toBeNull();
  });

  it("explains a check that is not available or a broken link", async () => {
    const { router } = renderRoutes("/reports/audit_gone", fakeServer(state()));
    expect(
      await screen.findByText("This check is not available"),
    ).toBeVisible();
    await act(() => router.navigate("/reports/not%20an%20id"));
    expect(
      await screen.findByText("This report link is not valid"),
    ).toBeVisible();
  });
});

function CheckReportSection() {
  const api = usePublicAPI();
  const { auditId = "" } = useParams();
  const check = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    queryFn: () => getAudit(api, auditId),
  });
  return check.data === undefined ? null : (
    <AuditReportView audit={check.data} api={api} />
  );
}

describe("AuditReportView", () => {
  const routes = [
    {
      path: "/projects/:projectId/audits/:auditId/report",
      element: <CheckReportSection />,
    },
  ];

  it("shows the report on the check page with the decision inline", async () => {
    renderRoutes(
      "/projects/project_payment/audits/audit_proposed/report",
      fakeServer(state()),
      routes,
    );
    const section = await screen.findByRole("region", { name: "Report" });
    expect(
      within(section).getByRole("heading", { level: 2, name: "Report" }),
    ).toBeVisible();
    expect(
      await within(section).findByText("Waiting for acceptance"),
    ).toBeVisible();
    expect(
      within(section).getByText("This report is awaiting owner acceptance."),
    ).toBeVisible();
    const acceptance = within(section).getByRole("region", {
      name: "Report acceptance",
    });
    expect(
      within(acceptance).getByRole("region", { name: "Your decision" }),
    ).toBeVisible();
    expect(
      within(section).getByRole("button", { name: "Download JSON" }),
    ).toBeVisible();
  });

  it("offers no decision on a ready report", async () => {
    renderRoutes(
      "/projects/project_payment/audits/audit_ready/report",
      fakeServer(state()),
      routes,
    );
    const section = await screen.findByRole("region", { name: "Report" });
    expect(await within(section).findByText("Ready")).toBeVisible();
    expect(
      within(section).queryByRole("region", { name: "Report acceptance" }),
    ).toBeNull();
  });
});
