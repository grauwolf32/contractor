import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { DIGEST } from "../decisions/test-support";
import {
  check,
  issue,
  project,
  renderIssues,
  server as fakeServer,
} from "./test-support";

const shop = project("project_shop", "Shop");
const billing = project("project_billing", "Billing");

/**
 * Two projects: Shop with a running trace (an older possible issue on an
 * endpoint) and Billing with a finished check (a newer possible issue and a
 * confirmed issue).
 */
function twoProjects() {
  return fakeServer({
    projects: [shop, billing],
    audits: [
      check(shop.projectId, "audit_shop", "active"),
      check(
        billing.projectId,
        "audit_billing",
        "completed",
        "owasp-top10-2025-source-risk",
      ),
    ],
    findings: [
      issue(
        "audit_shop",
        "finding_orders",
        "Any user can read any order",
        "2026-10-04T10:00:00Z",
        {},
        {
          subject: { kind: "openapi-operation", key: "GET /orders/{id}" },
          standard_refs: [
            { scheme: "CWE", version: "4.20", requirement_id: "CWE-639" },
          ],
          severity_suggestion: "high",
        },
      ),
      issue(
        "audit_billing",
        "finding_invoices",
        "Invoices leak through a debug endpoint",
        "2026-10-05T09:00:00Z",
        {},
        { severity_suggestion: "" },
      ),
      issue(
        "audit_billing",
        "finding_refunds",
        "Refunds skip the approval step",
        "2026-10-03T09:00:00Z",
        {
          state: "confirmed",
          analystVerdict: "true_positive",
          analystSeverity: "high",
        },
        { severity_suggestion: "low" },
      ),
    ],
  });
}

function rowTitles() {
  return [
    ...screen
      .getByRole("region", { name: "Possible issues" })
      .querySelectorAll("li.ui-row"),
  ].map((row) => row.querySelector(".ui-row-title")?.textContent);
}

describe("Issues", () => {
  it("lists possible issues across projects, newest first, with filters in the URL", async () => {
    const server = twoProjects();
    const { router } = renderIssues(server, "/issues");
    const list = await screen.findByRole("region", { name: "Possible issues" });
    expect(
      within(list).getByRole("heading", { level: 1, name: "Possible issues" }),
    ).toBeVisible();
    expect(
      within(list).getByText("Across all projects, newest first"),
    ).toBeVisible();
    await waitFor(() =>
      expect(rowTitles()).toEqual([
        "Invoices leak through a debug endpoint",
        "Any user can read any order",
      ]),
    );
    const states = within(list).getByRole("group", { name: "Filter by state" });
    expect(
      await within(states).findByRole("button", { name: "Needs review 2" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(
      within(states).getByRole("button", { name: "Confirmed 1" }),
    ).toBeVisible();
    expect(within(states).getByRole("button", { name: "All 3" })).toBeVisible();

    const orders = screen
      .getByRole("link", { name: "Any user can read any order" })
      .closest("li") as HTMLElement;
    expect(within(orders).getByText("GET")).toHaveClass("ui-method-chip");
    expect(within(orders).getByText("/orders/{id}")).toBeVisible();
    expect(within(orders).getByText("Shop")).toBeVisible();
    expect(within(orders).getByText("CWE-639")).toBeVisible();
    expect(within(orders).getByText("Needs review")).toBeVisible();
    expect(within(orders).getByText("Severity not set")).toBeVisible();
    expect(within(orders).getByText("AI suggestion: High")).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Any user can read any order" }),
    ).toHaveAttribute("href", "/issues/audit_shop/finding_orders");

    const user = userEvent.setup();
    await user.click(
      within(states).getByRole("button", { name: "Confirmed 1" }),
    );
    expect(router.state.location.search).toBe("?state=confirmed");
    await waitFor(() =>
      expect(rowTitles()).toEqual(["Refunds skip the approval step"]),
    );
    expect(screen.getByText("Severity: High")).toBeVisible();

    await user.selectOptions(screen.getByLabelText("Project"), "Billing");
    expect(router.state.location.search).toBe(
      "?state=confirmed&project=project_billing",
    );
    expect(within(list).getByText("In Billing, newest first")).toBeVisible();
    await user.click(within(states).getByRole("button", { name: /^All/ }));
    expect(router.state.location.search).toBe(
      "?state=all&project=project_billing",
    );
    await waitFor(() =>
      expect(rowTitles()).toEqual([
        "Invoices leak through a debug endpoint",
        "Refunds skip the approval step",
      ]),
    );
  });

  it("filters severity by the analyst's rating, never the AI suggestion", async () => {
    const server = twoProjects();
    server.findings.push(
      issue(
        "audit_shop",
        "finding_low",
        "Verbose errors on login",
        "2026-10-02T09:00:00Z",
        {
          state: "confirmed",
          analystVerdict: "true_positive",
          analystSeverity: "low",
        },
        { severity_suggestion: "high" },
      ),
    );
    const { requests, router } = renderIssues(server, "/issues?state=all");
    await waitFor(() => expect(rowTitles()).toHaveLength(4));
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Severity"), "high");
    expect(router.state.location.search).toBe("?state=all&severity=high");
    // Only the issue the analyst rated High; the AI suggested High for the
    // possible issue on /orders and the low-rated issue.
    await waitFor(() =>
      expect(rowTitles()).toEqual(["Refunds skip the approval step"]),
    );
    const reads = requests
      .map((request) => new URL(request.url))
      .filter((url) => url.pathname.endsWith("/findings"));
    expect(
      reads.some((url) => url.searchParams.get("severity") === "high"),
    ).toBe(true);
    expect(
      reads.every(
        (url) =>
          !url.searchParams.has("severity") ||
          url.searchParams.get("severity") === "high",
      ),
    ).toBe(true);
  });

  it("opens a possible issue with Summary, Evidence and History tabs and masks credentials", async () => {
    const server = twoProjects();
    const orders = server.findings[0]!;
    orders.firstProposal.document.locations = [
      { file: "orders/views.py", line: 42 },
      { url: "https://shop.example/orders/1", method: "GET" },
    ];
    orders.firstProposal.document.evidence_ids = ["response"];
    orders.firstProposal.document.http_exchange = {
      request_id: 7,
      request_tag: "probe",
      response_body_evidence_id: "response",
      attempts: [
        {
          method: "GET",
          url: "https://shop.example/orders/1",
          headers: [
            { name: "Accept", value: "application/json" },
            { name: "Authorization", value: "Bearer secret-token" },
          ],
          body_base64: btoa('{"note":"probe"}'),
          status: 200,
          response_headers: [
            { name: "Set-Cookie", value: "session=abc123" },
            { name: "Content-Type", value: "application/json" },
          ],
        },
      ],
    };
    orders.firstProposal.evidence = [
      {
        ref: { namespace: "audit-evidence", name: "body", revision: "r1" },
        digest: DIGEST,
        mediaType: "application/json",
        sizeBytes: 64,
      },
    ];
    renderIssues(server, "/issues/audit_shop/finding_orders");
    const review = await screen.findByRole("region", { name: "Review" });
    expect(
      await within(review).findByText("Possible issue 2 of 2"),
    ).toBeVisible();
    expect(
      within(review).getByRole("link", { name: "View in its check" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_shop/audits/audit_shop/findings?finding=finding_orders",
    );
    const tabs = within(review).getByRole("tablist", {
      name: "Review sections",
    });
    expect(within(tabs).getByRole("tab", { name: "Summary" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      await within(review).findByRole("heading", {
        level: 2,
        name: "Any user can read any order",
      }),
    ).toBeVisible();
    expect(
      within(review).getByRole("heading", {
        name: "Sources used by this check",
      }),
    ).toBeVisible();

    const user = userEvent.setup();
    await user.click(within(tabs).getByRole("tab", { name: "Evidence 4" }));
    const panel = within(review).getByRole("tabpanel", { name: "Evidence 4" });
    expect(
      within(panel).getByRole("heading", { name: "Captured HTTP exchange" }),
    ).toBeVisible();
    expect(within(panel).getByText("orders/views.py:42")).toBeVisible();
    expect(
      within(panel).getByText("GET https://shop.example/orders/1"),
    ).toBeVisible();
    expect(within(panel).getByText("HTTP 200")).toBeVisible();
    expect(within(panel).getByText('{"note":"probe"}')).toBeVisible();
    expect(
      within(panel).getByText("Accept", { selector: "dt" }).nextElementSibling,
    ).toHaveTextContent("application/json");
    // Credentials stay out of the page until shown, one value at a time.
    expect(review).not.toHaveTextContent("Bearer secret-token");
    expect(review).not.toHaveTextContent("session=abc123");
    await user.click(
      within(panel).getByRole("button", { name: "Show Authorization value" }),
    );
    expect(within(panel).getByText("Bearer secret-token")).toBeVisible();
    expect(review).not.toHaveTextContent("session=abc123");
    await user.click(
      within(panel).getByRole("button", { name: "Hide Authorization value" }),
    );
    expect(review).not.toHaveTextContent("Bearer secret-token");
    expect(
      within(panel).getByRole("button", { name: "Show Set-Cookie value" }),
    ).toBeVisible();
    expect(within(panel).getByText(/Response body ·/)).toBeVisible();

    // Arrow keys move between the tabs.
    within(tabs).getByRole("tab", { name: "Evidence 4" }).focus();
    await user.keyboard("{ArrowRight}");
    const history = within(tabs).getByRole("tab", { name: "History" });
    expect(history).toHaveFocus();
    expect(history).toHaveAttribute("aria-selected", "true");
    const historyPanel = within(review).getByRole("tabpanel", {
      name: "History",
    });
    expect(
      await within(historyPanel).findByText(
        "No decision has been recorded yet.",
      ),
    ).toBeVisible();
    await user.click(
      within(historyPanel).getByRole("button", { name: "Show provenance" }),
    );
    expect(
      await within(historyPanel).findByText("source-review", { exact: true }),
    ).toBeVisible();
  });

  it("records a decision, announces it and moves on when the possible issue leaves the list", async () => {
    const server = twoProjects();
    const { router, sent } = renderIssues(
      server,
      "/issues/audit_billing/finding_invoices",
    );
    const review = await screen.findByRole("region", { name: "Review" });
    expect(
      await within(review).findByText("Possible issue 1 of 2"),
    ).toBeVisible();
    const bar = await within(review).findByRole("region", {
      name: "Your decision",
    });
    const user = userEvent.setup();
    await user.click(
      await within(bar).findByRole("button", { name: "Confirm issue" }),
    );
    await user.click(within(bar).getByRole("radio", { name: "High" }));
    await user.type(
      within(bar).getByRole("textbox", { name: "Why" }),
      "The debug endpoint returns invoices without a session.",
    );
    await user.click(
      within(bar).getByRole("button", { name: "Record decision" }),
    );

    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/issues/audit_shop/finding_orders",
      ),
    );
    expect(
      screen.getByText(
        "Decision recorded: Confirmed · High. Showing the next possible issue: Any user can read any order.",
      ),
    ).toBeVisible();
    expect(sent("POST", "/decisions")).toHaveLength(1);
    // The decided issue leaves the Needs review list once it is read again.
    await waitFor(() =>
      expect(rowTitles()).toEqual(["Any user can read any order"]),
    );
    expect(
      await within(screen.getByRole("region", { name: "Review" })).findByText(
        "Possible issue 1 of 1",
      ),
    ).toBeVisible();
    expect(
      screen
        .getByRole("region", { name: "Review" })
        .querySelector(".issues-detail-bar"),
    ).toHaveFocus();
  });

  it("returns to the list after deciding the last possible issue, and stays on one opened from a link", async () => {
    const server = twoProjects();
    const { router } = renderIssues(
      server,
      "/issues/audit_shop/finding_orders?project=project_shop",
    );
    const review = await screen.findByRole("region", { name: "Review" });
    expect(
      await within(review).findByText("Possible issue 1 of 1"),
    ).toBeVisible();
    const user = userEvent.setup();
    let bar = await within(review).findByRole("region", {
      name: "Your decision",
    });
    // R chooses "Not an issue" and moves to the reason.
    await user.click(within(review).getByRole("tab", { name: "Summary" }));
    await user.keyboard("r");
    const reason = within(bar).getByRole("textbox", { name: "Why" });
    expect(reason).toHaveFocus();
    await user.type(reason, "Orders are scoped by the session.");
    await user.click(
      within(bar).getByRole("button", { name: "Record decision" }),
    );
    await waitFor(() => expect(router.state.location.pathname).toBe("/issues"));
    expect(router.state.location.search).toBe("?project=project_shop");
    expect(
      screen.getByText(
        "Decision recorded: Not an issue. That was the last possible issue in this list.",
      ),
    ).toBeVisible();

    // A possible issue outside the list (here: already decided) stays open.
    await act(async () => {
      await router.navigate(
        "/issues/audit_billing/finding_invoices?state=rejected",
      );
    });
    bar = await screen.findByRole("region", { name: "Your decision" });
    expect(await screen.findByText("Not in this list")).toBeVisible();
    await user.click(
      within(bar).getByRole("button", { name: "Needs evidence" }),
    );
    await user.type(
      within(bar).getByRole("textbox", { name: "Why" }),
      "Attach the debug response.",
    );
    await user.click(
      within(bar).getByRole("button", { name: "Record decision" }),
    );
    expect(
      await screen.findByRole("region", { name: "Current decision" }),
    ).toBeVisible();
    expect(router.state.location.pathname).toBe(
      "/issues/audit_billing/finding_invoices",
    );
  });

  it("moves with J and K and the previous and next buttons", async () => {
    const server = twoProjects();
    const { router } = renderIssues(server, "/issues");
    await waitFor(() => expect(rowTitles()).toHaveLength(2));
    expect(
      screen.getByText("Choose a possible issue", { exact: false }),
    ).toBeVisible();
    const user = userEvent.setup();
    await user.keyboard("j");
    expect(router.state.location.pathname).toBe(
      "/issues/audit_billing/finding_invoices",
    );
    await user.keyboard("j");
    expect(router.state.location.pathname).toBe(
      "/issues/audit_shop/finding_orders",
    );
    expect(
      screen.getByRole("link", { name: "Any user can read any order" }),
    ).toHaveAttribute("aria-current", "true");
    await user.keyboard("k");
    expect(router.state.location.pathname).toBe(
      "/issues/audit_billing/finding_invoices",
    );
    const next = await screen.findByRole("button", {
      name: "Next possible issue",
    });
    expect(next).toHaveAttribute("aria-keyshortcuts", "J");
    expect(
      screen.getByRole("button", { name: "Previous possible issue" }),
    ).toBeDisabled();
    await user.click(next);
    expect(router.state.location.pathname).toBe(
      "/issues/audit_shop/finding_orders",
    );
  });

  it("lists what loaded when a check fails, with Retry, and names checks with more", async () => {
    const server = twoProjects();
    server.failing.add("audit_billing");
    server.more.add("audit_shop");
    renderIssues(server, "/issues");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "Some possible issues could not be loaded.",
    );
    expect(alert).toHaveTextContent(
      "Possible issues of OWASP Top 10 · Source risks on Billing could not be loaded: Findings unavailable",
    );
    expect(rowTitles()).toEqual(["Any user can read any order"]);
    const note = screen.getByRole("note");
    expect(note).toHaveTextContent("Some possible issues are not listed here.");
    expect(
      within(note).getByRole("link", {
        name: "OpenAPI · Operation trace on Shop",
      }),
    ).toHaveAttribute(
      "href",
      "/projects/project_shop/audits/audit_shop/findings?state=proposed",
    );
    server.failing.delete("audit_billing");
    await act(async () => {
      await userEvent
        .setup()
        .click(within(alert).getByRole("button", { name: "Retry" }));
    });
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(rowTitles()).toEqual([
      "Invoices leak through a debug endpoint",
      "Any user can read any order",
    ]);
  });

  it("says when everything was reviewed and a check is still running", async () => {
    const server = twoProjects();
    server.findings = server.findings.filter(
      (finding) => finding.state !== "proposed",
    );
    renderIssues(server, "/issues");
    expect(
      await screen.findByText("That is everything that needs review."),
    ).toBeVisible();
    expect(
      screen.getByText(
        "OpenAPI · Operation trace on Shop is still running, so more may arrive.",
      ),
    ).toBeVisible();
  });

  it("blocks deciding on a review link that no longer matches the possible issue", async () => {
    const server = twoProjects();
    const orders = server.findings[0]!;
    orders.revision = 3;
    server.reviews.push({
      requestId: "review_old",
      auditId: "audit_shop",
      findingId: orders.findingId,
      subjectKind: "finding",
      subjectId: orders.findingId,
      kind: "finding-triage",
      subjectRevision: 2,
      subjectDigest: DIGEST,
      requestedActions: ["true_positive", "false_positive"],
      state: "pending",
      revision: 1,
      createdAt: "2026-10-05T10:00:00Z",
      updatedAt: "2026-10-05T10:00:00Z",
    });
    const { router } = renderIssues(
      server,
      "/issues/audit_shop/finding_orders?review=review_old",
    );
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "The review this link names no longer matches this possible issue.",
    );
    expect(screen.queryByRole("region", { name: "Your decision" })).toBeNull();
    await userEvent.setup().click(
      within(alert).getByRole("button", {
        name: "Decide on the current version",
      }),
    );
    expect(router.state.location.search).toBe("");
    expect(
      await screen.findByRole("region", { name: "Your decision" }),
    ).toBeVisible();
  });
});
