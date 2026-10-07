import {
  act,
  fireEvent,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  Audit,
  AuditCoverageRow,
  AuditEvent,
  AuditFinding,
  AuditItem,
  AuditReviewRequest,
} from "../../../api/audits";
import { queryKeys } from "../../../api/query-keys";
import * as queryClientFactory from "../../../app/query-client";
import {
  DIGEST,
  emptyPage,
  endpointRow,
  fakeAPI,
  jsonResponse,
  makeAttempt,
  makeAudit,
  makeFinding,
  makeItem,
  makeProject,
  makeReview,
  renderApplication,
  requirementRow,
  workspaceOf,
  type Handler,
} from "./check-test-support";

const ROOT = "/projects/project_example/audits/audit_example";
const project = makeProject("project_example", "Payment service");

function auditAt(
  state: Audit["state"],
  revision: number,
  overrides: Partial<Audit> & { profileName?: string } = {},
): Audit {
  return makeAudit("audit_example", "project_example", state, {
    profileName: "owasp-top10-2025-source-risk",
    revision,
    eventSequence: revision,
    updatedAt: `2026-10-05T10:0${revision}:00Z`,
    outstandingRunCount: state === "active" ? 1 : 0,
    ...overrides,
  });
}

/** Answers of the fake Server for one check; the rest is an empty page. */
interface CheckServer {
  audit: () => Audit;
  coverage?: (url: URL) => Response | AuditCoverageRow[];
  items?: (url: URL) => Response | AuditItem[];
  findings?: (url: URL) => Response | AuditFinding[];
  reviews?: (url: URL) => Response | AuditReviewRequest[];
  workspace?: (audit: Audit) => Response | object;
  report?: () => unknown;
  events?: (url: URL) => Response | AuditEvent[];
  /** Asked first, e.g. for mutations. */
  handle?: Handler;
}

function page(items: unknown[], audit: Audit) {
  return jsonResponse({
    items,
    page: { hasMore: false },
    total: items.length,
    auditRevision: audit.revision,
    asOf: audit.updatedAt,
  });
}

function serveCheck(server: CheckServer) {
  return fakeAPI(async (request, url) => {
    const handled = await server.handle?.(request, url);
    if (handled !== undefined) return handled;
    const path = url.pathname;
    const audit = server.audit();
    if (path === "/v1/projects/project_example")
      return jsonResponse(project, { headers: { ETag: '"1"' } });
    if (path === "/v1/audits/audit_example")
      return jsonResponse(audit, {
        headers: { ETag: `"${audit.revision}"` },
      });
    const answer = <T,>(
      read: ((url: URL) => Response | T[]) | undefined,
    ): Response => {
      if (read === undefined) return page([], audit);
      const value = read(url);
      return value instanceof Response ? value : page(value, audit);
    };
    if (path === `/v1/audits/audit_example/coverage`)
      return answer(server.coverage);
    if (path === `/v1/audits/audit_example/items`) return answer(server.items);
    if (path === `/v1/audits/audit_example/findings`)
      return answer(server.findings);
    if (path === `/v1/audits/audit_example/reviews`)
      return answer(server.reviews);
    if (path === `/v1/audits/audit_example/workspace`) {
      const value = server.workspace?.(audit) ?? workspaceOf(audit);
      return value instanceof Response ? value : jsonResponse(value);
    }
    if (path === `/v1/audits/audit_example/report`)
      return jsonResponse(server.report?.() ?? { status: "pending" });
    if (path === `/v1/audits/audit_example/events`) {
      const events = server.events?.(url) ?? [];
      return events instanceof Response
        ? events
        : jsonResponse({
            items: events,
            page: { hasMore: false },
            total: events.length,
            throughSequence: events[0]?.sequence ?? 0,
          });
    }
    return emptyPage();
  });
}

function regions() {
  return {
    list: () => screen.getByRole("region", { name: /^Check and / }),
    detail: () =>
      screen
        .getAllByRole("region")
        .find((region) =>
          region.classList.contains("ui-panes-detail"),
        ) as HTMLElement,
  };
}

let previousScroll: typeof HTMLElement.prototype.scrollIntoView;

function stubScroll() {
  previousScroll = HTMLElement.prototype.scrollIntoView;
  const scroll = vi.fn();
  HTMLElement.prototype.scrollIntoView = scroll;
  return scroll;
}

afterEach(() => {
  if (previousScroll !== undefined)
    HTMLElement.prototype.scrollIntoView = previousScroll;
  vi.useRealTimers();
});

describe("Check page", () => {
  it("redirects the old Checks section and selects the linked item with its attempts", async () => {
    const scroll = stubScroll();
    const current = auditAt("completed", 3);
    const row = (
      itemId: string,
      itemKey: string,
      ordinal: number,
      subjectKey: string,
    ): AuditCoverageRow => ({
      roundId: "round_1",
      itemId,
      ordinal,
      itemKey,
      subjectKey,
      coverage: {
        status: "satisfied",
        requested: [],
        completed: [],
        gaps: [],
      },
      updatedAt: current.updatedAt,
    });
    const item = makeItem("item_attempted", 0, "check", {
      itemKey: "check_attempted",
      subjectKey: "Attempted check",
      origin: {
        ...makeItem("x", 0, "check").origin,
        entryKey: "check_attempted",
        entryVersion: "2",
      },
      finalDisposition: "accepted-result",
      acceptedResult: {
        ref: {
          namespace: "audit-results",
          name: "check_attempted",
          revision: "result-r1",
        },
        digest: DIGEST,
      },
      attempts: [
        makeAttempt("item_attempted", 1, {
          terminalOutcome: "failed",
          collectionDisposition: "execution-failed",
          runId: "run_failed",
          runDeleted: true,
        }),
        makeAttempt("item_attempted", 2, {
          runId: "run_accepted",
          result: {
            ref: {
              namespace: "audit-results",
              name: "check_attempted",
              revision: "result-r1",
            },
            digest: DIGEST,
          },
        }),
      ],
    });
    const itemReads = vi.fn();
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        row("item_attempted", "check_attempted", 0, "Attempted check"),
        row("item_pending", "check_pending", 1, "Pending check"),
      ],
      items: () => {
        itemReads();
        return [item];
      },
    });
    const { router } = renderApplication(
      api,
      `${ROOT}/checks#check-item_attempted`,
    );
    const view = await screen.findByRole("article", {
      name: "Attempted check",
    });
    expect(router.state.location.pathname).toBe(`${ROOT}/coverage`);
    expect(router.state.location.hash).toBe("#check-item_attempted");
    const list = regions().list();
    expect(
      within(list).getByRole("link", { name: "Attempted check" }),
    ).toHaveAttribute("aria-current", "true");
    expect(
      within(list).getByRole("link", { name: "Pending check" }),
    ).not.toHaveAttribute("aria-current");
    expect(
      within(view).getByText("Retried automatically · attempt 2 of 3"),
    ).toBeVisible();
    expect(
      within(view).getByRole("link", { name: "Open the run" }),
    ).toHaveAttribute("href", "/runs/run_accepted");
    // No per-item retry: the item retries by itself.
    expect(within(view).queryByRole("button", { name: /retry/iu })).toBeNull();
    const user = userEvent.setup();
    await user.click(within(view).getByText("Technical details"));
    const attempts = within(view).getByRole("table", { name: "Attempts" });
    expect(
      within(attempts).getByRole("link", { name: "Deleted run provenance" }),
    ).toHaveAttribute("href", "/runs/run_failed");
    expect(
      within(attempts).getByRole("link", { name: "run_accepted" }),
    ).toHaveAttribute("href", "/runs/run_accepted");
    expect(within(attempts).getByText("execution-failed")).toBeVisible();
    expect(within(view).getByText("Task package")).toBeVisible();
    expect(within(view).getByText("Accepted result")).toBeVisible();
    expect(within(view).getByText("Produced result")).toBeVisible();
    expect(within(view).getByText("check_attempted@2")).toBeVisible();
    expect(
      within(view).getByText("check_attempted", { selector: "code" }),
    ).toBeVisible();
    expect(itemReads).toHaveBeenCalledTimes(1);
    // The search covers attempt outcomes.
    await user.type(
      within(list).getByRole("searchbox", { name: "Search items" }),
      "execution-failed",
    );
    expect(await within(list).findByText("Showing 1 of 2 items")).toBeVisible();
    expect(
      within(list).queryByRole("link", { name: "Pending check" }),
    ).toBeNull();
    expect(router.state.location.hash).toBe("#check-item_attempted");
    expect(scroll).toHaveBeenCalled();
  });

  it("shows the exact retained standard identity in the technical details", async () => {
    const current = auditAt("completed", 3);
    current.baseline = top10Baseline(current);
    current.baseline.inventory!.standardSelection = {
      scope: "ASVS 5.0 Level 1 source pilot",
      levels: ["1"],
      entryIds: ["v5.0.0-1.2.4", "v5.0.0-2.1.1"],
    };
    const { api } = serveCheck({ audit: () => current });
    renderApplication(api, ROOT);
    const detail = await screen.findByRole("region", { name: "All activity" });
    await userEvent
      .setup()
      .click(within(detail).getByText("Technical details"));
    const standards = within(detail).getByTestId("audit-baseline-standards");
    expect(within(standards).getByText("OWASP Top 10:2025")).toBeVisible();
    expect(within(standards).getByText("owasp-web-top10@2025")).toBeVisible();
    expect(standards).toHaveTextContent("CC-BY-SA-4.0");
    expect(
      within(standards).getByRole("link", { name: "source" }),
    ).toHaveAttribute("href", "https://owasp.org/Top10/2025/");
    const selection = within(detail).getByTestId(
      "audit-baseline-standard-selection",
    );
    expect(selection).toHaveTextContent("ASVS 5.0 Level 1 source pilot");
    expect(selection).toHaveTextContent("Level 1 · 2 requirements");
    expect(selection).toHaveTextContent("v5.0.0-2.1.1");
    // Every setup fact stays reachable.
    for (const fact of [
      "Profile digest",
      "Current revision",
      "Dispatch",
      "Evidence hold",
      "Objective",
      "Target",
      "Authorization scope",
      "Runtime labels",
      "Rounds",
      "Batch size",
      "Attempts per item",
      "Runs submitted",
      "Runs outstanding",
      "Evidence retained",
      "Deadline",
      "Source content",
      "Inventory",
      "Worklist",
      "Skills",
      "Runtime configs",
    ])
      expect(within(detail).getByText(fact, { selector: "dt" })).toBeVisible();
    expect(
      within(detail).getByRole("link", {
        name: /sources\/service-source@r1/u,
      }),
    ).toHaveAttribute(
      "href",
      "/projects/project_example/artifacts/sources/service-source?revision=r1",
    );
  });

  it("does not link a non-http standard source", async () => {
    const current = auditAt("completed", 3);
    current.baseline = top10Baseline(current);
    current.baseline.standards[0]!.source.url = "javascript:alert(1)";
    const { api } = serveCheck({ audit: () => current });
    renderApplication(api, ROOT);
    const detail = await screen.findByRole("region", { name: "All activity" });
    await userEvent
      .setup()
      .click(within(detail).getByText("Technical details"));
    const standards = within(detail).getByTestId("audit-baseline-standards");
    expect(within(standards).getByText("source")).toBeVisible();
    expect(
      within(standards).queryByRole("link", { name: "source" }),
    ).not.toBeInTheDocument();
  });

  it("lists mixed results with their status words and polls a running check", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let auditReads = 0;
    const active = auditAt("active", 2);
    const { api } = serveCheck({
      audit: () => {
        auditReads += 1;
        return active;
      },
      coverage: () => [
        requirementRow(
          "item_1",
          0,
          "A01:2025",
          "Check access control.",
          "violated",
          { rationale: "Missing authorization check" },
        ),
        requirementRow(
          "item_2",
          1,
          "A02:2025",
          "Check sessions.",
          "inconclusive",
          { gaps: ["tests unavailable"] },
        ),
      ],
    });
    renderApplication(api, `${ROOT}/coverage`);
    const list = await screen.findByRole("region", {
      name: "Check and requirements",
    });
    const first = await within(list).findByRole("link", {
      name: "A01:2025 Check access control.",
    });
    expect(first.closest("li")).toHaveTextContent("Issue found");
    const second = within(list).getByRole("link", {
      name: "A02:2025 Check sessions.",
    });
    expect(second.closest("li")).toHaveTextContent("Inconclusive");
    expect(screen.queryByText("tests unavailable")).toBeNull();
    fireEvent.click(second);
    const view = await screen.findByRole("article", {
      name: "A02:2025 Check sessions.",
    });
    expect(within(view).getByText("Why it is inconclusive")).toBeVisible();
    expect(within(view).getByText("tests unavailable")).toBeVisible();
    fireEvent.click(first);
    expect(
      await screen.findByText("Missing authorization check"),
    ).toBeVisible();
    const readsBeforePoll = auditReads;
    await vi.advanceTimersByTimeAsync(1_100);
    await vi.waitFor(() => expect(auditReads).toBeGreaterThan(readsBeforePoll));
  });

  it("reads every coverage page, filters by result and searches task, result and evidence", async () => {
    const completed = auditAt("completed", 4);
    const cursors: Array<string | null> = [];
    const custom: AuditCoverageRow = {
      ...requirementRow(
        "custom-check",
        0,
        "custom-check",
        "Verify that expired invitations cannot be reused.",
        "violated",
        { resultSummary: "An expired invitation can still be accepted." },
      ),
      details: {
        objective: "Verify that expired invitations cannot be reused.",
        methods: ["custom-method"],
        taskDocument: {
          schema: "contractor.audit.item-task.v1",
          checklist: {
            statement: "Verify that expired invitations cannot be reused.",
          },
        },
        resultSummary: "An expired invitation can still be accepted.",
        evidence: [
          {
            id: "proof",
            kind: "observation",
            summary: "The invitation endpoint accepts an expired token.",
          },
        ],
      },
    };
    const trace = requirementRow(
      "trace-check",
      1,
      "trace-check",
      "",
      "traced-complete",
    );
    delete trace.details;
    const { api } = serveCheck({
      audit: () => completed,
      coverage: (url) => {
        expect(url.searchParams.get("round")).toBe("round_1");
        const cursor = url.searchParams.get("cursor");
        cursors.push(cursor);
        return jsonResponse(
          cursor === null
            ? {
                items: [custom],
                page: { hasMore: true, nextCursor: "next-check" },
              }
            : { items: [trace], page: { hasMore: false } },
        );
      },
    });
    const { router } = renderApplication(api, `${ROOT}/coverage`);
    const user = userEvent.setup();
    const list = await screen.findByRole("region", {
      name: "Check and requirements",
    });
    const customLink = await within(list).findByRole("link", {
      name: "custom-check Verify that expired invitations cannot be reused.",
    });
    expect(within(list).getByText("Showing 2 of 2 requirements")).toBeVisible();
    expect(cursors).toEqual([null, "next-check"]);
    const filters = within(list).getByRole("group", {
      name: "Filter by result",
    });
    await user.click(
      within(filters).getByRole("button", { name: "Issues found 1" }),
    );
    expect(router.state.location.search).toBe("?result=issues");
    expect(within(list).getByText("Showing 1 of 2 requirements")).toBeVisible();
    expect(
      within(list).queryByRole("link", { name: "trace-check" }),
    ).toBeNull();
    await user.click(
      within(list).getByRole("button", { name: "Clear filters" }),
    );
    await user.type(
      within(list).getByRole("searchbox", { name: "Search requirements" }),
      "endpoint",
    );
    expect(router.state.location.search).toBe("?q=endpoint");
    expect(customLink).toBeVisible();
    expect(
      within(list).queryByRole("link", { name: "trace-check" }),
    ).toBeNull();
    await user.click(customLink);
    const view = await screen.findByRole("article", {
      name: "custom-check Verify that expired invitations cannot be reused.",
    });
    expect(router.state.location.search).toBe("?q=endpoint");
    expect(
      await within(
        within(view).getByRole("region", { name: "Conclusion" }),
      ).findByText("An expired invitation can still be accepted."),
    ).toBeVisible();
    expect(
      await within(view).findByText(
        "The invitation endpoint accepts an expired token.",
      ),
    ).toBeVisible();
    expect(within(view).getByText("Method: custom-method")).toBeVisible();
    await user.click(within(view).getByText("Technical details"));
    expect(
      within(view).getByText("Item key", { selector: "dt" }).parentElement,
    ).toHaveTextContent("custom-check");
    await user.click(
      within(list).getByRole("button", { name: "Clear filters" }),
    );
    const traceLink = within(list).getByRole("link", { name: "trace-check" });
    expect(traceLink.closest("li")).toHaveTextContent("Fully traced");
    await user.click(traceLink);
    expect(
      await screen.findByText(
        "All requested parts of this operation were traced. This is not a security verdict.",
      ),
    ).toBeVisible();
    expect(
      within(
        screen.getByRole("navigation", { name: "Check sections" }),
      ).getByRole("link", { name: "Possible issues" }),
    ).toHaveAttribute("href", `${ROOT}/findings`);
  });

  it("moves through the items with J and K and keeps the filters", async () => {
    const current = auditAt("completed", 3);
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        requirementRow("r1", 0, "A01:2025", "Access control.", "violated"),
        requirementRow("r2", 1, "A02:2025", "Configuration.", "satisfied"),
        requirementRow("r3", 2, "A03:2025", "Supply chain.", "violated"),
      ],
    });
    const { router } = renderApplication(api, `${ROOT}?result=issues`);
    const user = userEvent.setup();
    const list = await screen.findByRole("region", {
      name: "Check and requirements",
    });
    await within(list).findByRole("link", { name: "A03:2025 Supply chain." });
    expect(
      within(list).getByRole("link", { name: "All activity" }),
    ).toHaveAttribute("aria-current", "true");
    // The rows declare their keys; the footer hint is for sighted users.
    for (const name of ["All activity", "A03:2025 Supply chain."])
      expect(within(list).getByRole("link", { name })).toHaveAttribute(
        "aria-keyshortcuts",
        "J K ArrowDown ArrowUp Home End",
      );
    expect(list.querySelector(".checks-key-hint")).toHaveAttribute(
      "aria-hidden",
      "true",
    );
    await user.keyboard("j");
    await waitFor(() => expect(router.state.location.hash).toBe("#check-r1"));
    expect(router.state.location.pathname).toBe(`${ROOT}/coverage`);
    expect(router.state.location.search).toBe("?result=issues");
    const view = await screen.findByRole("article", {
      name: "A01:2025 Access control.",
    });
    expect(within(view).getByText("Requirement 1 of 2")).toBeVisible();
    expect(
      within(view).getByRole("button", { name: "Previous requirement" }),
    ).toBeDisabled();
    const next = within(view).getByRole("link", { name: "Next requirement" });
    expect(next).toHaveAttribute("aria-keyshortcuts", "J");
    await user.keyboard("j");
    await waitFor(() => expect(router.state.location.hash).toBe("#check-r3"));
    await user.keyboard("k");
    await waitFor(() => expect(router.state.location.hash).toBe("#check-r1"));
    await user.keyboard("k");
    await waitFor(() => expect(router.state.location.pathname).toBe(ROOT));
    expect(router.state.location.search).toBe("?result=issues");
  });

  it("starts J and K at the first item of the list", async () => {
    const current = auditAt("completed", 3);
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        requirementRow("r1", 0, "A01:2025", "Access control.", "violated"),
        requirementRow("r2", 1, "A02:2025", "Configuration.", "satisfied"),
      ],
    });
    const { router } = renderApplication(api, `${ROOT}/coverage`);
    const list = await screen.findByRole("region", {
      name: "Check and requirements",
    });
    await within(list).findByRole("link", { name: "A02:2025 Configuration." });
    // Nothing is selected: the detail asks for a choice.
    expect(
      screen.getByRole("region", { name: "Selected requirement" }),
    ).toHaveTextContent("Choose a requirement");
    await userEvent.setup().keyboard("k");
    await waitFor(() => expect(router.state.location.hash).toBe("#check-r1"));
    expect(router.state.location.pathname).toBe(`${ROOT}/coverage`);
  });

  it("names requirement and scenario checks and every requirement status", async () => {
    const current = auditAt("completed", 3, {
      profileName: "owasp-wstg-4-2-source-review",
    });
    current.baseline = top10Baseline(current);
    current.baseline.standards[0]!.reference = {
      scheme: "owasp-wstg",
      version: "4.2",
    };
    const statuses: AuditCoverageRow["coverage"]["status"][] = [
      "satisfied",
      "violated",
      "not-applicable",
      "inconclusive",
      "unmapped",
      "excluded",
    ];
    const { api } = serveCheck({
      audit: () => current,
      coverage: () =>
        statuses.map((status, index) =>
          requirementRow(
            `s${index}`,
            index,
            `WSTG-v42-ATHN-0${index + 1}`,
            `Scenario ${index + 1}.`,
            status,
          ),
        ),
    });
    renderApplication(api, `${ROOT}/coverage`);
    const list = await screen.findByRole("region", {
      name: "Check and scenarios",
    });
    expect(
      await within(list).findByRole("heading", { name: "Scenarios" }),
    ).toBeVisible();
    expect(
      within(list).getByRole("searchbox", { name: "Search scenarios" }),
    ).toBeVisible();
    const words = within(list)
      .getAllByRole("listitem")
      .filter((row) => row.id.startsWith("check-s"))
      .map((row) => row.querySelector(".checks-status-word")?.textContent);
    expect(words).toEqual([
      "Met",
      "Issue found",
      "Not applicable",
      "Inconclusive",
      "Unmapped",
      "Excluded",
    ]);
    const filters = within(list).getByRole("group", {
      name: "Filter by result",
    });
    expect(
      within(filters)
        .getAllByRole("button")
        .map((button) => button.textContent),
    ).toEqual([
      "All 6",
      "Issues found 1",
      "Need follow-up 2",
      "Not checked yet 0",
      "Met 1",
      "Not applicable / excluded 2",
    ]);
    expect(
      within(
        screen.getByRole("navigation", { name: "Check sections" }),
      ).getByRole("link", { name: "Scenarios", current: "page" }),
    ).toHaveAttribute("href", `${ROOT}/coverage`);
  });

  it("groups endpoints under their shared path and shows the possible issues found on one", async () => {
    const current = auditAt("active", 3, {
      profileName: "openapi-operation-trace",
    });
    const finding = makeFinding(
      "audit_example",
      "finding_1",
      "Any user reads any service report",
      "GET /workshop/api/mechanic/mechanic_report",
    );
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        endpointRow(
          "item_1",
          0,
          "GET",
          "/workshop/api/mechanic/mechanic_report",
          "traced-partial",
          {
            gaps: ["The identity service is outside this code."],
            resultSummary: "Anyone signed in can read any report.",
          },
        ),
        endpointRow(
          "item_2",
          1,
          "POST",
          "/workshop/api/shop/orders",
          "not-tested",
        ),
      ],
      items: () => [
        makeItem("item_1", 0, "operation-trace", {
          attempts: [makeAttempt("item_1", 1)],
        }),
        makeItem("item_2", 1, "operation-trace", {
          state: "submitted",
          attempts: [
            makeAttempt("item_2", 1, {
              state: "submitted",
              terminalOutcome: undefined,
              collectionDisposition: undefined,
              collectedAt: undefined,
            }),
          ],
        }),
      ],
      findings: () => [finding],
      workspace: (audit) =>
        workspaceOf(audit, {
          totalChecks: 2,
          gaps: 1,
          unchecked: 1,
          findings: 1,
          unreviewedFindings: 1,
        }),
    });
    renderApplication(api, `${ROOT}/coverage#check-item_1`);
    const list = await screen.findByRole("region", {
      name: "Check and endpoints",
    });
    const section = await within(list).findByRole("region", {
      name: "Endpoints",
    });
    expect(section).toHaveTextContent("under /workshop/api/");
    const reportRow = within(section)
      .getByRole("link", { name: "GET mechanic/mechanic_report" })
      .closest("li")!;
    expect(reportRow).toHaveTextContent("Partially traced");
    expect(reportRow).toHaveTextContent("1 possible issue");
    expect(
      within(section)
        .getByRole("link", { name: "POST shop/orders" })
        .closest("li"),
    ).toHaveTextContent("Checking now");
    const view = await screen.findByRole("article", {
      name: "GET /workshop/api/mechanic/mechanic_report",
    });
    expect(
      within(view).getByText("Endpoint 1 of 2 · mechanic area"),
    ).toBeVisible();
    expect(within(view).getByText("Why only partially")).toBeVisible();
    const issue = within(view).getByRole("link", {
      name: "Any user reads any service report",
    });
    expect(issue).toHaveAttribute("href", "/issues/audit_example/finding_1");
    const issueRow = issue.closest("li")!;
    expect(issueRow).toHaveTextContent("CWE-639");
    expect(issueRow).toHaveTextContent("Severity not set");
    expect(issueRow).toHaveTextContent("Needs review");
    expect(
      within(
        within(view).getByRole("list", { name: "Activity on this endpoint" }),
      ).getByText("Proposed a possible issue:"),
    ).toBeVisible();
    const progress = screen.getByRole("region", { name: "Check progress" });
    expect(
      within(progress).getByRole("link", { name: "Review 1 possible issue" }),
    ).toHaveAttribute("href", "/issues?project=project_example&state=proposed");
    expect(
      within(progress).getByRole("img", {
        name: "0 of 2 endpoints done: 1 partially traced, 1 checking now",
      }),
    ).toBeVisible();
    expect(within(regions().list()).getByText("Live")).toBeVisible();
  });

  it("keeps every section reachable under the new names", async () => {
    const current = auditAt("completed", 3);
    const { api } = serveCheck({ audit: () => current });
    renderApplication(api, `${ROOT}/runs`);
    const tabs = await screen.findByRole("navigation", {
      name: "Check sections",
    });
    expect(
      within(tabs)
        .getAllByRole("link")
        .map((link) => [link.textContent, link.getAttribute("href")]),
    ).toEqual([
      ["Overview", ROOT],
      ["Requirements", `${ROOT}/coverage`],
      ["Possible issues", `${ROOT}/findings`],
      ["Decisions", `${ROOT}/reviews`],
      ["Runs", `${ROOT}/runs`],
      ["Report", `${ROOT}/report`],
    ]);
    expect(within(tabs).getByRole("link", { name: "Runs" })).toHaveAttribute(
      "aria-current",
      "page",
    );
    const detail = screen.getByRole("region", { name: "Runs" });
    expect(
      await within(detail).findByRole("link", { name: "Global Runs →" }),
    ).toHaveAttribute("href", "/runs");
    expect(
      screen.getByRole("link", { name: "Back to requirements" }),
    ).toHaveAttribute("href", `${ROOT}/coverage`);
  });

  it("shows the whole check: decisions in place, possible issues, report and activity", async () => {
    const current = auditAt("waiting_review", 5, {
      startedAt: "2026-10-05T10:01:00Z",
    });
    const review = makeReview("audit_example", "review_item", {
      subjectId: "item_1",
      kind: "requirement-applicability",
      requestedActions: ["approve", "reject", "not_applicable"],
    });
    const report = makeReview("audit_example", "review_report", {
      subjectKind: "audit-report",
      subjectId: "audit_example",
      kind: "report-acceptance",
    });
    const finding = makeFinding(
      "audit_example",
      "finding_1",
      "Refunds skip the owner check",
      "A01:2025",
    );
    const { api } = serveCheck({
      audit: () => current,
      events: () => [
        {
          auditId: current.auditId,
          sequence: 3,
          kind: "review.requested",
          entityId: "review_report",
          summary: { kind: "report-acceptance" },
          createdAt: current.updatedAt,
        },
        {
          auditId: current.auditId,
          sequence: 2,
          kind: "review.requested",
          entityId: "review_item",
          summary: { kind: "requirement-applicability" },
          createdAt: current.updatedAt,
        },
        {
          auditId: current.auditId,
          sequence: 1,
          kind: "round.accepted",
          entityId: "round_1",
          summary: { round: 1, items: 1 },
          createdAt: current.startedAt!,
        },
      ],
      coverage: () => [
        requirementRow("item_1", 0, "A01:2025", "Access control.", "violated"),
      ],
      findings: () => [finding],
      reviews: () => [review, report],
      workspace: (audit) =>
        workspaceOf(audit, {
          totalChecks: 1,
          completedChecks: 1,
          issues: 1,
          findings: 1,
          unreviewedFindings: 1,
          pendingReviews: 2,
        }),
      report: () => ({
        status: "proposed",
        summary: "Summary",
        summaryArtifact: {
          ref: { namespace: "audit-x", name: "report.md", revision: "r1" },
          digest: DIGEST,
          mediaType: "text/markdown",
          sizeBytes: 7,
        },
        review: report,
      }),
    });
    renderApplication(api, ROOT);
    const detail = await screen.findByRole("region", { name: "All activity" });
    const decisions = within(detail).getByRole("region", {
      name: "Decisions waiting",
    });
    expect(
      await within(decisions).findByText("Requirement applicability"),
    ).toBeVisible();
    expect(
      within(decisions).getByRole("region", { name: "Your decision" }),
    ).toBeVisible();
    expect(
      within(decisions).getByRole("button", { name: "Not applicable" }),
    ).toBeVisible();
    expect(
      within(decisions).getByRole("link", { name: "Review the report →" }),
    ).toHaveAttribute("href", `${ROOT}/report?review=review_report`);
    const issues = within(detail).getByRole("region", {
      name: "Issues and possible issues",
    });
    expect(
      await within(issues).findByRole("link", {
        name: "Refunds skip the owner check",
      }),
    ).toHaveAttribute("href", "/issues/audit_example/finding_1");
    expect(
      within(issues).getByText("This check has 1 possible issue."),
    ).toBeVisible();
    const reportBlock = within(detail).getByRole("region", { name: "Report" });
    expect(
      await within(reportBlock).findByText("Waiting for acceptance"),
    ).toBeVisible();
    expect(
      within(reportBlock).getByRole("link", { name: "Open the report →" }),
    ).toHaveAttribute("href", `${ROOT}/report`);
    const log = within(detail).getByRole("list", {
      name: "Activity on this check",
    });
    expect(
      within(log).getByText("Waiting for your decisions before it can go on."),
    ).toBeVisible();
    expect(await within(log).findByText("Check started.")).toBeVisible();
    expect(within(log).getAllByText("Waiting for your decision:")).toHaveLength(
      2,
    );
  });

  it("links the progress counts to filtered lists and to the pending decisions", async () => {
    stubScroll();
    const completed = auditAt("completed", 4);
    const reviewReads: string[] = [];
    const review = makeReview("audit_example", "review_report", {
      subjectKind: "audit-report",
      subjectId: "report_example",
      kind: "report-acceptance",
    });
    const { api } = serveCheck({
      audit: () => completed,
      coverage: (url) =>
        jsonResponse(
          url.searchParams.get("cursor") === null
            ? {
                items: [requirementRow("met", 0, "met", "", "satisfied")],
                page: { hasMore: true, nextCursor: "coverage-next" },
              }
            : {
                items: [
                  requirementRow("gap", 1, "gap", "", "blocked"),
                  requirementRow("waiting", 2, "waiting", "", "not-tested"),
                ],
                page: { hasMore: false },
              },
        ),
      reviews: (url) => {
        reviewReads.push(url.search);
        return jsonResponse({
          items: [review],
          page: { hasMore: false },
          auditRevision: 4,
          asOf: completed.updatedAt,
          total: 1,
        });
      },
      workspace: (audit) =>
        workspaceOf(audit, {
          totalChecks: 3,
          completedChecks: 1,
          gaps: 1,
          unchecked: 1,
          findings: 51,
          unreviewedFindings: 51,
          pendingReviews: 1,
        }),
    });
    const { router } = renderApplication(api, ROOT);
    const progress = await screen.findByRole("region", {
      name: "Check progress",
    });
    expect(await within(progress).findByText("1 of 3 done")).toBeVisible();
    expect(
      await within(progress).findByRole("img", {
        name: "1 of 3 requirements done: 1 met, 1 blocked, 1 not checked yet",
      }),
    ).toBeVisible();
    const legend = within(progress).getByRole("list", { name: "Legend" });
    expect(
      within(legend).getByRole("link", { name: "Met: 1" }),
    ).toHaveAttribute("href", `${ROOT}/coverage?result=complete`);
    expect(
      within(legend).getByRole("link", { name: "Blocked: 1" }),
    ).toHaveAttribute("href", `${ROOT}/coverage?result=uncertain`);
    expect(
      within(progress).getByRole("link", {
        name: "51 possible issues not reviewed yet",
      }),
    ).toHaveAttribute(
      "href",
      `${ROOT}/findings?verdict=unreviewed&auditRevision=4`,
    );
    expect(
      within(progress).getByRole("link", { name: "Review 51 possible issues" }),
    ).toHaveAttribute("href", "/issues?project=project_example&state=proposed");
    const user = userEvent.setup();
    await user.click(
      within(progress).getByRole("link", {
        name: "1 decision waiting for you",
      }),
    );
    const states = await screen.findByRole("group", {
      name: "Filter decisions by state",
    });
    expect(
      within(states).getByRole("button", { name: "Waiting for you" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(router.state.location.search).toBe("?state=pending&auditRevision=4");
    expect(router.state.location.hash).toBe("");
    await waitFor(() =>
      expect(document.getElementById("review-review_report")).toBeVisible(),
    );
    // The queue's first page is pinned to the counts' revision.
    expect(reviewReads).toContain("?limit=50&auditRevision=4&state=pending");
    expect(reviewReads.every((search) => !search.includes("cursor"))).toBe(
      true,
    );
    expect(
      screen.queryByRole("button", { name: "Next page" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Refresh context" }),
    ).toBeEnabled();
    await user.click(within(states).getByRole("button", { name: "All" }));
    expect(router.state.location.search).toBe("");
  });

  it("approves an exact item decision without treating model text as authority", async () => {
    let current = auditAt("waiting_review", 3);
    const item = makeItem("item_active_check", 50, "check", {
      subjectKey: "POST /orders/{id} · Authorization check",
      state: "awaiting_review",
      approvalKind: "active-check-approval",
    });
    let review = makeReview("audit_example", "review_active_check", {
      subjectId: "item_active_check",
    });
    const { api, requests } = serveCheck({
      audit: () => current,
      items: (url) =>
        jsonResponse(
          url.searchParams.has("cursor")
            ? { items: [item], page: { hasMore: false } }
            : { items: [], page: { hasMore: true, nextCursor: "later" } },
        ),
      reviews: () => [review],
      handle: async (request, url) => {
        if (
          url.pathname ===
            "/v1/audits/audit_example/reviews/review_active_check/decisions" &&
          request.method === "POST"
        ) {
          expect(request.headers.get("If-Match")).toBe('"1"');
          expect(await request.json()).toEqual({
            action: "approve",
            rationale: "The target and exact active request are **approved**.",
          });
          const decision = {
            decisionId: "decision_active_check",
            requestId: review.requestId,
            auditId: review.auditId,
            action: "approve" as const,
            actorId: "user_local",
            rationale: "The target and exact active request are **approved**.",
            subjectRevision: review.subjectRevision,
            subjectDigest: review.subjectDigest,
            createdAt: current.updatedAt,
          };
          review = { ...review, state: "decided", revision: 2, decision };
          current = auditAt("active", 4);
          return jsonResponse({ request: review, decision, replayed: false });
        }
        return undefined;
      },
    });
    renderApplication(api, `${ROOT}/reviews`);
    const user = userEvent.setup();
    const card = (
      await screen.findByRole("heading", { name: item.subjectKey })
    ).closest("li")!;
    expect(
      within(card).getByText("Active test approval", {
        selector: ".checks-eyebrow",
      }),
    ).toBeVisible();
    expect(within(card).getByText("Waiting for you")).toBeVisible();
    expect(
      within(card).getByRole("link", { name: "View requirement →" }),
    ).toHaveAttribute("href", `${ROOT}/coverage#check-item_active_check`);
    const decision = within(card).getByRole("region", {
      name: "Your decision",
    });
    // Only the actions the request offers.
    expect(
      within(decision).queryByRole("button", { name: "Not applicable" }),
    ).toBeNull();
    await user.click(within(decision).getByRole("button", { name: "Approve" }));
    await user.type(
      within(decision).getByRole("textbox", { name: "Why" }),
      "The target and exact active request are **approved**.",
    );
    await user.click(
      within(decision).getByRole("button", { name: "Record decision" }),
    );
    expect(await within(card).findByText("Approved")).toBeVisible();
    expect(
      await within(card).findByText("approved", { selector: "strong" }),
    ).toBeVisible();
    expect(
      requests.filter((request) => request.method === "POST"),
    ).toHaveLength(1);
  });

  it("keeps confirmed issues apart from possible issues on the check and its items", async () => {
    const current = auditAt("completed", 3);
    const on = (
      id: string,
      title: string,
      state: AuditFinding["state"],
      minute: number,
    ) =>
      makeFinding("audit_example", id, title, "A01:2025", {
        state,
        createdAt: `2026-10-05T10:${minute}:00Z`,
      });
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        requirementRow("item_1", 0, "A01:2025", "Access control.", "violated"),
      ],
      findings: () => [
        on("f_confirmed", "Refunds skip the owner check", "confirmed", 15),
        on("f_possible", "Orders leak to other users", "proposed", 25),
        on("f_rejected", "Admin route reads any order", "rejected", 35),
      ],
      workspace: (audit) =>
        workspaceOf(audit, { findings: 3, unreviewedFindings: 1 }),
    });
    const { router } = renderApplication(api, ROOT);
    const issues = await screen.findByRole("region", {
      name: "Issues and possible issues",
    });
    expect(
      await within(issues).findByText(
        "This check has 1 issue and 1 possible issue. 1 was marked not an issue or duplicate.",
      ),
    ).toBeVisible();
    // Newest first; the finding set aside is counted, not listed.
    expect(
      within(issues)
        .getAllByRole("listitem")
        .map((row) => within(row).getByRole("link").textContent),
    ).toEqual(["Orders leak to other users", "Refunds skip the owner check"]);
    expect(within(issues).getByText("Confirmed")).toBeVisible();
    expect(within(issues).getByText("Needs review")).toBeVisible();
    expect(
      within(issues).queryByRole("link", {
        name: "Admin route reads any order",
      }),
    ).toBeNull();
    expect(
      within(issues).getByRole("link", {
        name: "All issues and possible issues of this check →",
      }),
    ).toHaveAttribute("href", `${ROOT}/findings`);
    await act(() => router.navigate(`${ROOT}/coverage#check-item_1`));
    const view = await screen.findByRole("article", {
      name: "A01:2025 Access control.",
    });
    const onItem = within(view).getByRole("region", {
      name: "Issues and possible issues on this requirement",
    });
    expect(
      within(onItem)
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual(["Refunds skip the owner check", "Orders leak to other users"]);
    const setAside = within(view).getByRole("region", { name: "Set aside" });
    expect(
      within(setAside).getByRole("link", {
        name: "Admin route reads any order",
      }),
    ).toHaveAttribute("href", "/issues/audit_example/f_rejected");
    expect(within(setAside).getByText("Not an issue")).toBeVisible();
  });

  it("does not call a check's only confirmed issue a possible issue", async () => {
    const current = auditAt("completed", 3);
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        requirementRow("item_1", 0, "A01:2025", "Access control.", "violated"),
      ],
      findings: () => [
        makeFinding(
          "audit_example",
          "f_1",
          "Refunds skip the owner check",
          "A01:2025",
          {
            state: "confirmed",
          },
        ),
      ],
      workspace: (audit) => workspaceOf(audit, { findings: 1 }),
    });
    renderApplication(api, ROOT);
    const issues = await screen.findByRole("region", {
      name: "Issues and possible issues",
    });
    expect(
      await within(issues).findByText("This check has 1 issue."),
    ).toBeVisible();
    const row = within(issues)
      .getByRole("link", { name: "Refunds skip the owner check" })
      .closest("li")!;
    expect(row).toHaveTextContent("Confirmed");
    expect(screen.queryByText(/possible issues? in this check/u)).toBeNull();
    const list = regions().list();
    expect(
      within(list)
        .getByRole("link", { name: "A01:2025 Access control." })
        .closest("li"),
    ).toHaveTextContent("1 issue");
    await userEvent
      .setup()
      .click(
        within(list).getByRole("link", { name: "A01:2025 Access control." }),
      );
    const view = await screen.findByRole("article", {
      name: "A01:2025 Access control.",
    });
    expect(
      within(view).getByRole("region", { name: "Issue on this requirement" }),
    ).toBeVisible();
    expect(
      within(view).queryByRole("region", { name: /possible issue/iu }),
    ).toBeNull();
  });

  it("says a draft has no runs yet on its Runs tab", async () => {
    const current = auditAt("draft", 1);
    const itemReads = vi.fn();
    const { api } = serveCheck({
      audit: () => current,
      items: () => {
        itemReads();
        return [];
      },
    });
    renderApplication(api, `${ROOT}/runs`);
    const detail = await screen.findByRole("region", { name: "Runs" });
    expect(
      await within(detail).findByText(
        "No runs yet. A draft starts its runs only once it is started.",
      ),
    ).toBeVisible();
    expect(within(detail).queryByText("Loading runs…")).toBeNull();
    expect(
      within(detail).getByRole("link", { name: "Global Runs →" }),
    ).toHaveAttribute("href", "/runs");
    expect(itemReads).not.toHaveBeenCalled();
  });

  it("keeps a typed decision reason when the technical details open from the list", async () => {
    const scroll = stubScroll();
    const current = auditAt("waiting_review", 3);
    const review = makeReview("audit_example", "review_item", {
      subjectId: "item_1",
    });
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        requirementRow("item_1", 0, "A01:2025", "Access control.", "violated"),
      ],
      reviews: () => [review],
    });
    const { router } = renderApplication(api, ROOT);
    const user = userEvent.setup();
    const decisions = await screen.findByRole("region", {
      name: "Decisions waiting",
    });
    const decision = await within(decisions).findByRole("region", {
      name: "Your decision",
    });
    await user.click(within(decision).getByRole("button", { name: "Approve" }));
    await user.type(
      within(decision).getByRole("textbox", { name: "Why" }),
      "The request stays inside the agreed scope.",
    );
    const technical = document.querySelector("#technical-details details");
    expect(technical).not.toHaveAttribute("open");
    await user.click(
      within(regions().list()).getByRole("link", { name: "Technical details" }),
    );
    await waitFor(() =>
      expect(router.state.location.hash).toBe("#technical-details"),
    );
    expect(
      document.querySelector("#technical-details details"),
    ).toHaveAttribute("open");
    expect(scroll).toHaveBeenCalled();
    expect(
      within(
        screen.getByRole("region", { name: "Decisions waiting" }),
      ).getByRole("textbox", { name: "Why" }),
    ).toHaveValue("The request stays inside the agreed scope.");
  });

  it("loads the details of an item beyond the first batch in place", async () => {
    const current = auditAt("completed", 3, {
      profileName: "openapi-operation-trace",
    });
    const cursors: Array<string | null> = [];
    const { api } = serveCheck({
      audit: () => current,
      coverage: () =>
        Array.from({ length: 6 }, (_, index) =>
          endpointRow(
            `i${index + 1}`,
            index,
            "GET",
            `/api/orders/${index + 1}`,
            "traced-complete",
          ),
        ),
      items: (url) => {
        const cursor = url.searchParams.get("cursor");
        cursors.push(cursor);
        const index =
          cursor === null ? 0 : Number(cursor.slice("page-".length));
        return jsonResponse({
          items: [
            makeItem(`i${index + 1}`, index, "operation-trace", {
              attempts: [makeAttempt(`i${index + 1}`, 1)],
            }),
          ],
          page:
            index + 1 < 6
              ? { hasMore: true, nextCursor: `page-${index + 1}` }
              : { hasMore: false },
        });
      },
    });
    renderApplication(api, `${ROOT}/coverage#check-i6`);
    const view = await screen.findByRole("article", {
      name: "GET /api/orders/6",
    });
    expect(
      await within(view).findByText(
        "Its attempts, run and activity are not loaded: this endpoint is beyond the endpoints read so far.",
      ),
    ).toBeVisible();
    expect(
      within(view).queryByRole("link", { name: "Open the run" }),
    ).toBeNull();
    expect(
      within(view).getByText(
        /^Attempts are not loaded: this endpoint is beyond the endpoints read so far/u,
      ),
    ).toBeInTheDocument();
    expect(cursors).toEqual([null, "page-1", "page-2", "page-3", "page-4"]);
    await userEvent.setup().click(
      within(view).getByRole("button", {
        name: "Load more endpoint details",
      }),
    );
    expect(
      await within(view).findByRole("link", { name: "Open the run" }),
    ).toHaveAttribute("href", "/runs/run_i6_1");
    expect(cursors.at(-1)).toBe("page-5");
    expect(
      within(view).queryByRole("button", {
        name: "Load more endpoint details",
      }),
    ).toBeNull();
    expect(
      within(view).getByRole("table", { name: "Attempts" }),
    ).toBeInTheDocument();
  });

  it("walks mixed scenarios and requirements in the order the list shows them", async () => {
    const current = auditAt("completed", 3, {
      profileName: "owasp-wstg-4-2-source-review",
    });
    const mapped = (itemId: string, ordinal: number, scheme: string) =>
      makeItem(itemId, ordinal, "standard-mapping", {
        origin: {
          ...makeItem(itemId, ordinal, "standard-mapping").origin,
          standard: {
            scheme,
            version: "1",
            mappingKey: itemId,
            entryIds: [itemId],
            evidenceContract: { id: "source-review", version: "1" },
          },
        },
      });
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        requirementRow("r1", 0, "WSTG-ATHN-01", "Scenario one.", "satisfied"),
        requirementRow("r2", 1, "V2.1.1", "Requirement two.", "satisfied"),
        requirementRow("r3", 2, "WSTG-ATHN-02", "Scenario three.", "satisfied"),
      ],
      items: () => [
        mapped("r1", 0, "owasp-wstg"),
        mapped("r2", 1, "owasp-asvs"),
        mapped("r3", 2, "owasp-wstg"),
      ],
    });
    const { router } = renderApplication(api, `${ROOT}/coverage#check-r1`);
    const list = await screen.findByRole("region", {
      name: "Check and scenarios",
    });
    await within(list).findByRole("heading", { name: "Requirements" });
    const order = within(list)
      .getAllByRole("listitem")
      .filter((row) => row.id.startsWith("check-r"))
      .map((row) => row.id);
    expect(order).toEqual(["check-r1", "check-r3", "check-r2"]);
    expect(
      await screen.findByRole("article", {
        name: "WSTG-ATHN-01 Scenario one.",
      }),
    ).toHaveTextContent("Scenario 1 of 3");
    const user = userEvent.setup();
    await user.keyboard("j");
    await waitFor(() => expect(router.state.location.hash).toBe("#check-r3"));
    const third = await screen.findByRole("article", {
      name: "WSTG-ATHN-02 Scenario three.",
    });
    expect(third).toHaveTextContent("Scenario 2 of 3");
    expect(
      within(third).getByRole("link", { name: "Next scenario" }),
    ).toHaveAttribute("href", `${ROOT}/coverage#check-r2`);
    await user.keyboard("j");
    await waitFor(() => expect(router.state.location.hash).toBe("#check-r2"));
    expect(
      await screen.findByRole("article", { name: "V2.1.1 Requirement two." }),
    ).toHaveTextContent("Requirement 3 of 3");
    await user.keyboard("k");
    await waitFor(() => expect(router.state.location.hash).toBe("#check-r3"));
  });
});

describe("Check lifecycle controls", () => {
  it("keeps the started check when an older detail read resolves later", async () => {
    let current = auditAt("draft", 2);
    let releaseStale: (() => void) | undefined;
    const queryClient = queryClientFactory.createApplicationQueryClient();
    vi.spyOn(
      queryClientFactory,
      "createApplicationQueryClient",
    ).mockReturnValue(queryClient);
    const detailKey = queryKeys.audits.detail("audit_example");
    const { api } = serveCheck({
      audit: () => current,
      handle: async (_request, url) => {
        if (url.pathname === "/v1/audits/audit_example") {
          const snapshot = current;
          if (releaseStale === undefined && snapshot.state === "draft")
            await new Promise<void>((resolve) => {
              releaseStale = resolve;
            });
          return jsonResponse(snapshot, {
            headers: { ETag: `"${snapshot.revision}"` },
          });
        }
        if (url.pathname.endsWith("/start")) {
          current = auditAt("active", 3);
          return jsonResponse(
            { audit: current, round: { roundId: "round_1" }, items: [] },
            { headers: { ETag: '"3"' } },
          );
        }
        return undefined;
      },
    });
    queryClient.setQueryData(detailKey, current);
    renderApplication(api, ROOT);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Start check" }),
    );
    const dialog = screen.getByRole("dialog", { name: "Start check" });
    expect(releaseStale).toBeDefined();
    await user.click(
      within(dialog).getByRole("button", { name: "Start check" }),
    );
    expect(
      await screen.findByRole("button", { name: "Pause new work" }),
    ).toBeVisible();
    releaseStale!();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(queryClient.getQueryData<Audit>(detailKey)?.revision).toBe(3);
  });

  it.each(["draft", "paused"] as const)(
    "chooses no time limit before starting or continuing a %s check",
    async (state) => {
      let current = auditAt(state, 2);
      if (state === "paused")
        current = {
          ...current,
          stopReason: {
            code: "deadline_exhausted",
            message: "The Audit wall-time deadline was reached",
          },
        };
      const writes: Request[] = [];
      const { api } = serveCheck({
        audit: () => current,
        handle: async (request, url) => {
          if (
            url.pathname.endsWith("/start") ||
            url.pathname.endsWith("/resume")
          ) {
            writes.push(request.clone());
            expect(await request.json()).toEqual({ deadlineSeconds: 0 });
            expect(request.headers.get("If-Match")).toBe('"2"');
            current = auditAt("active", 3);
            return jsonResponse(
              url.pathname.endsWith("/start")
                ? { audit: current, round: { roundId: "round_1" }, items: [] }
                : current,
              { headers: { ETag: '"3"' } },
            );
          }
          return undefined;
        },
      });
      renderApplication(api, ROOT);
      const user = userEvent.setup();
      const name = state === "draft" ? "Start check" : "Continue check";
      await user.click(await screen.findByRole("button", { name }));
      const dialog = screen.getByRole("dialog", { name });
      expect(writes).toHaveLength(0);
      const limit = within(dialog).getByLabelText("Time limit");
      expect(limit).toHaveFocus();
      expect(limit).toHaveValue("604800");
      expect(
        within(dialog).queryByRole("option", { name: "Keep remaining time" }),
      ).toBeNull();
      await user.selectOptions(limit, "0");
      await user.click(within(dialog).getByRole("button", { name }));
      expect(
        await screen.findByRole("button", { name: "Pause new work" }),
      ).toBeVisible();
      expect(writes).toHaveLength(1);
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    },
  );

  it("offers to keep the remaining time when continuing an ordinary pause", async () => {
    const current = auditAt("paused", 2);
    const { api } = serveCheck({ audit: () => current });
    renderApplication(api, ROOT);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Continue check" }),
    );
    const dialog = screen.getByRole("dialog", { name: "Continue check" });
    expect(within(dialog).getByLabelText("Time limit")).toHaveValue(
      "remaining",
    );
    await user.selectOptions(
      within(dialog).getByLabelText("Time limit"),
      "custom",
    );
    const hours = within(dialog).getByLabelText("Time limit in hours");
    await user.clear(hours);
    await user.type(hours, "9000");
    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "no longer than 365 days",
    );
    expect(
      within(dialog).getByRole("button", { name: "Continue check" }),
    ).toBeDisabled();
  });

  it.each(["completed", "failed"] as const)(
    "keeps an ended %s check with an old time limit reason final",
    async (state) => {
      const current = {
        ...auditAt(state, 2),
        stopReason: {
          code: "deadline_exhausted",
          message: "The Audit wall-time deadline was reached",
        },
      };
      const { api, requests } = serveCheck({ audit: () => current });
      renderApplication(api, ROOT);
      const progress = await screen.findByRole("region", {
        name: "Check progress",
      });
      await userEvent
        .setup()
        .click(within(progress).getByLabelText("Check actions"));
      expect(
        within(progress).getByRole("button", { name: "Delete check" }),
      ).toBeVisible();
      expect(
        screen.queryByRole("button", { name: "Continue check" }),
      ).not.toBeInTheDocument();
      expect(screen.queryByText(/deadline_exhausted/u)).not.toBeInTheDocument();
      expect(
        screen.queryByText(current.stopReason.message),
      ).not.toBeInTheDocument();
      const banner = within(progress)
        .getByText(
          "Stopped by the time limit: no new work was started after it.",
        )
        .closest(".checks-banner")!;
      expect(banner).toHaveAttribute("role", "status");
      expect(banner).toHaveAttribute(
        "data-tone",
        state === "failed" ? "blocked" : "warning",
      );
      expect(
        screen.queryByText(/Continue with a longer limit/u),
      ).not.toBeInTheDocument();
      expect(requests.every((request) => request.method === "GET")).toBe(true);
    },
  );

  it("recovers a stale pause from the current revision", async () => {
    let current = auditAt("active", 2);
    let pauseRequested = false;
    const { api } = serveCheck({
      audit: () => current,
      handle: (request, url) => {
        if (url.pathname === "/v1/audits/audit_example/pause") {
          expect(request.headers.get("If-Match")).toBe('"2"');
          pauseRequested = true;
          current = auditAt("paused", 3);
          return jsonResponse(
            {
              code: "precondition_failed",
              message: "Audit revision changed",
              retryable: false,
              requestId: "request_stale",
            },
            { status: 412 },
          );
        }
        return undefined;
      },
    });
    renderApplication(api, ROOT);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Pause new work" }),
    );
    expect(
      within(await screen.findByRole("alert")).getByText(
        "Audit revision changed",
      ),
    ).toBeVisible();
    await vi.waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Continue check" }),
      ).toBeVisible(),
    );
    expect(pauseRequested).toBe(true);
    expect(
      within(regions().list()).getByText("Paused", {
        selector: ".ui-status-chip",
      }),
    ).toBeVisible();
  });

  it.each([
    ["answers", 200],
    ["is refused", 412],
  ] as const)(
    "refreshes the cross-project lists when a pause %s, without waiting for them",
    async (_answer, status) => {
      let current = auditAt("active", 2);
      // The rail's Inbox badge reads the project index and the project's
      // checks; once the pause is answered, those reads stay unanswered.
      let held = false;
      let release = () => {};
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      const listReads: string[] = [];
      const { api } = serveCheck({
        audit: () => current,
        handle: async (request, url) => {
          if (
            request.method === "GET" &&
            (url.pathname === "/v1/projects" || url.pathname === "/v1/audits")
          ) {
            listReads.push(url.pathname);
            if (held) await gate;
            const state = url.searchParams.get("state");
            return jsonResponse({
              items:
                url.pathname === "/v1/projects"
                  ? [project]
                  : [current].filter(
                      (audit) => state === null || audit.state === state,
                    ),
              page: { hasMore: false },
            });
          }
          if (url.pathname === "/v1/audits/audit_example/pause") {
            held = true;
            current = auditAt("paused", 3);
            return status === 200
              ? jsonResponse(current, { headers: { ETag: '"3"' } })
              : jsonResponse(
                  {
                    code: "precondition_failed",
                    message: "Audit revision changed",
                    retryable: false,
                    requestId: "request_stale",
                  },
                  { status: 412 },
                );
          }
          return undefined;
        },
      });
      renderApplication(api, ROOT);
      const user = userEvent.setup();
      const pause = await screen.findByRole("button", {
        name: "Pause new work",
      });
      await waitFor(() => expect(listReads).toContain("/v1/audits"));
      const readsBefore = listReads.length;
      await user.click(pause);

      // The check's own controls are usable again while the lists load.
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Continue check" }),
        ).toBeEnabled(),
      );
      expect(listReads.slice(readsBefore)).toEqual(
        expect.arrayContaining(["/v1/projects", "/v1/audits"]),
      );
      if (status === 412)
        expect(
          within(screen.getByRole("alert")).getByText("Audit revision changed"),
        ).toBeVisible();
      await act(async () => {
        release();
        await gate;
      });
    },
  );

  it("confirms stopping and deleting a check without duplicate requests", async () => {
    let current = auditAt("active", 2);
    const mutations: Request[] = [];
    let releaseCancel: (() => void) | undefined;
    const cancelGate = new Promise<void>((resolve) => {
      releaseCancel = resolve;
    });
    const { api } = serveCheck({
      audit: () => current,
      handle: async (request, url) => {
        if (url.pathname === "/v1/audits/audit_example/cancel") {
          mutations.push(request.clone());
          await cancelGate;
          current = auditAt("cancelled", 3);
          return jsonResponse(current, {
            status: 202,
            headers: { ETag: '"3"' },
          });
        }
        if (
          url.pathname === "/v1/audits/audit_example" &&
          request.method === "DELETE"
        ) {
          mutations.push(request.clone());
          current = auditAt("deleting", 4);
          return jsonResponse(current, {
            status: 202,
            headers: { ETag: '"4"' },
          });
        }
        return undefined;
      },
    });
    renderApplication(api, ROOT);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Stop check" }));
    let dialog = screen.getByRole("alertdialog", { name: "Stop this check?" });
    expect(within(dialog).getByText(/Payment service/u)).toBeVisible();
    expect(within(dialog).getByText("audit_example")).toBeVisible();
    expect(
      within(dialog).getByText("owasp-top10-2025-source-risk@1"),
    ).toBeVisible();
    expect(within(dialog).getByText("Running · revision 2")).toBeVisible();
    expect(
      within(dialog).getByRole("button", { name: "Keep the check running" }),
    ).toHaveFocus();
    fireEvent.click(dialog.parentElement!);
    expect(dialog).toBeVisible();
    expect(mutations).toHaveLength(0);
    await user.keyboard("{Escape}");
    expect(
      screen.queryByRole("alertdialog", { name: "Stop this check?" }),
    ).toBeNull();
    expect(mutations).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Stop check" }));
    dialog = screen.getByRole("alertdialog", { name: "Stop this check?" });
    await user.dblClick(
      within(dialog).getByRole("button", { name: "Stop check" }),
    );
    await vi.waitFor(() => expect(mutations).toHaveLength(1));
    expect(mutations[0]?.headers.get("If-Match")).toBe('"2"');
    expect(mutations[0]?.headers.get("Idempotency-Key")).toMatch(
      /^mutate-audit-ui-/u,
    );
    releaseCancel?.();
    await vi.waitFor(() =>
      expect(
        within(regions().list()).getByText("Stopped", {
          selector: ".ui-status-chip",
        }),
      ).toBeVisible(),
    );
    await user.click(screen.getByLabelText("Check actions"));
    await user.click(screen.getByRole("button", { name: "Delete check" }));
    dialog = screen.getByRole("alertdialog", { name: "Delete this check?" });
    expect(
      within(dialog).getByText(/Permanently delete this check/u),
    ).toBeVisible();
    expect(
      within(dialog).getByRole("button", { name: "Keep the check" }),
    ).toHaveFocus();
    await user.click(
      within(dialog).getByRole("button", { name: "Delete check" }),
    );
    await vi.waitFor(() => expect(mutations).toHaveLength(2));
    expect(mutations[1]?.method).toBe("DELETE");
    expect(mutations[1]?.headers.get("If-Match")).toBe('"3"');
    await vi.waitFor(() =>
      expect(
        within(regions().list()).getByText("Deleting", {
          selector: ".ui-status-chip",
        }),
      ).toBeVisible(),
    );
  });

  it("refreshes a stale stop confirmation before an explicit retry", async () => {
    let current = auditAt("active", 2);
    const cancelRevisions: string[] = [];
    const { api } = serveCheck({
      audit: () => current,
      handle: (request, url) => {
        if (url.pathname === "/v1/audits/audit_example/cancel") {
          cancelRevisions.push(request.headers.get("If-Match") ?? "");
          if (cancelRevisions.length === 1) {
            current = auditAt("paused", 3);
            return jsonResponse(
              {
                code: "precondition_failed",
                message: "Audit revision changed",
                retryable: false,
                requestId: "request_stale_cancel",
              },
              { status: 412 },
            );
          }
          current = auditAt("cancelled", 4);
          return jsonResponse(current, {
            status: 202,
            headers: { ETag: '"4"' },
          });
        }
        return undefined;
      },
    });
    renderApplication(api, ROOT);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Stop check" }));
    let dialog = screen.getByRole("alertdialog", { name: "Stop this check?" });
    await user.click(
      within(dialog).getByRole("button", { name: "Stop check" }),
    );
    expect(
      await within(dialog).findByText(/The check changed in the meantime/u),
    ).toBeVisible();
    await vi.waitFor(() =>
      expect(within(dialog).getByText("Paused · revision 3")).toBeVisible(),
    );
    expect(cancelRevisions).toEqual(['"2"']);
    dialog = screen.getByRole("alertdialog", { name: "Stop this check?" });
    await user.click(
      within(dialog).getByRole("button", { name: "Stop check" }),
    );
    await vi.waitFor(() => expect(cancelRevisions).toEqual(['"2"', '"3"']));
    await vi.waitFor(() =>
      expect(
        within(regions().list()).getByText("Stopped", {
          selector: ".ui-status-chip",
        }),
      ).toBeVisible(),
    );
  });
});

describe("Check refresh correctness", () => {
  function refreshRow(audit: Audit): AuditCoverageRow {
    return {
      ...requirementRow(
        "item_refresh",
        0,
        "check_refresh",
        "Refresh check.",
        "satisfied",
      ),
      coverage: {
        status: "satisfied",
        requested: [],
        completed: [],
        gaps: [],
        rationale: `revision ${audit.revision}`,
      },
    };
  }

  function refreshItem(audit: Audit): AuditItem {
    return makeItem("item_refresh", 0, "check", {
      state: "submitted",
      attempts: [
        makeAttempt("item_refresh", 1, {
          state: "submitted",
          terminalOutcome: undefined,
          collectionDisposition: undefined,
          collectedAt: undefined,
          runId: "run_refresh",
          createdAt: audit.createdAt,
        }),
      ],
    });
  }

  it("signals a newer revision without replacing pinned rows", async () => {
    let current = auditAt("completed", 2);
    const queryClient = queryClientFactory.createApplicationQueryClient();
    vi.spyOn(
      queryClientFactory,
      "createApplicationQueryClient",
    ).mockReturnValue(queryClient);
    const coverageReads = vi.fn();
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => {
        coverageReads();
        return [refreshRow(current)];
      },
    });
    const { router } = renderApplication(
      api,
      `${ROOT}/coverage?auditRevision=2#check-item_refresh`,
    );
    const view = await screen.findByRole("article", {
      name: "check_refresh Refresh check.",
    });
    expect(await within(view).findByText("revision 2")).toBeVisible();
    current = auditAt("completed", 3);
    act(() =>
      queryClient.setQueryData(
        queryKeys.audits.detail(current.auditId),
        current,
      ),
    );
    expect(
      await screen.findByText("A newer revision of this check is available."),
    ).toBeVisible();
    expect(screen.getByText("This list shows revision 2.")).toBeVisible();
    expect(within(view).getByText("revision 2")).toBeVisible();
    expect(coverageReads).toHaveBeenCalledTimes(1);
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Show the latest results" }));
    await waitFor(() =>
      expect(
        within(
          screen.getByRole("article", { name: "check_refresh Refresh check." }),
        ).getByText("revision 3"),
      ).toBeVisible(),
    );
    expect(router.state.location.search).toBe("");
    expect(router.state.location.hash).toBe("#check-item_refresh");
    expect(coverageReads).toHaveBeenCalledTimes(2);
  });

  it.each(["coverage", "runs"] as const)(
    "keeps loaded %s rows after a failed background read",
    async (section) => {
      const current = auditAt("completed", 3);
      const queryClient = queryClientFactory.createApplicationQueryClient();
      vi.spyOn(
        queryClientFactory,
        "createApplicationQueryClient",
      ).mockReturnValue(queryClient);
      let failRead = false;
      const failure = () =>
        jsonResponse(
          {
            code: "temporarily_unavailable",
            message: "Temporary read failure",
            retryable: true,
            requestId: "retry_refresh",
          },
          { status: 503 },
        );
      const { api } = serveCheck({
        audit: () => current,
        coverage: () =>
          failRead && section === "coverage"
            ? failure()
            : [refreshRow(current)],
        items: () =>
          failRead && section === "runs" ? failure() : [refreshItem(current)],
      });
      renderApplication(
        api,
        section === "coverage"
          ? `${ROOT}/coverage#check-item_refresh`
          : `${ROOT}/runs`,
      );
      const row =
        section === "coverage"
          ? (
              await within(
                await screen.findByRole("region", {
                  name: "Check and requirements",
                }),
              ).findByRole("link", { name: "check_refresh Refresh check." })
            ).closest("li")!
          : (await screen.findByRole("link", { name: "run_refresh" })).closest(
              "tr",
            )!;
      failRead = true;
      await act(async () => {
        await queryClient.invalidateQueries({
          queryKey:
            section === "coverage"
              ? queryKeys.audits.allCoverage(current.auditId)
              : queryKeys.audits.allItems(current.auditId),
        });
      });
      expect(
        await screen.findByText(
          "Could not refresh; showing the last loaded data.",
        ),
      ).toBeVisible();
      expect(row).toBeInTheDocument();
      if (section === "coverage")
        expect(
          screen.getByRole("article", { name: "check_refresh Refresh check." }),
        ).toBeVisible();
      failRead = false;
      await userEvent
        .setup()
        .click(screen.getByRole("button", { name: "Try again" }));
      await waitFor(() =>
        expect(
          screen.queryByText(
            "Could not refresh; showing the last loaded data.",
          ),
        ).not.toBeInTheDocument(),
      );
      expect(row).toBeInTheDocument();
    },
  );

  it("counts a list pinned to an older revision from its own rows", async () => {
    let current = auditAt("completed", 2);
    const queryClient = queryClientFactory.createApplicationQueryClient();
    vi.spyOn(
      queryClientFactory,
      "createApplicationQueryClient",
    ).mockReturnValue(queryClient);
    const workspaceReads = vi.fn();
    const { api } = serveCheck({
      audit: () => current,
      coverage: () => [
        requirementRow("met", 0, "A01:2025", "Access control.", "satisfied"),
        requirementRow("open", 1, "A02:2025", "Configuration.", "not-tested"),
      ],
      workspace: (audit) => {
        workspaceReads(audit.revision);
        return audit.revision === 2
          ? workspaceOf(audit, { totalChecks: 2, completedChecks: 1 })
          : workspaceOf(audit, { totalChecks: 5, completedChecks: 3 });
      },
    });
    renderApplication(api, `${ROOT}/coverage?auditRevision=2`);
    const progress = await screen.findByRole("region", {
      name: "Check progress",
    });
    expect(await within(progress).findByText("1 of 2 done")).toBeVisible();
    current = auditAt("completed", 3);
    act(() =>
      queryClient.setQueryData(
        queryKeys.audits.detail(current.auditId),
        current,
      ),
    );
    // The check moved on: its counts are read again, but the list stays on
    // revision 2, and so does the header.
    await waitFor(() => expect(workspaceReads).toHaveBeenCalledWith(3));
    expect(
      await within(progress).findByText(
        "Counts describe revision 2, the one the list shows.",
      ),
    ).toBeVisible();
    expect(within(progress).getByText("1 of 2 done")).toBeVisible();
    expect(within(progress).queryByText("3 of 5 done")).toBeNull();
    expect(
      within(progress).getByRole("img", {
        name: "1 of 2 requirements done: 1 met, 1 not checked yet",
      }),
    ).toBeVisible();
    expect(
      within(within(progress).getByRole("list", { name: "Legend" })).getByRole(
        "link",
        { name: "Met: 1" },
      ),
    ).toHaveAttribute(
      "href",
      `${ROOT}/coverage?result=complete&auditRevision=2`,
    );
    expect(
      screen.getByText("A newer revision of this check is available."),
    ).toBeVisible();
  });

  it("reads the counts at most every 5 s while the check keeps moving", async () => {
    let revision = 2;
    const workspaceReads = vi.fn();
    const { api } = serveCheck({
      audit: () => {
        // Every read sees a new revision, as a busy check does.
        revision += 1;
        return auditAt("active", revision, {
          updatedAt: "2026-10-05T10:05:00Z",
        });
      },
      workspace: (audit) => {
        workspaceReads(audit.revision);
        return workspaceOf(audit);
      },
    });
    renderApplication(api, ROOT);
    await screen.findByRole("region", { name: "Check progress" });
    await waitFor(() => expect(workspaceReads).toHaveBeenCalledOnce());
    const shownRevision = () =>
      Number(
        screen.getByText("Current revision", { selector: "dt" })
          .nextElementSibling?.textContent,
      );
    const first = shownRevision();
    // The page polls the check every second and shows its new revision…
    await waitFor(() => expect(shownRevision()).toBeGreaterThan(first));
    await new Promise((resolve) => setTimeout(resolve, 500));
    // …but the counts wait for their own 5 s interval.
    expect(workspaceReads).toHaveBeenCalledOnce();
  }, 10_000);

  it("reads the counts again for each revision until the last run drains", async () => {
    let current = { ...auditAt("paused", 3), outstandingRunCount: 1 };
    const detailReads = vi.fn();
    const workspaceReads = vi.fn();
    const { api } = serveCheck({
      audit: () => {
        detailReads();
        return current;
      },
      workspace: (audit) => {
        workspaceReads(audit.revision);
        return workspaceOf(audit);
      },
    });
    renderApplication(api, ROOT);
    await screen.findByRole("region", { name: "Check progress" });
    await waitFor(() => expect(workspaceReads).toHaveBeenCalledWith(3));
    current = { ...auditAt("paused", 4), outstandingRunCount: 0 };
    await waitFor(() => expect(workspaceReads).toHaveBeenCalledWith(4));
    expect(workspaceReads).toHaveBeenCalledTimes(2);
    const readsAfterDrain = detailReads.mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 1_150));
    expect(detailReads).toHaveBeenCalledTimes(readsAfterDrain);
  });

  it("reads the counts again right after pausing a check", async () => {
    let current = auditAt("active", 2);
    const workspaceReads = vi.fn();
    const { api } = serveCheck({
      audit: () => current,
      workspace: (audit) => {
        workspaceReads(audit.revision);
        return workspaceOf(audit);
      },
      handle: (_request, url) => {
        if (url.pathname === "/v1/audits/audit_example/pause") {
          current = { ...auditAt("paused", 3), outstandingRunCount: 1 };
          return jsonResponse(current, { headers: { ETag: '"3"' } });
        }
        return undefined;
      },
    });
    renderApplication(api, ROOT);
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Pause new work" }));
    await waitFor(() => expect(workspaceReads).toHaveBeenCalledWith(3));
    expect(workspaceReads).toHaveBeenCalledTimes(2);
  });

  it("caps the item list at the page budget and continues from the cursor on Load more", async () => {
    const completed = auditAt("completed", 4);
    const cursors: Array<string | null> = [];
    const { api } = serveCheck({
      audit: () => completed,
      coverage: (url) => {
        const cursor = url.searchParams.get("cursor");
        cursors.push(cursor);
        const ordinal =
          cursor === null ? 0 : Number(cursor.slice("page-".length));
        return jsonResponse({
          items: [
            requirementRow(
              `check-${ordinal + 1}`,
              ordinal,
              `check-${ordinal + 1}`,
              "",
              "satisfied",
            ),
          ],
          page:
            ordinal + 1 < 6
              ? { hasMore: true, nextCursor: `page-${ordinal + 1}` }
              : { hasMore: false },
        });
      },
    });
    renderApplication(api, `${ROOT}/coverage`);
    const list = await screen.findByRole("region", {
      name: "Check and requirements",
    });
    const loadMore = await within(list).findByRole("button", {
      name: "Load more",
    });
    expect(
      within(loadMore.parentElement!).getByText("Showing 5 of ≥5 requirements"),
    ).toBeVisible();
    const rows = () =>
      within(list)
        .getAllByRole("listitem")
        .filter((row) => row.id.startsWith("check-check-"));
    expect(rows()).toHaveLength(5);
    expect(cursors).toEqual([null, "page-1", "page-2", "page-3", "page-4"]);
    await userEvent.setup().click(loadMore);
    expect(
      await within(list).findByText("Showing 6 of 6 requirements"),
    ).toBeVisible();
    expect(rows()).toHaveLength(6);
    expect(cursors.at(-1)).toBe("page-5");
    expect(
      within(list).queryByRole("button", { name: "Load more" }),
    ).toBeNull();
  });
});

describe("Project Checks tab", () => {
  it("lists the project's checks with compact controls and starts new checks in Start", async () => {
    const completed = makeAudit("audit_done", "project_example", "completed", {
      profileName: "owasp-top10-2025-source-risk",
    });
    const waiting = makeAudit(
      "audit_wait",
      "project_example",
      "waiting_review",
      {
        profileName: "openapi-operation-trace",
      },
    );
    const failed = makeAudit("audit_failed", "project_example", "failed", {
      profileName: "owasp-wstg-4-2-source-review",
      stopReason: {
        code: "controller_contract_invalid",
        message: "controller contract",
      },
    });
    const { api } = fakeAPI((_request, url) => {
      if (url.pathname === "/v1/projects/project_example")
        return jsonResponse(project, { headers: { ETag: '"1"' } });
      if (url.pathname === "/v1/projects/project_example/audits")
        return jsonResponse({
          items: [waiting, completed, failed],
          page: { hasMore: false },
        });
      return undefined;
    });
    renderApplication(api, "/projects/project_example/audits");
    const user = userEvent.setup();
    const open = await screen.findByRole("link", {
      name: "OWASP Top 10 · Source risks",
    });
    expect(open).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_done",
    );
    expect(screen.getByRole("link", { name: "Start a check" })).toHaveAttribute(
      "href",
      "/checks/new?project=project_example",
    );
    expect(screen.queryByRole("button", { name: "New Audit" })).toBeNull();
    const waitingRow = screen
      .getByRole("link", { name: "OpenAPI · Operation trace" })
      .closest("li")!;
    expect(waitingRow).toHaveTextContent("Waiting for you");
    expect(
      within(waitingRow).getByRole("link", { name: "Review decisions" }),
    ).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_wait/reviews?state=pending",
    );
    expect(
      within(waitingRow).getByRole("button", { name: "Pause new work" }),
    ).toBeVisible();
    expect(
      screen.getByText(
        "The check stopped because its check type is configured incorrectly.",
      ),
    ).toBeVisible();
    const doneRow = open.closest("li")!;
    expect(doneRow).toHaveTextContent("Runs: 2 submitted, 0 in progress");
    await user.click(within(doneRow).getByText("Check identity"));
    expect(
      within(doneRow).getByRole("button", {
        name: "Copy check ID of OWASP Top 10 · Source risks",
      }),
    ).toBeVisible();
    expect(
      within(doneRow).getByText("owasp-top10-2025-source-risk@1"),
    ).toBeVisible();
    await user.click(
      within(doneRow).getByRole("button", { name: "Delete check" }),
    );
    expect(
      screen.getByRole("alertdialog", { name: "Delete this check?" }),
    ).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Keep the check" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
  });
});

function top10Baseline(audit: Audit): NonNullable<Audit["baseline"]> {
  const exactStandard = {
    artifact: {
      namespace: "audit-audit_example",
      name: "standard-owasp-web-top10-2025",
      revision: "standard-r1",
    },
    digest: `sha256:${"c".repeat(64)}`,
    mediaType: "application/vnd.contractor.audit-standard+zip" as const,
    sizeBytes: 4096,
  };
  return {
    inputs: audit.inputs,
    scope: audit.scope,
    runtimeLabels: [],
    runtimeConfig: {
      default: {
        label: "default",
        explicit: false,
        bindingRevision: 1,
        config: {
          name: "default-runtime",
          version: "1",
          digest: `sha256:${"d".repeat(64)}`,
        },
      },
      labels: [],
    },
    skills: [],
    standards: [
      {
        reference: { scheme: "owasp-web-top10", version: "2025" },
        title: "OWASP Top 10:2025",
        source: {
          name: "OWASP Top 10:2025",
          url: "https://owasp.org/Top10/2025/",
          revision: "66ebc4798d2ca72973967a20264bdeb70dcf0a13",
        },
        license: {
          id: "CC-BY-SA-4.0",
          url: "https://creativecommons.org/licenses/by-sa/4.0/",
          attribution: "OWASP Foundation, OWASP Top 10:2025.",
          disclosure: "full",
        },
        catalog: {
          ...exactStandard,
          artifact: {
            namespace: "audit-standards",
            name: "std-owasp-web-top10-2025",
            revision: "catalog-r1",
          },
        },
        retained: exactStandard,
      },
    ],
    inventory: {
      sourceContentDigest: exactStandard.digest,
      canonicalInventoryDigest: `sha256:${"e".repeat(64)}`,
      gaps: [],
      worklist: {
        ref: {
          namespace: "audit-audit_example",
          name: "round-1-worklist",
          revision: "worklist-r1",
        },
        digest: `sha256:${"f".repeat(64)}`,
        mediaType: "application/zip",
        sizeBytes: 2048,
      },
    },
  };
}
