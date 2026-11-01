import { ARTIFACT_STORAGE_TIMESTAMPS } from "../../../test/builders";
import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";

import type { PublicAPI } from "../../../api/client";
import type {
  Audit,
  AuditCoverageRow,
  ProjectFinding,
  AuditItem,
  AuditProfile,
  AuditReviewRequest,
} from "../../../api/audits";
import { createApplicationQueryClient } from "../../../app/query-client";
import { queryKeys } from "../../../api/query-keys";
import { renderApplication } from "../../../test/application";
import * as build from "../../../test/builders";
import { fakeServer, invalidJSON } from "../../../test/fake-server";
import { json, page } from "../../../test/responses";
import { testAPI, testSession } from "../../../test/render";

/** A PublicAPI over `handle` whose answers the fake Server checks. */
function checkedAPI(
  handle: (input: Request) => Response | Promise<Response>,
): PublicAPI {
  return testAPI(fakeServer({}, (request) => handle(request))).api;
}

const session = testSession(["user", "operations"]);

const project = build.project("project_example", {
  name: "Payment service",
  description: "Reusable service analysis",
  createdAt: "2026-09-06T10:00:00Z",
  updatedAt: "2026-09-06T10:00:00Z",
});

const profile: AuditProfile = build.conforming("AuditProfile", {
  ref: {
    name: "owasp-top10-2025-source-risk",
    version: "1",
    digest: `sha256:${"2".repeat(64)}`,
  },
  mode: "risk-assessment",
  standards: [{ scheme: "owasp-web-top10", version: "2025" }],
  inputs: { source: { required: true, mediaTypes: ["application/zip"] } },
  inventory: {
    implementation: "standard-mappings@1",
    itemWorkflowRole: "check",
  },
  execution: {
    roundMode: "fixed-barrier",
    maxRounds: 1,
    batchSize: 1,
    maxItemsPerRound: 10,
    maxItemsTotal: 10,
    maxSubmittedRuns: 30,
    maxItemRunAttempts: 3,
    deadlineSeconds: 600,
    maxEvidenceBytes: 1_048_576,
    incompleteRound: "assess-with-gaps",
  },
  interaction: {
    activeChecks: "prohibited",
    notApplicable: "human-required",
    reportAcceptance: "automatic",
  },
  serverCompatible: true,
  requiresInputValidation: true,
  compatibilityReasons: [],
});

const sourceArtifact = {
  canUploadNewVersion: true,
  storageTimestamps: ARTIFACT_STORAGE_TIMESTAMPS,
  artifact: {
    namespace: "sources",
    name: "payment-service",
    revision: "revision-7",
  },
  mediaType: "application/zip",
  size: 2048,
  current: true,
  frozen: false,
  createdAt: "2026-09-06T10:00:00Z",
};

function auditAt(state: Audit["state"], revision: number): Audit {
  return build.conforming("Audit", {
    auditId: "audit_example",
    projectId: project.projectId,
    profile: { ...profile.ref },
    inputs: {
      source: {
        ref: { ...sourceArtifact.artifact },
        digest: `sha256:${"1".repeat(64)}`,
        mediaType: sourceArtifact.mediaType,
        sizeBytes: sourceArtifact.size,
      },
    },
    scope: { objective: "Review source controls" },
    runtimeLabels: ["debug"],
    state,
    phase: state === "draft" ? "not-started" : "rounds",
    ...(state === "draft" ? {} : { currentRoundId: "round_example" }),
    revision,
    dispatchState: state === "active" ? "open" : "closed",
    holdState: state === "draft" ? "pending" : "held",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 8,
      maxItemsTotal: 8,
      maxSubmittedRuns: 8,
      maxItemRunAttempts: 2,
      maxEvidenceBytes: 1_048_576,
    },
    reservedRunCount: state === "draft" ? 0 : 1,
    submittedRunCount: state === "draft" ? 0 : 1,
    outstandingRunCount: state === "active" ? 1 : 0,
    retainedEvidenceBytes: 0,
    eventSequence: revision,
    createdAt: "2026-09-06T10:00:00Z",
    updatedAt: `2026-09-06T10:0${revision}:00Z`,
  } as Audit);
}

function findingAt(
  state: ProjectFinding["state"],
  revision: number,
): ProjectFinding {
  return build.finding(
    project.projectId,
    "finding_example",
    { state, revision },
    {
      title: "Missing object authorization",
      subject: { kind: "component", key: "orders" },
    },
  );
}

/** A page of a check's collection with the summary members it carries. */
function auditPage(
  audit: Audit,
  items: unknown[],
  total = items.length,
  nextCursor?: string,
) {
  return {
    auditRevision: audit.revision,
    asOf: audit.updatedAt,
    total,
    items,
    page:
      nextCursor === undefined
        ? { hasMore: false }
        : { hasMore: true, nextCursor },
  };
}

/** An empty answer for one of `audit`'s collections at `path`. */
function emptyCollection(path: string, audit: Audit): Response {
  if (path.endsWith("/workspace")) return json(build.workspace(audit));
  if (path.endsWith("/coverage"))
    return json({
      auditRevision: audit.revision,
      items: [],
      page: { hasMore: false },
    });
  if (/^\/v1\/audits\/[^/]+\/findings$/u.test(path))
    return json({
      items: [],
      page: { hasMore: false },
      auditRevision: audit.revision,
      projectRevision: 1,
      asOf: audit.updatedAt,
    });
  if (/^\/v1\/audits\/[^/]+\/reviews$/u.test(path))
    return json(auditPage(audit, []));
  return json({ items: [], page: { hasMore: false } });
}

// The routes these tests open (the check page, the project's possible
// issues, Issues) and the Markdown preview of a possible issue's description
// load lazily. Without this, the first test to use each loads it inside its
// own 5 s budget: alone the first test took 265–285 ms instead of 96–99 ms
// and the first decision on Issues 582–672 ms instead of 250 ms; in the full
// parallel suite they took 770 ms and 867 ms (vitest JSON reports,
// 2026-10-23).
beforeAll(async () => {
  await Promise.all([
    import("./detail"),
    import("./findings"),
    import("../detail"),
    import("../../issues"),
    import("../../artifacts/previews/markdown"),
  ]);
});

describe("Project Audit routes", () => {
  it.each(["coverage", "report"] as const)(
    "refreshes the final %s projection when the parent stops polling",
    async (section) => {
      const previousScroll = HTMLElement.prototype.scrollIntoView;
      HTMLElement.prototype.scrollIntoView = vi.fn();
      let current = auditAt("active", 2);
      const queryClient = createApplicationQueryClient();
      const item: AuditItem = {
        findingIds: [],
        itemId: "item_final",
        roundId: "round_example",
        itemKey: "check_final",
        ordinal: 0,
        kind: "check",
        subjectKey: "Final retained check",
        task: current.inputs.source!,
        origin: {
          schema: "contractor.audit.item-origin.v1",
          entryKey: "check_final",
          sourceRef: current.inputs.source!.ref,
          sourceContentDigest: current.inputs.source!.digest,
          sourceMediaType: "application/zip",
          canonicalInventoryDigest: `sha256:${"c".repeat(64)}`,
        },
        workflowRole: "check",
        state: "settled",
        approvalKind: "none",
        attempts: [],
        createdAt: current.createdAt,
        updatedAt: current.updatedAt,
      };
      const coverageRow: AuditCoverageRow = {
        findingIds: [],
        roundId: item.roundId,
        itemId: item.itemId,
        ordinal: item.ordinal,
        itemKey: item.itemKey,
        subjectKey: item.subjectKey,
        coverage: {
          status: "satisfied",
          requested: [],
          completed: [],
          gaps: [],
        },
        updatedAt: current.updatedAt,
      };
      const reads = vi.fn();
      const api = checkedAPI(async (input) => {
        const path = new URL((input as Request).url).pathname;
        if (path === "/v1/auth/session") return json(session);
        if (path === "/v1/projects/project_example")
          return json(project, 200, { ETag: '"1"' });
        if (path === "/v1/audits/audit_example")
          return json(current, 200, { ETag: `"${current.revision}"` });
        if (path.endsWith("/coverage"))
          return json({
            auditRevision: current.revision,
            items: [coverageRow],
            page: { hasMore: false },
          });
        // Each section counts its own projection: the check page reads
        // the item collection on every section.
        if (path.endsWith("/items")) {
          if (section === "coverage") reads();
          return json({
            items: current.state === "active" ? [] : [item],
            page: { hasMore: false },
          });
        }
        if (path.endsWith("/report")) {
          if (section === "report") reads();
          return json(
            current.state === "active"
              ? { status: "pending" }
              : {
                  status: "ready",
                  summary: "Final retained report",
                  summaryArtifact: {
                    ref: {
                      namespace: "audit-example",
                      name: "report.md",
                      revision: "report-r1",
                    },
                    digest: `sha256:${"1".repeat(64)}`,
                    mediaType: "text/markdown",
                    sizeBytes: 21,
                  },
                },
          );
        }
        return emptyCollection(path, current);
      });
      renderApplication(
        `/projects/project_example/audits/audit_example/${section}${section === "coverage" ? "#check-item_final" : ""}`,
        { api, queryClient },
      );
      await screen.findByText(
        section === "coverage"
          ? /^Attempts are not loaded for this/u
          : "The check has not reached report generation.",
      );
      current = auditAt("completed", 3);
      act(() =>
        queryClient.setQueryData(
          queryKeys.audits.detail(current.auditId),
          current,
        ),
      );
      await screen.findByText(
        section === "coverage" ? "No run submitted." : "Final retained report",
      );
      if (section === "coverage")
        expect(
          screen.getByText("check_final", { selector: "code" }),
        ).toBeInTheDocument();
      expect(reads).toHaveBeenCalledTimes(2);
      HTMLElement.prototype.scrollIntoView = previousScroll;
    },
  );

  it.each(["text/markdown", "text/plain"])(
    "only previews and downloads current Markdown reports: %s",
    async (mediaType) => {
      const markdown = mediaType === "text/markdown";
      const summary =
        "# Coverage summary\n\n| Status | Count |\n| --- | ---: |\n| satisfied | 2 |\n";
      const api = checkedAPI(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return json(session);
        if (path === "/v1/projects/project_example")
          return json(project, 200, { ETag: '"1"' });
        if (path === "/v1/audits/audit_example")
          return json(auditAt("completed", 3), 200, { ETag: '"3"' });
        if (path === "/v1/audits/audit_example/report")
          // A text/plain summary breaks the contract on purpose.
          return (markdown ? json : invalidJSON)({
            status: "ready",
            summary,
            summaryArtifact: {
              ref: {
                namespace: "audit-example",
                name: markdown ? "report.md" : "report.txt",
                revision: "report-r1",
              },
              digest: `sha256:${"1".repeat(64)}`,
              mediaType,
              sizeBytes: summary.length,
            },
          });
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return emptyCollection(path, auditAt("completed", 3));
        throw new Error(`unexpected ${request.method} ${path}`);
      });
      renderApplication(
        "/projects/project_example/audits/audit_example/report",
        { api },
      );
      if (!markdown) {
        expect(
          await screen.findByText(
            /The Server sent an answer this page cannot read/u,
          ),
        ).toBeVisible();
        expect(
          screen.queryByRole("button", { name: "Download summary" }),
        ).not.toBeInTheDocument();
        expect(
          screen.queryByRole("heading", { name: "Coverage summary" }),
        ).not.toBeInTheDocument();
        return;
      }
      const button = await screen.findByRole("button", {
        name: "Download summary",
      });
      expect(
        await screen.findByRole("heading", { name: "Coverage summary" }),
      ).toBeVisible();
      expect(screen.getByRole("table")).toHaveTextContent("satisfied");
      let downloadedName = "";
      const click = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(function (this: HTMLAnchorElement) {
          downloadedName = this.download;
        });
      const createURL = vi.fn<(blob: Blob) => string>(() => "blob:report-test");
      const originalCreate = URL.createObjectURL;
      const originalRevoke = URL.revokeObjectURL;
      URL.createObjectURL = createURL;
      URL.revokeObjectURL = vi.fn();
      try {
        await userEvent.setup().click(button);
        expect(downloadedName).toBe("audit_example-report.md");
        expect(createURL.mock.calls[0]?.[0].type).toBe(mediaType);
      } finally {
        click.mockRestore();
        URL.createObjectURL = originalCreate;
        URL.revokeObjectURL = originalRevoke;
      }
    },
  );

  it("lists canonical Project records without requiring their source checks and filters loaded rows", async () => {
    const first = findingAt(null, 1);
    const rated = build.finding(
      project.projectId,
      "finding_rated",
      {
        state: "confirmed",
        analystVerdict: "true_positive",
        analystSeverity: "low",
        rationale: "Checked",
        actorId: "analyst",
        assessedAt: first.createdAt,
      },
      { title: "Trace information exposure", severity_suggestion: "medium" },
    );
    const reads: string[] = [];
    const api = checkedAPI((request) => {
      const url = new URL(request.url);
      reads.push(url.pathname);
      if (url.pathname === "/v1/auth/session") return json(session);
      if (url.pathname === "/v1/projects/project_example")
        return json(project, 200, { ETag: '"1"' });
      if (url.pathname === "/v1/projects/project_example/findings")
        return json({
          items: [first, rated].map(build.findingSummary),
          projectRevision: 4,
          page: { hasMore: false },
        });
      throw new Error(`unexpected ${url}`);
    });
    const { router } = renderApplication("/projects/project_example/findings", {
      api,
    });
    const user = userEvent.setup();
    expect(
      await screen.findByText("2 of 2 possible issues loaded"),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: first.title })).toHaveAttribute(
      "href",
      "/issues/project_example/finding_example?state=all&project=project_example",
    );
    expect(
      reads.some(
        (path) => path.startsWith("/v1/audits/") && path.endsWith("/findings"),
      ),
    ).toBe(false);
    await user.selectOptions(screen.getByLabelText("Severity"), "medium");
    expect(screen.getByText("No matching possible issues")).toBeVisible();
    await user.selectOptions(screen.getByLabelText("Severity"), "low");
    expect(screen.getByRole("link", { name: rated.title })).toBeVisible();
    expect(router.state.location.search).toBe("?severity=low");
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await user.click(screen.getByRole("button", { name: "Not assessed 1" }));
    expect(screen.queryByRole("link", { name: rated.title })).toBeNull();
    expect(router.state.location.search).toBe("?state=unassessed");
  });

  it("marks only an authorized applicability review as not applicable", async () => {
    const currentAudit = auditAt("active", 3);
    const review: AuditReviewRequest = {
      requestId: "review_applicability",
      auditId: currentAudit.auditId,
      subjectKind: "audit-item-action",
      subjectId: "item_asvs_documentation",
      kind: "requirement-applicability",
      subjectRevision: 1,
      subjectDigest: `sha256:${"c".repeat(64)}`,
      requestedActions: ["approve", "reject", "not_applicable"],
      state: "pending",
      revision: 1,
      createdAt: currentAudit.createdAt,
      updatedAt: currentAudit.updatedAt,
    };
    let decided = false;
    const api = checkedAPI(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const path = new URL(request.url).pathname;
      if (path === "/v1/auth/session") return json(session);
      if (path === "/v1/projects/project_example") {
        return json(project, 200, { ETag: '"1"' });
      }
      if (path === "/v1/audits/audit_example") {
        return json(currentAudit, 200, { ETag: '"3"' });
      }
      if (
        path === "/v1/audits/audit_example/reviews" &&
        request.method === "GET"
      ) {
        return json({
          items: decided
            ? [
                {
                  ...review,
                  state: "decided",
                  revision: 2,
                  decision: {
                    decisionId: "decision_not_applicable",
                    requestId: review.requestId,
                    auditId: review.auditId,
                    action: "not_applicable",
                    actorId: session.principal.userId,
                    rationale:
                      "Documentation is outside this exact application scope.",
                    subjectRevision: review.subjectRevision,
                    subjectDigest: review.subjectDigest,
                    createdAt: currentAudit.updatedAt,
                  },
                },
              ]
            : [review],
          page: { hasMore: false },
          auditRevision: currentAudit.revision,
          asOf: currentAudit.updatedAt,
          total: 1,
        });
      }
      if (
        path ===
          "/v1/audits/audit_example/reviews/review_applicability/decisions" &&
        request.method === "POST"
      ) {
        expect(request.headers.get("If-Match")).toBe('"1"');
        expect(await request.json()).toEqual({
          action: "not_applicable",
          rationale: "Documentation is outside this exact application scope.",
        });
        decided = true;
        return json({
          request: review,
          decision: {
            decisionId: "decision_not_applicable",
            requestId: review.requestId,
            auditId: review.auditId,
            action: "not_applicable",
            actorId: session.principal.userId,
            rationale: "Documentation is outside this exact application scope.",
            subjectRevision: review.subjectRevision,
            subjectDigest: review.subjectDigest,
            createdAt: currentAudit.updatedAt,
          },
          replayed: false,
        });
      }
      if (path === "/v1/audits/audit_example/items") {
        return json({ items: [], page: { hasMore: false } });
      }
      if (path === "/v1/audits/audit_example/report") {
        return json({ status: "pending" });
      }
      if (path.endsWith("/coverage") || path.endsWith("/reviews"))
        return emptyCollection(path, currentAudit);
      throw new Error(`unexpected ${request.method} ${path}`);
    });
    renderApplication(
      "/projects/project_example/audits/audit_example/reviews",
      { api },
    );
    const user = userEvent.setup();
    const decision = await screen.findByRole("region", {
      name: "Your decision",
    });
    await user.click(
      within(decision).getByRole("button", { name: "Not applicable" }),
    );
    await user.type(
      within(decision).getByRole("textbox", { name: "Why" }),
      "Documentation is outside this exact application scope.",
    );
    await user.click(
      within(decision).getByRole("button", { name: "Record decision" }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("region", { name: "Your decision" }),
      ).not.toBeInTheDocument(),
    );
    // The recorded outcome, not the "Not applicable" result filter of the
    // item list beside it.
    expect(
      screen.getByText("Not applicable", { selector: ".ui-status-chip" }),
    ).toBeVisible();
    expect(decided).toBe(true);
  });
});

describe("Bounded Project finding collections", () => {
  it("loads five pages per explicit batch, retains a failed tail and refreshes from the head", async () => {
    const cursors: (string | null)[] = [];
    let fail = false;
    const api = checkedAPI((request) => {
      const url = new URL(request.url);
      if (url.pathname === "/v1/auth/session") return json(session);
      if (url.pathname === "/v1/projects/project_example")
        return json(project, 200, { ETag: '"1"' });
      if (url.pathname === "/v1/projects/project_example/findings") {
        const cursor = url.searchParams.get("cursor");
        cursors.push(cursor);
        if (fail && cursor !== null)
          return json(
            {
              code: "conflict",
              message: "Project changed",
              retryable: false,
              requestId: "stale",
            },
            409,
          );
        const ordinal = Number(cursor ?? 0);
        const row = build.finding(
          project.projectId,
          `finding_${ordinal}`,
          {},
          { title: `Finding ${ordinal}` },
        );
        return json({
          projectRevision: 1,
          items: [build.findingSummary(row)],
          page:
            ordinal < 5
              ? { hasMore: true, nextCursor: String(ordinal + 1) }
              : { hasMore: false },
        });
      }
      throw new Error(`unexpected ${url}`);
    });
    renderApplication("/projects/project_example/findings", { api });
    const user = userEvent.setup();
    expect(
      await screen.findByText("5 of 5 possible issues loaded"),
    ).toBeVisible();
    expect(cursors).toEqual([null, "1", "2", "3", "4"]);
    fail = true;
    await user.click(
      screen.getByRole("button", { name: "Load more possible issues" }),
    );
    expect(
      await screen.findByText(/Loading more failed: Project changed/),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "Finding 0" })).toBeVisible();
    fail = false;
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      await screen.findByText("6 of 6 possible issues loaded"),
    ).toBeVisible();
    const before = cursors.length;
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(cursors.slice(before)).toEqual([null, "1", "2", "3", "4"]),
    );
    expect(
      await screen.findByText("5 of 5 possible issues loaded"),
    ).toBeVisible();
  });
});

describe("Audit workspace snapshot navigation", () => {
  it("pins canonical continuations and resets stale cursors on filter changes", async () => {
    const audit = auditAt("completed", 5);
    const queries: URLSearchParams[] = [];
    const api = checkedAPI(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      if (url.pathname === "/v1/auth/session") return json(session);
      if (url.pathname === "/v1/projects/project_example")
        return json(project, 200, { ETag: '"1"' });
      if (url.pathname === "/v1/audits/audit_example")
        return json(audit, 200, { ETag: '"5"' });
      if (url.pathname === "/v1/audits/audit_example/findings") {
        // The check page's own unfiltered read of every possible issue.
        if ([...url.searchParams.keys()].every((key) => key === "limit"))
          return json({
            items: [],
            page: { hasMore: false },
            projectRevision: 1,
            auditRevision: 5,
            asOf: audit.updatedAt,
          });
        queries.push(url.searchParams);
        if (url.searchParams.has("cursor"))
          return json(
            {
              code: "conflict",
              message: "Audit changed",
              retryable: false,
              requestId: "stale",
            },
            409,
          );
        return json({
          items: [build.findingSummary(findingAt("proposed", 1))],
          page: { hasMore: true, nextCursor: "next-exact" },
          projectRevision: 1,
          auditRevision: 5,
          asOf: audit.updatedAt,
        });
      }
      // The shell's owner-wide read of review requests.
      if (url.pathname === "/v1/reviews") return json(page([]));
      if (url.pathname.endsWith("/reviews"))
        return json({
          items: [],
          page: { hasMore: false },
          total: 0,
          auditRevision: 5,
          asOf: audit.updatedAt,
        });
      throw new Error(`unexpected ${url}`);
    });
    const { router } = renderApplication(
      "/projects/project_example/audits/audit_example/findings?verdict=unreviewed",
      { api },
    );
    const user = userEvent.setup();
    expect(await screen.findByText(/Showing 1 matching records/)).toBeVisible();
    expect(queries).toHaveLength(1);
    expect(screen.getByLabelText("Decision")).toHaveValue("unreviewed");
    expect(
      screen.getByRole("link", { name: "Missing object authorization" }),
    ).toHaveAttribute(
      "href",
      "/issues/project_example/finding_example?state=all&project=project_example",
    );
    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(await screen.findByText(/This check changed/)).toBeVisible();
    expect(queries.at(-1)?.get("cursor")).toBe("next-exact");
    expect(queries.at(-1)?.get("auditRevision")).toBe("5");
    expect(queries.at(-1)?.get("verdict")).toBe("unreviewed");
    await user.selectOptions(screen.getByLabelText("Severity"), "high");
    expect(await screen.findByText(/Showing 1 matching records/)).toBeVisible();
    expect(router.state.location.search).toBe(
      "?verdict=unreviewed&severity=high",
    );
    expect(queries.at(-1)?.has("cursor")).toBe(false);
    expect(queries.at(-1)?.has("auditRevision")).toBe(false);
    await user.click(
      within(screen.getByRole("group", { name: "Filter by state" })).getByRole(
        "button",
        { name: "Needs review" },
      ),
    );
    expect(router.state.location.search).toBe(
      "?verdict=unreviewed&severity=high&state=proposed",
    );
    await waitFor(() => expect(queries.at(-1)?.get("state")).toBe("proposed"));
    expect(
      screen.getByRole("link", { name: "Missing object authorization" }),
    ).toHaveAttribute(
      "href",
      "/issues/project_example/finding_example?state=proposed&project=project_example&severity=high",
    );
  });

  it("opens the exact report from a paged review queue, preserves return context and blocks a different review", async () => {
    const audit = auditAt("waiting_review", 5);
    const review: AuditReviewRequest = {
      requestId: "report-review",
      auditId: audit.auditId,
      subjectKind: "audit-report",
      subjectId: audit.auditId,
      kind: "report-acceptance",
      subjectRevision: 4,
      subjectDigest: `sha256:${"b".repeat(64)}`,
      requestedActions: ["approve", "reject"],
      state: "pending",
      revision: 1,
      createdAt: audit.createdAt,
      updatedAt: audit.updatedAt,
    };
    const mutations: string[] = [];
    const api = checkedAPI(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const url = new URL(request.url);
      if (request.method !== "GET") mutations.push(request.method);
      if (url.pathname === "/v1/auth/session") return json(session);
      if (url.pathname === "/v1/projects/project_example")
        return json(project, 200, { ETag: '"1"' });
      if (url.pathname === "/v1/audits/audit_example")
        return json(audit, 200, { ETag: '"5"' });
      // The shell's owner-wide read of review requests.
      if (url.pathname === "/v1/reviews") return json(page([]));
      if (url.pathname.endsWith("/reviews"))
        return json({
          items: [review],
          page: { hasMore: false },
          total: 51,
          auditRevision: 5,
          asOf: audit.updatedAt,
        });
      if (url.pathname.endsWith("/report"))
        return json({
          status: "proposed",
          summary: "Exact retained evidence for owner acceptance.",
          summaryArtifact: {
            ref: {
              namespace: "audit-example",
              name: "report.md",
              revision: "report-r1",
            },
            digest: `sha256:${"1".repeat(64)}`,
            mediaType: "text/markdown",
            sizeBytes: 44,
          },
          review,
        });
      throw new Error(`unexpected ${url}`);
    });
    const queue =
      "/projects/project_example/audits/audit_example/reviews?state=pending&cursor=page-two&auditRevision=5";
    const { router } = renderApplication(queue, { api });
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("link", { name: "Review report →" }),
    );
    expect(
      await screen.findByText("Exact retained evidence for owner acceptance."),
    ).toBeVisible();
    const decision = screen.getByRole("region", { name: "Your decision" });
    expect(
      within(decision).getByRole("button", { name: "Approve" }),
    ).toHaveAttribute("aria-pressed", "false");
    expect(
      within(decision).getByRole("button", { name: "Record decision" }),
    ).toBeDisabled();
    await user.click(screen.getByRole("link", { name: "← Check decisions" }));
    expect(router.state.location.pathname + router.state.location.search).toBe(
      queue,
    );
    await router.navigate(
      "/projects/project_example/audits/audit_example/report?review=removed-review",
    );
    expect(
      await screen.findByText(/requested report review is unavailable/),
    ).toBeVisible();
    expect(
      screen.queryByRole("region", { name: "Your decision" }),
    ).not.toBeInTheDocument();
    expect(mutations).toEqual([]);
  });
});
