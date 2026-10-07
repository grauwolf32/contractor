import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../../../config/runtime-config";
import { PublicAPI } from "../../../api/client";
import type {
  Audit,
  AuditCoverageRow,
  AuditFinding,
  AuditItem,
  AuditProfile,
  AuditReviewRequest,
} from "../../../api/audits";
import { Application } from "../../../app/application";
import * as queryClientFactory from "../../../app/query-client";
import { applicationRoutes } from "../../../app/router";
import { queryKeys } from "../../../api/query-keys";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-06T20:00:00Z",
  absoluteExpiresAt: "2026-09-07T12:00:00Z",
};

const project = {
  projectId: "project_example",
  kind: "project",
  name: "Payment service",
  description: "Reusable service analysis",
  lifecycle: "active",
  revision: "1",
  createdAt: "2026-09-06T10:00:00Z",
  updatedAt: "2026-09-06T10:00:00Z",
};

const profile: AuditProfile = {
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
    findingConfirmation: "human-required",
    notApplicable: "human-required",
    reportAcceptance: "automatic",
  },
  serverCompatible: true,
  requiresInputValidation: true,
  compatibilityReasons: [],
};

const sourceArtifact = {
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
  return {
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
  };
}

function findingAt(
  state: AuditFinding["state"],
  revision: number,
): AuditFinding {
  return {
    findingId: "finding_example",
    auditId: "audit_example",
    state,
    firstProposal: {
      receiptId: "receipt_example",
      proposalId: "proposal_example",
      requestDigest: `sha256:${"3".repeat(64)}`,
      clientKey: "candidate-authz",
      proposal: {
        ref: {
          namespace: "audit-findings",
          name: "candidate-authz",
          revision: "proposal-r1",
        },
        digest: `sha256:${"4".repeat(64)}`,
        mediaType: "application/json",
        sizeBytes: 512,
      },
      document: {
        schema: "contractor.audit.finding-proposal.v1",
        client_key: "candidate-authz",
        title: "Missing object authorization",
        description:
          "The **order endpoint** may read another owner's record.\n\n- Check `ownerId` before reading.",
        subject: { kind: "component", key: "orders" },
        preconditions: [],
        standard_refs: [],
        evidence_ids: [],
        proposed_checks: [],
        severity_suggestion: "high",
        limitations: [],
      },
      evidence: [],
      origin: {
        runId: "run_source",
        stageExecutionId: "stage_source",
        allocationId: "allocation_source",
        invocationId: "invocation_source",
        logicalAgentName: "reviewer",
        workflow: {
          name: "source-review",
          version: "1",
          schemaVersion: "contractor/v1alpha1",
          configurationRef: { name: "source-review", version: "1" },
          closureDigest: `sha256:${"5".repeat(64)}`,
        },
        runDeleted: true,
      },
      retention: "audit-held",
      auditHolds: [],
      createdAt: "2026-09-06T10:00:00Z",
    },
    revision,
    createdAt: "2026-09-06T10:00:00Z",
    updatedAt: "2026-09-06T10:00:00Z",
  };
}

function jsonResponse(value: unknown, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(JSON.stringify(value), { ...options, headers });
}

function renderApplication(api: PublicAPI, path: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

describe("Project Audit routes", () => {
  it.each(["coverage", "report"] as const)(
    "refreshes the final %s projection when the parent stops polling",
    async (section) => {
      const previousScroll = HTMLElement.prototype.scrollIntoView;
      HTMLElement.prototype.scrollIntoView = vi.fn();
      let current = auditAt("active", 2);
      const queryClient = queryClientFactory.createApplicationQueryClient();
      vi.spyOn(
        queryClientFactory,
        "createApplicationQueryClient",
      ).mockReturnValue(queryClient);
      const item: AuditItem = {
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
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const path = new URL((input as Request).url).pathname;
          if (path === "/v1/auth/session") return jsonResponse(session);
          if (path === "/v1/projects/project_example")
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          if (path === "/v1/audits/audit_example")
            return jsonResponse(current, {
              headers: { ETag: `"${current.revision}"` },
            });
          if (path.endsWith("/coverage"))
            return jsonResponse({
              items: [coverageRow],
              page: { hasMore: false },
            });
          // Each section counts its own projection: the check page reads
          // the item collection on every section.
          if (path.endsWith("/items")) {
            if (section === "coverage") reads();
            return jsonResponse({
              items: current.state === "active" ? [] : [item],
              page: { hasMore: false },
            });
          }
          if (path.endsWith("/report")) {
            if (section === "report") reads();
            return jsonResponse(
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
          return jsonResponse({ items: [], page: { hasMore: false } });
        }),
      );
      renderApplication(
        api,
        `/projects/project_example/audits/audit_example/${section}${section === "coverage" ? "#check-item_final" : ""}`,
      );
      await screen.findByText(
        section === "coverage"
          ? /^Attempts are not loaded for this/u
          : /The Audit has not reached report generation/,
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
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          const path = new URL(request.url).pathname;
          if (path === "/v1/auth/session") return jsonResponse(session);
          if (path === "/v1/projects/project_example")
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          if (path === "/v1/audits/audit_example")
            return jsonResponse(auditAt("completed", 3), {
              headers: { ETag: '"3"' },
            });
          if (path === "/v1/audits/audit_example/report")
            return jsonResponse({
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
            return jsonResponse({ items: [], page: { hasMore: false } });
          throw new Error(`unexpected ${request.method} ${path}`);
        }),
      );
      renderApplication(
        api,
        "/projects/project_example/audits/audit_example/report",
      );
      if (!markdown) {
        expect(
          await screen.findByText("Server returned an invalid Audit response"),
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

  it("combines every audit and finding page, filters results, and keeps duplicate reviews within their audit", async () => {
    const firstAudit = auditAt("paused", 3);
    const otherAudit = {
      ...firstAudit,
      auditId: "audit_trace",
      profile: { ...profile.ref, name: "openapi-operation-trace" },
    };
    const first = findingAt("proposed", 1);
    const sibling = {
      ...findingAt("proposed", 1),
      findingId: "finding_sibling",
    };
    sibling.firstProposal.document.title = "Missing rate limit";
    sibling.firstProposal.document.severity_suggestion = "medium";
    const other = { ...findingAt("proposed", 1), auditId: otherAudit.auditId };
    other.firstProposal.document.title = "Trace information exposure";
    other.firstProposal.document.severity_suggestion = "low";
    const review: AuditReviewRequest = {
      requestId: "review_duplicate",
      auditId: firstAudit.auditId,
      findingId: first.findingId,
      subjectId: first.findingId,
      subjectKind: "finding",
      kind: "finding-triage",
      subjectRevision: first.revision,
      subjectDigest: `sha256:${"6".repeat(64)}`,
      requestedActions: [
        "true_positive",
        "false_positive",
        "duplicate",
        "reopen",
        "needs_evidence",
      ],
      state: "pending",
      revision: 1,
      createdAt: firstAudit.createdAt,
      updatedAt: firstAudit.updatedAt,
    };
    const requested: string[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        const path = url.pathname;
        expect(request.method).toBe("GET");
        requested.push(`${path}:${url.searchParams.get("cursor") ?? "first"}`);
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (path === "/v1/projects/project_example/audits")
          return jsonResponse(
            url.searchParams.has("cursor")
              ? { items: [otherAudit], page: { hasMore: false } }
              : {
                  items: [firstAudit],
                  page: { hasMore: true, nextCursor: "other-audits" },
                },
          );
        if (path === "/v1/audits/audit_example/findings")
          return jsonResponse(
            url.searchParams.has("cursor")
              ? { items: [sibling], page: { hasMore: false } }
              : {
                  items: [first],
                  page: { hasMore: true, nextCursor: "more-findings" },
                },
          );
        if (path === "/v1/audits/audit_trace/findings")
          return jsonResponse({ items: [other], page: { hasMore: false } });
        if (path === "/v1/audits/audit_example/reviews") {
          expect(url.searchParams.get("state")).toBe("pending");
          return jsonResponse({
            items: [review],
            page: { hasMore: false },
          });
        }
        if (path === "/v1/audits/audit_trace/reviews")
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${path}`);
      }),
    );
    const { container, router } = renderApplication(
      api,
      "/projects/project_example/findings",
    );
    const user = userEvent.setup();
    expect(await screen.findByText("3 of 3 findings · 2 audits")).toBeVisible();
    const filters = within(
      screen.getByRole("region", { name: "Finding filters" }),
    );
    expect(requested).toEqual(
      expect.arrayContaining([
        "/v1/projects/project_example/audits:other-audits",
        "/v1/audits/audit_example/findings:more-findings",
        "/v1/audits/audit_example/reviews:first",
      ]),
    );
    const cards = container.querySelectorAll(".audit-finding-card");
    expect(cards).toHaveLength(3);
    expect(new Set([...cards].map((card) => card.id)).size).toBe(3);
    const firstCard = within(
      screen
        .getByRole("heading", { name: first.firstProposal.document.title })
        .closest(".audit-finding-card") as HTMLElement,
    );
    await user.click(
      await firstCard.findByRole("button", { name: "More decisions" }),
    );
    await user.click(firstCard.getByRole("button", { name: "Duplicate…" }));
    const original = firstCard.getByRole("group", { name: "Duplicate of" });
    expect(
      await within(original).findByRole("radio", {
        name: "Missing rate limit finding_sibling",
      }),
    ).not.toBeChecked();
    expect(
      within(original)
        .getAllByRole("radio")
        .map((option) => option.closest("label")?.textContent),
    ).toEqual(["Missing rate limit finding_sibling"]);
    expect(
      screen.getByRole("link", {
        name: "OpenAPI · Operation trace · it_trace",
      }),
    ).toHaveAttribute(
      "href",
      "/projects/project_example/audits/audit_trace/findings",
    );

    await user.selectOptions(filters.getByLabelText("Audit"), "audit_trace");
    expect(screen.getByText("1 of 3 findings · 2 audits")).toBeVisible();
    expect(
      screen.queryByRole("heading", {
        name: first.firstProposal.document.title,
      }),
    ).not.toBeInTheDocument();
    expect(router.state.location.search).toBe("?audit=audit_trace");
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await user.selectOptions(filters.getByLabelText("Severity"), "medium");
    expect(
      screen.getByRole("heading", { name: "Missing rate limit" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("heading", { name: "Trace information exposure" }),
    ).not.toBeInTheDocument();
    await user.selectOptions(filters.getByLabelText("Severity"), "");
    await user.type(
      screen.getByLabelText("Search findings"),
      "TRACE INFORMATION",
    );
    expect(screen.getByText("1 of 3 findings · 2 audits")).toBeVisible();
    await user.type(screen.getByLabelText("Search findings"), " missing");
    expect(
      screen.getByRole("heading", { name: "No matching findings" }),
    ).toBeVisible();
  });

  it("keeps available findings visible when another audit fails and retries the missing audit", async () => {
    const firstAudit = auditAt("paused", 3);
    const otherAudit = {
      ...firstAudit,
      auditId: "audit_trace",
      profile: { ...profile.ref, name: "openapi-operation-trace" },
    };
    let failed = true;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (path === "/v1/projects/project_example/audits")
          return jsonResponse({
            items: [firstAudit, otherAudit],
            page: { hasMore: false },
          });
        if (path === "/v1/audits/audit_example/findings")
          return jsonResponse({
            items: [findingAt("proposed", 1)],
            page: { hasMore: false },
          });
        if (path === "/v1/audits/audit_trace/findings" && failed)
          return jsonResponse(
            {
              code: "unavailable",
              message: "Findings unavailable",
              retryable: true,
              requestId: "request_failed",
            },
            { status: 503 },
          );
        if (
          path.endsWith("/reviews") ||
          path === "/v1/audits/audit_trace/findings"
        )
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/findings");
    const user = userEvent.setup();
    expect(
      await screen.findByRole("heading", {
        name: "Missing object authorization",
      }),
    ).toBeVisible();
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Findings unavailable: OpenAPI · Operation trace",
    );
    expect(screen.getByText("1 of 1 findings loaded · 2 audits")).toBeVisible();
    expect(screen.queryByText("No findings yet")).not.toBeInTheDocument();
    failed = false;
    await user.click(screen.getByRole("button", { name: "Retry findings" }));
    expect(await screen.findByText("1 of 1 findings · 2 audits")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each(["audit", "project"] as const)(
    "reviews a finding with exact revisions from the %s view and renders immutable history",
    async (scope) => {
      const requests: Request[] = [];
      let currentAudit = auditAt("completed", 2);
      let currentFinding = findingAt("proposed", 1);
      currentFinding.firstProposal.document.description +=
        "\n\n" +
        "Retained evidence must remain readable without another action. ".repeat(
          8,
        ) +
        "\n\nThe final paragraph contains the complete remediation context.";
      let reviews: AuditReviewRequest[] = [];
      const pendingReview: AuditReviewRequest = {
        requestId: "review_example",
        auditId: currentAudit.auditId,
        findingId: currentFinding.findingId,
        subjectKind: "finding",
        subjectId: currentFinding.findingId,
        kind: "finding-triage",
        subjectRevision: 1,
        subjectDigest: `sha256:${"6".repeat(64)}`,
        requestedActions: [
          "true_positive",
          "false_positive",
          "duplicate",
          "reopen",
          "needs_evidence",
        ],
        state: "pending",
        revision: 1,
        createdAt: currentAudit.createdAt,
        updatedAt: currentAudit.updatedAt,
      };
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input instanceof Request ? input : new Request(input);
          requests.push(request.clone());
          const path = new URL(request.url).pathname;
          if (path === "/v1/auth/session") return jsonResponse(session);
          if (path === "/v1/projects/project_example") {
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          }
          if (path === "/v1/projects/project_example/audits") {
            return jsonResponse({
              items: [currentAudit],
              page: { hasMore: false },
            });
          }
          if (path === "/v1/audits/audit_example") {
            return jsonResponse(currentAudit, {
              headers: { ETag: `"${currentAudit.revision}"` },
            });
          }
          if (path === "/v1/audits/audit_example/findings") {
            return jsonResponse({
              items: [currentFinding],
              page: { hasMore: false },
            });
          }
          if (path === "/v1/audits/audit_example/reviews") {
            return jsonResponse({ items: reviews, page: { hasMore: false } });
          }
          if (
            path ===
            "/v1/audits/audit_example/findings/finding_example/provenance"
          ) {
            return jsonResponse({
              auditRevision: currentAudit.revision,
              findingRevision: currentFinding.revision,
              items: [
                {
                  recordId: "attempt:receipt_example:execution_item_example",
                  kind: "check-attempt",
                  receiptId: "receipt_example",
                  relation: "verification",
                  proposal: currentFinding.firstProposal.proposal,
                  origin: currentFinding.firstProposal.origin,
                  supportsCurrentAssessment: true,
                  createdAt: currentAudit.createdAt,
                  assessment: {
                    assessmentId: "assessment_example",
                    semanticAssessment: "supported",
                    result: {
                      ref: {
                        namespace: "audit-results",
                        name: "check-one",
                        revision: "result-r2",
                      },
                      digest: `sha256:${"8".repeat(64)}`,
                    },
                    receiptId: "receipt_example",
                    directVerification: false,
                    acceptedAt: currentAudit.createdAt,
                  },
                  attempt: {
                    executionItemId: "execution_item_example",
                    executionId: "execution_example",
                    itemId: "item_example",
                    itemAttempt: 2,
                    role: "check",
                    workflowRole: "authorization-check",
                    state: "settled",
                    collectionDisposition: "accepted-result",
                    terminalOutcome: "succeeded",
                    runId: "run_verification",
                    runDeleted: true,
                    runProvenance: {
                      schema: "contractor.audit.run-provenance.v1",
                      runId: "run_verification",
                      workflow: {
                        name: "verify-authorization",
                        version: "1",
                        schemaVersion: "contractor/v1alpha1",
                        configurationRef: {
                          name: "verify-authorization",
                          version: "1",
                        },
                        closureDigest: `sha256:${"9".repeat(64)}`,
                      },
                    },
                    task: {
                      ref: {
                        namespace: "audit-task-packages",
                        name: "check-one",
                        revision: "task-r1",
                      },
                      digest: `sha256:${"a".repeat(64)}`,
                    },
                    itemOrigin: {
                      schema: "contractor.audit.item-origin.v1",
                      entryKey: "check-one",
                      sourceRef: {
                        namespace: "inputs",
                        name: "checks",
                        revision: "checks-r1",
                      },
                      sourceContentDigest: `sha256:${"b".repeat(64)}`,
                      sourceMediaType: "application/json",
                      canonicalInventoryDigest: `sha256:${"c".repeat(64)}`,
                    },
                    result: {
                      ref: {
                        namespace: "audit-results",
                        name: "check-one",
                        revision: "result-r2",
                      },
                      digest: `sha256:${"8".repeat(64)}`,
                    },
                    createdAt: currentAudit.createdAt,
                    collectedAt: currentAudit.updatedAt,
                  },
                },
              ],
              page: { hasMore: false },
            });
          }
          if (
            path ===
              "/v1/audits/audit_example/findings/finding_example/reviews" &&
            request.method === "POST"
          ) {
            expect(request.headers.get("If-Match")).toBe('"1"');
            reviews = [pendingReview];
            currentAudit = auditAt("completed", 3);
            return jsonResponse(pendingReview, {
              status: 201,
              headers: { ETag: '"1"' },
            });
          }
          if (
            path ===
              "/v1/audits/audit_example/reviews/review_example/decisions" &&
            request.method === "POST"
          ) {
            expect(request.headers.get("If-Match")).toBe('"1"');
            const body = (await request.json()) as Record<string, unknown>;
            expect(body).toEqual({
              verdict: "true_positive",
              severity: "high",
              rationale: "Confirmed from exact source evidence.",
            });
            const decision = {
              decisionId: "decision_example",
              requestId: pendingReview.requestId,
              auditId: currentAudit.auditId,
              findingId: currentFinding.findingId,
              actorId: session.principal.userId,
              verdict: "true_positive" as const,
              severity: "high" as const,
              rationale: "Confirmed from exact source evidence.",
              subjectRevision: 1,
              subjectDigest: pendingReview.subjectDigest,
              createdAt: currentAudit.updatedAt,
            };
            currentFinding = {
              ...currentFinding,
              state: "confirmed",
              revision: 2,
              analystVerdict: "true_positive",
              analystSeverity: "high",
              analystDecision: decision,
            };
            reviews = [
              { ...pendingReview, state: "decided", revision: 2, decision },
            ];
            currentAudit = auditAt("completed", 4);
            return jsonResponse({
              finding: currentFinding,
              request: reviews[0],
              decision,
              replayed: false,
            });
          }
          if (path.endsWith("/coverage") || path.endsWith("/reviews"))
            return jsonResponse({ items: [], page: { hasMore: false } });
          throw new Error(`unexpected ${request.method} ${path}`);
        }),
      );
      renderApplication(
        api,
        scope === "audit"
          ? "/projects/project_example/audits/audit_example/findings"
          : "/projects/project_example/findings",
      );
      const user = userEvent.setup();

      expect(
        await screen.findByRole("heading", {
          name: "Missing object authorization",
        }),
      ).toBeVisible();
      expect(screen.getByText("Unreviewed", { selector: "dd" })).toBeVisible();
      expect(
        await screen.findByText(
          "The final paragraph contains the complete remediation context.",
        ),
      ).toBeVisible();
      expect(
        await screen.findByText("order endpoint", { selector: "strong" }),
      ).toBeVisible();
      expect(screen.getByText("ownerId", { selector: "code" })).toBeVisible();
      const sourceLink = within(
        screen.getByRole("region", { name: "Source artifacts" }),
      ).getByRole("link");
      expect(sourceLink).toHaveAttribute(
        "href",
        `/projects/project_example/artifacts/sources/${sourceArtifact.artifact.name}?revision=${sourceArtifact.artifact.revision}`,
      );

      await user.click(screen.getByRole("button", { name: "Show provenance" }));
      expect(
        await screen.findByText(
          "attempt 2 · check/authorization-check · settled",
        ),
      ).toBeVisible();
      expect(screen.getByText("verify-authorization@1")).toBeVisible();
      expect(screen.getByText("inventory entry check-one")).toBeVisible();
      const findingCard = within(
        screen
          .getByRole("heading", { name: "Missing object authorization" })
          .closest(".audit-finding-card") as HTMLElement,
      );
      await user.click(
        await findingCard.findByRole("button", { name: "Confirm issue" }),
      );
      await user.click(findingCard.getByRole("radio", { name: "High" }));
      await user.type(
        findingCard.getByRole("textbox", { name: "Why" }),
        "Confirmed from exact source evidence.",
      );
      await user.click(
        findingCard.getByRole("button", { name: "Record decision" }),
      );

      expect(await screen.findByText("true_positive · high")).toBeVisible();
      expect(
        within(
          findingCard.getByRole("region", { name: "Current decision" }),
        ).getByText("Confirmed · High"),
      ).toBeVisible();
      expect(
        findingCard.getByRole("button", { name: "Change decision" }),
      ).toBeVisible();
      const mutationRequests = requests.filter(
        (request) => request.method === "POST",
      );
      expect(mutationRequests).toHaveLength(2);
      expect(mutationRequests[0]?.headers.get("Idempotency-Key")).toMatch(
        /^audit-finding-review-ui-/u,
      );
      expect(mutationRequests[1]?.headers.get("Idempotency-Key")).toMatch(
        /^audit-finding-review-ui-/u,
      );

      if (scope === "project") {
        await user.click(
          screen.getByRole("link", {
            name: "OWASP Top 10 · Source risks · _example",
          }),
        );
      }
      await user.click(await screen.findByRole("link", { name: "Decisions" }));
      expect(
        await screen.findByRole("heading", { name: "Decisions" }),
      ).toBeVisible();
      expect(
        screen.getByText("Confirmed from exact source evidence."),
      ).toBeVisible();
    },
  );

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
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (path === "/v1/audits/audit_example") {
          return jsonResponse(currentAudit, { headers: { ETag: '"3"' } });
        }
        if (
          path === "/v1/audits/audit_example/reviews" &&
          request.method === "GET"
        ) {
          return jsonResponse({
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
          return jsonResponse({
            request: review,
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
            replayed: false,
          });
        }
        if (path === "/v1/audits/audit_example/items") {
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/audits/audit_example/report") {
          return jsonResponse({ status: "pending" });
        }
        if (path.endsWith("/coverage") || path.endsWith("/reviews"))
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/reviews",
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
    expect(screen.getByText("Not applicable")).toBeVisible();
    expect(decided).toBe(true);
  });
});

describe("Bounded Audit collections", () => {
  /** Six pages of one record each: five fill the batch, one waits behind it. */
  function pagedResponse<T>(
    cursor: string | null,
    record: (ordinal: number) => T,
    pages = 6,
  ): Response {
    const ordinal = cursor === null ? 0 : Number(cursor.slice("page-".length));
    return jsonResponse({
      items: [record(ordinal)],
      page:
        ordinal + 1 < pages
          ? { hasMore: true, nextCursor: `page-${ordinal + 1}` }
          : { hasMore: false },
    });
  }

  it("caps project findings per audit and offers one Load more for the aggregate", async () => {
    const paused = auditAt("paused", 3);
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const url = new URL((input as Request).url);
        const path = url.pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (path === "/v1/projects/project_example/audits")
          return jsonResponse({ items: [paused], page: { hasMore: false } });
        if (path === "/v1/audits/audit_example/findings")
          return pagedResponse(url.searchParams.get("cursor"), (ordinal) => {
            const finding = findingAt("proposed", 1);
            finding.findingId = `finding_${ordinal + 1}`;
            finding.firstProposal.document.title = `Finding ${ordinal + 1}`;
            return finding;
          });
        if (path === "/v1/audits/audit_example/reviews")
          return jsonResponse({ items: [], page: { hasMore: false } });
        throw new Error(`unexpected ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/findings");
    expect(await screen.findByText("5 of 5 findings · 1 audits")).toBeVisible();
    expect(screen.getByText("Showing 5 of ≥5 findings")).toBeVisible();
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", {
        name: "Some audits have more findings — load more",
      }),
    );
    expect(await screen.findByText("6 of 6 findings · 1 audits")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Finding 6" })).toBeVisible();
    expect(screen.queryByText(/Showing 5 of/u)).toBeNull();
    await user.type(screen.getByLabelText("Search findings"), "Finding 6");
    expect(screen.getByText("1 of 6 findings · 1 audits")).toBeVisible();
  });

  it.each([true, false])(
    "opens a project finding decision after 250 historical reviews (already pending: %s)",
    async (alreadyPending) => {
      const audit = auditAt("completed", 4);
      const finding = findingAt("proposed", 2);
      const pending: AuditReviewRequest = {
        requestId: "review_pending",
        auditId: audit.auditId,
        subjectKind: "finding",
        subjectId: finding.findingId,
        findingId: finding.findingId,
        kind: "finding-triage",
        subjectRevision: finding.revision,
        subjectDigest: `sha256:${"b".repeat(64)}`,
        requestedActions: ["true_positive", "false_positive", "duplicate"],
        state: "pending",
        revision: 1,
        createdAt: audit.createdAt,
        updatedAt: audit.updatedAt,
      };
      const reviews: AuditReviewRequest[] = Array.from(
        { length: 250 },
        (_, index) => ({
          ...pending,
          requestId: `review_history_${index}`,
          state: "decided",
        }),
      );
      reviews.push({
        ...pending,
        requestId: "review_stale",
        subjectRevision: finding.revision - 1,
      });
      if (alreadyPending) reviews.push(pending);
      const reads: URLSearchParams[] = [];
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const request = input as Request;
          const url = new URL(request.url);
          if (url.pathname === "/v1/auth/session") return jsonResponse(session);
          if (url.pathname === "/v1/projects/project_example")
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          if (url.pathname === "/v1/projects/project_example/audits")
            return jsonResponse({ items: [audit], page: { hasMore: false } });
          if (url.pathname === "/v1/audits/audit_example/findings")
            return jsonResponse({ items: [finding], page: { hasMore: false } });
          if (url.pathname === "/v1/audits/audit_example/reviews") {
            reads.push(url.searchParams);
            const matching = reviews.filter(
              (review) =>
                (!url.searchParams.has("finding") ||
                  review.findingId === url.searchParams.get("finding")) &&
                (!url.searchParams.has("state") ||
                  review.state === url.searchParams.get("state")),
            );
            const offset = Number(url.searchParams.get("cursor") ?? "0");
            return jsonResponse({
              items: matching.slice(offset, offset + 50),
              page:
                offset + 50 < matching.length
                  ? { hasMore: true, nextCursor: String(offset + 50) }
                  : { hasMore: false },
            });
          }
          if (
            url.pathname ===
              "/v1/audits/audit_example/findings/finding_example/reviews" &&
            request.method === "POST"
          ) {
            reviews.push(pending);
            return jsonResponse(pending, {
              status: 201,
              headers: { ETag: '"1"' },
            });
          }
          throw new Error(`unexpected ${request.method} ${url.pathname}`);
        }),
      );
      renderApplication(api, "/projects/project_example/findings");
      const decision = await screen.findByRole("region", {
        name: "Your decision",
      });
      expect(
        within(decision).getByRole("button", { name: "Record decision" }),
      ).toBeVisible();
      // An open request offers only its requested verdicts; without one,
      // recording opens a request that offers every verdict.
      expect(
        within(decision).queryByRole("button", { name: "Needs evidence" }) ===
          null,
      ).toBe(alreadyPending);
      expect(reads).not.toHaveLength(0);
      for (const query of reads) {
        expect(query.has("finding")).toBe(false);
        expect(query.get("state")).toBe("pending");
        expect(query.has("cursor")).toBe(false);
      }
    },
  );

  it("reads pending review status once per Audit with 250 loaded findings", async () => {
    const audit = auditAt("completed", 4);
    const allFindings = Array.from({ length: 250 }, (_, index) => ({
      ...findingAt("proposed", 1),
      findingId: `finding_${index}`,
    }));
    let reviewReads = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const url = new URL((input as Request).url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (url.pathname === "/v1/projects/project_example/audits")
          return jsonResponse({ items: [audit], page: { hasMore: false } });
        if (url.pathname === "/v1/audits/audit_example/findings") {
          const offset = Number(url.searchParams.get("cursor") ?? "0");
          return jsonResponse({
            items: allFindings.slice(offset, offset + 50),
            page:
              offset + 50 < allFindings.length
                ? { hasMore: true, nextCursor: String(offset + 50) }
                : { hasMore: false },
          });
        }
        if (url.pathname === "/v1/audits/audit_example/reviews") {
          expect(url.searchParams.get("state")).toBe("pending");
          expect(url.searchParams.has("finding")).toBe(false);
          reviewReads += 1;
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        throw new Error(`unexpected ${url.pathname}`);
      }),
    );
    renderApplication(api, "/projects/project_example/findings");
    expect(
      await screen.findByText("250 of 250 findings · 1 audits"),
    ).toBeVisible();
    expect(reviewReads).toBe(1);
  });

  it("keeps new review creation closed until later pending pages are read", async () => {
    const audit = auditAt("completed", 4);
    const finding = findingAt("proposed", 2);
    const current: AuditReviewRequest = {
      requestId: "review_current",
      auditId: audit.auditId,
      findingId: finding.findingId,
      subjectKind: "finding",
      subjectId: finding.findingId,
      kind: "finding-triage",
      subjectRevision: finding.revision,
      subjectDigest: `sha256:${"a".repeat(64)}`,
      requestedActions: ["true_positive", "false_positive"],
      state: "pending",
      revision: 1,
      createdAt: audit.createdAt,
      updatedAt: audit.updatedAt,
    };
    const pending: AuditReviewRequest[] = Array.from(
      { length: 250 },
      (_, index) => ({
        ...current,
        requestId: `review_other_${index}`,
        findingId: `finding_other_${index}`,
        subjectId: `finding_other_${index}`,
      }),
    );
    pending.push(current);
    const reviewReads: URLSearchParams[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const url = new URL((input as Request).url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (url.pathname === "/v1/projects/project_example/audits")
          return jsonResponse({ items: [audit], page: { hasMore: false } });
        if (url.pathname === "/v1/audits/audit_example/findings")
          return jsonResponse({ items: [finding], page: { hasMore: false } });
        if (url.pathname === "/v1/audits/audit_example/reviews") {
          reviewReads.push(url.searchParams);
          const offset = Number(url.searchParams.get("cursor") ?? "0");
          return jsonResponse({
            items: pending.slice(offset, offset + 50),
            page:
              offset + 50 < pending.length
                ? { hasMore: true, nextCursor: String(offset + 50) }
                : { hasMore: false },
          });
        }
        throw new Error(`unexpected ${url.pathname}`);
      }),
    );
    renderApplication(api, "/projects/project_example/findings");
    const loadMore = await screen.findAllByRole("button", {
      name: "Load more pending reviews",
    });
    expect(screen.queryByRole("region", { name: "Your decision" })).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Record decision" }),
    ).toBeNull();
    expect(reviewReads).toHaveLength(5);
    await userEvent.setup().click(loadMore[0]!);
    expect(
      await screen.findByRole("region", { name: "Your decision" }),
    ).toBeVisible();
    expect(reviewReads).toHaveLength(6);
    expect(reviewReads.every((query) => query.get("state") === "pending")).toBe(
      true,
    );
  });

  it.each([
    ["completed", false],
    ["active", true],
  ] as const)(
    "polls project findings of a %s audit: %s",
    async (state, polls) => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      const audit = auditAt(state, 3);
      const requests = { audits: 0, findings: 0, reviews: 0 };
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          const path = new URL((input as Request).url).pathname;
          if (path === "/v1/auth/session") return jsonResponse(session);
          if (path === "/v1/projects/project_example")
            return jsonResponse(project, { headers: { ETag: '"1"' } });
          if (path === "/v1/projects/project_example/audits") {
            requests.audits += 1;
            return jsonResponse({ items: [audit], page: { hasMore: false } });
          }
          if (path === "/v1/audits/audit_example/findings") {
            requests.findings += 1;
            return jsonResponse({
              items: [findingAt("proposed", 1)],
              page: { hasMore: false },
            });
          }
          if (path === "/v1/audits/audit_example/reviews") {
            requests.reviews += 1;
            return jsonResponse({ items: [], page: { hasMore: false } });
          }
          throw new Error(`unexpected ${path}`);
        }),
      );
      renderApplication(api, "/projects/project_example/findings");
      expect(
        await screen.findByText("1 of 1 findings · 1 audits"),
      ).toBeVisible();
      const before = { ...requests };
      await vi.advanceTimersByTimeAsync(11_000);
      if (polls) {
        await vi.waitFor(() => {
          expect(requests.audits).toBeGreaterThan(before.audits);
          expect(requests.findings).toBeGreaterThan(before.findings);
          expect(requests.reviews).toBeGreaterThan(before.reviews);
        });
      } else {
        expect(requests).toEqual(before);
      }
      vi.useRealTimers();
    },
  );
});

describe("Audit workspace snapshot navigation", () => {
  it("uses whole-filter totals, pins continuations, and resets stale cursors on filter changes", async () => {
    const audit = auditAt("completed", 5);
    const queries: URLSearchParams[] = [];
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (url.pathname === "/v1/audits/audit_example")
          return jsonResponse(audit, { headers: { ETag: '"5"' } });
        if (url.pathname.endsWith("/findings")) {
          // The check page's own unfiltered read of every possible issue.
          if ([...url.searchParams.keys()].every((key) => key === "limit"))
            return jsonResponse({
              items: [],
              page: { hasMore: false },
              total: 0,
              auditRevision: 5,
              asOf: audit.updatedAt,
            });
          queries.push(url.searchParams);
          if (url.searchParams.has("cursor"))
            return jsonResponse(
              {
                code: "conflict",
                message: "Audit changed",
                retryable: false,
                requestId: "stale",
              },
              { status: 409 },
            );
          return jsonResponse({
            items: [findingAt("proposed", 1)],
            page: { hasMore: true, nextCursor: "next-exact" },
            total: url.searchParams.has("severity") ? 7 : 65,
            auditRevision: 5,
            asOf: audit.updatedAt,
          });
        }
        if (url.pathname.endsWith("/reviews"))
          return jsonResponse({
            items: [],
            page: { hasMore: false },
            total: 0,
            auditRevision: 5,
            asOf: audit.updatedAt,
          });
        throw new Error(`unexpected ${url}`);
      }),
    );
    const { router } = renderApplication(
      api,
      "/projects/project_example/audits/audit_example/findings?verdict=unreviewed",
    );
    const user = userEvent.setup();
    expect(
      await screen.findByText(/Showing 1 of 65 matching records/),
    ).toBeVisible();
    expect(queries).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: "Next page" }));
    expect(await screen.findByText(/This Audit changed/)).toBeVisible();
    expect(queries.at(-1)?.get("cursor")).toBe("next-exact");
    expect(queries.at(-1)?.get("auditRevision")).toBe("5");
    expect(queries.at(-1)?.get("verdict")).toBe("unreviewed");
    await user.selectOptions(screen.getByLabelText("Analyst severity"), "high");
    expect(
      await screen.findByText(/Showing 1 of 7 matching records/),
    ).toBeVisible();
    expect(router.state.location.search).toBe(
      "?verdict=unreviewed&severity=high",
    );
    expect(queries.at(-1)?.has("cursor")).toBe(false);
    expect(queries.at(-1)?.has("auditRevision")).toBe(false);
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
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (request.method !== "GET") mutations.push(request.method);
        if (url.pathname === "/v1/auth/session") return jsonResponse(session);
        if (url.pathname === "/v1/projects/project_example")
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        if (url.pathname === "/v1/audits/audit_example")
          return jsonResponse(audit, { headers: { ETag: '"5"' } });
        if (url.pathname.endsWith("/reviews"))
          return jsonResponse({
            items: [review],
            page: { hasMore: false },
            total: 51,
            auditRevision: 5,
            asOf: audit.updatedAt,
          });
        if (url.pathname.endsWith("/report"))
          return jsonResponse({
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
      }),
    );
    const queue =
      "/projects/project_example/audits/audit_example/reviews?state=pending&cursor=page-two&auditRevision=5";
    const { router } = renderApplication(api, queue);
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
