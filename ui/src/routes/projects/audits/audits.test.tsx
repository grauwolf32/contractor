import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../../../config/runtime-config";
import { PublicAPI } from "../../../api/client";
import type {
  Audit,
  AuditFinding,
  AuditProfile,
  AuditReviewRequest,
} from "../../../api/audits";
import { Application } from "../../../app/application";
import { applicationRoutes } from "../../../app/router";

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
        description: "The order endpoint may read another owner's record.",
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
  it.each(["text/markdown", "text/plain"])(
    "previews and downloads an exact %s report",
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
          throw new Error(`unexpected ${request.method} ${path}`);
        }),
      );
      renderApplication(
        api,
        "/projects/project_example/audits/audit_example/report",
      );
      const button = await screen.findByRole("button", {
        name: "Download exact summary",
      });
      if (markdown) {
        expect(
          await screen.findByRole("heading", { name: "Coverage summary" }),
        ).toBeVisible();
        expect(screen.getByRole("table")).toHaveTextContent("satisfied");
      } else {
        expect(screen.getByText(/# Coverage summary/u)).toBeVisible();
        expect(
          screen.queryByRole("heading", { name: "Coverage summary" }),
        ).toBeNull();
      }
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
        expect(downloadedName).toBe(
          `audit_example-report.${markdown ? "md" : "txt"}`,
        );
        expect(createURL.mock.calls[0]?.[0].type).toBe(mediaType);
      } finally {
        click.mockRestore();
        URL.createObjectURL = originalCreate;
        URL.revokeObjectURL = originalRevoke;
      }
    },
  );

  it("creates a draft from one compatible exact Project Artifact", async () => {
    const requests: Request[] = [];
    const draft = auditAt("draft", 1);
    const selectedProfile: AuditProfile = {
      ...profile,
      inventory: {
        ...profile.inventory,
        standardSelection: {
          scope: "ASVS 5.0 Level 1 source pilot",
          levels: ["1"],
          entryIds: ["v5.0.0-1.2.4", "v5.0.0-1.2.5"],
        },
      },
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
        if (path === "/v1/audit-profiles") {
          return jsonResponse({
            items: [selectedProfile],
            page: { hasMore: false },
          });
        }
        if (
          path === "/v1/audit-profiles/owasp-top10-2025-source-risk/versions/1"
        ) {
          return jsonResponse(selectedProfile, {
            headers: { ETag: `"${selectedProfile.ref.digest}"` },
          });
        }
        if (path === "/v1/projects/project_example/artifacts") {
          return jsonResponse({
            items: [sourceArtifact],
            page: { hasMore: false },
          });
        }
        if (path === "/v1/projects/project_example/audits") {
          if (request.method === "POST") {
            return jsonResponse(draft, {
              status: 201,
              headers: { ETag: '"1"' },
            });
          }
          return jsonResponse({ items: [], page: { hasMore: false } });
        }
        if (path === "/v1/audits/audit_example") {
          return jsonResponse(draft, { headers: { ETag: '"1"' } });
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    const { router } = renderApplication(
      api,
      "/projects/project_example/audits",
    );
    const user = userEvent.setup();

    await screen.findByRole("heading", { name: "New Audit" });
    const createButton = screen.getByRole("button", {
      name: "Create Audit draft",
    });
    expect(createButton).toBeDisabled();
    expect(
      await screen.findByText(
        "Exact standards pinned at start: owasp-web-top10@2025",
      ),
    ).toBeVisible();
    expect(
      await screen.findByTestId("audit-profile-standard-selection"),
    ).toHaveTextContent("ASVS 5.0 Level 1 source pilot");
    expect(
      screen.getByTestId("audit-profile-standard-selection"),
    ).toHaveTextContent("2 exact requirements");
    await user.selectOptions(
      await screen.findByLabelText("Input source"),
      screen.getByRole("option", {
        name: /sources\/payment-service@revision-7/u,
      }),
    );
    expect(createButton).toBeEnabled();
    await user.type(screen.getByLabelText("Objective"), "Map attack surface");
    await user.click(createButton);

    await vi.waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/projects/project_example/audits/audit_example",
      ),
    );
    expect(
      await screen.findByRole("heading", { name: "audit_example" }),
    ).toBeVisible();
    const create = requests.find(
      (request) =>
        request.method === "POST" &&
        new URL(request.url).pathname === "/v1/projects/project_example/audits",
    );
    expect(create?.headers.get("Idempotency-Key")).toMatch(
      /^create-audit-ui-/u,
    );
    expect(create?.headers.get("X-CSRF-Token")).toBe(session.csrfToken);
    await expect(create?.json()).resolves.toEqual({
      profile: { name: "owasp-top10-2025-source-risk", version: "1" },
      inputs: { source: sourceArtifact.artifact },
      scope: { objective: "Map attack surface" },
    });
  });

  it("shows the exact retained standard identity on the Audit baseline", async () => {
    const current = auditAt("completed", 3);
    current.baseline = top10Baseline(current);
    current.baseline.inventory.standardSelection = {
      scope: "ASVS 5.0 Level 1 source pilot",
      levels: ["1"],
      entryIds: ["v5.0.0-1.2.4", "v5.0.0-2.1.1"],
    };
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
          return jsonResponse(current, { headers: { ETag: '"3"' } });
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/audits/audit_example");

    const standards = await screen.findByTestId("audit-baseline-standards");
    expect(within(standards).getByText("OWASP Top 10:2025")).toBeVisible();
    expect(within(standards).getByText("owasp-web-top10@2025")).toBeVisible();
    expect(standards).toHaveTextContent("CC-BY-SA-4.0");
    expect(
      within(standards).getByRole("link", { name: "source" }),
    ).toHaveAttribute("href", "https://owasp.org/Top10/2025/");
    const selection = screen.getByTestId("audit-baseline-standard-selection");
    expect(selection).toHaveTextContent("ASVS 5.0 Level 1 source pilot");
    expect(selection).toHaveTextContent("v5.0.0-1.2.4");
    expect(selection).toHaveTextContent("v5.0.0-2.1.1");
  });

  it("renders mixed coverage as assessments and refetches active Audits", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let auditReads = 0;
    const active = auditAt("active", 2);
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
          auditReads += 1;
          return jsonResponse(active, { headers: { ETag: '"2"' } });
        }
        if (path.endsWith("/coverage")) {
          return jsonResponse({
            items: [
              {
                roundId: "round_example",
                itemId: "item_1",
                ordinal: 0,
                itemKey: "check-auth",
                subjectKey: "Authentication controls",
                coverage: {
                  status: "violated",
                  requested: ["implementation"],
                  completed: ["implementation"],
                  gaps: [],
                  rationale: "Missing authorization check",
                },
                updatedAt: active.updatedAt,
              },
              {
                roundId: "round_example",
                itemId: "item_2",
                ordinal: 1,
                itemKey: "check-session",
                subjectKey: "Session controls",
                coverage: {
                  status: "inconclusive",
                  requested: ["implementation", "tests"],
                  completed: ["implementation"],
                  gaps: ["tests unavailable"],
                },
                updatedAt: active.updatedAt,
              },
            ],
            page: { hasMore: false },
          });
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/coverage",
    );

    expect(
      await screen.findByRole("heading", { name: "Coverage matrix" }),
    ).toBeVisible();
    const rows = screen.getAllByRole("row");
    expect(within(rows[1]!).getByText("violated")).toBeVisible();
    expect(within(rows[2]!).getByText("inconclusive")).toBeVisible();
    expect(screen.getByText("tests unavailable")).toBeVisible();
    const readsBeforePoll = auditReads;
    await vi.advanceTimersByTimeAsync(1_100);
    await vi.waitFor(() => expect(auditReads).toBeGreaterThan(readsBeforePoll));
    vi.useRealTimers();
  });

  it("reviews a finding with exact revisions and renders immutable history", async () => {
    const requests: Request[] = [];
    let currentAudit = auditAt("completed", 2);
    let currentFinding = findingAt("proposed", 1);
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
                    provenanceIncomplete: true,
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
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/findings",
    );
    const user = userEvent.setup();

    expect(
      await screen.findByRole("heading", {
        name: "Missing object authorization",
      }),
    ).toBeVisible();
    expect(screen.getByText("Unreviewed")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Show provenance" }));
    expect(
      await screen.findByText(
        "attempt 2 · check/authorization-check · settled",
      ),
    ).toBeVisible();
    expect(screen.getByText("verify-authorization@1")).toBeVisible();
    expect(screen.getByText("inventory entry check-one")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Review finding" }));
    await user.selectOptions(await screen.findByLabelText("Severity"), "high");
    await user.type(
      screen.getByLabelText("Analyst rationale"),
      "Confirmed from exact source evidence.",
    );
    await user.click(screen.getByRole("button", { name: "Record decision" }));

    expect(await screen.findByText("true_positive · high")).toBeVisible();
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

    await user.click(screen.getByRole("link", { name: "Reviews" }));
    expect(
      await screen.findByRole("heading", { name: "Human reviews" }),
    ).toBeVisible();
    expect(
      screen.getByText("Confirmed from exact source evidence."),
    ).toBeVisible();
  });

  it("approves an exact non-finding review without treating model text as authority", async () => {
    let currentAudit = auditAt("waiting_review", 3);
    let review: AuditReviewRequest = {
      requestId: "review_active_check",
      auditId: currentAudit.auditId,
      subjectKind: "audit-item-action",
      subjectId: "item_active_check",
      kind: "active-check-approval",
      subjectRevision: 1,
      subjectDigest: `sha256:${"b".repeat(64)}`,
      requestedActions: ["approve", "reject"],
      state: "pending",
      revision: 1,
      createdAt: currentAudit.createdAt,
      updatedAt: currentAudit.updatedAt,
    };
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
          return jsonResponse(currentAudit, {
            headers: { ETag: `"${currentAudit.revision}"` },
          });
        }
        if (
          path === "/v1/audits/audit_example/reviews" &&
          request.method === "GET"
        ) {
          return jsonResponse({ items: [review], page: { hasMore: false } });
        }
        if (
          path ===
            "/v1/audits/audit_example/reviews/review_active_check/decisions" &&
          request.method === "POST"
        ) {
          expect(request.headers.get("If-Match")).toBe('"1"');
          expect(request.headers.get("Idempotency-Key")).toMatch(
            /^audit-action-review-ui-/u,
          );
          expect(await request.json()).toEqual({
            action: "approve",
            rationale: "The target and exact active request are approved.",
          });
          const decision = {
            decisionId: "decision_active_check",
            requestId: review.requestId,
            auditId: review.auditId,
            action: "approve" as const,
            actorId: session.principal.userId,
            rationale: "The target and exact active request are approved.",
            subjectRevision: review.subjectRevision,
            subjectDigest: review.subjectDigest,
            createdAt: currentAudit.updatedAt,
          };
          review = { ...review, state: "decided", revision: 2, decision };
          currentAudit = auditAt("active", 4);
          return jsonResponse({ request: review, decision, replayed: false });
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/reviews",
    );
    const user = userEvent.setup();
    expect(
      await screen.findByText("active-check-approval", { exact: false }),
    ).toBeVisible();
    await user.type(
      screen.getByLabelText("Rationale"),
      "The target and exact active request are approved.",
    );
    await user.click(
      screen.getByRole("button", { name: "Approve exact subject" }),
    );
    expect(
      await screen.findByText("approve", { exact: false, selector: "span" }),
    ).toBeVisible();
    expect(
      screen.getByText("The target and exact active request are approved."),
    ).toBeVisible();
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
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(
      api,
      "/projects/project_example/audits/audit_example/reviews",
    );
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText("Rationale"),
      "Documentation is outside this exact application scope.",
    );
    await user.click(
      screen.getByRole("button", { name: "Mark not applicable" }),
    );
    expect(
      await screen.findByText("not_applicable", {
        exact: false,
        selector: "span",
      }),
    ).toBeVisible();
    expect(decided).toBe(true);
  });

  it("recovers a stale pause from the authoritative revision", async () => {
    let current = auditAt("active", 2);
    let pauseRequested = false;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const path = new URL(request.url).pathname;
        if (path === "/v1/auth/session") return jsonResponse(session);
        if (path === "/v1/projects/project_example") {
          return jsonResponse(project, { headers: { ETag: '"1"' } });
        }
        if (path === "/v1/audits/audit_example/pause") {
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
        if (path === "/v1/audits/audit_example") {
          return jsonResponse(current, {
            headers: { ETag: `"${current.revision}"` },
          });
        }
        throw new Error(`unexpected ${request.method} ${path}`);
      }),
    );
    renderApplication(api, "/projects/project_example/audits/audit_example");
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: "Pause new Runs" }),
    );
    expect(
      within(await screen.findByRole("alert")).getByText(
        "Audit revision changed",
      ),
    ).toBeVisible();
    await vi.waitFor(() =>
      expect(screen.getByRole("button", { name: "Resume" })).toBeVisible(),
    );
    expect(pauseRequested).toBe(true);
    expect(screen.getByText("revision 3")).toBeVisible();
  });
});
