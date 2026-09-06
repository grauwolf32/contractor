import { expect, test, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const PROJECT_ID = "project_audit_browser";
const PROFILE_DIGEST = `sha256:${"2".repeat(64)}`;

function headers(extra: Record<string, string> = {}): Record<string, string> {
  return {
    "cache-control": "no-store",
    "content-type": "application/json",
    "x-contractor-api-version": API_VERSION,
    ...extra,
  };
}

async function fulfillJSON(
  route: Route,
  value: unknown,
  status = 200,
  extraHeaders: Record<string, string> = {},
): Promise<void> {
  const origin = route.request().headers()["origin"];
  await route.fulfill({
    status,
    body: JSON.stringify(value),
    headers: headers({
      ...(origin === undefined
        ? {}
        : {
            "access-control-allow-origin": origin,
            "access-control-allow-credentials": "true",
            "access-control-expose-headers":
              "X-Contractor-API-Version, ETag, X-Request-ID",
          }),
      ...extraHeaders,
    }),
  });
}

test("Project Audit pins exact input and exposes authoritative coverage", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const apiOrigin =
    process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
  const project = {
    projectId: PROJECT_ID,
    kind: "project",
    name: "Audit browser fixture",
    description: "Exact Project input",
    lifecycle: "active",
    revision: "1",
    createdAt: "2026-09-06T10:00:00Z",
    updatedAt: "2026-09-06T10:00:00Z",
  };
  const artifact = {
    artifact: {
      namespace: "sources",
      name: "browser-service",
      revision: "revision-browser-3",
    },
    mediaType: "application/zip",
    size: 3,
    current: true,
    frozen: false,
    createdAt: "2026-09-06T10:00:00Z",
  };
  const exactArtifact = {
    ref: artifact.artifact,
    digest: `sha256:${"1".repeat(64)}`,
    mediaType: artifact.mediaType,
    sizeBytes: artifact.size,
  };
  const profile = {
    ref: {
      name: "source-checklist",
      version: "1",
      digest: PROFILE_DIGEST,
    },
    mode: "custom-checklist",
    standards: [],
    inputs: {
      sources: { required: true, mediaTypes: ["application/zip"] },
    },
    inventory: {
      implementation: "checklist@1",
      sourceInput: "sources",
      itemWorkflowRole: "check",
    },
    execution: {
      roundMode: "fixed-barrier",
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 8,
      maxItemsTotal: 8,
      maxSubmittedRuns: 8,
      maxItemRunAttempts: 2,
      deadlineSeconds: 600,
      maxEvidenceBytes: 1048576,
      incompleteRound: "assess-with-gaps",
    },
    interaction: {
      activeChecks: "prohibited",
      findingConfirmation: "disabled",
      notApplicable: "profile-rule",
      reportAcceptance: "automatic",
    },
    serverCompatible: true,
    requiresInputValidation: true,
    compatibilityReasons: [],
  };
  const baseAudit = {
    auditId: "audit_browser",
    projectId: PROJECT_ID,
    profile: profile.ref,
    inputs: { sources: exactArtifact },
    scope: {},
    runtimeLabels: [],
    dispatchState: "closed",
    holdState: "pending",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 8,
      maxItemsTotal: 8,
      maxSubmittedRuns: 8,
      maxItemRunAttempts: 2,
      maxEvidenceBytes: 1048576,
    },
    reservedRunCount: 0,
    submittedRunCount: 0,
    outstandingRunCount: 0,
    retainedEvidenceBytes: 0,
    eventSequence: 1,
    createdAt: "2026-09-06T10:00:00Z",
    updatedAt: "2026-09-06T10:00:00Z",
  };
  let audit = { ...baseAudit, state: "draft", revision: 1 };
  const createRequests: Array<{
    headers: Record<string, string>;
    body: unknown;
  }> = [];

  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: apiOrigin,
      },
    }),
  );
  await page.route(`${apiOrigin}/v1/**`, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "access-control-allow-origin": new URL(configuredBaseURL).origin,
          "access-control-allow-credentials": "true",
          "access-control-allow-methods": "GET, POST, DELETE, OPTIONS",
          "access-control-allow-headers":
            "content-type, idempotency-key, if-match, x-csrf-token",
        },
      });
      return;
    }
    if (path === "/v1/auth/session") {
      await fulfillJSON(route, {
        principal: {
          userId: "user_browser",
          username: "browser",
          capabilities: ["user", "operations"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}`) {
      await fulfillJSON(route, project, 200, { etag: '"1"' });
      return;
    }
    if (path === "/v1/audit-profiles") {
      await fulfillJSON(route, { items: [profile], page: { hasMore: false } });
      return;
    }
    if (path === "/v1/audit-profiles/source-checklist/versions/1") {
      await fulfillJSON(route, profile, 200, {
        etag: `"${PROFILE_DIGEST}"`,
      });
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}/artifacts`) {
      await fulfillJSON(route, { items: [artifact], page: { hasMore: false } });
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}/audits`) {
      if (request.method() === "POST") {
        createRequests.push({
          headers: request.headers(),
          body: request.postDataJSON(),
        });
        await fulfillJSON(route, audit, 201, { etag: '"1"' });
      } else {
        await fulfillJSON(route, { items: [], page: { hasMore: false } });
      }
      return;
    }
    if (path === "/v1/audits/audit_browser/start") {
      audit = {
        ...audit,
        state: "active",
        revision: 2,
        dispatchState: "open",
        holdState: "held",
        submittedRunCount: 1,
        outstandingRunCount: 1,
      };
      await fulfillJSON(
        route,
        {
          audit,
          round: {
            roundId: "round_browser",
            ordinal: 1,
            manifest: exactArtifact,
            state: "executing",
            expectedItemCount: 1,
            revision: 1,
            createdAt: audit.createdAt,
            updatedAt: audit.updatedAt,
          },
          items: [],
        },
        200,
        { etag: '"2"' },
      );
      return;
    }
    if (path === "/v1/audits/audit_browser") {
      await fulfillJSON(route, audit, 200, { etag: `"${audit.revision}"` });
      return;
    }
    if (path === "/v1/audits/audit_browser/coverage") {
      await fulfillJSON(route, {
        items: [
          {
            roundId: "round_browser",
            itemId: "item_browser",
            ordinal: 0,
            itemKey: "check-authz",
            subjectKey: "Authorization controls",
            coverage: {
              status: "inconclusive",
              requested: ["implementation", "tests"],
              completed: ["implementation"],
              gaps: ["test evidence missing"],
            },
            updatedAt: audit.updatedAt,
          },
        ],
        page: { hasMore: false },
      });
      return;
    }
    await fulfillJSON(
      route,
      {
        code: "not_found",
        message: "not found",
        retryable: false,
        requestId: "request-browser",
      },
      404,
    );
  });

  await page.goto(`/projects/${PROJECT_ID}/audits`);
  await expect(page.getByRole("heading", { name: "New Audit" })).toBeVisible();
  const input = page.getByLabel("Input sources");
  const artifactOption = input
    .locator("option")
    .filter({ hasText: "browser-service@revision-browser-3" });
  const artifactValue = await artifactOption.getAttribute("value");
  if (artifactValue === null || artifactValue === "") {
    throw new Error("exact Artifact option is missing");
  }
  await input.selectOption(artifactValue);
  await page.getByRole("button", { name: "Create Audit draft" }).click();
  await expect(page).toHaveURL(/\/audits\/audit_browser$/u);
  await page.getByRole("button", { name: "Start Audit" }).click();
  await expect(page.getByText("revision 2")).toBeVisible();
  await page.getByRole("link", { name: "Coverage" }).click();
  await expect(page.getByText("inconclusive")).toBeVisible();
  await expect(page.getByText("test evidence missing")).toBeVisible();

  expect(createRequests).toHaveLength(1);
  expect(createRequests[0]!.headers["idempotency-key"]).toMatch(
    /^create-audit-ui-/u,
  );
  expect(createRequests[0]!.headers["x-csrf-token"]).toBe("a".repeat(43));
  expect(createRequests[0]!.body).toEqual({
    profile: { name: "source-checklist", version: "1" },
    inputs: { sources: artifact.artifact },
  });
  const storage = await page.evaluate(() => ({
    local: { ...localStorage },
    session: { ...sessionStorage },
    search: location.search,
  }));
  expect(JSON.stringify(storage)).not.toContain("PROFILE_PACKAGE");
  expect(storage.search).toBe("");
});

test("Audit UI reviews exact evidence and completes destructive lifecycle controls", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const apiOrigin =
    process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
  const auditId = "audit_review_browser";
  const now = "2026-09-06T11:00:00Z";
  const exact = (name: string, revision: string) => ({
    ref: { namespace: "audit-review-browser", name, revision },
    digest: `sha256:${"0".repeat(64)}`,
    mediaType: name.endsWith(".json") ? "application/json" : "text/markdown",
    sizeBytes: 64,
  });
  // Keep the digest expression explicit because the browser contract requires
  // a complete sha256 value, even in a mocked independently served UI flow.
  const machineArtifact = {
    ...exact("report.json", "report-machine-r1"),
    digest: `sha256:${"7".repeat(64)}`,
  };
  const summaryArtifact = {
    ...exact("report.md", "report-summary-r1"),
    digest: `sha256:${"8".repeat(64)}`,
  };
  const project = {
    projectId: PROJECT_ID,
    kind: "project",
    name: "Audit review fixture",
    description: "Browser lifecycle fixture",
    lifecycle: "active",
    revision: "1",
    createdAt: now,
    updatedAt: now,
  };
  const baseAudit = {
    auditId,
    projectId: PROJECT_ID,
    profile: {
      name: "source-checklist",
      version: "1",
      digest: PROFILE_DIGEST,
    },
    inputs: {},
    scope: { objective: "Review retained evidence." },
    runtimeLabels: [],
    dispatchState: "closed",
    holdState: "held",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 8,
      maxItemsTotal: 8,
      maxSubmittedRuns: 8,
      maxItemRunAttempts: 2,
      maxEvidenceBytes: 1_048_576,
    },
    reservedRunCount: 1,
    submittedRunCount: 1,
    outstandingRunCount: 0,
    retainedEvidenceBytes: 512,
    eventSequence: 5,
    createdAt: now,
    updatedAt: now,
  };
  let audit: Record<string, unknown> = {
    ...baseAudit,
    state: "waiting_review",
    revision: 1,
  };
  const proposal = {
    receiptId: "receipt_browser",
    proposalId: "proposal_browser",
    requestDigest: `sha256:${"3".repeat(64)}`,
    clientKey: "candidate-browser",
    proposal: {
      ref: {
        namespace: "audit-findings",
        name: "candidate-browser",
        revision: "proposal-r1",
      },
      digest: `sha256:${"4".repeat(64)}`,
      mediaType: "application/json",
      sizeBytes: 128,
    },
    document: {
      schema: "contractor.audit.finding-proposal.v1",
      client_key: "candidate-browser",
      title: "Retained authorization bypass",
      description: "Exact source evidence requires an analyst decision.",
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
      runId: "run_deleted_source",
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
    createdAt: now,
  };
  const findingId = "finding_browser";
  let finding: Record<string, unknown> = {
    findingId,
    auditId,
    state: "proposed",
    firstProposal: proposal,
    revision: 1,
    createdAt: now,
    updatedAt: now,
  };
  const reviewId = "review_browser";
  const reviewSubjectDigest = `sha256:${"6".repeat(64)}`;
  let review: Record<string, unknown> = {
    requestId: reviewId,
    auditId,
    findingId,
    subjectKind: "finding",
    subjectId: findingId,
    kind: "finding-triage",
    subjectRevision: 1,
    subjectDigest: reviewSubjectDigest,
    requestedActions: [
      "true_positive",
      "false_positive",
      "duplicate",
      "reopen",
      "needs_evidence",
    ],
    state: "pending",
    revision: 1,
    createdAt: now,
    updatedAt: now,
  };
  const mutations: Array<{
    method: string;
    path: string;
    ifMatch: string | undefined;
  }> = [];

  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: apiOrigin,
      },
    }),
  );
  await page.route(`${apiOrigin}/v1/**`, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "access-control-allow-origin": new URL(configuredBaseURL).origin,
          "access-control-allow-credentials": "true",
          "access-control-allow-methods": "GET, POST, DELETE, OPTIONS",
          "access-control-allow-headers":
            "content-type, idempotency-key, if-match, x-csrf-token",
        },
      });
      return;
    }
    if (path === "/v1/auth/session") {
      await fulfillJSON(route, {
        principal: {
          userId: "user_browser",
          username: "browser",
          capabilities: ["user", "operations"],
        },
        csrfToken: "b".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}`) {
      await fulfillJSON(route, project, 200, { etag: '"1"' });
      return;
    }
    if (path === `/v1/audits/${auditId}` && request.method() === "GET") {
      await fulfillJSON(route, audit, 200, { etag: `"${audit.revision}"` });
      return;
    }
    if (path === `/v1/audits/${auditId}/findings`) {
      await fulfillJSON(route, { items: [finding], page: { hasMore: false } });
      return;
    }
    if (path === `/v1/audits/${auditId}/reviews`) {
      await fulfillJSON(route, { items: [review], page: { hasMore: false } });
      return;
    }
    if (path === `/v1/audits/${auditId}/findings/${findingId}/provenance`) {
      await fulfillJSON(route, {
        auditRevision: audit.revision,
        findingRevision: finding.revision,
        items: [
          {
            recordId: "source:receipt_browser",
            kind: "source-proposal",
            receiptId: proposal.receiptId,
            relation: "source",
            proposal: proposal.proposal,
            origin: proposal.origin,
            supportsCurrentAssessment: false,
            createdAt: now,
          },
        ],
        page: { hasMore: false },
      });
      return;
    }
    if (
      path === `/v1/audits/${auditId}/reviews/${reviewId}/decisions` &&
      request.method() === "POST"
    ) {
      mutations.push({
        method: request.method(),
        path,
        ifMatch: request.headers()["if-match"],
      });
      const decision = {
        decisionId: "decision_browser",
        requestId: reviewId,
        auditId,
        findingId,
        actorId: "user_browser",
        verdict: "true_positive",
        severity: "high",
        rationale: "Confirmed against the retained exact source revision.",
        subjectRevision: 1,
        subjectDigest: reviewSubjectDigest,
        createdAt: now,
      };
      finding = {
        ...finding,
        state: "confirmed",
        revision: 2,
        analystVerdict: "true_positive",
        analystSeverity: "high",
        analystDecision: decision,
      };
      review = { ...review, state: "decided", revision: 2, decision };
      audit = { ...audit, revision: 2, eventSequence: 6 };
      await fulfillJSON(route, {
        finding,
        request: review,
        decision,
        replayed: false,
      });
      return;
    }
    if (path === `/v1/audits/${auditId}/report`) {
      await fulfillJSON(route, {
        status: "proposed",
        machineArtifact,
        summaryArtifact,
        machine: {
          conclusion: "completed-with-gaps",
          certification: false,
        },
        summary:
          "One exact finding awaits report acceptance; this is not a certification.",
      });
      return;
    }
    if (
      path === `/v1/audits/${auditId}/cancel` &&
      request.method() === "POST"
    ) {
      mutations.push({
        method: request.method(),
        path,
        ifMatch: request.headers()["if-match"],
      });
      audit = {
        ...audit,
        state: "cancelled",
        revision: 3,
        eventSequence: 7,
        holdState: "released",
      };
      await fulfillJSON(route, audit, 200, { etag: '"3"' });
      return;
    }
    if (path === `/v1/audits/${auditId}` && request.method() === "DELETE") {
      mutations.push({
        method: request.method(),
        path,
        ifMatch: request.headers()["if-match"],
      });
      audit = {
        ...audit,
        state: "deleting",
        revision: 4,
        eventSequence: 8,
      };
      await fulfillJSON(route, audit, 200, { etag: '"4"' });
      return;
    }
    await fulfillJSON(
      route,
      {
        code: "not_found",
        message: "not found",
        retryable: false,
        requestId: "request-review-browser",
      },
      404,
    );
  });

  await page.goto(`/projects/${PROJECT_ID}/audits/${auditId}/findings`);
  await expect(
    page.getByRole("heading", { name: "Retained authorization bypass" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Show provenance" }).click();
  await expect(page.getByText("source-review")).toBeVisible();
  await expect(page.getByText(/Run deleted/u)).toBeVisible();
  await page.getByLabel("Severity").selectOption("high");
  await page
    .getByLabel("Analyst rationale")
    .fill("Confirmed against the retained exact source revision.");
  await page.getByRole("button", { name: "Record decision" }).click();
  await expect(page.getByText("true_positive · high")).toBeVisible();

  await page.getByRole("link", { name: "Report" }).click();
  await expect(
    page.getByText("This exact report is awaiting owner acceptance."),
  ).toBeVisible();
  await expect(page.getByText(/not a certification/u)).toBeVisible();
  await page.getByRole("button", { name: "Cancel" }).click();
  await expect(page.getByText("cancelled", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Delete" }).click();
  await expect(page.getByText("deleting", { exact: true })).toBeVisible();

  expect(mutations).toEqual([
    {
      method: "POST",
      path: `/v1/audits/${auditId}/reviews/${reviewId}/decisions`,
      ifMatch: '"1"',
    },
    {
      method: "POST",
      path: `/v1/audits/${auditId}/cancel`,
      ifMatch: '"2"',
    },
    {
      method: "DELETE",
      path: `/v1/audits/${auditId}`,
      ifMatch: '"3"',
    },
  ]);
});
