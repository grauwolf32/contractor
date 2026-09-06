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

test("audit program library renders exact Top 10 and ASVS evidence boundaries", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const apiOrigin =
    process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
  const now = "2026-09-06T10:30:00Z";
  const digest = (value: string) => `sha256:${value.repeat(64)}`;
  const exact = (namespace: string, name: string, revision: string) => ({
    ref: { namespace, name, revision },
    digest: digest("a"),
    mediaType: "application/zip",
    sizeBytes: 1024,
  });
  const standard = (
    scheme: string,
    version: string,
    title: string,
    sourceUrl: string,
    sourceRevision: string,
  ) => ({
    reference: { scheme, version },
    title,
    source: { name: title, url: sourceUrl, revision: sourceRevision },
    license: {
      id: "CC-BY-SA-4.0",
      url: "https://creativecommons.org/licenses/by-sa/4.0/",
      attribution: `OWASP Foundation, ${title}.`,
      disclosure: "full",
    },
    catalog: {
      artifact: {
        namespace: "audit-standards",
        name: `std-${scheme}-${version}`,
        revision: `catalog-${scheme}-r1`,
      },
      digest: digest("b"),
      mediaType: "application/vnd.contractor.audit-standard+zip",
      sizeBytes: 4096,
    },
    retained: {
      artifact: {
        namespace: "audit-program-browser",
        name: `standard-${scheme}-${version}`,
        revision: `retained-${scheme}-r1`,
      },
      digest: digest("b"),
      mediaType: "application/vnd.contractor.audit-standard+zip",
      sizeBytes: 4096,
    },
  });
  const project = {
    projectId: PROJECT_ID,
    kind: "project",
    name: "Audit program library fixture",
    description: "Exact Top 10 and ASVS reports",
    lifecycle: "active",
    revision: "1",
    createdAt: now,
    updatedAt: now,
  };
  const runtimeConfig = {
    default: {
      label: "default",
      explicit: false,
      bindingRevision: 1,
      config: { name: "default-runtime", version: "1", digest: digest("c") },
    },
    labels: [],
  };
  const makeAudit = (
    auditId: string,
    profileName: string,
    pinnedStandard: ReturnType<typeof standard>,
    selected?: { scope: string; levels: string[]; entryIds: string[] },
  ) => ({
    auditId,
    projectId: PROJECT_ID,
    profile: { name: profileName, version: "1", digest: digest("d") },
    inputs: {},
    scope: { objective: "Render exact program evidence." },
    runtimeLabels: [],
    state: "completed",
    revision: 7,
    dispatchState: "closed",
    holdState: "held",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 16,
      maxItemsTotal: 16,
      maxSubmittedRuns: 32,
      maxItemRunAttempts: 2,
      maxEvidenceBytes: 1_048_576,
    },
    reservedRunCount: 10,
    submittedRunCount: 10,
    outstandingRunCount: 0,
    retainedEvidenceBytes: 2048,
    eventSequence: 24,
    createdAt: now,
    updatedAt: now,
    startedAt: now,
    finishedAt: now,
    baseline: {
      inputs: {},
      scope: { objective: "Render exact program evidence." },
      runtimeLabels: [],
      runtimeConfig,
      skills: [
        {
          name: "trace",
          source: { namespace: "skills", name: "trace", revision: "trace-r1" },
          sourceDigest: digest("e"),
          sourceSize: 2048,
        },
      ],
      standards: [pinnedStandard],
      inventory: {
        sourceContentDigest: digest("f"),
        canonicalInventoryDigest: digest("1"),
        gaps: [],
        worklist: exact(
          "audit-program-browser",
          `${auditId}-worklist`,
          "worklist-r1",
        ),
        ...(selected === undefined ? {} : { standardSelection: selected }),
      },
    },
  });
  const top10 = makeAudit(
    "audit_top10_browser",
    "owasp-top10-2025-source-risk",
    standard(
      "owasp-web-top10",
      "2025",
      "OWASP Top 10:2025",
      "https://owasp.org/Top10/2025/",
      "66ebc4798d2ca72973967a20264bdeb70dcf0a13",
    ),
  );
  const asvsEntryIds = [
    "v5.0.0-1.2.4",
    "v5.0.0-1.2.5",
    "v5.0.0-1.3.2",
    "v5.0.0-1.5.1",
    "v5.0.0-2.1.1",
  ];
  const asvs = makeAudit(
    "audit_asvs_browser",
    "owasp-asvs-5-0-l1-source-review",
    standard(
      "owasp-asvs",
      "5.0.0",
      "OWASP ASVS 5.0.0",
      "https://github.com/OWASP/ASVS/tree/v5.0.0_release/5.0",
      "5cf9b032440be53ce345ab3c130fda46ba1ce7a2",
    ),
    {
      scope: "ASVS 5.0 Level 1 source and documentation pilot (5 requirements)",
      levels: ["1"],
      entryIds: asvsEntryIds,
    },
  );
  const coverageRow = (
    auditId: string,
    itemKey: string,
    ordinal: number,
    status: string,
  ) => ({
    roundId: `${auditId}-round-1`,
    itemId: `${auditId}-item-${ordinal}`,
    ordinal,
    itemKey,
    subjectKey: itemKey,
    coverage: {
      status,
      requested: ["observation"],
      completed:
        status === "satisfied" || status === "violated" ? ["observation"] : [],
      gaps:
        status === "inconclusive" || status === "not-tested"
          ? ["bounded-evidence-gap"]
          : [],
    },
    updatedAt: now,
  });
  const top10Statuses = [
    "violated",
    "satisfied",
    "inconclusive",
    "not-tested",
    "violated",
    "satisfied",
    "satisfied",
    "inconclusive",
    "violated",
    "not-tested",
  ];
  const asvsStatuses = [
    "violated",
    "satisfied",
    "inconclusive",
    "not-tested",
    "not-applicable",
  ];
  const programs = {
    [top10.auditId]: {
      audit: top10,
      coverage: top10Statuses.map((status, index) =>
        coverageRow(
          top10.auditId,
          `A${String(index + 1).padStart(2, "0")}:2025`,
          index,
          status,
        ),
      ),
      summary:
        "Bounded OWASP Top 10 risk-awareness results. This is not a security or compliance certification.",
    },
    [asvs.auditId]: {
      audit: asvs,
      coverage: asvsStatuses.map((status, index) =>
        coverageRow(asvs.auditId, asvsEntryIds[index]!, index, status),
      ),
      summary:
        "Exact selected requirements: 5. This is not a security or compliance certification.",
    },
  };

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
          "access-control-allow-methods": "GET, OPTIONS",
          "access-control-allow-headers": "content-type, x-csrf-token",
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
        csrfToken: "p".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}`) {
      await fulfillJSON(route, project, 200, { etag: '"1"' });
      return;
    }
    const match = /^\/v1\/audits\/([^/]+)(?:\/(coverage|report))?$/u.exec(path);
    const program =
      match === null ? undefined : programs[match[1] as keyof typeof programs];
    if (match !== null && program !== undefined) {
      if (match[2] === "coverage") {
        await fulfillJSON(route, {
          items: program.coverage,
          page: { hasMore: false },
        });
      } else if (match[2] === "report") {
        await fulfillJSON(route, {
          status: "ready",
          machineArtifact: {
            ...exact(
              "audit-program-browser",
              `${match[1]}-report.json`,
              "report-json-r1",
            ),
            mediaType: "application/json",
          },
          summaryArtifact: {
            ...exact(
              "audit-program-browser",
              `${match[1]}-report.md`,
              "report-md-r1",
            ),
            mediaType: "text/markdown",
          },
          machine: { conclusion: "completed-with-gaps", certification: false },
          summary: program.summary,
        });
      } else {
        await fulfillJSON(route, program.audit, 200, { etag: '"7"' });
      }
      return;
    }
    await fulfillJSON(
      route,
      {
        code: "not_found",
        message: "not found",
        retryable: false,
        requestId: "request-program-library",
      },
      404,
    );
  });

  await page.goto(`/projects/${PROJECT_ID}/audits/${top10.auditId}`);
  const top10Standards = await page.getByTestId("audit-baseline-standards");
  await expect(top10Standards).toContainText("OWASP Top 10:2025");
  await expect(top10Standards).toContainText("owasp-web-top10@2025");
  await expect(top10Standards).toContainText("CC-BY-SA-4.0");
  await page.getByRole("link", { name: "Coverage" }).click();
  await expect(page.locator("tbody tr")).toHaveCount(10);
  await expect(
    page.getByText("inconclusive", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText("not-tested", { exact: true }).first(),
  ).toBeVisible();
  await page.getByRole("link", { name: "Report" }).click();
  await expect(
    page.getByText(/not a security or compliance certification/u),
  ).toBeVisible();

  await page.goto(`/projects/${PROJECT_ID}/audits/${asvs.auditId}`);
  const asvsStandards = await page.getByTestId("audit-baseline-standards");
  await expect(asvsStandards).toContainText("OWASP ASVS 5.0.0");
  await expect(asvsStandards).toContainText("owasp-asvs@5.0.0");
  await expect(asvsStandards).toContainText("CC-BY-SA-4.0");
  const selected = page.getByTestId("audit-baseline-standard-selection");
  await expect(selected).toContainText("5 exact requirements");
  await expect(selected).toContainText("v5.0.0-2.1.1");
  await page.getByRole("link", { name: "Coverage" }).click();
  await expect(page.locator("tbody tr")).toHaveCount(5);
  await expect(page.getByText("not-applicable", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Report" }).click();
  await expect(page.getByText("Exact selected requirements: 5.")).toBeVisible();
  await expect(
    page.getByText(/not a security or compliance certification/u),
  ).toBeVisible();
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
  let confirmation = page.getByRole("alertdialog", {
    name: "Cancel this Audit?",
  });
  await expect(confirmation).toContainText(project.name);
  await expect(confirmation).toContainText(auditId);
  await expect(
    confirmation.getByRole("button", { name: "Keep Audit unchanged" }),
  ).toBeFocused();
  await confirmation
    .getByRole("button", { name: "Keep Audit unchanged" })
    .click();
  expect(mutations).toHaveLength(1);

  await page.getByRole("button", { name: "Cancel" }).click();
  confirmation = page.getByRole("alertdialog", {
    name: "Cancel this Audit?",
  });
  await confirmation
    .getByRole("button", { name: "Confirm cancellation" })
    .click();
  await expect(page.getByText("cancelled", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Delete" }).click();
  confirmation = page.getByRole("alertdialog", {
    name: "Delete this Audit?",
  });
  await expect(confirmation).toContainText("Deletion is asynchronous");
  await confirmation
    .getByRole("button", { name: "Begin Audit deletion" })
    .click();
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
