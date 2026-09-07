import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const digest = `sha256:${"1".repeat(64)}`;
const timestamp = "2026-09-07T10:00:00Z";
const exactSource = {
  namespace: "sources",
  name: "service",
  revision: "source-r1",
};
const sourceMetadata = {
  artifact: exactSource,
  mediaType: "application/zip",
  size: 42,
  current: false,
  frozen: false,
  createdAt: timestamp,
};
const workflow = {
  ref: { name: "repeat-workflow", version: "1" },
  entryStage: "inspect",
  parameters: { objective: { required: true } },
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
  },
  outputs: {},
  stages: {},
};
const runtimeConfiguration = {
  default: {
    label: "default",
    bindingRevision: "1",
    config: { name: "empty", version: "1", digest },
  },
  labels: [],
};

function headers(origin?: string): Record<string, string> {
  return {
    "cache-control": "no-store",
    "content-type": "application/json",
    "x-contractor-api-version": API_VERSION,
    ...(origin === undefined
      ? {}
      : {
          "access-control-allow-origin": origin,
          "access-control-allow-credentials": "true",
          "access-control-expose-headers":
            "X-Contractor-API-Version, ETag, X-Request-ID",
        }),
  };
}

async function json(
  route: Route,
  value: unknown,
  status = 200,
  extraHeaders: Record<string, string> = {},
): Promise<void> {
  await route.fulfill({
    status,
    body: JSON.stringify(value),
    headers: { ...headers(route.request().headers().origin), ...extraHeaders },
  });
}

async function installFixture(page: Page, apiOrigin: string) {
  const created: Array<{ body: unknown; idempotencyKey: string | null }> = [];
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
          "access-control-allow-origin": request.headers().origin ?? "*",
          "access-control-allow-credentials": "true",
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers":
            "content-type, idempotency-key, x-csrf-token",
        },
      });
      return;
    }
    if (path === "/v1/auth/session") {
      await json(route, {
        principal: {
          userId: "repeat-owner",
          username: "repeat-owner",
          capabilities: ["user", "operations"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (path === "/v1/runs/run-repeat-browser") {
      await json(route, {
        runId: "run-repeat-browser",
        workflow: "repeat-workflow@1",
        state: "failed",
        deletable: true,
        runtimeLabels: [],
        labels: { "eval.id": "sample-1", purpose: "eval" },
        runtimeConfiguration,
        parameters: { objective: "Inspect original source" },
        inputs: {
          source: {
            namespace: "inputs",
            name: "source",
            revision: "run-input-r1",
          },
        },
        attempts: [],
        transitions: [],
        outputs: {},
        outputPublications: [],
        createdAt: timestamp,
        updatedAt: timestamp,
        finishedAt: timestamp,
      });
      return;
    }
    if (path === "/v1/runs/run-repeat-browser/repeat-draft") {
      await json(route, {
        sourceRunId: "run-repeat-browser",
        authority: "ordinary",
        workflow: { name: "repeat-workflow", version: "1" },
        notices: [
          {
            code: "evaluation_labels_require_review",
            severity: "warning",
            field: "labels",
            message: "Review eval identity before submitting.",
          },
        ],
        draft: {
          parameters: { objective: "Inspect original source" },
          runtimeLabels: [],
          labels: { "eval.id": "sample-1", purpose: "eval" },
          executionConfig: { status: "available", value: {} },
          inputs: {
            source: {
              status: "available",
              sourceScope: "user",
              artifact: exactSource,
              metadata: sourceMetadata,
            },
          },
        },
      });
      return;
    }
    if (path === "/v1/runs/run-audit-browser") {
      await json(route, {
        runId: "run-audit-browser",
        projectId: "project-audit",
        workflow: "repeat-workflow@1",
        state: "failed",
        deletable: true,
        runtimeLabels: [],
        labels: { "audit.id": "audit-one" },
        runtimeConfiguration,
        attempts: [],
        transitions: [],
        outputs: {},
        outputPublications: [],
        createdAt: timestamp,
        updatedAt: timestamp,
        finishedAt: timestamp,
      });
      return;
    }
    if (path === "/v1/runs/run-audit-browser/repeat-draft") {
      await json(route, {
        sourceRunId: "run-audit-browser",
        authority: "audit-managed",
        workflow: { name: "repeat-workflow", version: "1" },
        projectId: "project-audit",
        auditId: "audit-one",
        notices: [
          {
            code: "audit_managed_run",
            severity: "blocking",
            field: "authority",
            message: "Continue from the owning Audit.",
          },
        ],
      });
      return;
    }
    if (path === "/v1/projects/project-audit") {
      await json(
        route,
        {
          projectId: "project-audit",
          kind: "project",
          name: "Audit fixture",
          description: "Repeat navigation fixture",
          lifecycle: "active",
          revision: "1",
          createdAt: timestamp,
          updatedAt: timestamp,
        },
        200,
        { etag: '"1"' },
      );
      return;
    }
    if (path === "/v1/audits/audit-one") {
      await json(
        route,
        {
          auditId: "audit-one",
          projectId: "project-audit",
          profile: { name: "fixture", version: "1", digest },
          inputs: {},
          scope: {},
          runtimeLabels: [],
          state: "failed",
          revision: 1,
          dispatchState: "closed",
          holdState: "released",
          limits: {
            maxRounds: 1,
            batchSize: 1,
            maxItemsPerRound: 1,
            maxItemsTotal: 1,
            maxSubmittedRuns: 1,
            maxItemRunAttempts: 1,
            maxEvidenceBytes: 1024,
          },
          reservedRunCount: 1,
          submittedRunCount: 1,
          outstandingRunCount: 0,
          retainedEvidenceBytes: 0,
          eventSequence: 1,
          createdAt: timestamp,
          updatedAt: timestamp,
          finishedAt: timestamp,
        },
        200,
        { etag: '"1"' },
      );
      return;
    }
    if (path === "/v1/workflows/repeat-workflow/versions/1") {
      await json(route, workflow);
      return;
    }
    if (path === "/v1/artifacts") {
      await json(route, { items: [sourceMetadata], page: { hasMore: false } });
      return;
    }
    if (path === "/v1/runs" && request.method() === "POST") {
      created.push({
        body: request.postDataJSON(),
        idempotencyKey: request.headers()["idempotency-key"] ?? null,
      });
      await json(
        route,
        {
          runId: "run-created-browser",
          state: "initializing",
          runtimeLabels: [],
          labels: { "eval.id": "sample-1", purpose: "eval" },
          runtimeConfiguration,
        },
        202,
      );
      return;
    }
    await json(
      route,
      {
        code: "not_found",
        message: `Fixture has no ${request.method()} ${path}`,
        retryable: false,
        requestId: "request-repeat-browser",
      },
      404,
    );
  });
  return created;
}

test("terminal Run creates a reviewed exact repeat draft and preserves conflicts", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const apiOrigin = new URL(configuredBaseURL).origin;
  const created = await installFixture(page, apiOrigin);
  await page.goto("/runs/run-repeat-browser");
  await page.getByRole("button", { name: "Configure another Run" }).click();

  await expect(page).toHaveURL(
    /\/catalog\/workflows\/repeat-workflow\/1#workflow-run-setup$/,
  );
  await expect(page.locator('[name="parameter-objective"]')).toHaveValue(
    "Inspect original source",
  );
  await expect(page.locator('[name="artifact-source"]')).toHaveValue(
    "sources/service@source-r1",
  );
  await expect(
    page.getByText("evaluation_labels_require_review"),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start Workflow Run" }),
  ).toBeDisabled();

  await page
    .locator('[name="parameter-objective"]')
    .fill("Local reviewed edit");
  await page.getByRole("link", { name: "Source Run" }).click();
  await page.getByRole("button", { name: "Configure another Run" }).click();
  await expect(page.getByText("Existing draft preserved")).toBeVisible();
  await page.getByRole("link", { name: "Open the existing draft" }).click();
  await expect(page.locator('[name="parameter-objective"]')).toHaveValue(
    "Local reviewed edit",
  );

  await page
    .getByLabel(/I reviewed the retained inputs, labels and execution settings/)
    .check();
  await page
    .getByRole("button", { name: "Confirm exact input for source" })
    .click();
  await page.getByRole("button", { name: "Start Workflow Run" }).click();
  await expect.poll(() => created.length).toBe(1);
  expect(created[0]?.body).toEqual({
    workflow: "repeat-workflow@1",
    runtimeLabels: [],
    labels: { "eval.id": "sample-1", purpose: "eval" },
    parameters: { objective: "Local reviewed edit" },
    artifacts: { source: exactSource },
  });
  expect(created[0]?.idempotencyKey).toMatch(/^run-ui-[0-9a-f]{32}$/);
});

test("Audit-managed Run returns to the owning Audit", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  await installFixture(page, new URL(configuredBaseURL).origin);
  await page.goto("/runs/run-audit-browser");
  await page.getByRole("button", { name: "Configure another Run" }).click();
  await expect(page).toHaveURL("/projects/project-audit/audits/audit-one");
});
