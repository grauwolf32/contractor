import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const RUN_ID = "run_0123456789abcdef0123456789abcdef";
const LONG_REVISION = `rev_${"a".repeat(64)}`;

function apiHeaders(): Record<string, string> {
  return {
    "cache-control": "no-store",
    "content-type": "application/json",
    "x-contractor-api-version": API_VERSION,
  };
}

async function fulfillJson(route: Route, value: unknown): Promise<void> {
  await route.fulfill({
    body: JSON.stringify(value),
    headers: apiHeaders(),
    status: 200,
  });
}

async function installRunAPI(page: Page, uiOrigin: string): Promise<void> {
  await page.route("**/runtime-config.json", async (route) => {
    await route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: uiOrigin,
      },
    });
  });
  await page.route(`${uiOrigin}/v1/**`, async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/auth/session") {
      await fulfillJson(route, {
        principal: {
          userId: "user_responsive",
          username: "responsive",
          capabilities: ["user", "operations"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (url.pathname === `/v1/runs/${RUN_ID}`) {
      await fulfillJson(route, {
        runId: RUN_ID,
        workflow: "responsive-layout@1",
        state: "succeeded",
        runtimeLabels: [],
        labels: {},
        runtimeConfiguration: {
          default: {
            label: "default",
            bindingRevision: "1",
            config: {
              name: "contractor-empty",
              version: "1",
              digest: `sha256:${"1".repeat(64)}`,
            },
          },
          labels: [],
        },
        parameters: {},
        inputs: {},
        attempts: [
          {
            stageExecutionId: "stage-responsive-1",
            stage: "validate",
            objective: "Validate the responsive layout.",
            attempt: 1,
            executionConfig: { variant: "base", agents: {} },
            state: "succeeded",
            result: {
              apiVersion: "contractor/v1alpha1",
              outcome: "succeeded",
              summary: `Validated the exact candidate at ${LONG_REVISION} without changing its immutable identity.`,
              artifacts: {
                validation_report: {
                  namespace: "outputs",
                  name: "validation_report",
                  revision: LONG_REVISION,
                },
              },
            },
            createdAt: "2026-09-04T08:00:00Z",
            updatedAt: "2026-09-04T08:01:00Z",
            terminalAt: "2026-09-04T08:01:00Z",
          },
        ],
        transitions: [],
        outputs: {
          validation_report: {
            namespace: "outputs",
            name: "validation_report",
            revision: LONG_REVISION,
          },
        },
        createdAt: "2026-09-04T08:00:00Z",
        updatedAt: "2026-09-04T08:01:00Z",
        startedAt: "2026-09-04T08:00:00Z",
        finishedAt: "2026-09-04T08:01:00Z",
      });
      return;
    }
    if (url.pathname === `/v1/runs/${RUN_ID}/artifacts`) {
      await fulfillJson(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.pathname === "/v1/runs") {
      await fulfillJson(route, {
        items: [
          {
            runId: RUN_ID,
            workflow: "responsive-layout-with-a-long-name@1",
            state: "succeeded",
            labels: {},
            createdAt: "2026-09-04T08:00:00Z",
            updatedAt: "2026-09-04T08:01:00Z",
            finishedAt: "2026-09-04T08:01:00Z",
          },
        ],
        page: { hasMore: false },
      });
      return;
    }
    await route.fulfill({ status: 404, body: "not found" });
  });
}

test("Run detail stays within a 320px viewport", async ({ page }, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const uiOrigin = new URL(configuredBaseURL).origin;
  await page.setViewportSize({ width: 320, height: 568 });
  await installRunAPI(page, uiOrigin);

  await page.goto(`/runs/${RUN_ID}`);
  await expect(page.getByRole("heading", { name: RUN_ID })).toBeVisible();
  await page.locator("details.run-output-preview > summary").click();
  await expect(page.locator(".run-output-detail-link")).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(() => ({
        viewport: window.innerWidth,
        document: document.documentElement.scrollWidth,
      })),
    )
    .toEqual({ viewport: 320, document: 320 });
});

test("Run list prioritizes compact facts on narrow screens", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const uiOrigin = new URL(configuredBaseURL).origin;
  await page.setViewportSize({ width: 320, height: 568 });
  await installRunAPI(page, uiOrigin);

  await page.goto("/runs");
  const runLink = page.getByRole("link", { name: RUN_ID });
  await expect(runLink).toBeVisible();
  await expect(runLink).toHaveText("run_01234567…89abcdef");
  await expect(runLink).toHaveAttribute("title", RUN_ID);

  const row = runLink.locator("xpath=ancestor::tr");
  await expect(row).toHaveCSS("display", "grid");
  await expect(row.locator(".run-list-workflow-cell")).toBeVisible();
  await expect(row.locator(".run-list-updated-cell")).toBeVisible();
  await expect(row.locator(".run-list-created-cell")).toBeHidden();
  await expect(row.locator(".run-list-finished-cell")).toBeHidden();
  await expect(row.locator(".run-list-labels-cell")).toBeHidden();
  await expect(row.locator(".state-badge")).toHaveCSS("white-space", "nowrap");
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
    .toBe(320);
});
