import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";

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

async function installRunsAPI(
  page: Page,
  uiOrigin: string,
  runListRequests: URL[],
  queueRequests: URL[],
): Promise<void> {
  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: uiOrigin,
      },
    }),
  );
  await page.route(`${uiOrigin}/v1/**`, async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/auth/session") {
      await fulfillJson(route, {
        principal: {
          userId: "user_runs",
          username: "runs",
          capabilities: ["user", "operations"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (url.pathname === "/v1/queue") {
      queueRequests.push(url);
      await fulfillJson(route, {
        items: [
          {
            runId: "run-active-browser",
            workflow: "browser-workflow@1",
            state: "running",
            labels: {},
            eventCursor: { generation: "events-browser", sequence: "1" },
            createdAt: "2026-09-05T08:00:00Z",
            updatedAt: "2026-09-05T08:01:00Z",
          },
        ],
        page: { hasMore: false },
      });
      return;
    }
    if (url.pathname === "/v1/runs") {
      runListRequests.push(url);
      await fulfillJson(route, {
        items: [
          {
            runId: "run-completed-browser",
            workflow: "browser-workflow@1",
            state: "succeeded",
            labels: {},
            createdAt: "2026-09-05T07:00:00Z",
            updatedAt: "2026-09-05T07:02:00Z",
            finishedAt: "2026-09-05T07:02:00Z",
          },
        ],
        page: { hasMore: false },
      });
      return;
    }
    await route.fulfill({ status: 404, body: "not found" });
  });
}

test("Runs defaults to Queue and keeps terminal history in Completed", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const uiOrigin = new URL(configuredBaseURL).origin;
  const runListRequests: URL[] = [];
  const queueRequests: URL[] = [];
  await installRunsAPI(page, uiOrigin, runListRequests, queueRequests);

  await page.goto("/runs");
  await expect(
    page.getByRole("link", { name: "run-active-browser" }),
  ).toBeVisible();
  const primary = page.getByRole("navigation", {
    name: "Primary navigation",
  });
  await expect(primary.getByRole("link", { name: "Queue" })).toHaveCount(0);
  await expect(primary.getByRole("link", { name: "Runs" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  const views = page.getByRole("navigation", { name: "Run views" });
  await expect(views.getByRole("link", { name: /Queue/ })).toHaveAttribute(
    "aria-current",
    "page",
  );

  await views.getByRole("link", { name: /Completed/ }).click();
  await expect(page).toHaveURL(/\/runs\?view=completed$/);
  await expect(
    page.getByRole("link", { name: "run-completed-browser" }),
  ).toBeVisible();
  await expect(views.getByRole("link", { name: /Completed/ })).toHaveAttribute(
    "aria-current",
    "page",
  );
  expect(runListRequests).toHaveLength(1);
  expect(runListRequests[0]?.searchParams.get("lifecycle")).toBe("terminal");

  await page.goto("/queue?membership=project");
  await expect(page).toHaveURL(/\/runs\?membership=project$/);
  await expect(
    page.getByRole("link", { name: "run-active-browser" }),
  ).toBeVisible();
  expect(queueRequests.at(-1)?.searchParams.get("membership")).toBe("project");

  await page.setViewportSize({ width: 320, height: 568 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
    .toBe(320);
});
