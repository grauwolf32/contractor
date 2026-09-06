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
  let queuePaused = false;
  let queueControlRevision = 0;
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
    if (url.pathname === "/v1/queue/control") {
      if (route.request().method() === "PUT") {
        const body = route.request().postDataJSON() as { paused: boolean };
        queuePaused = body.paused;
        queueControlRevision += 1;
      }
      await route.fulfill({
        body: JSON.stringify({
          paused: queuePaused,
          revision: String(queueControlRevision),
          ...(queueControlRevision === 0
            ? {}
            : { updatedAt: "2026-09-05T08:02:00Z" }),
        }),
        headers: {
          ...apiHeaders(),
          etag: `"${queueControlRevision}"`,
        },
        status: 200,
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

async function installRunResultAPI(
  page: Page,
  uiOrigin: string,
  resultRequests: URL[],
): Promise<void> {
  const source = "browser primary result";
  const digest = `sha256:${"1".repeat(64)}`;
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
    if (url.pathname === "/v1/runs/run-result-browser") {
      await fulfillJson(route, {
        runId: "run-result-browser",
        workflow: "browser-result@1",
        state: "succeeded",
        deletable: true,
        runtimeLabels: [],
        labels: {},
        runtimeConfiguration: {
          default: {
            label: "default",
            bindingRevision: "1",
            config: { name: "empty", version: "1", digest },
          },
          labels: [],
        },
        attempts: [],
        transitions: [],
        outputs: {
          report: {
            namespace: "outputs",
            name: "report",
            revision: "report-browser-r1",
          },
        },
        outputPublications: [],
        createdAt: "2026-09-07T08:00:00Z",
        updatedAt: "2026-09-07T08:01:00Z",
        finishedAt: "2026-09-07T08:01:00Z",
      });
      return;
    }
    if (url.pathname === "/v1/workflows/browser-result/versions/1") {
      resultRequests.push(url);
      await fulfillJson(route, {
        ref: { name: "browser-result", version: "1" },
        entryStage: "render",
        parameters: {},
        inputs: {},
        outputs: {
          report: {
            required: true,
            mediaTypes: ["text/plain"],
            primary: true,
          },
          trace: {
            required: false,
            mediaTypes: ["application/json"],
          },
        },
        stages: {},
      });
      return;
    }
    if (url.pathname === "/v1/runs/run-result-browser/artifacts") {
      await fulfillJson(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (
      url.pathname ===
      "/v1/runs/run-result-browser/artifacts/outputs/report/metadata"
    ) {
      resultRequests.push(url);
      await fulfillJson(route, {
        artifact: {
          namespace: "outputs",
          name: "report",
          revision: "report-browser-r1",
        },
        mediaType: "text/plain",
        size: source.length,
        current: true,
        frozen: true,
        createdAt: "2026-09-07T08:01:00Z",
      });
      return;
    }
    if (
      url.pathname === "/v1/runs/run-result-browser/artifacts/outputs/report"
    ) {
      resultRequests.push(url);
      await route.fulfill({
        body: source,
        headers: {
          ...apiHeaders(),
          "content-length": String(source.length),
          "content-type": "text/plain",
        },
        status: 200,
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
  await page.getByRole("button", { name: "Pause queue" }).click();
  await expect(
    page.getByRole("button", { name: "Resume queue" }),
  ).toBeVisible();
  await expect(page.getByText("Admission paused")).toBeVisible();
  await page.getByRole("button", { name: "Resume queue" }).click();
  await expect(page.getByRole("button", { name: "Pause queue" })).toBeVisible();

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

test("Run results expose the declared primary output with one bounded action", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const uiOrigin = new URL(configuredBaseURL).origin;
  const resultRequests: URL[] = [];
  await installRunResultAPI(page, uiOrigin, resultRequests);

  await page.goto("/runs/run-result-browser");
  await expect(page.getByText("Declared primary result")).toBeVisible();
  await expect(page.getByRole("heading", { name: "trace" })).toBeVisible();
  expect(resultRequests.map((url) => url.pathname)).toEqual([
    "/v1/workflows/browser-result/versions/1",
  ]);

  await page.getByRole("button", { name: "Preview result" }).click();
  await expect(page.locator("pre.artifact-preview")).toContainText(
    "browser primary result",
  );
  expect(resultRequests.map((url) => url.pathname)).toEqual([
    "/v1/workflows/browser-result/versions/1",
    "/v1/runs/run-result-browser/artifacts/outputs/report/metadata",
    "/v1/runs/run-result-browser/artifacts/outputs/report",
  ]);
  await expect(
    page.getByRole("button", { name: "Load preview", exact: true }),
  ).toHaveCount(0);
});
