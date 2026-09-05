import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const PROJECT_ID = "project_browser_example";
const RUNTIME_CONFIGURATION = {
  default: {
    label: "default",
    bindingRevision: "1",
    config: {
      name: "contractor-empty",
      version: "1",
      digest: `sha256:${"0".repeat(64)}`,
    },
  },
  labels: [],
};

interface ProjectAPIFixtureOptions {
  artifactStoredInitially?: boolean;
  exposeWorkflow?: boolean;
  runRequests?: Array<{
    url: string;
    headers: Record<string, string>;
    body: unknown;
  }>;
}

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
  const requestOrigin = route.request().headers()["origin"];
  await route.fulfill({
    body: JSON.stringify(value),
    headers: headers({
      ...(requestOrigin === undefined
        ? {}
        : {
            "access-control-allow-origin": requestOrigin,
            "access-control-allow-credentials": "true",
            "access-control-expose-headers":
              "X-Contractor-API-Version, ETag, X-Request-ID, Content-Length, Content-Type",
          }),
      ...extraHeaders,
    }),
    status,
  });
}

async function installProjectAPI(
  page: Page,
  uiOrigin: string,
  apiOrigin: string,
  uploads: Array<{
    url: string;
    headers: Record<string, string>;
    body: Buffer;
  }>,
  options: ProjectAPIFixtureOptions = {},
): Promise<void> {
  let artifactStored = options.artifactStoredInitially ?? false;
  const project = {
    projectId: PROJECT_ID,
    kind: "project",
    name: "Browser workspace",
    description: "Project UI browser fixture",
    lifecycle: "active",
    revision: "1",
    createdAt: "2026-09-01T10:00:00Z",
    updatedAt: "2026-09-01T10:00:00Z",
  };
  const artifact = {
    artifact: {
      namespace: "sources",
      name: "browser-source",
      revision: "revision-browser-1",
    },
    mediaType: "application/zip",
    size: 3,
    current: true,
    frozen: false,
    createdAt: "2026-09-01T10:01:00Z",
  };
  const workflow = {
    ref: { name: "openapi-from-source", version: "1" },
    entryStage: "analyze",
    parameters: {},
    inputs: {
      source: { required: true, mediaTypes: ["application/zip"] },
    },
    outputs: {
      openapi: {
        required: true,
        mediaTypes: ["application/yaml"],
        primary: true,
      },
    },
  };
  await page.route("**/runtime-config.json", async (route) => {
    await route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: apiOrigin,
      },
    });
  });
  await page.route(`${apiOrigin}/v1/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "access-control-allow-origin": uiOrigin,
          "access-control-allow-credentials": "true",
          "access-control-allow-methods": "GET, POST, PUT, PATCH, OPTIONS",
          "access-control-allow-headers":
            "content-type, idempotency-key, if-match, if-none-match, x-csrf-token",
        },
      });
      return;
    }
    if (url.pathname === "/v1/auth/session") {
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
    if (url.pathname === "/v1/projects" && request.method() === "GET") {
      await fulfillJSON(route, { items: [project], page: { hasMore: false } });
      return;
    }
    if (url.pathname === "/v1/workflows") {
      await fulfillJSON(route, {
        items: options.exposeWorkflow ? [workflow] : [],
        page: { hasMore: false },
      });
      return;
    }
    if (
      url.pathname === "/v1/workflows/openapi-from-source/versions/1" &&
      options.exposeWorkflow
    ) {
      await fulfillJSON(route, { ...workflow, stages: {} });
      return;
    }
    if (url.pathname === `/v1/projects/${PROJECT_ID}`) {
      await fulfillJSON(route, project, 200, { etag: '"1"' });
      return;
    }
    if (
      url.pathname === `/v1/projects/${PROJECT_ID}/artifacts` &&
      request.method() === "GET"
    ) {
      await fulfillJSON(route, {
        items: artifactStored ? [artifact] : [],
        page: { hasMore: false },
      });
      return;
    }
    if (
      url.pathname ===
        `/v1/projects/${PROJECT_ID}/artifacts/sources/browser-source` &&
      request.method() === "PUT"
    ) {
      uploads.push({
        url: request.url(),
        headers: request.headers(),
        body: request.postDataBuffer() ?? Buffer.alloc(0),
      });
      artifactStored = true;
      await fulfillJSON(
        route,
        {
          artifact: artifact.artifact,
          mediaType: artifact.mediaType,
          size: artifact.size,
        },
        201,
        { etag: '"revision-browser-1"' },
      );
      return;
    }
    if (
      url.pathname === `/v1/projects/${PROJECT_ID}/runs` &&
      request.method() === "POST"
    ) {
      options.runRequests?.push({
        url: request.url(),
        headers: request.headers(),
        body: request.postDataJSON(),
      });
      await fulfillJSON(
        route,
        {
          runId: "run_project_browser",
          projectId: PROJECT_ID,
          state: "initializing",
          runtimeLabels: [],
          labels: {},
          runtimeConfiguration: RUNTIME_CONFIGURATION,
        },
        202,
      );
      return;
    }
    if (
      url.pathname === `/v1/projects/${PROJECT_ID}/runs` &&
      request.method() === "GET"
    ) {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.pathname === "/v1/runs/run_project_browser") {
      await fulfillJSON(route, {
        runId: "run_project_browser",
        projectId: PROJECT_ID,
        workflow: "openapi-from-source@1",
        state: "initializing",
        runtimeLabels: [],
        labels: {},
        runtimeConfiguration: RUNTIME_CONFIGURATION,
        attempts: [],
        transitions: [],
        outputs: {},
        outputPublications: [],
      });
      return;
    }
    if (url.pathname === "/v1/runs/run_project_browser/artifacts") {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
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

  // Keep this assertion close to the routing setup: Artifact bytes target the
  // configured Go API origin, not the static UI origin.
  expect(apiOrigin).not.toBe(uiOrigin);
}

async function openProjectFromShell(page: Page): Promise<void> {
  await page.goto(`/projects/${PROJECT_ID}`);
}

test("Project file-drop uploads exact bytes directly to Go API", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const uiOrigin = new URL(configuredBaseURL).origin;
  const apiOrigin =
    process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
  const uploads: Array<{
    url: string;
    headers: Record<string, string>;
    body: Buffer;
  }> = [];
  await installProjectAPI(page, uiOrigin, apiOrigin, uploads);

  await openProjectFromShell(page);
  await expect(
    page.getByRole("heading", { name: "Browser workspace" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Sources" }).click();
  const dialog = page.getByRole("dialog", { name: "Sources" });
  await dialog.getByLabel("Drop a file here").setInputFiles({
    name: "browser-source.zip",
    mimeType: "application/zip",
    buffer: Buffer.from("zip"),
  });
  await expect(dialog.getByLabel("Namespace", { exact: true })).toHaveValue(
    "sources",
  );
  await expect(dialog.getByLabel("Name", { exact: true })).toHaveValue(
    "browser-source",
  );
  await dialog.getByRole("button", { name: "Create binding" }).click();

  await expect(
    page.getByRole("link", { name: "sources/browser-source", exact: true }),
  ).toBeVisible();
  expect(uploads).toHaveLength(1);
  expect(new URL(uploads[0]!.url).origin).toBe(apiOrigin);
  expect(uploads[0]!.headers["if-none-match"]).toBe("*");
  expect(uploads[0]!.headers["x-csrf-token"]).toBe("a".repeat(43));
  expect(uploads[0]!.body.toString()).toBe("zip");
});

test("Project dashboard remains usable at 320px", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const uiOrigin = new URL(configuredBaseURL).origin;
  await page.setViewportSize({ width: 320, height: 568 });
  await installProjectAPI(
    page,
    uiOrigin,
    process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080",
    [],
  );

  await openProjectFromShell(page);
  await expect(page.getByRole("button", { name: "Other" })).toBeVisible();
  await page.getByRole("button", { name: "Other" }).click();
  await expect(page.getByRole("dialog", { name: "Other" })).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
    .toBe(320);
});

test("Project recommendation launches an exact Project Run", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const uiOrigin = new URL(configuredBaseURL).origin;
  const apiOrigin =
    process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
  const runRequests: NonNullable<ProjectAPIFixtureOptions["runRequests"]> = [];
  await installProjectAPI(page, uiOrigin, apiOrigin, [], {
    artifactStoredInitially: true,
    exposeWorkflow: true,
    runRequests,
  });

  await openProjectFromShell(page);
  await page.getByRole("button", { name: "Run openapi-from-source@1" }).click();
  const dialog = page.getByRole("dialog", {
    name: "openapi-from-source@1",
  });
  await expect(
    dialog.getByRole("combobox", { name: /source required/ }),
  ).toHaveValue("sources/browser-source@revision-browser-1");
  await dialog
    .getByRole("button", { name: "Start Project Workflow Run" })
    .click();

  await expect(page).toHaveURL(/\/runs\/run_project_browser$/);
  await expect(
    page.getByRole("heading", { name: "run_project_browser" }),
  ).toBeVisible();
  expect(runRequests).toHaveLength(1);
  expect(new URL(runRequests[0]!.url).origin).toBe(apiOrigin);
  expect(runRequests[0]!.headers["idempotency-key"]).toMatch(/^run-ui-/);
  expect(runRequests[0]!.body).toMatchObject({
    workflow: "openapi-from-source@1",
    artifacts: {
      source: {
        namespace: "sources",
        name: "browser-source",
        revision: "revision-browser-1",
      },
    },
  });
});
