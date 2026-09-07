import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const PROJECT_ID = "project_run_draft";
const WORKFLOW_NAME = "draft-workflow";
const WORKFLOW_VERSION = "1";
const WORKFLOW_SELECTOR = `${WORKFLOW_NAME}@${WORKFLOW_VERSION}`;
const timestamp = "2026-09-06T12:00:00Z";

const workflow = {
  ref: { name: WORKFLOW_NAME, version: WORKFLOW_VERSION },
  entryStage: "inspect",
  parameters: {
    objective: { required: true },
  },
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
  },
  outputs: {},
  stages: {},
};

const existingArtifact = {
  artifact: {
    namespace: "sources",
    name: "existing",
    revision: "revision-existing",
  },
  mediaType: "application/zip",
  size: 3,
  current: true,
  frozen: false,
  createdAt: timestamp,
};

const uploadedArtifact = {
  artifact: {
    namespace: "artifacts",
    name: "replacement",
    revision: "revision-uploaded",
  },
  mediaType: "application/zip",
  size: 3,
  current: true,
  frozen: false,
  createdAt: timestamp,
};

function responseHeaders(origin?: string): Record<string, string> {
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

async function fulfill(
  route: Route,
  value: unknown,
  status = 200,
  extraHeaders: Record<string, string> = {},
): Promise<void> {
  await route.fulfill({
    status,
    body: JSON.stringify(value),
    headers: {
      ...responseHeaders(route.request().headers().origin),
      ...extraHeaders,
    },
  });
}

async function installFixture(page: Page, apiOrigin: string) {
  const uploads: Array<{
    path: string;
    headers: Record<string, string>;
    body: Buffer;
  }> = [];
  let uploaded = false;
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
          "access-control-allow-methods": "GET, POST, PUT, PATCH, OPTIONS",
          "access-control-allow-headers":
            "content-type, idempotency-key, if-match, if-none-match, x-csrf-token",
        },
      });
      return;
    }
    if (path === "/v1/auth/session") {
      await fulfill(route, {
        principal: {
          userId: "owner_run_draft",
          username: "draft-owner",
          capabilities: ["user", "operations"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (path === "/v1/workflows") {
      await fulfill(route, { items: [workflow], page: { hasMore: false } });
      return;
    }
    if (
      path === `/v1/workflows/${WORKFLOW_NAME}/versions/${WORKFLOW_VERSION}`
    ) {
      await fulfill(route, workflow);
      return;
    }
    if (path === "/v1/artifacts" && request.method() === "GET") {
      await fulfill(route, {
        items: [existingArtifact],
        page: { hasMore: false },
      });
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}`) {
      await fulfill(
        route,
        {
          projectId: PROJECT_ID,
          kind: "project",
          name: "Run draft workspace",
          description: "In-memory Run draft browser fixture",
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
    if (
      path === `/v1/projects/${PROJECT_ID}/artifacts` &&
      request.method() === "GET"
    ) {
      await fulfill(route, {
        items: uploaded
          ? [existingArtifact, uploadedArtifact]
          : [existingArtifact],
        page: { hasMore: false },
      });
      return;
    }
    if (
      path === `/v1/projects/${PROJECT_ID}/artifacts/artifacts/replacement` &&
      request.method() === "PUT"
    ) {
      uploads.push({
        path,
        headers: request.headers(),
        body: request.postDataBuffer() ?? Buffer.alloc(0),
      });
      uploaded = true;
      await fulfill(
        route,
        {
          artifact: uploadedArtifact.artifact,
          mediaType: uploadedArtifact.mediaType,
          size: uploadedArtifact.size,
        },
        201,
        { etag: '"revision-uploaded"' },
      );
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}/runs`) {
      await fulfill(route, { items: [], page: { hasMore: false } });
      return;
    }
    await fulfill(
      route,
      {
        code: "not_found",
        message: `No Run draft fixture for ${request.method()} ${path}`,
        retryable: false,
        requestId: "request-run-draft",
      },
      404,
    );
  });
  return uploads;
}

test("standalone Run draft survives SPA navigation without browser storage", async ({
  page,
}) => {
  const apiOrigin =
    process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
  await installFixture(page, apiOrigin);
  await page.goto(`/catalog/workflows/${WORKFLOW_NAME}/${WORKFLOW_VERSION}`);
  await page.locator('[name="parameter-objective"]').fill("Resume this Run");
  await page
    .locator('[name="artifact-source"]')
    .selectOption("sources/existing@revision-existing");

  await page.getByRole("link", { name: "Artifacts", exact: true }).click();
  await expect(page).toHaveURL(/\/artifacts$/);
  await page.goBack();

  await expect(page.locator('[name="parameter-objective"]')).toHaveValue(
    "Resume this Run",
  );
  await expect(page.locator('[name="artifact-source"]')).toHaveValue(
    "sources/existing@revision-existing",
  );
  expect(
    await page.evaluate(() => ({
      local: localStorage.length,
      session: sessionStorage.length,
    })),
  ).toEqual({ local: 0, session: 0 });
});

test("Project Run draft survives close and binds a local upload to its source slot", async ({
  page,
}) => {
  const apiOrigin =
    process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
  const uploads = await installFixture(page, apiOrigin);
  await page.goto(`/projects/${PROJECT_ID}`);
  const launcher = page.getByRole("button", {
    name: `Run ${WORKFLOW_SELECTOR}`,
  });
  await launcher.click();
  let runDialog = page.getByRole("dialog", { name: WORKFLOW_NAME });
  await expect(runDialog.locator(".project-dialog-heading code")).toHaveText(
    WORKFLOW_SELECTOR,
  );
  await runDialog
    .locator('[name="parameter-objective"]')
    .fill("Keep Project setup");
  await expect(runDialog.locator('[name="artifact-source"]')).toHaveValue(
    "sources/existing@revision-existing",
  );
  await expect(
    runDialog.getByRole("button", {
      name: "Confirm exact input for source",
    }),
  ).toBeVisible();
  await runDialog
    .getByRole("button", { name: "Close Workflow Run dialog" })
    .click();

  await launcher.click();
  runDialog = page.getByRole("dialog", { name: WORKFLOW_NAME });
  await expect(runDialog.locator('[name="parameter-objective"]')).toHaveValue(
    "Keep Project setup",
  );
  await runDialog
    .getByRole("button", { name: "Upload local file for source" })
    .click();
  const uploadDialog = page.getByRole("dialog", {
    name: "Upload local file for source",
  });
  await uploadDialog.getByLabel("Drop a file here").setInputFiles({
    name: "replacement.zip",
    mimeType: "application/zip",
    buffer: Buffer.from("zip"),
  });
  await expect(
    uploadDialog.getByLabel("Namespace", { exact: true }),
  ).toHaveValue("artifacts");
  await expect(
    uploadDialog.getByLabel("Namespace", { exact: true }),
  ).toBeDisabled();
  await uploadDialog
    .getByRole("button", { name: "Upload and select exact revision" })
    .click();

  await expect(uploadDialog).toBeHidden();
  await expect(runDialog.locator('[name="artifact-source"]')).toHaveValue(
    "artifacts/replacement@revision-uploaded",
  );
  await expect(
    runDialog.getByRole("region", { name: "Exact input review for source" }),
  ).toContainText("Confirmed");
  await expect(runDialog.locator('[name="parameter-objective"]')).toHaveValue(
    "Keep Project setup",
  );
  expect(uploads).toHaveLength(1);
  expect(uploads[0]!.headers["if-none-match"]).toBe("*");
  expect(uploads[0]!.headers["x-csrf-token"]).toBe("a".repeat(43));
  expect(uploads[0]!.body.toString()).toBe("zip");

  await runDialog
    .getByRole("button", { name: "Close Workflow Run dialog" })
    .click();
  await launcher.click();
  runDialog = page.getByRole("dialog", { name: WORKFLOW_NAME });
  await expect(runDialog.locator('[name="artifact-source"]')).toHaveValue(
    "artifacts/replacement@revision-uploaded",
  );
  await expect(
    runDialog.getByRole("region", { name: "Exact input review for source" }),
  ).toContainText("Confirmed");
});
