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

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test.describe(`${viewport.width}px journeys`, () => {
    test.use({ viewport });

    test("standalone Run draft survives SPA navigation without browser storage", async ({
      page,
    }) => {
      const apiOrigin =
        process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
      await installFixture(page, apiOrigin);
      await page.goto(
        `/catalog/workflows/${WORKFLOW_NAME}/${WORKFLOW_VERSION}`,
      );
      await page
        .getByRole("button", { name: "Configure Run", exact: true })
        .click();
      await page
        .locator('[name="parameter-objective"]')
        .fill("Resume this Run");
      await page
        .locator('[name="artifact-source"]')
        .selectOption("sources/existing@revision-existing");

      await page.getByRole("button", { name: "Close Run setup" }).click();
      if (viewport.width === 390)
        await page.getByRole("button", { name: "Menu", exact: true }).click();
      await page.getByRole("link", { name: "Artifacts", exact: true }).click();
      await expect(page).toHaveURL(/\/artifacts$/);
      await page.goBack();
      await page
        .getByRole("button", { name: "Configure Run", exact: true })
        .click();

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

    test("ambiguous launch retains exact request identity after leaving the draft", async ({
      page,
    }) => {
      const apiOrigin =
        process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
      await installFixture(page, apiOrigin);
      const submissions: Array<{ body: unknown; key: string | undefined }> = [];
      await page.route(`${apiOrigin}/v1/runs`, async (route) => {
        const request = route.request();
        if (request.method() !== "POST") return route.fallback();
        submissions.push({
          body: request.postDataJSON(),
          key: request.headers()["idempotency-key"],
        });
        if (submissions.length === 1) return route.abort("failed");
        await fulfill(
          route,
          {
            runId: "run-recovered-draft",
            state: "initializing",
            runtimeLabels: [],
            labels: {},
            runtimeConfiguration: {
              default: {
                label: "default",
                bindingRevision: "1",
                config: {
                  name: "empty",
                  version: "1",
                  digest: `sha256:${"1".repeat(64)}`,
                },
              },
              labels: [],
            },
          },
          202,
        );
      });
      await page.goto(
        `/catalog/workflows/${WORKFLOW_NAME}/${WORKFLOW_VERSION}`,
      );
      await page
        .getByRole("button", { name: "Configure Run", exact: true })
        .click();
      await page
        .locator('[name="parameter-objective"]')
        .fill("Retain exact submission");
      await page
        .locator('[name="artifact-source"]')
        .selectOption("sources/existing@revision-existing");
      await page.getByRole("button", { name: "Start Workflow Run" }).click();
      await expect(
        page.getByRole("button", { name: "Retry exact request" }),
      ).toBeVisible();
      expect(submissions).toHaveLength(1);
      await page.getByRole("button", { name: "Close Run setup" }).click();
      if (viewport.width === 390)
        await page.getByRole("button", { name: "Menu", exact: true }).click();
      await page.getByRole("link", { name: "Artifacts", exact: true }).click();
      await page.goBack();
      await page
        .getByRole("button", { name: "Configure Run", exact: true })
        .click();
      await expect(page.locator('[name="parameter-objective"]')).toHaveValue(
        "Retain exact submission",
      );
      expect(submissions).toHaveLength(1);
      await page.getByRole("button", { name: "Retry exact request" }).click();
      await expect(page).toHaveURL(/\/runs\/run-recovered-draft$/);
      expect(submissions).toHaveLength(2);
      expect(submissions[1]).toEqual(submissions[0]);
      expect(submissions[0]!.key).toMatch(/^run-ui-[0-9a-f]{32}$/);
    });

    test("Project Run draft survives close and binds a local upload to its source slot", async ({
      page,
    }) => {
      const apiOrigin =
        process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
      const uploads = await installFixture(page, apiOrigin);
      await page.goto(`/projects/${PROJECT_ID}/workflows`);
      const launcher = page.getByRole("button", {
        name: `Configure ${WORKFLOW_SELECTOR}`,
      });
      await launcher.click();
      let runDialog = page.getByRole("dialog", { name: "Configure Run" });
      await expect(
        runDialog.locator(".workflow-drawer-heading code"),
      ).toHaveText(WORKFLOW_SELECTOR);
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
      await runDialog.getByRole("button", { name: "Close Run setup" }).click();

      if (viewport.width === 390) {
        await page
          .getByRole("combobox", { name: "Project section", exact: true })
          .selectOption("settings");
        await expect(page).toHaveURL(/\/settings$/);
        await page
          .getByRole("combobox", { name: "Project section", exact: true })
          .selectOption("workflows");
      } else {
        const navigation = page.getByRole("navigation", {
          name: "Project sections",
        });
        await navigation
          .getByRole("link", { name: "Settings", exact: true })
          .click();
        await expect(page).toHaveURL(/\/settings$/);
        await navigation
          .getByRole("link", { name: "Workflows", exact: true })
          .click();
      }
      await launcher.click();
      runDialog = page.getByRole("dialog", { name: "Configure Run" });
      await expect(
        runDialog.locator('[name="parameter-objective"]'),
      ).toHaveValue("Keep Project setup");
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
      await page.keyboard.press("Escape");
      await expect(uploadDialog).toBeHidden();
      expect(uploads).toHaveLength(0);
      await expect(
        runDialog.locator('[name="parameter-objective"]'),
      ).toHaveValue("Keep Project setup");
      await expect(runDialog.locator('[name="artifact-source"]')).toHaveValue(
        "sources/existing@revision-existing",
      );
      await runDialog
        .getByRole("button", { name: "Upload local file for source" })
        .click();
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
        runDialog.getByRole("region", {
          name: "Exact input review for source",
        }),
      ).toContainText("Confirmed");
      await expect(
        runDialog.locator('[name="parameter-objective"]'),
      ).toHaveValue("Keep Project setup");
      expect(uploads).toHaveLength(1);
      expect(uploads[0]!.headers["if-none-match"]).toBe("*");
      expect(uploads[0]!.headers["x-csrf-token"]).toBe("a".repeat(43));
      expect(uploads[0]!.body.toString()).toBe("zip");

      await runDialog.getByRole("button", { name: "Close Run setup" }).click();
      await launcher.click();
      runDialog = page.getByRole("dialog", { name: "Configure Run" });
      await expect(runDialog.locator('[name="artifact-source"]')).toHaveValue(
        "artifacts/replacement@revision-uploaded",
      );
      await expect(
        runDialog.getByRole("region", {
          name: "Exact input review for source",
        }),
      ).toContainText("Confirmed");
    });
  });
}
