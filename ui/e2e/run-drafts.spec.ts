import { expect, test, type Page } from "@playwright/test";

import { project } from "../src/test/builders";
import {
  runRuntimeConfiguration,
  stageContract,
  workflowResource,
  workflowSummary,
} from "../src/test/builders-execution";
import {
  authSession,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
  mockedAPIOrigin,
} from "./support/mock-api";

const PROJECT_ID = "project_run_draft";
const WORKFLOW_NAME = "draft-workflow";
const WORKFLOW_VERSION = "1";
const WORKFLOW_SELECTOR = `${WORKFLOW_NAME}@${WORKFLOW_VERSION}`;
const timestamp = "2026-09-06T12:00:00Z";

const workflow = workflowResource(WORKFLOW_NAME, WORKFLOW_VERSION, {
  entryStage: "inspect",
  parameters: {
    objective: { required: true },
  },
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
  },
  stages: { inspect: stageContract() },
});

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

async function installFixture(page: Page, apiOrigin: string) {
  const uploads: Array<{
    path: string;
    headers: Record<string, string>;
    body: Buffer;
  }> = [];
  let uploaded = false;
  await mockAPI(
    page,
    async (route, request, url) => {
      const path = url.pathname;
      if (path === "/v1/workflows") {
        await fulfillJSON(route, {
          items: [workflowSummary(workflow)],
          page: { hasMore: false },
        });
        return;
      }
      if (
        path === `/v1/workflows/${WORKFLOW_NAME}/versions/${WORKFLOW_VERSION}`
      ) {
        await fulfillJSON(route, workflow);
        return;
      }
      if (path === "/v1/artifacts" && request.method() === "GET") {
        await fulfillJSON(route, {
          items: [existingArtifact],
          page: { hasMore: false },
        });
        return;
      }
      if (path === `/v1/projects/${PROJECT_ID}`) {
        await fulfillJSON(
          route,
          project(PROJECT_ID, {
            name: "Run draft workspace",
            description: "In-memory Run draft browser fixture",
          }),
          200,
          { etag: '"1"' },
        );
        return;
      }
      if (
        path === `/v1/projects/${PROJECT_ID}/artifacts` &&
        request.method() === "GET"
      ) {
        await fulfillJSON(route, {
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
        await fulfillJSON(
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
        await fulfillJSON(route, { items: [], page: { hasMore: false } });
        return;
      }
      await fulfillEmptyPageOrNotFound(
        route,
        `No Run draft fixture for ${request.method()} ${path}`,
      );
    },
    { apiOrigin, session: authSession("draft-owner") },
  );
  return uploads;
}

// Files (/artifacts) has no rail item in V3B: every Library page, the workflow
// page included, links to it from its section tabs, which wrap and stay in
// the page at every width (no Menu drawer needed on phones).
async function leaveForFiles(page: Page): Promise<void> {
  await page
    .getByRole("navigation", { name: "Library sections", exact: true })
    .getByRole("link", { name: "Files", exact: true })
    .click();
  await expect(page).toHaveURL(/\/artifacts$/);
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
      const apiOrigin = mockedAPIOrigin();
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
      await leaveForFiles(page);
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
      const apiOrigin = mockedAPIOrigin();
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
        await fulfillJSON(
          route,
          {
            runId: "run-recovered-draft",
            state: "initializing",
            runtimeLabels: [],
            labels: {},
            runtimeConfiguration: runRuntimeConfiguration(),
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
        page.getByRole("button", { name: "Retry same request" }),
      ).toBeVisible();
      expect(submissions).toHaveLength(1);
      await page.getByRole("button", { name: "Close Run setup" }).click();
      await leaveForFiles(page);
      await page.goBack();
      await page
        .getByRole("button", { name: "Configure Run", exact: true })
        .click();
      await expect(page.locator('[name="parameter-objective"]')).toHaveValue(
        "Retain exact submission",
      );
      expect(submissions).toHaveLength(1);
      await page.getByRole("button", { name: "Retry same request" }).click();
      await expect(page).toHaveURL(/\/runs\/run-recovered-draft$/);
      expect(submissions).toHaveLength(2);
      expect(submissions[1]).toEqual(submissions[0]);
      expect(submissions[0]!.key).toMatch(/^run-ui-[0-9a-f]{32}$/);
    });

    test("Project Run draft survives close and binds a local upload to its source slot", async ({
      page,
    }) => {
      const apiOrigin = mockedAPIOrigin();
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
          name: "Confirm input for source",
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
        .getByRole("button", { name: "Upload and select" })
        .click();

      await expect(uploadDialog).toBeHidden();
      await expect(runDialog.locator('[name="artifact-source"]')).toHaveValue(
        "artifacts/replacement@revision-uploaded",
      );
      await expect(
        runDialog.getByRole("region", {
          name: "Input review for source",
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
          name: "Input review for source",
        }),
      ).toContainText("Confirmed");
    });
  });
}
