import { ARTIFACT_STORAGE_TIMESTAMPS } from "../src/test/builders";
import { expect, test, type Page } from "@playwright/test";

import {
  audit as auditBuilder,
  finding as findingBuilder,
  project as projectBuilder,
} from "../src/test/builders";
import {
  runRuntimeConfiguration,
  runStatus,
  stageContract,
  workflowResource,
  workflowSummary,
} from "../src/test/builders-execution";
import {
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
  mockedAPIOrigin,
  uiOrigin,
} from "./support/mock-api";

const PROJECT_ID = "project_browser_example";

interface ProjectAPIFixtureOptions {
  artifactStoredInitially?: boolean;
  showResult?: boolean;
  outputReads?: string[];
  exposeWorkflow?: boolean;
  runRequests?: Array<{
    url: string;
    headers: Record<string, string>;
    body: unknown;
  }>;
}

async function installProjectAPI(
  page: Page,
  apiOrigin: string,
  uploads: Array<{
    url: string;
    headers: Record<string, string>;
    body: Buffer;
  }>,
  options: ProjectAPIFixtureOptions = {},
): Promise<void> {
  let artifactStored = options.artifactStoredInitially ?? false;
  const project = projectBuilder(PROJECT_ID, {
    name: "Browser workspace",
    description: "Project UI browser fixture",
    createdAt: "2026-09-01T10:00:00Z",
    updatedAt: "2026-09-01T10:00:00Z",
  });
  const artifact = {
    canUploadNewVersion: true,
    storageTimestamps: ARTIFACT_STORAGE_TIMESTAMPS,
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
  const reviewCheck = auditBuilder(
    PROJECT_ID,
    "check_review_browser",
    "active",
    {
      profile: {
        name: "source-checklist",
        version: "1",
        digest: `sha256:${"a".repeat(64)}`,
      },
    },
  );
  const reviewFinding = findingBuilder(
    reviewCheck.auditId,
    "finding_review_browser",
    {},
    { title: "Possible missing object authorization" },
  );
  const resultSource = "note: selected output";
  const resultRef = {
    namespace: "outputs",
    name: "openapi",
    revision: "result-r1",
  };
  const resultRun = runStatus("run_result_browser", {
    projectId: PROJECT_ID,
    workflow: "openapi-from-source@1",
    state: "succeeded",
    deletable: true,
    finishedAt: "2026-09-01T10:05:00Z",
    outputs: { openapi: resultRef },
  });
  const workflow = workflowResource("openapi-from-source", "1", {
    entryStage: "analyze",
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
    stages: { analyze: stageContract() },
  });
  await mockAPI(
    page,
    async (route, request, url) => {
      if (
        options.showResult &&
        url.pathname.startsWith("/v1/runs/run_result_browser/artifacts/")
      ) {
        options.outputReads?.push(request.url());
        expect(url.searchParams.get("revision")).toBe("result-r1");
        if (url.pathname.endsWith("/metadata"))
          await fulfillJSON(route, {
            ...artifact,
            artifact: resultRef,
            mediaType: "application/yaml",
            size: resultSource.length,
            canUploadNewVersion: false,
            frozen: true,
          });
        else await fulfillContent(route, resultSource, "application/yaml");
        return;
      }
      if (
        options.showResult &&
        url.pathname === "/v1/runs/run_result_browser"
      ) {
        await fulfillJSON(route, resultRun);
        return;
      }
      if (url.pathname === "/v1/projects" && request.method() === "GET") {
        await fulfillJSON(route, {
          items: [project],
          page: { hasMore: false },
        });
        return;
      }
      if (url.pathname === `/v1/projects/${PROJECT_ID}/audits`) {
        await fulfillJSON(route, {
          items: options.showResult ? [reviewCheck] : [],
          page: { hasMore: false },
        });
        return;
      }
      if (
        options.showResult &&
        url.pathname === `/v1/audits/${reviewCheck.auditId}/findings`
      ) {
        await fulfillJSON(route, {
          items: [reviewFinding],
          total: 1,
          auditRevision: 1,
          asOf: "2026-11-01T08:30:00Z",
          page: { hasMore: false },
        });
        return;
      }
      if (url.pathname === "/v1/workflows") {
        await fulfillJSON(route, {
          items: options.exposeWorkflow ? [workflowSummary(workflow)] : [],
          page: { hasMore: false },
        });
        return;
      }
      if (
        url.pathname === "/v1/workflows/openapi-from-source/versions/1" &&
        options.exposeWorkflow
      ) {
        await fulfillJSON(route, workflow);
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
      if (url.pathname === "/v1/artifacts" && request.method() === "GET") {
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
            runtimeConfiguration: runRuntimeConfiguration(),
          },
          202,
        );
        return;
      }
      if (
        url.pathname === `/v1/projects/${PROJECT_ID}/runs` &&
        request.method() === "GET"
      ) {
        const {
          runId,
          projectId,
          workflow,
          state,
          deletable,
          labels,
          createdAt,
          updatedAt,
          finishedAt,
        } = resultRun;
        await fulfillJSON(route, {
          items: options.showResult
            ? [
                {
                  runId,
                  projectId,
                  workflow,
                  state,
                  deletable,
                  labels,
                  createdAt,
                  updatedAt,
                  finishedAt,
                },
              ]
            : [],
          page: { hasMore: false },
        });
        return;
      }
      if (url.pathname === "/v1/runs/run_project_browser") {
        await fulfillJSON(
          route,
          runStatus("run_project_browser", {
            projectId: PROJECT_ID,
            workflow: "openapi-from-source@1",
            state: "initializing",
          }),
        );
        return;
      }
      if (url.pathname === "/v1/runs/run_project_browser/artifacts") {
        await fulfillJSON(route, { items: [], page: { hasMore: false } });
        return;
      }
      await fulfillEmptyPageOrNotFound(route);
    },
    { apiOrigin },
  );
}

async function openProjectFromShell(page: Page): Promise<void> {
  await page.goto(`/projects/${PROJECT_ID}`);
}

test("Project file-drop uploads exact bytes directly to Go API", async ({
  page,
}, testInfo) => {
  const apiOrigin = mockedAPIOrigin();
  const uploads: Array<{
    url: string;
    headers: Record<string, string>;
    body: Buffer;
  }> = [];
  // Artifact bytes target the configured Go API origin, not the static UI
  // origin.
  expect(apiOrigin).not.toBe(uiOrigin(testInfo));
  await installProjectAPI(page, apiOrigin, uploads);

  await openProjectFromShell(page);
  await expect(
    page.getByRole("heading", { name: "Browser workspace" }),
  ).toBeVisible();
  await page
    .getByRole("navigation", { name: "Project sections" })
    .getByRole("link", { name: "Materials", exact: true })
    .click();
  await page.getByRole("button", { name: "Add material", exact: true }).click();
  await page
    .getByRole("button", { name: "Source code ZIP", exact: true })
    .click();
  const dialog = page.getByRole("dialog", { name: "Source code ZIP" });
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

test("Project dashboard remains usable at 320px", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await installProjectAPI(page, mockedAPIOrigin(), []);

  await openProjectFromShell(page);
  await page
    .getByRole("navigation", { name: "Project sections" })
    .getByRole("link", { name: "Materials", exact: true })
    .click();
  await page.getByRole("button", { name: "Add material", exact: true }).click();
  await expect(page.getByRole("button", { name: "Other" })).toBeVisible();
  await page.getByRole("button", { name: "Other" }).click();
  await expect(page.getByRole("dialog", { name: "Other" })).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
    .toBe(320);
});

test("Project recommendation launches an exact Project Run", async ({
  page,
}) => {
  const apiOrigin = mockedAPIOrigin();
  const runRequests: NonNullable<ProjectAPIFixtureOptions["runRequests"]> = [];
  await installProjectAPI(page, apiOrigin, [], {
    artifactStoredInitially: true,
    exposeWorkflow: true,
    runRequests,
  });

  await openProjectFromShell(page);
  await page
    .getByRole("navigation", { name: "Project sections" })
    .getByRole("link", { name: "Workflows", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Configure openapi-from-source@1" })
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Configure Run",
  });
  await expect(dialog.locator(".workflow-drawer-heading code")).toHaveText(
    "openapi-from-source@1",
  );
  await expect(
    dialog.getByRole("combobox", { name: /source required/ }),
  ).toHaveValue("sources/browser-source@revision-browser-1");
  await expect(dialog.getByText("Review 1")).toBeVisible();
  await expect(
    dialog.getByText(/Suggested only because application\/zip/),
  ).toBeVisible();
  await dialog
    .getByRole("button", { name: "Confirm input for source" })
    .click();
  await expect(dialog.getByText("Ready", { exact: true })).toBeVisible();
  await dialog
    .getByRole("button", { name: "Start Project Workflow Run" })
    .click();

  await expect(page).toHaveURL(/\/runs\/run_project_browser$/);
  await expect(
    page.getByRole("heading", { name: "openapi-from-source" }),
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

test("Project sections open inputs and Run setup, with explicit confirmed deletion", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await installProjectAPI(page, mockedAPIOrigin(), [], {
    artifactStoredInitially: true,
    exposeWorkflow: true,
  });

  await openProjectFromShell(page);
  const navigation = page.getByRole("navigation", { name: "Project sections" });
  await expect(navigation).toBeInViewport();
  await expect(
    navigation.getByRole("link", { name: "Overview", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  await expect(
    page.getByRole("button", { name: "Delete Project" }),
  ).toBeVisible();
  // The overview's Materials chips end with the link that opens the sheet.
  await page
    .getByRole("region", { name: "Project materials" })
    .getByRole("link", { name: "Add material", exact: true })
    .click();
  const chooser = page.getByRole("dialog", { name: "Add material" });
  await expect(chooser).toBeVisible();
  await chooser
    .getByRole("button", { name: "Source code ZIP", exact: true })
    .click();
  const upload = page.getByRole("dialog", { name: "Source code ZIP" });
  await expect(
    upload.getByRole("button", { name: "Close upload dialog" }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(upload).toBeHidden();
  await expect(
    chooser.getByRole("button", { name: "Source code ZIP", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(chooser).toBeHidden();
  await expect(page).not.toHaveURL(/add=artifact/);
  await navigation
    .getByRole("link", { name: "Workflows", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Configure openapi-from-source@1" })
    .click();
  const setup = page.getByRole("dialog", { name: "Configure Run" });
  await expect(setup).toBeVisible();
  await expect(
    setup.getByText("Inputs come from this Project.", {
      exact: false,
    }),
  ).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(setup).toBeHidden();
  await expect(
    page.getByRole("button", { name: "Configure openapi-from-source@1" }),
  ).toBeFocused();
  const deleteProject = page.getByRole("button", { name: "Delete Project" });
  await expect(deleteProject).toBeVisible();
  await deleteProject.click();
  const deletion = page.getByRole("alertdialog", {
    name: "Delete Browser workspace?",
  });
  await expect(deletion.getByRole("button", { name: "Cancel" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(deletion).toBeHidden();
  await expect(deleteProject).toBeFocused();
});

test("Workflow overview leads with Run setup in a dismissible drawer", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await installProjectAPI(page, mockedAPIOrigin(), [], {
    artifactStoredInitially: true,
    exposeWorkflow: true,
  });

  await page.goto("/catalog/workflows/openapi-from-source/1");
  const configure = page.getByRole("button", { name: "Configure Run" });
  const setup = page.locator("#workflow-run-setup");
  const technical = page.locator("details.workflow-technical-details");
  await expect(configure).toBeInViewport();
  await expect(
    page.getByRole("region", { name: "Inputs and results" }),
  ).toBeInViewport();
  await expect(setup).toHaveCount(0);
  await expect(technical).not.toHaveAttribute("open", "");
  await configure.click();
  const drawer = page.getByRole("dialog", { name: "Configure Run" });
  await expect(drawer).toBeVisible();
  await expect(
    drawer.getByRole("button", { name: "Close Run setup" }),
  ).toBeFocused();
  await drawer
    .getByLabel(/^source/)
    .selectOption("sources/browser-source@revision-browser-1");
  await page.keyboard.press("Escape");
  await expect(drawer).toHaveCount(0);
  await expect(configure).toBeFocused();
  await configure.click();
  await expect(drawer.getByLabel(/^source/)).toHaveValue(
    "sources/browser-source@revision-browser-1",
  );
  await page.keyboard.press("Escape");
  await technical
    .getByText("Agents, execution settings and transitions")
    .click();
  await expect(technical).toHaveAttribute("open", "");

  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
    .toBe(390);
  await page.screenshot({
    path: testInfo.outputPath("workflow-technical-mobile.png"),
    fullPage: true,
  });
});

for (const theme of ["light", "dark", "black"] as const) {
  for (const width of [1440, 390, 320]) {
    test(`Project Overview hierarchy and exact actions at ${width}px in ${theme}`, async ({
      page,
    }, testInfo) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (theme) => localStorage.setItem("contractor.theme", theme),
        theme,
      );
      const outputReads: string[] = [];
      await installProjectAPI(page, mockedAPIOrigin(), [], {
        artifactStoredInitially: true,
        exposeWorkflow: true,
        showResult: true,
        outputReads,
      });
      await page.goto(`/projects/${PROJECT_ID}`);
      const composer = page.getByRole("form", {
        name: "What do you want to check?",
      });
      const results = page.getByRole("region", { name: "Recent results" });
      const output = results.getByRole("article", { name: "openapi" });
      await expect(
        output.getByTitle("outputs/openapi@result-r1"),
      ).toBeVisible();
      const field = composer.getByRole("textbox");
      await expect(field).toHaveAttribute("maxlength", "4000");
      await expect(
        composer.getByRole("button", { name: "Start a check" }),
      ).toBeVisible();
      expect(outputReads).toHaveLength(0);
      // The reviewed design uses an open canvas with independent document
      // cards, not white panels nested inside another white panel.
      const presentation = await page.evaluate(() => {
        const color = (selector: string) =>
          getComputedStyle(document.querySelector(selector)!).backgroundColor;
        return {
          canvas: color(".projects-workspace > .ui-detail-pane"),
          card: color(".projects-output"),
        };
      });
      expect(presentation.canvas).not.toBe(presentation.card);
      await expect(results).toHaveCSS("border-top-width", "0px");
      const glance = page.getByRole("region", { name: "At a glance" });
      const materials = page.getByRole("region", { name: "Project materials" });
      expect((await glance.boundingBox())!.y).toBeLessThan(
        (await materials.boundingBox())!.y,
      );
      const timeline = page.getByRole("region", { name: "Timeline" });
      const disclosure = timeline.locator("details");
      await expect(disclosure).not.toHaveAttribute("open");
      await disclosure.locator("summary").click();
      await expect(timeline.locator(".ui-activity").first()).toBeVisible();
      expect(
        await timeline.locator(".ui-activity svg").count(),
      ).toBeGreaterThan(0);
      await disclosure.locator("summary").click();
      const positions = await page.evaluate(() => {
        const box = (selector: string) =>
          document.querySelector(selector)!.getBoundingClientRect().toJSON();
        return {
          composer: box(".projects-composer"),
          result: box(".projects-output"),
          actions: box(".projects-output-actions"),
          footer: box(".projects-output-footer"),
          details: box(".projects-output-footer > a"),
          provenance: box(".projects-result-provenance"),
          documentWidth: document.documentElement.scrollWidth,
        };
      });
      expect(positions.composer.top).toBeLessThan(positions.result.top);
      expect(positions.actions.top).toBeLessThan(positions.footer.top);
      expect(positions.details.top).toBeGreaterThanOrEqual(
        positions.provenance.bottom,
      );
      expect(positions.documentWidth).toBeLessThanOrEqual(width);
      await output
        .getByRole("button", { name: "Preview", exact: true })
        .click();
      await expect(
        output.getByText("note: selected output", { exact: true }),
      ).toBeVisible();
      expect(outputReads).toHaveLength(2);
      await expect(
        output.getByRole("button", { name: "Download openapi" }),
      ).toBeVisible();
      await page.screenshot({
        path: testInfo.outputPath(`project-overview-${width}-${theme}.png`),
        fullPage: true,
      });
      await page.goto(`/projects/${PROJECT_ID}/workflows`);
      const navigation = page.getByRole("navigation", {
        name: "Project sections",
      });
      await expect(navigation.getByRole("link")).toHaveCount(8);
      await expect(
        navigation.getByRole("link", { name: "Workflows", exact: true }),
      ).toHaveAttribute("aria-current", "page");
      for (const resized of width === 1440 ? [390, 320, 1440] : [width]) {
        await page.setViewportSize({ width: resized, height: 1000 });
        await expect
          .poll(() =>
            navigation.evaluate((nav) => {
              const active = nav
                .querySelector('[aria-current="page"]')!
                .getBoundingClientRect();
              const bounds = nav.getBoundingClientRect();
              return (
                active.left >= bounds.left - 1 &&
                active.right <= bounds.right + 1
              );
            }),
          )
          .toBe(true);
        await expect
          .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
          .toBeLessThanOrEqual(resized);
      }
      await navigation
        .getByRole("link", { name: "Overview", exact: true })
        .click();
      const refresh = page.getByRole("button", {
        name: "Refresh project details",
        exact: true,
      });
      await expect(refresh).toHaveCount(1);
      await refresh.scrollIntoViewIfNeeded();
      await expect(refresh).toBeInViewport();
      await expect
        .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
        .toBeLessThanOrEqual(width);
    });
  }
}
