import { ARTIFACT_STORAGE_TIMESTAMPS } from "../src/test/builders";
import { expect, test, type Page } from "@playwright/test";

import {
  runStatus,
  stageContract,
  workflowResource,
} from "../src/test/builders-execution";
import {
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

async function installRunsAPI(page: Page): Promise<void> {
  await mockAPI(page, async (route, _request, url) => {
    if (url.pathname === "/v1/runs/run-completed-browser") {
      await fulfillJSON(
        route,
        runStatus("run-completed-browser", {
          workflow: "browser-workflow@1",
          state: "succeeded",
          deletable: true,
        }),
      );
      return;
    }
    if (url.pathname === "/v1/workflows/browser-workflow/versions/1") {
      await fulfillJSON(route, workflowResource("browser-workflow", "1"));
      return;
    }
    if (url.pathname === "/v1/queue/control") {
      await fulfillJSON(route, { paused: false, revision: "0" }, 200, {
        etag: '"0"',
      });
      return;
    }
    if (url.pathname === "/v1/queue") {
      await fulfillJSON(route, {
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
      await fulfillJSON(route, {
        items: [
          {
            runId: "run-completed-browser",
            workflow: "browser-workflow@1",
            state: "succeeded",
            deletable: true,
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
    await fulfillEmptyPageOrNotFound(route);
  });
}

async function installRunResultAPI(
  page: Page,
  resultRequests: URL[],
): Promise<void> {
  const source = "browser primary result";
  const report = {
    namespace: "outputs",
    name: "report",
    revision: "report-browser-r1",
  };
  await mockAPI(page, async (route, _request, url) => {
    if (url.pathname === "/v1/runs/run-result-browser") {
      await fulfillJSON(
        route,
        runStatus("run-result-browser", {
          workflow: "browser-result@1",
          outputs: { report },
          createdAt: "2026-09-07T08:00:00Z",
          updatedAt: "2026-09-07T08:01:00Z",
          finishedAt: "2026-09-07T08:01:00Z",
        }),
      );
      return;
    }
    if (url.pathname === "/v1/workflows/browser-result/versions/1") {
      resultRequests.push(url);
      await fulfillJSON(
        route,
        workflowResource("browser-result", "1", {
          entryStage: "render",
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
          stages: { render: stageContract() },
        }),
      );
      return;
    }
    if (
      url.pathname ===
      "/v1/runs/run-result-browser/artifacts/outputs/report/metadata"
    ) {
      resultRequests.push(url);
      await fulfillJSON(route, {
        canUploadNewVersion: false,
        storageTimestamps: ARTIFACT_STORAGE_TIMESTAMPS,
        artifact: report,
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
      await fulfillContent(route, source, "text/plain");
      return;
    }
    await fulfillEmptyPageOrNotFound(route);
  });
}

test("Runs defaults to Queue and keeps terminal history in Completed", async ({
  page,
}) => {
  await installRunsAPI(page);

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
  await page.setViewportSize({ width: 320, height: 568 });
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
    .toBe(320);
});

test("Run results expose the declared primary output with one bounded action", async ({
  page,
}) => {
  const resultRequests: URL[] = [];
  await installRunResultAPI(page, resultRequests);

  await page.goto("/runs/run-result-browser");
  await expect(page.getByText("Declared primary result")).toBeVisible();
  await expect(page.getByRole("heading", { name: "trace" })).toBeVisible();
  expect(resultRequests.map((url) => url.pathname)).toEqual([
    "/v1/workflows/browser-result/versions/1",
  ]);

  await page.getByRole("button", { name: "Preview" }).click();
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

for (const theme of ["light", "dark", "black"]) {
  test(`retains the source Runs list without switches in ${theme}`, async ({
    page,
  }, testInfo) => {
    await page.addInitScript(
      (theme) => localStorage.setItem("contractor.theme", theme),
      theme,
    );
    await installRunsAPI(page);
    const requests: string[] = [];
    page.on("request", (request) => {
      if (new URL(request.url()).pathname === "/v1/runs")
        requests.push(request.url());
    });
    const source = "/runs?view=completed&state=succeeded&cursor=kept-page";
    await page.goto(source);
    await page
      .getByRole("link", { name: "run-completed-browser", exact: true })
      .click();
    const pane = page.getByRole("region", { name: "Source Runs", exact: true });
    const row = pane.getByRole("link", { name: /run-completed-browser/ });
    await expect(row).toHaveAttribute("aria-current", "true");
    await expect(pane.getByRole("combobox")).toHaveCount(0);
    await expect(pane.getByRole("tab")).toHaveCount(0);
    await expect(
      pane.getByRole("link", { name: "Queue", exact: true }),
    ).toHaveCount(0);
    expect(requests.length).toBeGreaterThan(0);
    for (const request of requests) {
      const params = new URL(request).searchParams;
      expect(params.get("cursor")).toBe("kept-page");
      expect(params.get("state")).toBe("succeeded");
      expect(params.get("lifecycle")).toBe("terminal");
    }
    await page.screenshot({
      path: testInfo.outputPath(`source-runs-${theme}-1440.png`),
    });
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await expect(pane).toBeHidden();
      await expect(
        page.getByRole("heading", { name: "browser-workflow", exact: true }),
      ).toBeVisible();
      await expect
        .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
        .toBe(width);
      await page.screenshot({
        path: testInfo.outputPath(`source-runs-${theme}-${width}.png`),
      });
    }
    await page
      .getByRole("link", { name: "Back to Completed", exact: true })
      .click();
    await expect(page).toHaveURL(
      new RegExp("/runs\\?view=completed&state=succeeded&cursor=kept-page$"),
    );
    await expect(
      page.getByRole("link", { name: "run-completed-browser", exact: true }),
    ).toBeVisible();
  });
}
