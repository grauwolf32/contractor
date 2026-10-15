import { expect, test, type Page } from "@playwright/test";

import { audit, project } from "../src/test/builders";
import {
  runRuntimeConfiguration,
  stageContract,
  workflowResource,
  workflowSummary,
} from "../src/test/builders-execution";
import {
  authSession,
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
  uiOrigin,
} from "./support/mock-api";

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
const workflow = workflowResource("repeat-workflow", "1", {
  entryStage: "inspect",
  parameters: { objective: { required: true } },
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
  },
  stages: { inspect: stageContract() },
});
const runtimeConfiguration = runRuntimeConfiguration();

async function installFixture(
  page: Page,
  apiOrigin: string,
  withResult = false,
) {
  const report = {
    namespace: "outputs",
    name: "report",
    revision: "report-r1",
  };
  const source =
    "Retained primary result\n<script>document.body.dataset.injected = 'yes'</script>";
  const selectedWorkflow = withResult
    ? workflowResource("repeat-workflow", "1", {
        ...workflow,
        outputs: {
          report: { primary: true, required: true, mediaTypes: ["text/plain"] },
        },
      })
    : workflow;
  const created: Array<{ body: unknown; idempotencyKey: string | null }> = [];
  await mockAPI(
    page,
    async (route, request, url) => {
      const path = url.pathname;
      if (path === "/v1/runs/run-repeat-browser") {
        await fulfillJSON(route, {
          runId: "run-repeat-browser",
          workflow: "repeat-workflow@1",
          state: withResult ? "succeeded" : "failed",
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
          outputs: withResult ? { report } : {},
          outputPublications: [],
          createdAt: timestamp,
          updatedAt: timestamp,
          finishedAt: timestamp,
        });
        return;
      }
      if (
        withResult &&
        path.startsWith("/v1/runs/run-repeat-browser/artifacts/outputs/report")
      ) {
        expect(url.searchParams.get("revision")).toBe(report.revision);
        if (path.endsWith("/metadata")) {
          await fulfillJSON(route, {
            artifact: report,
            mediaType: "text/plain",
            size: Buffer.byteLength(source),
            current: true,
            frozen: true,
            createdAt: timestamp,
          });
        } else {
          await fulfillContent(route, source, "text/plain");
        }
        return;
      }
      if (path === "/v1/runs/run-repeat-browser/repeat-draft") {
        await fulfillJSON(route, {
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
        await fulfillJSON(route, {
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
        await fulfillJSON(route, {
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
        await fulfillJSON(
          route,
          project("project-audit", {
            name: "Audit fixture",
            description: "Repeat navigation fixture",
            createdAt: timestamp,
            updatedAt: timestamp,
          }),
          200,
          { etag: '"1"' },
        );
        return;
      }
      if (path === "/v1/audits/audit-one") {
        await fulfillJSON(
          route,
          audit("project-audit", "audit-one", "failed", {
            currentRoundId: "round-one",
            holdState: "released",
            reservedRunCount: 1,
            submittedRunCount: 1,
            createdAt: timestamp,
            updatedAt: timestamp,
            finishedAt: timestamp,
          }),
          200,
          { etag: '"1"' },
        );
        return;
      }
      if (path === "/v1/workflows") {
        await fulfillJSON(route, {
          items: [workflowSummary(selectedWorkflow)],
          page: { hasMore: false },
        });
        return;
      }
      if (path === "/v1/workflows/repeat-workflow/versions/1") {
        await fulfillJSON(route, selectedWorkflow);
        return;
      }
      if (path === "/v1/artifacts") {
        await fulfillJSON(route, {
          items: [sourceMetadata],
          page: { hasMore: false },
        });
        return;
      }
      if (path === "/v1/runs" && request.method() === "POST") {
        created.push({
          body: request.postDataJSON(),
          idempotencyKey: request.headers()["idempotency-key"] ?? null,
        });
        await fulfillJSON(
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
      await fulfillEmptyPageOrNotFound(
        route,
        `Fixture has no ${request.method()} ${path}`,
      );
    },
    { apiOrigin, session: authSession("repeat-owner") },
  );
  return created;
}

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test.describe(`${viewport.width}px journeys`, () => {
    test.use({ viewport });

    test("terminal Run creates a reviewed exact repeat draft and preserves conflicts", async ({
      page,
    }, testInfo) => {
      const apiOrigin = uiOrigin(testInfo);
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
        .getByLabel(
          /I reviewed the retained inputs, labels and execution settings/,
        )
        .check();
      await page
        .getByRole("button", { name: "Confirm input for source" })
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

    test("primary output previews in place before preparing an exact repeat", async ({
      page,
    }, testInfo) => {
      const apiOrigin = uiOrigin(testInfo);
      const created = await installFixture(page, apiOrigin, true);
      const reads: string[] = [];
      page.on("request", (request) => {
        if (request.url().includes("/artifacts/outputs/report"))
          reads.push(request.url());
      });
      await page.goto("/runs/run-repeat-browser");
      const result = page.locator(".run-result-primary");
      await expect(result.getByText("Primary", { exact: true })).toBeVisible();
      expect(reads).toHaveLength(0);
      await result.getByRole("button", { name: "Preview result" }).click();
      await expect(result.locator("pre.artifact-preview")).toContainText(
        "Retained primary result",
      );
      await expect(page).toHaveURL(/\/runs\/run-repeat-browser$/);
      expect(reads).toHaveLength(2);
      expect(
        await page.evaluate(() => document.body.dataset.injected),
      ).toBeUndefined();
      await page.getByRole("button", { name: "Configure another Run" }).click();
      await expect(page.locator('[name="artifact-source"]')).toHaveValue(
        "sources/service@source-r1",
      );
      await expect(
        page.getByRole("button", { name: "Start Workflow Run" }),
      ).toBeDisabled();
      expect(created).toHaveLength(0);
    });

    for (const [code, message] of [
      [
        "repeat_request_unavailable",
        "The original Run request is unavailable. Configure a new Run from the Workflow.",
      ],
      [
        "repeat_request_invalid",
        "The saved Run request could not be verified. Configure a new Run from the Workflow.",
      ],
    ]) {
      test(`${code} blocks Repeat without navigation or submission`, async ({
        page,
      }, testInfo) => {
        const apiOrigin = uiOrigin(testInfo);
        const created = await installFixture(page, apiOrigin);
        let repeatReads = 0;
        await page.route(
          `${apiOrigin}/v1/runs/run-repeat-browser/repeat-draft`,
          async (route) => {
            repeatReads++;
            await fulfillJSON(route, {
              sourceRunId: "run-repeat-browser",
              authority: "ordinary",
              workflow: { name: "repeat-workflow", version: "1" },
              notices: [{ code, severity: "blocking", message }],
            });
          },
        );
        await page.goto("/runs/run-repeat-browser");
        for (let attempt = 1; attempt <= 2; attempt++) {
          await page
            .getByRole("button", { name: "Configure another Run" })
            .click();
          await expect.poll(() => repeatReads).toBe(attempt);
          await expect(page.getByText(message!)).toBeVisible();
          await expect(page).toHaveURL(/\/runs\/run-repeat-browser$/);
          await expect(
            page.getByRole("button", { name: "Start Workflow Run" }),
          ).toHaveCount(0);
          expect(created).toHaveLength(0);
        }
      });
    }

    test("Audit-managed Run returns to the owning Audit", async ({
      page,
    }, testInfo) => {
      await installFixture(page, uiOrigin(testInfo));
      await page.goto("/runs/run-audit-browser");
      await page.getByRole("button", { name: "Configure another Run" }).click();
      await expect(page).toHaveURL("/projects/project-audit/audits/audit-one");
    });
  });
}
