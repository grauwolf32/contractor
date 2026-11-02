import { expect, test, type Page } from "@playwright/test";

import {
  runRuntimeConfiguration,
  workflowResource,
} from "../src/test/builders-execution";
import {
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

const RUN_ID = "run_0123456789abcdef0123456789abcdef";
const LONG_REVISION = `rev_${"a".repeat(64)}`;

async function installRunAPI(page: Page): Promise<void> {
  await mockAPI(page, async (route, _request, url) => {
    if (url.pathname === "/v1/workflows/responsive-layout/versions/1") {
      await fulfillJSON(
        route,
        workflowResource("responsive-layout", "1", {
          outputs: {
            validation_report: {
              required: true,
              primary: true,
              mediaTypes: ["text/markdown"],
            },
          },
        }),
      );
      return;
    }
    if (url.pathname === `/v1/runs/${RUN_ID}`) {
      await fulfillJSON(route, {
        runId: RUN_ID,
        workflow: "responsive-layout@1",
        state: "succeeded",
        deletable: true,
        runtimeLabels: [],
        labels: {},
        runtimeConfiguration: runRuntimeConfiguration(),
        parameters: {},
        inputs: {},
        attempts: [
          {
            stageExecutionId: "stage-responsive-1",
            stage: "validate",
            objective: "Validate the responsive layout.",
            attempt: 1,
            executionConfig: { variant: "base", agents: { validator: {} } },
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
        outputPublications: [],
        createdAt: "2026-09-04T08:00:00Z",
        updatedAt: "2026-09-04T08:01:00Z",
        startedAt: "2026-09-04T08:00:00Z",
        finishedAt: "2026-09-04T08:01:00Z",
      });
      return;
    }
    if (url.pathname.endsWith("/runtime-diagnostics")) {
      await fulfillJSON(route, {
        status: "available",
        items: [
          {
            logicalAgent: "validator",
            status: "partial",
            complete: false,
            stopReasonTruncated: false,
            durationMs: "9007199254740993",
            adapters: [
              {
                ref: "otlp-http@1",
                operations: "18446744073709551615",
                failedOperations: "1",
                flushAttempted: true,
                flushSucceeded: false,
                lastErrorCode: "flush_failed",
                droppedSpans: {
                  queueOverflow: "0",
                  encodingFailed: "0",
                  nonRetryable: "0",
                  retryExhausted: "0",
                  collectorRejected: "0",
                  deadlineExceeded: "0",
                  cancelled: "0",
                  shutdown: "9007199254740993",
                },
              },
            ],
          },
        ],
        page: { hasMore: false },
      });
      return;
    }
    if (url.pathname.endsWith("/tool-calls")) {
      const worker = url.searchParams.get("participant") === "worker";
      const next = url.searchParams.has("cursor");
      await fulfillJSON(route, {
        status: "available",
        participant: worker ? "worker" : "planner",
        ...(worker ? { logicalAgent: "validator" } : {}),
        complete: !worker,
        detailTruncated: worker,
        items: [
          {
            callReference: (next ? "b" : "a").repeat(64),
            tool: next ? "second_probe" : "probe",
            toolTruncated: false,
            outcome: "failed",
            argumentsJson:
              '{"large":9007199254740993,"unsigned":18446744073709551615}',
            argumentsTruncated: worker,
            durationMs: "9223372036854775807",
            resultSizeBytes: "9007199254740993",
            error: {
              code: "tool_call_failed",
              message: "The probe failed with a retained diagnostic.",
              messageTruncated: false,
              retryable: false,
            },
          },
        ],
        page:
          worker || next
            ? { hasMore: false }
            : { hasMore: true, nextCursor: "next-tool-page" },
      });
      return;
    }
    if (url.pathname === `/v1/runs/${RUN_ID}/artifacts`) {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.pathname === "/v1/runs") {
      await fulfillJSON(route, {
        items: [
          {
            runId: RUN_ID,
            workflow: "responsive-layout-with-a-long-name@1",
            state: "succeeded",
            deletable: true,
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
    if (url.pathname === "/v1/queue/control") {
      await fulfillJSON(route, { paused: false, revision: "0" });
      return;
    }
    await fulfillEmptyPageOrNotFound(route);
  });
}

test("Run detail stays within a 320px viewport", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await installRunAPI(page);

  await page.goto(`/runs/${RUN_ID}`);
  await expect(
    page.getByRole("heading", { name: "responsive-layout" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Preview" })).toBeVisible();
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
}) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await installRunAPI(page);

  await page.goto("/runs?view=completed");
  const runLink = page.getByRole("link", { name: RUN_ID });
  await expect(runLink).toBeVisible();
  await expect(runLink).toHaveText("run_01234567…89abcdef");
  await expect(runLink).toHaveAttribute("title", RUN_ID);

  const row = runLink.locator("xpath=ancestor::tr");
  await expect(row).toHaveCSS("display", "grid");
  await expect(row.locator(".run-list-workflow-cell")).toBeVisible();
  await expect(row.locator(".run-list-finished-cell")).toBeVisible();
  await expect(row.locator(".run-list-labels-cell")).toBeVisible();
  // The state is a V3B status chip: a glyph and the capitalized state word.
  await expect(row.getByText("Succeeded", { exact: true })).toHaveCSS(
    "white-space",
    "nowrap",
  );
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
    .toBe(320);
});

test("Reopening a failed Run refreshes both navigation-cancelled diagnostic reads", async ({
  page,
}) => {
  let failed = false;
  const reads = { runtime: 0, tool: 0 };
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => {
    release = resolve;
  });
  let drained = 0;
  await mockAPI(page, async (route, _request, url) => {
    if (url.pathname === `/v1/runs/${RUN_ID}`) {
      await fulfillJSON(route, {
        runId: RUN_ID,
        workflow: "diagnostics-reopen@1",
        state: failed ? "failed" : "running",
        deletable: false,
        runtimeLabels: [],
        labels: {},
        runtimeConfiguration: runRuntimeConfiguration(),
        attempts: [
          {
            stageExecutionId: "stage-reopen",
            stage: "diagnose",
            attempt: 1,
            executionConfig: { variant: "base", agents: { validator: {} } },
            state: failed ? "failed" : "running",
            ...(failed
              ? {
                  result: {
                    apiVersion: "contractor/v1alpha1",
                    outcome: "failed",
                    summary: "The probe failed.",
                    artifacts: {},
                    error: {
                      code: "probe_failed",
                      message: "The probe failed.",
                      retryable: false,
                    },
                  },
                  terminalAt: "2026-09-04T08:01:00Z",
                }
              : {}),
          },
        ],
        transitions: [],
        outputs: {},
        outputPublications: [],
        ...(failed
          ? { finishedAt: "2026-09-04T08:01:00Z" }
          : { activeStageExecutionId: "stage-reopen" }),
        createdAt: "2026-09-04T08:00:00Z",
        updatedAt: failed ? "2026-09-04T08:01:00Z" : "2026-09-04T08:00:00Z",
      });
      return;
    }
    const kind = url.pathname.endsWith("/runtime-diagnostics")
      ? "runtime"
      : url.pathname.endsWith("/tool-calls")
        ? "tool"
        : undefined;
    if (kind !== undefined) {
      const count = ++reads[kind];
      if (count === 2) {
        // Keep the terminal read in flight until navigation has aborted it.
        // Cleanup only settles the deliberately abandoned mocked requests.
        await blocked;
        await route.abort("aborted").catch(() => undefined);
        drained++;
        return;
      }
      const value = count === 1 ? "1" : "2";
      await fulfillJSON(
        route,
        kind === "runtime"
          ? {
              status: "available",
              items: [
                {
                  logicalAgent: "validator",
                  status: count === 1 ? "partial" : "available",
                  complete: count !== 1,
                  durationMs: value,
                  stopReasonTruncated: false,
                  adapters: [],
                },
              ],
              page: { hasMore: false },
            }
          : {
              status: "available",
              participant: "planner",
              complete: count !== 1,
              detailTruncated: false,
              items: [
                {
                  callReference: "a".repeat(64),
                  tool: "probe",
                  toolTruncated: false,
                  outcome: "succeeded",
                  argumentsTruncated: false,
                  durationMs: value,
                  resultSizeBytes: null,
                },
              ],
              page: { hasMore: false },
            },
      );
      return;
    }
    if (url.pathname === "/v1/queue/control") {
      await fulfillJSON(route, { paused: false, revision: "0" });
      return;
    }
    await fulfillEmptyPageOrNotFound(route);
  });
  const panel = (name: string) =>
    page
      .locator(".runs-block")
      .filter({ has: page.getByRole("heading", { name, exact: true }) });
  try {
    await page.goto(`/runs/${RUN_ID}`);
    await expect(
      panel("Runtime adapter diagnostics").getByText("1", { exact: true }),
    ).toBeVisible();
    await expect(
      panel("Tool calls").getByText("1", { exact: true }),
    ).toBeVisible();
    expect(reads).toEqual({ runtime: 1, tool: 1 });
    failed = true;
    await page.getByRole("button", { name: "Refresh", exact: true }).click();
    await expect.poll(() => reads).toEqual({ runtime: 2, tool: 2 });
    await page.getByRole("link", { name: "Completed", exact: true }).click();
    await page.waitForURL(/\/runs\?view=completed/);
    await page.goBack();
    await expect(
      panel("Runtime adapter diagnostics").getByText("2", { exact: true }),
    ).toBeVisible();
    await expect(
      panel("Tool calls").getByText("2", { exact: true }),
    ).toBeVisible();
    expect(reads).toEqual({ runtime: 3, tool: 3 });
  } finally {
    release();
    await expect.poll(() => drained).toBe(2);
  }
});

for (const theme of ["light", "dark", "black"] as const) {
  test(`Tool-call detail remains exact at 390 px in ${theme}`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (theme) => localStorage.setItem("contractor.theme", theme),
      theme,
    );
    await installRunAPI(page);
    await page.goto(`/runs/${RUN_ID}`);
    const attempts = page.locator("#run-attempts");
    if ((await attempts.getAttribute("open")) === null)
      await attempts.locator(":scope > summary").click();
    const attempt = page.locator(".runs-attempt").first();
    if ((await attempt.getAttribute("open")) === null)
      await attempt.locator(":scope > summary").click();
    const panel = attempt.locator(".runs-block").filter({
      has: page.getByRole("heading", { name: "Tool calls", exact: true }),
    });
    await expect(
      panel.getByText("9223372036854775807", { exact: true }),
    ).toBeVisible();
    await panel.getByText("Sanitized arguments", { exact: true }).click();
    await expect(
      panel.getByText(
        '{"large":9007199254740993,"unsigned":18446744073709551615}',
        { exact: true },
      ),
    ).toBeVisible();
    await panel.getByRole("button", { name: "Load more tool calls" }).click();
    await expect(
      panel.getByText("second_probe", { exact: true }),
    ).toBeVisible();
    await panel
      .getByLabel("Tool-call participant")
      .selectOption("worker:validator");
    await expect(
      panel.getByText("The selected report is incomplete."),
    ).toBeVisible();
    await expect(
      panel.getByText(
        "Retained detail was shortened; older records may have been removed.",
      ),
    ).toBeVisible();
    await panel.getByText("Sanitized arguments", { exact: true }).click();
    await expect
      .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
      .toBe(390);
    await panel
      .getByRole("heading", { name: "Tool calls", exact: true })
      .click();
    await panel.screenshot({
      path: testInfo.outputPath(`tool-diagnostics-${theme}.png`),
    });
  });
  test(`Runtime delivery diagnostics remain exact at 390 px in ${theme}`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (theme) => localStorage.setItem("contractor.theme", theme),
      theme,
    );
    await installRunAPI(page);
    await page.goto(`/runs/${RUN_ID}`);
    const attempts = page.locator("#run-attempts");
    if ((await attempts.getAttribute("open")) === null)
      await attempts.locator(":scope > summary").click();
    const attempt = page.locator(".runs-attempt").first();
    if ((await attempt.getAttribute("open")) === null)
      await attempt.locator(":scope > summary").click();
    await expect(
      attempt.getByRole("heading", { name: "Runtime adapter diagnostics" }),
    ).toBeVisible();
    await expect(
      attempt.getByText("18446744073709551615", { exact: true }),
    ).toBeVisible();
    await expect(
      attempt.getByText("flush_failed", { exact: true }),
    ).toBeVisible();
    await expect
      .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
      .toBe(390);
    await attempt.screenshot({
      path: testInfo.outputPath(`runtime-diagnostics-${theme}.png`),
    });
  });
}

for (const theme of ["light", "dark", "black"] as const) {
  for (const width of [1440, 390, 320]) {
    test(`Run design fidelity at ${width}px in ${theme}`, async ({
      page,
    }, testInfo) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (theme) => localStorage.setItem("contractor.theme", theme),
        theme,
      );
      let artifactReads = 0;
      page.on("request", (request) => {
        if (new URL(request.url()).pathname.includes("/artifacts/"))
          artifactReads++;
      });
      await installRunAPI(page);
      await page.goto(`/runs/${RUN_ID}`);
      const frame = page.locator(".runs-run");
      const card = page.locator(".run-result-card").first();
      await expect(card).toBeVisible();
      await expect(frame).toHaveCSS("border-top-width", "0px");
      await expect(
        card.getByRole("button", { name: "Preview", exact: true }),
      ).toBeVisible();
      await expect(
        card.getByRole("button", { name: /Download/ }),
      ).toBeVisible();
      expect(artifactReads).toBe(0);
      const details = card.locator(".run-output-detail-link");
      if (width <= 390) {
        const actionLines = await card.evaluate((node) => {
          const actions = node.querySelector(".runs-result-actions")!;
          const badges = Array.from(
            actions.querySelectorAll(".ui-status-chip"),
          );
          const preview = Array.from(actions.querySelectorAll("button")).find(
            (button) => button.textContent?.includes("Preview"),
          )!;
          return {
            badgeCount: badges.length,
            previewTop: preview.getBoundingClientRect().top,
            badgesBottom: Math.max(
              ...badges.map((badge) => badge.getBoundingClientRect().bottom),
            ),
          };
        });
        expect(actionLines.badgeCount).toBeGreaterThan(0);
        expect(actionLines.previewTop).toBeGreaterThan(
          actionLines.badgesBottom,
        );
      }
      await expect(details).toHaveText("details");
      await expect(details).toHaveAttribute("href", /\?revision=/);
      const fonts = await page.evaluate(() => ({
        base: getComputedStyle(document.body).fontFamily,
        heading: getComputedStyle(document.querySelector("h1")!).fontFamily,
        button: getComputedStyle(
          document.querySelector(".runs-result-actions button")!,
        ).fontFamily,
        documentWidth: document.documentElement.scrollWidth,
      }));
      expect(fonts.heading).toBe(fonts.base);
      expect(fonts.button).toBe(fonts.base);
      expect(fonts.base).not.toMatch(/Manrope|JetBrains/);
      expect(fonts.documentWidth).toBe(width);
      const sections = page.locator(".runs-sections");
      await expect(sections).toHaveCSS("border-top-width", "0px");
      await page.screenshot({
        path: testInfo.outputPath(`run-fidelity-${width}-${theme}.png`),
        fullPage: true,
      });
    });
  }
}
