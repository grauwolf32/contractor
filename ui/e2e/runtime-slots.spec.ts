import { expect, test } from "@playwright/test";
import {
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

for (const theme of ["light", "dark", "black"]) {
  test(`Runtime capacity and allocation at 390px in ${theme}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    await mockAPI(page, async (route, _request, url) => {
      if (url.pathname === "/v1/operations/snapshot") {
        return fulfillJSON(route, {
          cursor: { generation: "slots-browser", revision: "1" },
          runtimeAgents: [
            {
              instanceId: "runtime-four",
              softwareVersion: "1.0.0",
              supportedRuntimes: ["tool@1"],
              supportedToolsets: [],
              supportedSandboxProfiles: ["local-workdir@1"],
              supportedRuntimeAdapters: [],
              observedState: "allocated",
              state: "busy",
              allocationSlots: 4,
              availableSlots: 3,
              busySlots: 1,
              reservedSlots: 0,
              drainingSlots: 0,
              fencedSlots: 0,
              observedAllocations: [
                { allocationId: "allocation-one", observedState: "allocated" },
              ],
              authoritativeAllocationIds: ["allocation-one"],
              pendingReleaseAllocationIds: [],
            },
          ],
          allocations: [
            {
              allocationId: "allocation-one",
              runId: "run-one",
              stageExecutionId: "stage-one",
              runtimeAgentInstanceId: "runtime-four",
              logicalWorker: "scanner",
              agentTemplate: {
                templateId: "scanner",
                version: "1",
                digest: `sha256:${"a".repeat(64)}`,
              },
              executionConfig: {},
              authoritativePhase: "active",
              observedPhase: "prepared",
              metrics: {
                reportsComplete: false,
                modelCalls: 0,
                inputTokens: 0,
                outputTokens: 0,
                totalTokens: 0,
                toolCalls: 0,
                toolFailures: 0,
                errorCount: 0,
                truncated: false,
              },
            },
          ],
        });
      }
      return fulfillEmptyPageOrNotFound(route, url.pathname);
    });
    await page.goto("/operations");
    await expect(
      page.getByRole("img", { name: /4 Runtime Agent slots: 3 idle, 1 busy/ }),
    ).toBeVisible();
    await expect(
      page.getByText("1 / 4 occupied · 3 free · 0 fenced"),
    ).toBeVisible();
    await page.getByText("runtime-four", { exact: true }).first().click();
    await expect(
      page.getByRole("link", { name: "Inspect allocation allocation-one →" }),
    ).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  });
}
