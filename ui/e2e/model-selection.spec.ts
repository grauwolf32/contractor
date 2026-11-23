import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "@playwright/test";
import { operationsSnapshot } from "../src/test/builders-execution";
import {
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

for (const theme of ["light", "dark", "black"]) {
  test(`model selection saves at 390px in ${theme}`, async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (theme) => localStorage.setItem("contractor.theme", theme),
      theme,
    );
    const digest = `sha256:${"a".repeat(64)}`;
    let selection: { defaultModel: string; roles: Record<string, string> } = {
      defaultModel: "alpha",
      roles: {},
    };
    let revision = "1";
    await mockAPI(page, async (route, request, url) => {
      const path = url.pathname;
      if (path === "/v1/operations/settings/models") {
        if (request.method() === "PUT") {
          expect(request.headers()["if-match"]).toBe(`"1:${digest.slice(7)}"`);
          selection = request.postDataJSON();
          revision = "2";
        }
        return fulfillJSON(
          route,
          {
            configured: true,
            selection,
            revision,
            catalogDigest: digest,
            updatedAt: "2026-10-21T00:00:00Z",
            models: ["alpha", "beta"].map((id) => ({
              id,
              digest,
              upstreamModel: `fixture/${id}`,
              contextWindowTokens: 1000000,
              maxOutputTokens: 32768,
              policies: {
                planner: `planner_${id}@1`,
                worker: `worker_${id}@1`,
                summarizer: `summarizer_${id}@1`,
                audit_completion_worker: `audit_${id}@1`,
              },
            })),
          },
          200,
          {
            ETag: `"${revision}:${digest.slice(7)}"`,
            "Cache-Control": "no-store",
          },
        );
      }
      if (path === "/v1/operations/snapshot")
        return fulfillJSON(route, operationsSnapshot());
      if (path === "/v1/settings/git-key")
        return fulfillJSON(route, { configured: false });
      if (path === "/v1/operations/settings/scheduler")
        return fulfillJSON(
          route,
          {
            maxConcurrentRuns: 1,
            maxConcurrentToolRuns: 2,
            revision: "1",
            updatedAt: "2026-10-21T00:00:00Z",
          },
          200,
          { ETag: '"1"' },
        );
      return fulfillEmptyPageOrNotFound(route);
    });
    await page.goto("/operations/settings#model-selection");
    const section = page.locator("#model-selection");
    await expect(
      section.getByRole("heading", { name: "Models", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("combobox", { name: "Default model", exact: true })
      .selectOption("beta");
    await page.getByText("Models by role · all use default").click();
    await page
      .getByRole("combobox", { name: "Summarizer model", exact: true })
      .selectOption("alpha");
    await expect(
      section.getByText("Unsaved changes", { exact: true }),
    ).toBeVisible();
    await expect(section.getByLabel("Model changes to save")).toContainText(
      "alpha → beta",
    );
    await page.getByRole("button", { name: "Save model defaults" }).click();
    await expect(
      page.getByText("Model defaults saved for new executions."),
    ).toBeVisible();
    expect(selection).toEqual({
      defaultModel: "beta",
      roles: { summarizer: "alpha" },
    });
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    const overflow = await section.evaluate(
      (element) => element.scrollWidth > element.clientWidth,
    );
    expect(overflow).toBe(false);
    await page
      .getByRole("combobox", { name: "Default model", exact: true })
      .focus();
    await section.screenshot({
      path: join(tmpdir(), `contractor-model-selection-${theme}.png`),
    });
  });
}
