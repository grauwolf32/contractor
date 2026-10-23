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
  test(`both scheduling limits fit and save at 390px in ${theme}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (theme) => localStorage.setItem("contractor.theme", theme),
      theme,
    );
    let settings = {
      maxConcurrentRuns: 1,
      maxConcurrentToolRuns: 2,
      revision: "1",
      updatedAt: "2026-10-23T00:00:00Z",
    };
    const writes: { ifMatch: string | undefined; body: unknown }[] = [];
    await mockAPI(page, async (route, request, url) => {
      const path = url.pathname;
      if (path === "/v1/operations/settings/scheduler") {
        if (request.method() === "PUT") {
          writes.push({
            ifMatch: request.headers()["if-match"],
            body: request.postDataJSON(),
          });
          settings = {
            ...settings,
            ...request.postDataJSON(),
            revision: "2",
            updatedAt: "2026-10-23T00:01:00Z",
          };
        }
        return fulfillJSON(route, settings, 200, {
          ETag: `"${settings.revision}"`,
          "Cache-Control": "no-store",
        });
      }
      if (path === "/v1/operations/snapshot")
        return fulfillJSON(route, operationsSnapshot());
      if (path === "/v1/settings/git-key")
        return fulfillJSON(route, { configured: false });
      return fulfillEmptyPageOrNotFound(route);
    });
    await page.goto("/operations/settings#workflow-scheduling");

    const section = page.locator("#workflow-scheduling");
    const model = section.getByLabel("Concurrent Runs using a model");
    const tool = section.getByLabel("Concurrent Runs without a model");
    await expect(model).toHaveValue("1");
    await expect(tool).toHaveValue("2");
    await section.getByText("Technical details").click();
    await expect(
      section.getByText("maxConcurrentToolRuns", { exact: true }),
    ).toBeVisible();
    for (const field of [model, tool]) {
      const box = await field.boundingBox();
      expect(box).not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(390);
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await section.screenshot({
      path: join(tmpdir(), `contractor-scheduler-settings-${theme}.png`),
    });

    await tool.fill("6");
    await section
      .getByRole("button", { name: "Save scheduling limits" })
      .click();
    await expect(
      section.getByText("Saved 1 using a model and 6 without a model."),
    ).toBeVisible();
    expect(writes).toEqual([
      {
        ifMatch: '"1"',
        body: { maxConcurrentRuns: 1, maxConcurrentToolRuns: 6 },
      },
    ]);
    await expect(tool).toHaveValue("6");
  });
}
