import { expect, test } from "@playwright/test";

import {
  audit as auditBuilder,
  auditInput,
  project,
} from "../src/test/builders";
import {
  authSession,
  fulfillError,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

const date = "2026-10-07T10:00:00Z";
const audit = auditBuilder("project_history", "audit_history", "completed", {
  inputs: { source: auditInput("history-source") },
  revision: 51,
  eventSequence: 51,
  holdState: "released",
  createdAt: date,
  updatedAt: date,
  startedAt: date,
  finishedAt: date,
});
const events = Array.from({ length: 51 }, (_, i) => ({
  auditId: audit.auditId,
  sequence: 51 - i,
  kind: i === 50 ? "audit.created" : "task.settled",
  entityId: i === 50 ? audit.auditId : `task_${i}`,
  summary: i === 50 ? {} : { taskCount: 1 },
  createdAt: date,
}));

for (const width of [1440, 390, 320]) {
  test(`check history keeps durable pages and retry at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    let fail = true;
    const cursors: string[] = [];
    await mockAPI(
      page,
      async (route, _request, url) => {
        let value: unknown = { items: [], page: { hasMore: false } };
        if (url.pathname === "/v1/projects/project_history")
          value = project("project_history", { name: "History project" });
        else if (url.pathname === "/v1/audits/audit_history") value = audit;
        else if (url.pathname.endsWith("/events")) {
          const cursor = url.searchParams.get("cursor");
          cursors.push(cursor ?? "head");
          if (cursor !== null && fail) {
            await fulfillError(route, 500, "internal_error", "Unavailable");
            return;
          }
          value = {
            items: cursor === null ? events.slice(0, 50) : events.slice(50),
            throughSequence: 51,
            total: 51,
            page:
              cursor === null
                ? { hasMore: true, nextCursor: "older" }
                : { hasMore: false },
          };
        }
        await fulfillJSON(route, value, 200, {
          ETag:
            url.pathname === "/v1/projects/project_history" ? '"1"' : '"51"',
        });
      },
      { session: authSession("owner", ["user"]) },
    );
    await page.goto("/projects/project_history/audits/audit_history");
    await page
      .getByRole("button", { name: "Show activity", exact: true })
      .click();
    const activity = page.getByRole("main").getByRole("list");
    await expect(activity.getByRole("listitem")).toHaveCount(50);
    await expect(page.getByRole("alert")).toHaveCount(0);
    const pages = page.getByRole("navigation", { name: "Activity pages" });
    await pages.getByRole("button", { name: "Next", exact: true }).click();
    await expect(page.getByRole("alert")).toContainText("Unavailable");
    fail = false;
    await page.getByRole("button", { name: "Try again" }).click();
    await expect(activity.getByRole("listitem")).toHaveCount(1);
    await expect(
      activity.getByText("audit.created", { exact: true }),
    ).toBeVisible();
    await expect(
      pages.getByRole("button", { name: "Next", exact: true }),
    ).toBeDisabled();
    await pages.getByRole("button", { name: "Previous", exact: true }).click();
    await expect(activity.getByRole("listitem")).toHaveCount(50);
    expect(cursors.slice(0, 3)).toEqual(["head", "older", "older"]);
    expect(errors).toEqual([]);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: testInfo.outputPath(`check-events-${width}.png`),
      fullPage: true,
    });
  });
}
