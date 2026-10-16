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
  kind: i === 50 ? "audit.created" : "review.decided",
  entityId: i === 50 ? audit.auditId : `decision_${i}`,
  summary: i === 50 ? {} : { action: "approve" },
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
        // Owner lists and Audit pages are empty; findings and reviews pages
        // also carry the Audit revision they were read at.
        let value: unknown = /^\/v1\/audits\/[^/]+\/(findings|reviews)$/.test(
          url.pathname,
        )
          ? {
              items: [],
              page: { hasMore: false },
              total: 0,
              auditRevision: 51,
              asOf: date,
            }
          : { items: [], page: { hasMore: false } };
        if (url.pathname === "/v1/projects/project_history")
          value = project("project_history", { name: "History project" });
        else if (url.pathname === "/v1/audits/audit_history") value = audit;
        else if (url.pathname.endsWith("/workspace"))
          value = {
            auditId: audit.auditId,
            auditRevision: 51,
            asOf: date,
            executionState: "completed",
            outstandingRuns: 0,
            totalChecks: 0,
            completedChecks: 0,
            issues: 0,
            gaps: 0,
            unchecked: 0,
            findings: 0,
            unreviewedFindings: 0,
            pendingReviews: 0,
          };
        else if (url.pathname.endsWith("/report"))
          value = { status: "unavailable" };
        else if (url.pathname.endsWith("/coverage"))
          value = { items: [], page: { hasMore: false }, auditRevision: 51 };
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
    const activity = page.getByRole("list", { name: "Activity on this check" });
    await expect(activity.getByRole("listitem")).toHaveCount(50);
    await expect(page.getByRole("alert")).toHaveCount(0);
    await page.getByRole("button", { name: "Load older activity" }).click();
    await expect(page.getByRole("alert")).toContainText(
      "Older activity could not be loaded.",
    );
    await expect(activity.getByRole("listitem")).toHaveCount(50);
    fail = false;
    await page.getByRole("button", { name: "Try again" }).click();
    await expect(activity.getByRole("listitem")).toHaveCount(51);
    await expect(activity.getByText("Check created.")).toBeVisible();
    await expect(
      page.getByText("Showing 51 of 51 recorded events."),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Load older activity" }),
    ).toHaveCount(0);
    expect(cursors).toEqual(["head", "older", "older"]);
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
