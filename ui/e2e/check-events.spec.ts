import { expect, test } from "@playwright/test";
import packageMetadata from "../package.json" with { type: "json" };

const date = "2026-10-07T10:00:00Z";
const digest = `sha256:${"a".repeat(64)}`;
const audit = {
  auditId: "audit_history",
  projectId: "project_history",
  profile: { name: "openapi-operation-trace", version: "1", digest },
  inputs: {},
  scope: {},
  runtimeLabels: [],
  state: "completed",
  phase: "rounds",
  currentRoundId: "round_1",
  revision: 51,
  eventSequence: 51,
  dispatchState: "closed",
  holdState: "released",
  limits: {
    maxRounds: 1,
    batchSize: 1,
    maxItemsPerRound: 8,
    maxItemsTotal: 8,
    maxSubmittedRuns: 8,
    maxItemRunAttempts: 2,
    maxEvidenceBytes: 1048576,
  },
  reservedRunCount: 0,
  submittedRunCount: 0,
  outstandingRunCount: 0,
  retainedEvidenceBytes: 0,
  createdAt: date,
  updatedAt: date,
  startedAt: date,
  finishedAt: date,
};
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
    const origin = new URL(String(testInfo.project.use.baseURL)).origin;
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    let fail = true;
    const cursors: string[] = [];
    await page.route("**/runtime-config.json", (route) =>
      route.fulfill({
        json: {
          uiVersion: packageMetadata.version,
          supportedApiVersions: ["contractor.public.v1"],
          apiBaseUrl: origin,
        },
      }),
    );
    await page.route(`${origin}/v1/**`, async (route) => {
      const url = new URL(route.request().url());
      let value: unknown = {
        items: [],
        page: { hasMore: false },
        total: 0,
        auditRevision: 51,
        asOf: date,
      };
      if (url.pathname === "/v1/auth/session")
        value = {
          principal: {
            userId: "owner",
            username: "owner",
            capabilities: ["user"],
          },
          csrfToken: "a".repeat(43),
          idleExpiresAt: "2099-01-01T00:00:00Z",
          absoluteExpiresAt: "2099-01-02T00:00:00Z",
        };
      else if (url.pathname === "/v1/projects/project_history")
        value = {
          projectId: "project_history",
          kind: "project",
          name: "History project",
          description: "",
          lifecycle: "active",
          revision: "1",
          createdAt: date,
          updatedAt: date,
        };
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
      else if (url.pathname.endsWith("/events")) {
        const cursor = url.searchParams.get("cursor");
        cursors.push(cursor ?? "head");
        if (cursor !== null && fail) {
          await route.fulfill({
            status: 500,
            json: {
              code: "internal_error",
              message: "Unavailable",
              requestId: "history-request",
            },
            headers: { "x-contractor-api-version": "contractor.public.v1" },
          });
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
      await route.fulfill({
        json: value,
        headers: {
          "x-contractor-api-version": "contractor.public.v1",
          ETag:
            url.pathname === "/v1/projects/project_history" ? '"1"' : '"51"',
        },
      });
    });
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
