import { expect, test } from "@playwright/test";

import packageMetadata from "../package.json" with { type: "json" };

const digest = `sha256:${"a".repeat(64)}`;
const date = "2026-10-07T10:00:00Z";
const projects = Array.from({ length: 51 }, (_, i) => ({
  projectId: `project_${i}`,
  kind: "project",
  name: `Project ${i + 1}`,
  description: "",
  lifecycle: "active",
  revision: "1",
  createdAt: date,
  updatedAt: date,
}));
const checks = Array.from({ length: 201 }, (_, i) => ({
  auditId: `audit_${i}`,
  projectId: "project_50",
  profile: { name: "openapi-operation-trace", version: "1", digest },
  inputs: {},
  scope: {},
  runtimeLabels: [],
  state: "draft",
  phase: "not-started",
  revision: 1,
  dispatchState: "closed",
  holdState: "pending",
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
  eventSequence: 1,
  createdAt: date,
  updatedAt: date,
}));

for (const width of [1440, 390]) {
  test(`owner lists follow every page at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    const origin = new URL(String(testInfo.project.use.baseURL)).origin;
    const requests: URL[] = [];
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
      requests.push(url);
      let value: unknown;
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
      else {
        const items =
          url.pathname === "/v1/projects"
            ? projects
            : url.pathname === "/v1/audits"
              ? checks.filter(
                  (check) =>
                    !url.searchParams.has("state") ||
                    url.searchParams
                      .get("state")!
                      .split(",")
                      .includes(check.state),
                )
              : [];
        const start = Number(url.searchParams.get("cursor") ?? 0);
        const limit = Number(url.searchParams.get("limit") ?? 50);
        const hasMore = start + limit < items.length;
        value = {
          items: items.slice(start, start + limit),
          page: {
            hasMore,
            ...(hasMore ? { nextCursor: String(start + limit) } : {}),
          },
        };
      }
      await route.fulfill({
        json: value,
        headers: { "x-contractor-api-version": "contractor.public.v1" },
      });
    });
    await page.goto("/checks");
    const list = page.getByRole("region", { name: "Checks", exact: true });
    await expect(list.locator("li.ui-row")).toHaveCount(201);
    await expect(
      page
        .getByRole("combobox", { name: "Project", exact: true })
        .getByRole("option", { name: "Project 51", exact: true }),
    ).toHaveCount(1);
    await page
      .getByRole("combobox", { name: "Project", exact: true })
      .selectOption("project_50");
    await expect(page).toHaveURL(/project=project_50/);
    await expect(list.locator("li.ui-row")).toHaveCount(201);
    expect(
      requests.some(
        (url) =>
          url.pathname === "/v1/projects" && url.searchParams.has("cursor"),
      ),
    ).toBe(true);
    expect(
      requests.some(
        (url) =>
          url.pathname === "/v1/audits" && url.searchParams.has("cursor"),
      ),
    ).toBe(true);
    expect(
      requests.some((url) =>
        /^\/v1\/projects\/[^/]+\/audits$/.test(url.pathname),
      ),
    ).toBe(false);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  });
}
