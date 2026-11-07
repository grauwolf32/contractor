import { expect, test } from "@playwright/test";

import {
  audit,
  auditInput,
  project,
  finding,
  findingSummary,
} from "../src/test/builders";
import { ownerListResponse } from "../src/test/owner-lists";
import {
  authSession,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  fulfillResponse,
  mockAPI,
} from "./support/mock-api";

const date = "2026-10-07T10:00:00Z";
const projects = Array.from({ length: 51 }, (_, i) =>
  project(`project_${i}`, {
    name: `Project ${i + 1}`,
    createdAt: date,
    updatedAt: date,
  }),
);
const checks = Array.from({ length: 201 }, (_, i) =>
  audit("project_50", `audit_${i}`, "draft", {
    inputs: { source: auditInput("owner-source") },
    createdAt: date,
    updatedAt: date,
  }),
);

test("Inbox badge describes a failed findings continuation as a lower bound", async ({
  page,
}) => {
  const owner = project("project_partial");
  const issues = [
    findingSummary(finding(owner.projectId, "finding_one")),
    findingSummary(finding(owner.projectId, "finding_two")),
  ];
  await mockAPI(
    page,
    async (route, _request, url) => {
      if (url.pathname === "/v1/projects")
        return fulfillJSON(route, { items: [owner], page: { hasMore: false } });
      if (
        url.pathname === "/v1/findings" &&
        url.searchParams.get("verdict") === "unreviewed"
      ) {
        if (url.searchParams.has("cursor"))
          return fulfillJSON(
            route,
            {
              code: "storage_transaction_conflict",
              message: "Later findings unavailable",
              retryable: true,
            },
            503,
          );
        return fulfillJSON(route, {
          items: issues,
          page: { hasMore: true, nextCursor: "later" },
        });
      }
      return fulfillEmptyPageOrNotFound(route);
    },
    { session: authSession("owner", ["user"]) },
  );
  await page.goto("/checks");
  await expect(
    page.getByRole("link", { name: "Inbox (At least 2 need your decision)" }),
  ).toBeVisible();
});

for (const width of [1440, 390]) {
  test(`owner lists follow every page at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const requests: URL[] = [];
    await mockAPI(
      page,
      async (route, _request, url) => {
        requests.push(url);
        const owner = ownerListResponse(url, { projects, audits: checks });
        if (owner !== undefined) return fulfillResponse(route, owner);
        if (url.pathname !== "/v1/projects")
          return fulfillEmptyPageOrNotFound(route);
        const start = Number(url.searchParams.get("cursor") ?? 0);
        const limit = Number(url.searchParams.get("limit") ?? 50);
        const hasMore = start + limit < projects.length;
        return fulfillJSON(route, {
          items: projects.slice(start, start + limit),
          page: {
            hasMore,
            ...(hasMore ? { nextCursor: String(start + limit) } : {}),
          },
        });
      },
      { session: authSession("owner", ["user"]) },
    );
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
