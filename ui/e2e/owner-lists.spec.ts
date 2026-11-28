import { expect, test } from "@playwright/test";

import {
  audit,
  auditInput,
  project,
  finding,
  findingSummary,
  report,
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

for (const theme of ["light", "dark", "black"])
  for (const width of [1440, 390]) {
    test(`Check deletion stays visible in both views · ${theme} · ${width}`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.addInitScript(
        (value) => localStorage.setItem("contractor.theme", value),
        theme,
      );
      const owner = project("project_delete", { name: "Delete review" });
      const selected = audit(owner.projectId, "audit_delete", "draft");
      let deletions = 0;
      await mockAPI(page, async (route, request, url) => {
        if (request.method() === "DELETE") deletions++;
        const list = ownerListResponse(url, {
          projects: [owner],
          audits: [selected],
        });
        if (list) return fulfillResponse(route, list);
        if (url.pathname === "/v1/projects")
          return fulfillJSON(route, {
            items: [owner],
            page: { hasMore: false },
          });
        if (url.pathname === `/v1/projects/${owner.projectId}`)
          return fulfillJSON(route, owner, 200, { ETag: '"1"' });
        if (url.pathname === `/v1/projects/${owner.projectId}/audits`)
          return fulfillJSON(route, {
            items: [selected],
            page: { hasMore: false },
          });
        if (url.pathname === `/v1/audits/${selected.auditId}`)
          return fulfillJSON(route, selected, 200, {
            ETag: `"${selected.revision}"`,
          });
        return fulfillEmptyPageOrNotFound(route);
      });
      for (const path of [
        `/checks?state=all&check=${selected.auditId}`,
        `/projects/${owner.projectId}/audits/${selected.auditId}`,
      ]) {
        await page.goto(path);
        const deletion = page.getByRole("button", {
          name: "Delete check",
          exact: true,
        });
        await expect(deletion).toBeVisible();
        await expect(
          page.getByLabel("Check actions", { exact: true }),
        ).toHaveCount(0);
        await expect
          .poll(() =>
            page.evaluate(
              () => document.documentElement.scrollWidth <= innerWidth,
            ),
          )
          .toBe(true);
        await deletion.click();
        await expect(
          page.getByRole("alertdialog", { name: "Delete this check?" }),
        ).toBeVisible();
        expect(deletions).toBe(0);
        await page
          .getByRole("button", { name: "Keep the check", exact: true })
          .click();
        expect(deletions).toBe(0);
      }
    });
  }

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
        url.searchParams.get("verdict") === "needs_review"
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
  test(`Reports reads one check page and preserves it on selection at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    const owner = project("project_reports", { name: "Report workspace" });
    const audits = Array.from({ length: 25 }, (_, index) =>
      audit(owner.projectId, `audit_report_${index}`, "completed"),
    );
    const reportReads: string[] = [];
    await mockAPI(
      page,
      async (route, _request, url) => {
        const list = ownerListResponse(url, { projects: [owner], audits });
        if (list) return fulfillResponse(route, list);
        if (url.pathname === "/v1/projects")
          return fulfillJSON(route, {
            items: [owner],
            page: { hasMore: false },
          });
        if (url.pathname === `/v1/projects/${owner.projectId}`)
          return fulfillJSON(route, owner, 200, { ETag: '"1"' });
        const match = /^\/v1\/audits\/([^/]+)(\/report)?$/.exec(url.pathname);
        if (match?.[2]) {
          reportReads.push(match[1]!);
          return fulfillJSON(
            route,
            report(match[1]!, "ready", {
              summary: `Saved content for ${match[1]}`,
            }),
          );
        }
        if (match) {
          const selected = audits.find((check) => check.auditId === match[1]);
          return fulfillJSON(route, selected, 200, {
            ETag: `"${selected!.revision}"`,
          });
        }
        return fulfillEmptyPageOrNotFound(route);
      },
      { session: authSession("owner", ["user"]) },
    );
    await page.goto(`/reports?project=${owner.projectId}`);
    const list = page.getByRole("region", { name: "Reports", exact: true });
    await expect(
      list.getByText("20 ready in this page of checks.", { exact: true }),
    ).toBeVisible();
    await expect(list.locator("li.ui-row")).toHaveCount(20);
    expect(reportReads).toHaveLength(20);
    await list
      .getByRole("navigation", { name: "Report pages" })
      .getByRole("button", { name: "Next" })
      .click();
    await expect(
      list.getByText("5 ready in this page of checks.", { exact: true }),
    ).toBeVisible();
    expect(reportReads).toHaveLength(25);
    await expect(page).toHaveURL(/project=project_reports&page=1$/);
    await list
      .locator(
        'a[href="/reports/audit_report_20?project=project_reports&page=1"]',
      )
      .click();
    await expect(
      page.getByText("Saved content for audit_report_20"),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: "Back to reports", includeHidden: true }),
    ).toHaveAttribute("href", "/reports?project=project_reports&page=1");
    expect(reportReads).toHaveLength(26);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  });

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
    await page.goto("/projects");
    const projectList = page.getByRole("region", {
      name: "Projects",
      exact: true,
    });
    await projectList
      .getByRole("searchbox", { name: "Filter projects" })
      .fill("Project 51");
    await expect(
      projectList.getByRole("link", { name: /Project 51/ }),
    ).toBeVisible();
    await expect(
      projectList.getByText("1 of 51 projects", { exact: true }),
    ).toBeVisible();
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

for (const width of [1440, 390]) {
  test(`Report scope changes close the old report and preserve the return path at ${width}px`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const owners = [
      project("project_invoice", { name: "Invoice Lab" }),
      project("project_payments", { name: "Payment Portal" }),
    ];
    const checks = owners.map((owner, index) =>
      audit(owner.projectId, `audit_scope_${index}`, "completed", {
        scope: {
          objective:
            index === 0
              ? "Review invoice authorization"
              : "Check payment callbacks",
        },
      }),
    );
    await mockAPI(page, async (route, _request, url) => {
      const listed = ownerListResponse(url, {
        projects: owners,
        audits: checks,
      });
      if (listed) return fulfillResponse(route, listed);
      if (url.pathname === "/v1/projects")
        return fulfillJSON(route, { items: owners, page: { hasMore: false } });
      const owner = owners.find(
        (p) => url.pathname === `/v1/projects/${p.projectId}`,
      );
      if (owner) return fulfillJSON(route, owner, 200, { ETag: '"1"' });
      const selected = checks.find((check) =>
        url.pathname.startsWith(`/v1/audits/${check.auditId}`),
      );
      if (selected) {
        if (url.pathname.endsWith("/report"))
          return fulfillJSON(
            route,
            report(selected.auditId, "ready", {
              summary: `Retained result for ${selected.projectId}.`,
            }),
          );
        return fulfillJSON(route, selected, 200, {
          ETag: `"${selected.revision}"`,
        });
      }
      return fulfillEmptyPageOrNotFound(route);
    });
    await page.goto("/reports?project=project_invoice");
    const list = page.getByRole("region", { name: "Reports", exact: true });
    await expect(
      list.getByRole("link", { name: "Review invoice authorization" }),
    ).toBeVisible();
    await expect(
      list.getByRole("group", { name: "Filter by status" }),
    ).toHaveCount(0);
    await list
      .getByRole("link", { name: "Review invoice authorization" })
      .click();
    await expect(
      page.getByText("Retained result for project_invoice.", { exact: true }),
    ).toBeVisible();
    if (width === 390)
      await page.getByRole("link", { name: "Back to reports" }).click();
    await list
      .getByRole("combobox", { name: "Project" })
      .selectOption("project_payments");
    await expect(page).toHaveURL(/\/reports\?project=project_payments$/);
    await expect(
      list.getByRole("link", { name: "Check payment callbacks" }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", {
        name: "Download summary",
        includeHidden: true,
      }),
    ).toHaveCount(0);
    await list.getByRole("link", { name: "Check payment callbacks" }).click();
    await expect(
      page.getByText("Retained result for project_payments.", { exact: true }),
    ).toBeVisible();
    const downloadEvent = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download summary" }).click();
    expect((await downloadEvent).suggestedFilename()).toBe(
      "audit_scope_1-report.md",
    );
    await expect(
      page.getByRole("link", { name: "Back to reports", includeHidden: true }),
    ).toHaveAttribute("href", "/reports?project=project_payments");
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: info.outputPath(`reports-${width}.png`),
      fullPage: true,
    });
  });
}

test("Project search explains a failed continuation and retries the complete inventory", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 900 });
  let unavailable = true;
  const owners = Array.from({ length: 51 }, (_, index) =>
    project(`project_search_${index}`, {
      name: index === 50 ? "Travel Ledger" : `Workspace ${index}`,
    }),
  );
  await mockAPI(page, async (route, _request, url) => {
    if (url.pathname === "/v1/projects") {
      const offset = Number(url.searchParams.get("cursor") ?? 0);
      if (offset > 0 && unavailable)
        return fulfillJSON(
          route,
          {
            code: "storage_transaction_conflict",
            message: "Project continuation unavailable",
            retryable: true,
          },
          503,
        );
      return fulfillJSON(route, {
        items: owners.slice(offset, offset + 50),
        page:
          offset === 0
            ? { hasMore: true, nextCursor: "50" }
            : { hasMore: false },
      });
    }
    const listed = ownerListResponse(url, { projects: owners, audits: [] });
    if (listed) return fulfillResponse(route, listed);
    return fulfillEmptyPageOrNotFound(route);
  });
  await page.goto("/projects");
  const list = page.getByRole("region", { name: "Projects", exact: true });
  await list
    .getByRole("searchbox", { name: "Filter projects" })
    .fill("Travel Ledger");
  await expect(
    list.getByText("No match in loaded projects", { exact: true }),
  ).toBeVisible();
  await expect(
    list.getByText(/Search covers 50 loaded projects/),
  ).toBeVisible();
  unavailable = false;
  await list.getByRole("button", { name: "Retry search" }).click();
  await expect(list.getByRole("link", { name: /Travel Ledger/ })).toBeVisible();
  await expect(list.getByText(/Search covers 50 loaded projects/)).toHaveCount(
    0,
  );
  await list.getByRole("searchbox").clear();
  await expect(list.getByRole("link", { name: /Travel Ledger/ })).toHaveCount(
    0,
  );
  await expect(
    list.getByRole("navigation", { name: "Projects pages" }),
  ).toBeVisible();
});
