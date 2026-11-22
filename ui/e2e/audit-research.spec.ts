import { expect, test } from "@playwright/test";
import { audit as auditBuilder } from "../src/test/builders";
import {
  RESEARCH_AUDIT,
  RESEARCH_PROJECT,
  researchFixture,
} from "../src/test/audit-research-fixtures";
import {
  authSession,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

test("an off-page selected Task receives its final collected result", async ({
  page,
}) => {
  const fixture = researchFixture("active");
  const collected = fixture.tasks[0]!;
  const ready = { ...collected, state: "ready" as const };
  delete ready.result;
  let settled = false;
  let selectedReads = 0;
  await page.clock.install();
  await mockAPI(page, async (route, request, url) => {
    if (url.pathname === `/v1/audits/${RESEARCH_AUDIT}/tasks`) {
      await fulfillJSON(route, {
        items: fixture.tasks.slice(1),
        page: { hasMore: false },
      });
      return;
    }
    if (
      url.pathname ===
      `/v1/audits/${RESEARCH_AUDIT}/tasks/${collected.document.task_id}`
    ) {
      selectedReads++;
      await fulfillJSON(route, settled ? collected : ready);
      return;
    }
    const response = await fixture.respond(
      new Request(url, { method: request.method() }),
    );
    if (!response) return fulfillEmptyPageOrNotFound(route);
    const headers: Record<string, string> = {};
    response.headers.forEach((value, name) => {
      headers[name] = value;
    });
    await fulfillJSON(route, await response.json(), response.status, headers);
  });
  await page.goto(
    `/projects/${RESEARCH_PROJECT}/audits/${RESEARCH_AUDIT}/work?task=${collected.document.task_id}`,
  );
  await expect(page.getByText("No accepted conclusion yet.")).toBeVisible();
  expect(selectedReads).toBe(1);
  settled = true;
  Object.assign(fixture.audit, {
    state: "completed",
    revision: fixture.audit.revision + 1,
    outstandingRunCount: 0,
    collectionPending: false,
  });
  await page.clock.fastForward(5000);
  await expect(
    page.getByRole("button", { name: "Read saved result" }),
  ).toBeVisible();
  await expect(page.getByText("No accepted conclusion yet.")).not.toBeVisible();
  expect(selectedReads).toBeGreaterThan(1);
});

test("studio live shows common coordination and links to retained tasks", async ({
  page,
}) => {
  const audit = auditBuilder("project_studio", "audit_studio", "completed", {
    taskCount: 7,
  });
  const mutations: string[] = [];
  await mockAPI(
    page,
    async (route, request, url) => {
      if (request.method() !== "GET") mutations.push(request.method());
      await fulfillJSON(
        route,
        url.pathname === "/v1/audits/audit_studio"
          ? audit
          : { items: [], page: { hasMore: false } },
        200,
        { ETag: `"${audit.revision}"` },
      );
    },
    { session: authSession("owner", ["user"]) },
  );
  await page.goto("/catalog/studio?audit=audit_studio");
  await expect(page.getByText("Accepted tasks", { exact: true })).toBeVisible();
  await expect(page.getByText("7", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("link", { name: "Open tasks and saved results" }),
  ).toHaveAttribute("href", "/projects/project_studio/audits/audit_studio");
  for (const theme of ["light", "dark", "black"]) {
    await page.evaluate((theme) => {
      document.documentElement.dataset.theme = theme;
    }, theme);
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
    }
  }
  expect(mutations).toEqual([]);
});
