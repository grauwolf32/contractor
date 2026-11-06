import { expect, test } from "@playwright/test";
import { audit as auditBuilder } from "../src/test/builders";
import { authSession, fulfillJSON, mockAPI } from "./support/mock-api";

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
