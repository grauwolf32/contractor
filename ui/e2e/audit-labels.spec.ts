import { expect, test } from "@playwright/test";
import {
  top10,
  sourceZip,
  projectFixture,
  draftAudit,
} from "../src/routes/checks/start/test-fixtures";
import {
  authSession,
  fulfillError,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

for (const theme of ["light", "dark", "black"] as const) {
  test(`Audit labels retain cloud selection at 390px in ${theme}`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    let labels: Record<string, string> = {};
    let startAttempts = 0;
    await mockAPI(
      page,
      async (route, request, url) => {
        let value: unknown = { items: [], page: { hasMore: false } };
        const headers: Record<string, string> = {};
        if (url.pathname === "/v1/projects/project_shop") {
          value = projectFixture();
          headers.ETag = '"1"';
        } else if (url.pathname === "/v1/audit-profiles")
          value = { items: [top10], page: { hasMore: false } };
        else if (url.pathname.startsWith("/v1/audit-profiles/")) {
          value = top10;
          headers.ETag = `"${top10.ref.digest}"`;
        } else if (url.pathname === "/v1/projects/project_shop/artifacts")
          value = { items: [sourceZip], page: { hasMore: false } };
        else if (
          url.pathname === "/v1/projects/project_shop/audits" &&
          request.method() === "POST"
        ) {
          labels = (
            request.postDataJSON() as { labels: Record<string, string> }
          ).labels;
          await fulfillJSON(route, draftAudit(top10, { labels }), 201, {
            ETag: '"1"',
          });
          return;
        } else if (url.pathname === "/v1/audits/audit_new/start") {
          startAttempts++;
          await fulfillError(
            route,
            503,
            "storage_transaction_conflict",
            "Please retry",
          );
          return;
        } else if (url.pathname === "/v1/audits/audit_new") {
          value = draftAudit(top10, { labels });
          headers.ETag = '"1"';
        }
        await fulfillJSON(route, value, 200, headers);
      },
      { session: authSession("owner", ["user"]) },
    );
    await page.goto(
      "/checks/new?project=project_shop&type=owasp-top10-2025-source-risk",
    );
    await page
      .getByLabel("Your objective, in your own words")
      .fill("Review authentication");
    await page.getByText("Advanced options", { exact: true }).click();
    await page.getByRole("button", { name: "+ cloud", exact: true }).click();
    await page.getByLabel("Label value 1", { exact: true }).fill("aws");
    await page
      .getByRole("button", { name: "+ Add label", exact: true })
      .click();
    await page.getByLabel("Label key 2", { exact: true }).fill("team");
    await page.getByLabel("Label value 2", { exact: true }).fill("appsec");
    const start = page.getByRole("button", {
      name: "Start check",
      exact: true,
    });
    await expect(start).toBeEnabled();
    await expect(
      page.getByLabel("Execution location", { exact: true }),
    ).toHaveCount(0);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page.locator(".start-advanced").screenshot({
      path: testInfo.outputPath(`audit-labels-${theme}-390.png`),
    });
    await start.click();
    await expect.poll(() => startAttempts).toBeGreaterThan(0);
    expect(labels).toEqual({ cloud: "aws", team: "appsec" });
    await expect(page.getByLabel("Label value 1", { exact: true })).toHaveValue(
      "aws",
    );
    await page.getByRole("button", { name: "+ local", exact: true }).click();
    await expect(
      page.getByRole("alert").filter({
        hasText: "Choose cloud or local; both cannot apply to one launch.",
      }),
    ).toBeVisible();
    await expect(start).toBeDisabled();
    await page
      .getByRole("button", { name: "Remove label 3", exact: true })
      .click();
    await expect(start).toBeEnabled();
    await page
      .getByRole("button", { name: "+ Add label", exact: true })
      .click();
    await expect(page.getByLabel("Label key 3", { exact: true })).toBeFocused();
    await expect(start).toBeDisabled();
    await page
      .getByRole("button", { name: "Remove label 3", exact: true })
      .click();
    await expect(start).toBeEnabled();
    expect(errors).toEqual([]);
  });
}
