import { expect, test } from "@playwright/test";

import {
  dynamicPresetFixture,
  newerPresetFixture,
  presetFixture,
} from "../src/test/audit-presets-fixture";
import { authSession, fulfillJSON, mockAPI } from "./support/mock-api";

test("Check types expose exact input and Workflow contracts and retain mobile return context", async ({
  page,
}, testInfo) => {
  const profiles = [presetFixture, newerPresetFixture, dynamicPresetFixture];
  const pageErrors: string[] = [];
  const versionsRead: string[] = [];
  const inventoryCursors: (string | null)[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await mockAPI(
    page,
    async (route, _request, url) => {
      const headers: Record<string, string> = {};
      let value: unknown = { items: [], page: { hasMore: false } };
      if (url.pathname === "/v1/audit-profiles") {
        inventoryCursors.push(url.searchParams.get("cursor"));
        value = {
          items: url.searchParams.has("cursor")
            ? profiles.slice(1)
            : profiles.slice(0, 1),
          page: url.searchParams.has("cursor")
            ? { hasMore: false }
            : { hasMore: true, nextCursor: "page-2" },
        };
      } else if (url.pathname.startsWith("/v1/audit-profiles/")) {
        const profile = profiles.find(
          (item) =>
            url.pathname ===
            `/v1/audit-profiles/${item.ref.name}/versions/${item.ref.version}`,
        )!;
        value = profile;
        versionsRead.push(profile.ref.version);
        headers.ETag = `"${profile.ref.digest}"`;
      }
      await fulfillJSON(route, value, 200, headers);
    },
    { session: authSession("owner", ["user"]) },
  );

  await page.goto("/catalog/audit-presets");
  await expect(
    page.getByRole("heading", { name: "Find your next check", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Search check types").fill("source review");
  await expect(
    page.getByLabel("Check type version", { exact: true }),
  ).toHaveValue("10");
  await expect(page.getByRole("article")).toHaveCount(1);
  await page
    .getByLabel("Check type version", { exact: true })
    .selectOption("1");
  await page.screenshot({
    path: testInfo.outputPath("audit-presets-desktop.png"),
    fullPage: true,
  });
  await expect(page).toHaveURL(
    /\/catalog\/audit-presets\/source-review\/1\?q=source\+review$/,
  );
  await expect(
    page.getByLabel("Check type version", { exact: true }),
  ).toHaveValue("1");
  const inputs = page.getByRole("region", {
    name: "What you'll need",
    exact: true,
  });
  await expect(inputs.getByText("Source code", { exact: true })).toBeVisible();
  await expect(inputs.getByText("Required", { exact: true })).toBeVisible();
  await expect(inputs).toContainText("ZIP");
  const workflows = page.getByRole("region", {
    name: "Workflows",
    exact: true,
  });
  await expect(
    workflows.getByRole("link", { name: /source-check@3/ }),
  ).toHaveAttribute("href", "/catalog/workflows/source-check/3");
  await expect(workflows).toContainText("Role check");
  await expect(
    page.getByRole("link", { name: "Set up check" }),
  ).toHaveAttribute("href", "/checks/new?type=source-review");
  await page.screenshot({
    path: testInfo.outputPath("audit-preset-details-desktop.png"),
    fullPage: true,
  });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(inputs).toBeVisible();
  await expect(workflows).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("audit-preset-details-mobile.png"),
    fullPage: true,
  });
  await page.reload();
  await expect(
    page.getByLabel("Check type version", { exact: true }),
  ).toHaveValue("1");
  await expect(inputs).toContainText("ZIP");
  await page
    .getByLabel("Check type version", { exact: true })
    .selectOption("10");
  await expect(page).toHaveURL(
    /\/catalog\/audit-presets\/source-review\/10\?q=source\+review$/,
  );
  await expect(
    page.getByLabel("Check type version", { exact: true }),
  ).toHaveValue("10");
  await expect(
    workflows.getByRole("link", { name: /source-check@3/ }),
  ).toBeVisible();
  await page
    .getByRole("link", { name: "All check types", exact: true })
    .click();
  await expect(page.getByLabel("Search check types")).toHaveValue(
    "source review",
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("audit-presets-mobile.png"),
    fullPage: true,
  });
  expect(pageErrors).toEqual([]);
  expect(inventoryCursors).toContain(null);
  expect(inventoryCursors).toContain("page-2");
  expect(versionsRead).toContain("1");
  expect(versionsRead).toContain("10");
});
