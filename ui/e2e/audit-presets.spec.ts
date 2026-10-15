import { expect, test } from "@playwright/test";

import {
  dynamicPresetFixture,
  newerPresetFixture,
  presetFixture,
  standardResponse,
} from "../src/test/audit-presets-fixture";
import { authSession, fulfillJSON, mockAPI } from "./support/mock-api";

test("Check types expose selected requirements and remain usable on mobile", async ({
  page,
}, testInfo) => {
  const profiles = [presetFixture, newerPresetFixture, dynamicPresetFixture];
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await mockAPI(
    page,
    async (route, _request, url) => {
      const headers: Record<string, string> = {};
      let value: unknown = { items: [], page: { hasMore: false } };
      if (url.pathname === "/v1/audit-profiles")
        value = {
          items: url.searchParams.has("cursor")
            ? profiles.slice(1)
            : profiles.slice(0, 1),
          page: url.searchParams.has("cursor")
            ? { hasMore: false }
            : { hasMore: true, nextCursor: "page-2" },
        };
      else if (url.pathname.startsWith("/v1/audit-profiles/")) {
        const profile = profiles.find(
          (item) =>
            url.pathname ===
            `/v1/audit-profiles/${item.ref.name}/versions/${item.ref.version}`,
        )!;
        value = profile;
        headers.ETag = `"${profile.ref.digest}"`;
      } else if (url.pathname.startsWith("/v1/audit-standards/"))
        value = standardResponse();
      await fulfillJSON(route, value, 200, headers);
    },
    { session: authSession("owner", ["user"]) },
  );

  await page.goto("/catalog/audit-presets");
  await expect(
    page.getByRole("heading", { name: "Check types", exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel("Version of source-review")).toHaveValue("10");
  await page.getByLabel("Search check types").fill("source review");
  await expect(page.getByRole("article")).toHaveCount(1);
  await page.getByLabel("Version of source-review").selectOption("1");
  await page.screenshot({
    path: testInfo.outputPath("audit-presets-desktop.png"),
    fullPage: true,
  });
  await page.getByRole("link", { name: "View requirements" }).click();
  await expect(page.getByText("2 requirements", { exact: true })).toBeVisible();
  await expect(page.getByText("Check transport security")).toHaveCount(0);
  await page.getByText("Check authorization", { exact: true }).click();
  await expect(
    page.getByText("Verify authorization before accessing a private record."),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("audit-preset-checks-desktop.png"),
    fullPage: true,
  });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByLabel("Search requirements")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("audit-preset-checks-mobile.png"),
    fullPage: true,
  });
  await page.getByLabel("Search requirements").fill("trust boundary");
  await expect(
    page.getByText("1 of 2 requirements", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Check authorization", { exact: true }),
  ).toHaveCount(0);
  await page.reload();
  await expect(page.getByLabel("Search requirements")).toHaveValue(
    "trust boundary",
  );
  await expect(
    page.getByText("1 of 2 requirements", { exact: true }),
  ).toBeVisible();
  await page
    .getByLabel("Check type version", { exact: true })
    .selectOption("10");
  await expect(page.getByText("3 requirements", { exact: true })).toBeVisible();
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
});
