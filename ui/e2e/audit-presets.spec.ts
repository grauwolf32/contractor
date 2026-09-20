import { expect, test } from "@playwright/test";

import packageMetadata from "../package.json" with { type: "json" };
import {
  dynamicPresetFixture,
  newerPresetFixture,
  presetFixture,
  standardFixture,
} from "../src/test/audit-presets-fixture";

test("Audit presets expose selected checks and remain usable on mobile", async ({
  page,
}, testInfo) => {
  const origin = new URL(String(testInfo.project.use.baseURL)).origin;
  const profiles = [presetFixture, newerPresetFixture, dynamicPresetFixture];
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
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
    const headers: Record<string, string> = {
      "X-Contractor-API-Version": "contractor.public.v1",
    };
    let value: unknown = { items: [], page: { hasMore: false } };
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
    else if (url.pathname === "/v1/audit-profiles")
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
      value = { apiVersion: "contractor/v1alpha1", standard: standardFixture };
    await route.fulfill({ json: value, headers });
  });

  await page.goto("/catalog/audit-presets");
  await expect(
    page.getByRole("heading", { name: "Audit presets", exact: true }),
  ).toBeVisible();
  await expect(page.getByLabel("Version of source-review")).toHaveValue("10");
  await page.getByLabel("Search audit presets").fill("source review");
  await expect(page.getByRole("article")).toHaveCount(1);
  await page.getByLabel("Version of source-review").selectOption("1");
  await page.screenshot({
    path: testInfo.outputPath("audit-presets-desktop.png"),
    fullPage: true,
  });
  await page.getByRole("link", { name: "View checks →" }).click();
  await expect(page.getByText("2 checks", { exact: true })).toBeVisible();
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
  await expect(page.getByLabel("Search checks")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("audit-preset-checks-mobile.png"),
    fullPage: true,
  });
  await page.getByLabel("Search checks").fill("trust boundary");
  await expect(page.getByText("1 of 2 checks", { exact: true })).toBeVisible();
  await expect(
    page.getByText("Check authorization", { exact: true }),
  ).toHaveCount(0);
  await page.reload();
  await expect(page.getByLabel("Search checks")).toHaveValue("trust boundary");
  await expect(page.getByText("1 of 2 checks", { exact: true })).toBeVisible();
  await page.getByLabel("Preset version").selectOption("10");
  await expect(page.getByText("3 checks", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "← Audit presets" }).click();
  await expect(page.getByLabel("Search audit presets")).toHaveValue(
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
