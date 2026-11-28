import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
const require = createRequire(
  new URL("../../../../../ui/package.json", import.meta.url),
);
const { expect, test } = require("@playwright/test");
const root = fileURLToPath(new URL(".", import.meta.url));
test.beforeEach(async ({ page }) => {
  page.on("pageerror", (error) => {
    throw error;
  });
  await page.route("**/*", (route) =>
    new URL(route.request().url()).hostname === "127.0.0.1"
      ? route.continue()
      : route.abort(),
  );
});
for (const theme of ["dark", "light", "black"])
  for (const width of [1440, 1024, 390]) {
    test(`Materials and chooser · ${theme} · ${width}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(`/prototype.html?theme=${theme}&screen=materials`);
      await expect(
        page.getByRole("heading", { name: "Materials", exact: true }),
      ).toBeVisible();
      expect(await page.locator("#picker").isVisible()).toBe(false);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await page.screenshot({
        path: `${root}/review/materials-${theme}-${width}.png`,
        fullPage: true,
      });
      await page
        .getByRole("searchbox", { name: "Search materials" })
        .fill("invoice-worker");
      await expect(
        page.getByRole("button", { name: "invoice-worker", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("button", { name: "invoice-service", exact: true }),
      ).toHaveCount(0);
      await page
        .getByRole("button", { name: "invoice-worker", exact: true })
        .click();
      await expect(
        page.getByRole("complementary", { name: "Preview invoice-worker" }),
      ).toBeVisible();
      if (width <= 820) {
        await page.getByRole("button", { name: "Back to materials" }).click();
        await expect(page.getByRole("searchbox")).toHaveValue("invoice-worker");
      }
      await page.goto(`/prototype.html?theme=${theme}&screen=check`);
      await page
        .getByRole("button", { name: "Change", exact: true })
        .first()
        .click();
      const picker = page.getByRole("dialog", { name: "Choose source code" });
      await expect(picker).toBeVisible();
      await picker
        .getByRole("button", { name: "invoice-worker", exact: true })
        .click();
      await expect(
        picker.getByRole("complementary", { name: "Preview invoice-worker" }),
      ).toBeVisible();
      await expect(
        picker.getByRole("button", { name: "Attach material" }),
      ).toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      expect(
        await picker.evaluate((el) => el.scrollWidth <= el.clientWidth),
      ).toBe(true);
      await page.screenshot({
        path: `${root}/review/chooser-${theme}-${width}.png`,
      });
      await picker.getByRole("button", { name: "Attach material" }).click();
      await expect(picker).not.toBeVisible();
      await expect(page.locator(".attached").first()).toContainText(
        "invoice-worker",
      );
      await expect(page.locator("#objective")).toHaveValue(
        "Can one customer read or export another customer’s invoices?",
      );
      await page.screenshot({
        path: `${root}/review/check-${theme}-${width}.png`,
        fullPage: true,
      });
    });
  }
for (const width of [1440, 390]) {
  test(`Chooser cancel, upload and check context · ${width}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/prototype.html?screen=check&theme=dark");
    await page.locator("#objective").fill("Review the worker export boundary");
    await page.locator("#scope").fill("Exports only, customer A and B");
    await page.locator("#time-limit").selectOption("7 days");
    await page
      .getByRole("button", { name: "Change", exact: true })
      .first()
      .click();
    const picker = page.getByRole("dialog", { name: "Choose source code" });
    await picker
      .getByRole("button", { name: "invoice-worker", exact: true })
      .click();
    await picker.getByRole("button", { name: "Cancel", exact: true }).click();
    await expect(page.locator(".attached").first()).toContainText(
      "invoice-service",
    );
    await page
      .getByRole("button", { name: "Change", exact: true })
      .first()
      .click();
    await picker.getByRole("searchbox").fill("worker");
    await expect(
      picker.getByRole("button", { name: "Attach material" }),
    ).toBeDisabled();
    await expect(
      picker.getByRole("button", { name: "invoice-worker", exact: true }),
    ).toBeVisible();
    await picker
      .getByRole("button", { name: "Add material", exact: true })
      .click();
    const upload = page.getByRole("dialog", {
      name: "Add material",
      exact: true,
    });
    await upload
      .locator("#upload-file")
      .setInputFiles({
        name: "worker-notes.txt",
        mimeType: "text/plain",
        buffer: Buffer.from("fixture"),
      });
    await upload.getByRole("button", { name: "Add and attach" }).click();
    await expect(upload.getByRole("alert")).toHaveText(
      "Choose a ZIP archive for source code.",
    );
    await upload
      .locator("#upload-file")
      .setInputFiles({
        name: "worker-new.zip",
        mimeType: "application/zip",
        buffer: Buffer.from("synthetic archive placeholder"),
      });
    await upload.getByRole("button", { name: "Add and attach" }).click();
    await expect(upload).not.toBeVisible();
    await expect(picker).not.toBeVisible();
    await expect(page.locator(".attached").first()).toContainText("worker-new");
    await expect(page.locator("#objective")).toHaveValue(
      "Review the worker export boundary",
    );
    await expect(page.locator("#scope")).toHaveValue(
      "Exports only, customer A and B",
    );
    await expect(page.locator("#time-limit")).toHaveValue("7 days");
    if (width <= 820)
      await page.locator("#mobile-check-type").selectOption("active");
    else await page.getByRole("button", { name: /Active HTTP checks/ }).click();
    await expect(
      page.getByRole("button", { name: "Start check", exact: true }),
    ).toBeDisabled();
    await page.getByRole("button", { name: "Choose", exact: true }).click();
    const engagement = page.getByRole("dialog", { name: "Choose engagement" });
    await engagement
      .getByRole("button", { name: "staging-access", exact: true })
      .click();
    await engagement.getByRole("button", { name: "Attach material" }).click();
    await expect(
      page.getByRole("button", { name: "Start check", exact: true }),
    ).toBeEnabled();
  });
}
test("Loading, empty, error and full search recovery", async ({ page }) => {
  for (const state of ["loading", "empty", "error"]) {
    await page.goto(
      `/prototype.html?screen=materials&state=${state}&theme=dark`,
    );
    await expect(
      page.getByRole("button", { name: "invoice-service", exact: true }),
    ).toHaveCount(0);
    if (state === "error") {
      await page
        .getByRole("button", { name: "Try again", exact: true })
        .click();
      await expect(
        page.getByRole("button", { name: "invoice-service", exact: true }),
      ).toBeVisible();
    }
  }
});
test("Details, exact copy and preview tabs", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/prototype.html?screen=materials&theme=dark");
  const reader = page.getByRole("complementary", {
    name: "Preview invoice-api.yaml",
  });
  await reader
    .getByRole("button", { name: "Copy material identifier" })
    .click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
    "openapi/invoice-api.yaml@api-r4",
  );
  await reader.getByRole("button", { name: "Raw", exact: true }).click();
  await expect(reader.locator("pre")).toContainText("openapi: 3.1.0");
  await reader.getByRole("button", { name: "details →", exact: true }).click();
  const detail = page.getByRole("dialog", { name: "Material details" });
  await expect(detail).toContainText("openapi/invoice-api.yaml@api-r4");
  await detail.getByRole("button", { name: "Done", exact: true }).click();
  await expect(detail).not.toBeVisible();
});
test("Review host preserves the selected viewport and opens both journeys", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1180, height: 1000 });
  await page.goto("/?screen=materials&theme=dark&width=1440&state=normal");
  const app = page.frameLocator("#preview");
  await expect(
    app.getByRole("heading", { name: "Materials", exact: true }),
  ).toBeVisible();
  expect(await app.locator("body").evaluate(() => innerWidth)).toBe(1440);
  await page.locator("#width").selectOption("390");
  await expect(
    app.getByRole("heading", { name: "Materials", exact: true }),
  ).toBeVisible();
  expect(await app.locator("body").evaluate(() => innerWidth)).toBe(390);
  await page.locator("#state").selectOption("empty");
  await expect(
    app.getByRole("heading", { name: "No materials in this view" }),
  ).toBeVisible();
  await page.locator("#screen").selectOption("check");
  await app
    .getByRole("button", { name: "Change", exact: true })
    .first()
    .click();
  await app
    .getByRole("dialog", { name: "Choose source code" })
    .getByRole("button", { name: "invoice-service", exact: true })
    .click();
  await expect(
    app.getByRole("complementary", { name: "Preview invoice-service" }),
  ).toBeVisible();
});
test("Secondary categories and unmatched search keep preview in scope", async ({
  page,
}) => {
  await page.goto("/prototype.html?screen=materials&theme=dark");
  await page
    .getByRole("combobox", { name: "More material categories" })
    .selectOption("diffs");
  await expect(
    page.getByRole("button", { name: "export-permissions.patch", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("complementary", {
      name: "Preview export-permissions.patch",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "invoice-api.yaml", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("searchbox").fill("missing fixture");
  await expect(
    page.getByRole("heading", { name: "No matching materials" }),
  ).toBeVisible();
  await expect(
    page.getByRole("complementary", {
      name: "Preview export-permissions.patch",
    }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Clear filters" }).click();
  await expect(
    page.getByRole("button", { name: "invoice-api.yaml", exact: true }),
  ).toBeVisible();
});
