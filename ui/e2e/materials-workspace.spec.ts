import { test, expect, type Route } from "@playwright/test";
import { createMaterialsReviewAPI } from "./support/materials-workspace";
import { fulfillResponse, fulfillContent, mockAPI } from "./support/mock-api";

async function fulfillReview(route: Route, response: Response) {
  const request = route.request();
  const url = new URL(request.url());
  if (
    request.method() === "GET" &&
    /\/artifacts\/[^/]+\/[^/]+$/.test(url.pathname) &&
    response.ok
  ) {
    const headers: Record<string, string> = {};
    response.headers.forEach((value, name) => {
      headers[name] = value;
    });
    await fulfillContent(
      route,
      await response.text(),
      response.headers.get("content-type")!,
      response.status,
      headers,
    );
  } else await fulfillResponse(route, response);
}

for (const theme of ["light", "dark", "black"])
  for (const width of [1440, 1024, 390]) {
    test(`Materials and confirmed chooser ${theme} ${width}px`, async ({
      page,
    }, info) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (theme) => localStorage.setItem("contractor.theme", theme),
        theme,
      );
      const api = createMaterialsReviewAPI();
      const reads: string[] = [],
        writes: string[] = [],
        errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await mockAPI(page, async (route, request, url) => {
        (request.method() === "GET" ? reads : writes).push(
          url.pathname + url.search,
        );
        await fulfillReview(
          route,
          await api.fetch(
            new Request(url, {
              method: request.method(),
              headers: request.headers(),
              ...(request.method() === "GET"
                ? {}
                : { body: new Uint8Array(request.postDataBuffer() ?? []) }),
            }),
          ),
        );
      });
      await page.goto("/projects/project_shop/artifacts");
      await expect(
        page.getByRole("heading", { name: "Materials", exact: true }),
      ).toBeVisible();
      await page
        .getByRole("searchbox", { name: "Search materials" })
        .fill("invoice-worker");
      const row = page.getByRole("link", {
        name: "sources/invoice-worker",
        exact: true,
      });
      await expect(row).toHaveAttribute("href", /revision=worker-r3/);
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(width);
      await page.screenshot({
        path: info.outputPath(`materials-${theme}-${width}.png`),
        fullPage: true,
      });
      await row.click();
      await expect(
        page.getByRole("heading", { level: 1, name: "sources/invoice-worker" }),
      ).toBeVisible();
      await expect(
        page.locator('nav[aria-label="Archive files"]'),
      ).toBeAttached();
      await expect(page.locator(".archive-file-content pre")).toContainText(
        "exportInvoices",
      );
      expect(
        reads.some((path) => path.endsWith("/archive?revision=worker-r3")),
      ).toBe(true);
      await expect(
        page.getByRole("button", { name: "Load preview", exact: true }),
      ).toHaveCount(0);
      const reader = await page
        .locator(".materials-detail-reader")
        .boundingBox();
      expect(reader!.width).toBeGreaterThan(
        width * (width <= 820 ? 0.78 : 0.5),
      );
      await page.screenshot({
        path: info.outputPath(`material-${theme}-${width}.png`),
        fullPage: true,
      });
      await page
        .getByRole("link", { name: "← Materials", exact: true })
        .click();
      await expect(
        page.getByRole("searchbox", { name: "Search materials" }),
      ).toHaveValue("invoice-worker");
      await page.goto(
        "/checks/new?project=project_shop&type=owasp-top10-2025-source-risk",
      );
      await page
        .getByRole("textbox", { name: "Your objective, in your own words" })
        .fill("Review exports for customer isolation");
      await page
        .getByLabel("Time limit", { exact: true })
        .selectOption("7 days");
      await page
        .getByRole("button", {
          name: "Choose source code material",
          exact: true,
        })
        .click();
      const picker = page.getByRole("dialog", {
        name: "Choose source code",
        exact: true,
      });
      await expect(picker).toBeVisible();
      await picker.getByRole("searchbox").fill("invoice-worker");
      await picker
        .getByRole("button", {
          name: "sources/invoice-worker@worker-r3",
          exact: true,
        })
        .click();
      await expect(
        picker.getByRole("button", { name: "Attach material", exact: true }),
      ).toBeEnabled();
      await page.screenshot({
        path: info.outputPath(`chooser-${theme}-${width}.png`),
      });
      await picker.getByRole("button", { name: "Cancel", exact: true }).click();
      await expect(
        page.getByRole("button", {
          name: "Choose source code material",
          exact: true,
        }),
      ).toBeVisible();
      await page
        .getByRole("button", {
          name: "Choose source code material",
          exact: true,
        })
        .click();
      await picker
        .getByRole("button", {
          name: "sources/invoice-worker@worker-r3",
          exact: true,
        })
        .click();
      await picker
        .getByRole("button", { name: "Attach material", exact: true })
        .click();
      await expect(
        page.getByRole("button", {
          name: "Change source code material",
          exact: true,
        }),
      ).toBeVisible();
      await expect(
        page.locator(".check-material-attachment").first(),
      ).toContainText("worker-r3");
      await expect(page.getByLabel("Time limit", { exact: true })).toHaveValue(
        "604800",
      );
      await expect(
        page.getByRole("textbox", {
          name: "Your objective, in your own words",
        }),
      ).toHaveValue("Review exports for customer isolation");
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(width);
      expect(writes).toEqual([]);
      expect(errors).toEqual([]);
      await page.screenshot({
        path: info.outputPath(`check-${theme}-${width}.png`),
        fullPage: true,
      });
    });
  }

test("inline upload attaches its saved revision and retains the composer", async ({
  page,
}) => {
  const api = createMaterialsReviewAPI();
  await mockAPI(page, async (route, request, url) =>
    fulfillReview(
      route,
      await api.fetch(
        new Request(url, {
          method: request.method(),
          headers: request.headers(),
          ...(request.method() === "GET"
            ? {}
            : { body: new Uint8Array(request.postDataBuffer() ?? []) }),
        }),
      ),
    ),
  );
  await page.goto(
    "/checks/new?project=project_shop&type=openapi-operation-trace",
  );
  await page
    .getByRole("textbox", { name: "Your objective, in your own words" })
    .fill("Keep this objective while adding a source");
  await page
    .getByRole("button", { name: "Choose source code material", exact: true })
    .click();
  const picker = page.getByRole("dialog", {
    name: "Choose source code",
    exact: true,
  });
  await picker
    .getByRole("button", { name: "Add material", exact: true })
    .click();
  const upload = page.getByRole("dialog", {
    name: "Add source code",
    exact: true,
  });
  await upload.getByLabel("Drop a file here").setInputFiles({
    name: "new-service.zip",
    mimeType: "application/zip",
    buffer: Buffer.from("synthetic ZIP fixture"),
  });
  await upload
    .getByRole("button", { name: "Add and attach", exact: true })
    .click();
  await expect(picker).not.toBeVisible();
  await expect(
    page
      .locator(".check-material-slot")
      .filter({ has: page.getByText("Source code", { exact: true }) }),
  ).toContainText("new-service");
  await expect(
    page.getByRole("textbox", { name: "Your objective, in your own words" }),
  ).toHaveValue("Keep this objective while adding a source");
});

test("returning from a material restores the phone list reading position", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 700 });
  const api = createMaterialsReviewAPI();
  await mockAPI(page, async (route, request, url) =>
    fulfillReview(
      route,
      await api.fetch(
        new Request(url, {
          method: request.method(),
          headers: request.headers(),
          ...(request.method() === "GET"
            ? {}
            : { body: new Uint8Array(request.postDataBuffer() ?? []) }),
        }),
      ),
    ),
  );
  await page.goto("/projects/project_shop/artifacts");
  const row = page.getByRole("link", {
    name: "outputs/discovered-routes.json",
    exact: true,
  });
  await row.scrollIntoViewIfNeeded();
  const readPosition = () =>
    page.locator(".materials-region").evaluate((el) => {
      const pane = el.closest<HTMLElement>(".ui-panes-detail");
      const owner =
        pane !== null &&
        pane.scrollHeight > pane.clientHeight &&
        /auto|scroll/.test(getComputedStyle(pane).overflowY)
          ? pane
          : document.scrollingElement;
      return owner?.scrollTop ?? 0;
    });
  const top = await readPosition();
  expect(top).toBeGreaterThan(100);
  await row.click();
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "outputs/discovered-routes.json",
    }),
  ).toBeVisible();
  await page.getByRole("link", { name: "← Materials", exact: true }).click();
  await expect(row).toBeAttached();
  await expect
    .poll(async () => Math.abs((await readPosition()) - top))
    .toBeLessThan(2);
});
