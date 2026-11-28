import { test, expect, type Page } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import fixture from "../src/test/fixtures/engagement-preview.json" with { type: "json" };
import { artifactMetadata, project } from "../src/test/builders";
import {
  ENGAGEMENT_MEDIA_TYPE,
  MAXIMUM_PREVIEW_BYTES,
} from "../src/api/artifacts";
import {
  mockAPI,
  fulfillJSON,
  fulfillContent,
  fulfillError,
  uiOrigin,
} from "./support/mock-api";

const path =
  "/projects/project_shop/artifacts/engagement/staging?revision=engagement-r1";
async function install(page: Page, source = JSON.stringify(fixture)) {
  const writes: string[] = [],
    credentials: string[] = [];
  await mockAPI(page, async (route, request, url) => {
    if (url.pathname.includes("credentials")) credentials.push(url.pathname);
    if (request.method() !== "GET") {
      writes.push(url.pathname);
      return fulfillError(route, 403, "forbidden");
    }
    if (url.pathname === "/v1/projects/project_shop")
      return fulfillJSON(
        route,
        project("project_shop", { name: "Shop API" }),
        200,
        { etag: '"1"' },
      );
    if (
      [
        "/v1/projects",
        "/v1/audits",
        "/v1/findings",
        "/v1/reviews",
        "/v1/projects/project_shop/artifacts",
      ].includes(url.pathname)
    )
      return fulfillJSON(route, { items: [], page: { hasMore: false } });
    const metadata = artifactMetadata("engagement", "staging", {
      artifact: {
        namespace: "engagement",
        name: "staging",
        revision: "engagement-r1",
      },
      mediaType: ENGAGEMENT_MEDIA_TYPE,
      size: Buffer.byteLength(source),
      canUploadNewVersion: false,
    });
    if (url.pathname.endsWith("/engagement/staging/metadata"))
      return fulfillJSON(route, metadata);
    if (url.pathname.endsWith("/engagement/staging/versions"))
      return fulfillJSON(route, {
        items: [metadata],
        page: { hasMore: false },
      });
    if (url.pathname.endsWith("/engagement/staging/lineage"))
      return fulfillJSON(route, { items: [], page: { hasMore: false } });
    if (url.pathname.endsWith("/engagement/staging")) {
      expect(url.searchParams.get("revision")).toBe("engagement-r1");
      return fulfillContent(route, source, ENGAGEMENT_MEDIA_TYPE);
    }
    return fulfillError(route, 404, "not_found");
  });
  return { writes, credentials };
}
async function load(page: Page) {
  await page.goto(path);
  return page.getByRole("article", { name: "Saved engagement", exact: true });
}
async function browse(page: Page, width: number) {
  if (width <= 820)
    await page.getByRole("button", { name: /Browse routes/ }).click();
  return page.getByRole("navigation", {
    name: "Engagement routes",
    exact: true,
  });
}
for (const theme of ["light", "dark", "black"])
  for (const width of [1440, 390]) {
    test(`engagement choices and preview ${theme} ${width}px`, async ({
      page,
    }, info) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (theme) => localStorage.setItem("contractor.theme", theme),
        theme,
      );
      const { writes, credentials } = await install(page);
      const errors: string[] = [],
        external: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      page.on("request", (request) => {
        if (
          ![
            uiOrigin(info),
            "http://127.0.0.3:8080",
            "http://127.0.0.1:1",
          ].includes(new URL(request.url()).origin)
        )
          external.push(request.url());
      });
      await page.goto("/projects/project_shop/artifacts?add=artifact");
      const choices = page.getByRole("dialog", {
        name: "Add material",
        exact: true,
      });
      const choice = choices.getByRole("button", {
        name: "Engagement",
        exact: true,
      });
      await expect(choice.locator("svg")).toHaveCount(1);
      await choice.click();
      const upload = page.getByRole("dialog", {
        name: "Engagement",
        exact: true,
      });
      await expect(upload.getByLabel("Namespace", { exact: true })).toHaveValue(
        "engagement",
      );
      await expect(
        upload.getByLabel("Media type", { exact: true }),
      ).toHaveValue(ENGAGEMENT_MEDIA_TYPE);
      await upload
        .getByRole("button", { name: "Close upload dialog", exact: true })
        .click();
      await expect(choice).toBeFocused();
      const preview = await load(page);
      await expect(
        preview.getByText("All HTTP methods", { exact: true }),
      ).toBeVisible();
      const nav = await browse(page, width);
      await nav.getByRole("button").filter({ hasText: "Exclude" }).click();
      await expect(
        preview.getByRole("article", {
          name: "Selected egress route",
          exact: true,
        }),
      ).toContainText("/v1/payments/live");
      await expect(
        preview.getByText("Exclusion", { exact: true }),
      ).toBeVisible();
      await preview
        .getByRole("tab", { name: "Identities", exact: true })
        .click();
      for (const id of ["anonymous", "reader", "analyst", "admin"])
        await expect(
          preview.getByRole("heading", { name: id, exact: true }),
        ).toBeVisible();
      for (const kind of [
        "target-basic@1",
        "target-bearer@1",
        "target-cookies@1",
      ])
        await expect(preview.getByText(kind, { exact: true })).toBeVisible();
      await expect(
        preview.getByText("Administrator", { exact: true }),
      ).toBeVisible();
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(width);
      await preview.getByRole("tab", { name: "Limits", exact: true }).click();
      const limits = preview.getByRole("region", {
        name: "Saved HTTP limits",
        exact: true,
      });
      await expect(
        limits.getByText("Requests per second", { exact: true }),
      ).toBeVisible();
      await expect(limits.getByText("5", { exact: true })).toBeVisible();
      await expect(
        limits.getByText("Concurrent requests", { exact: true }),
      ).toBeVisible();
      await expect(limits.getByText("3", { exact: true })).toBeVisible();
      await expect(
        preview.getByText("65,536 bytes", { exact: true }),
      ).toBeVisible();
      await preview.getByRole("tab", { name: "Egress", exact: true }).click();
      await preview
        .getByLabel("Find a route", { exact: true })
        .fill("no-such-route");
      await expect(
        preview.getByText("No matching route", { exact: true }),
      ).toBeVisible();
      await preview.getByLabel("Find a route", { exact: true }).fill("");
      await preview
        .getByRole("combobox", { name: "Show", exact: true })
        .selectOption("allowed");
      await expect(
        preview.getByText("Allowed route", { exact: true }),
      ).toBeVisible();
      if (process.env.CONTRACTOR_ENGAGEMENT_CAPTURE_DIR) {
        await mkdir(process.env.CONTRACTOR_ENGAGEMENT_CAPTURE_DIR, {
          recursive: true,
        });
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({
          path: join(
            process.env.CONTRACTOR_ENGAGEMENT_CAPTURE_DIR,
            `engagement-${theme}-${width}.png`,
          ),
          fullPage: true,
        });
      }
      await page.getByRole("tab", { name: "Source", exact: true }).click();
      await expect(
        page.getByRole("tabpanel", { name: "Source", exact: true }),
      ).toContainText('"contractor.engagement.v1"');
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(width);
      expect(writes).toEqual([]);
      expect(credentials).toEqual([]);
      expect(external).toEqual([]);
      expect(errors).toEqual([]);
    });
  }

test("optional limits stay absent and invalid/oversized policies retain Source/download", async ({
  page,
}) => {
  await install(
    page,
    JSON.stringify({ ...fixture, limits: { requestsPerSecond: 7 } }),
  );
  const preview = await load(page);
  await preview.getByRole("tab", { name: "Limits", exact: true }).click();
  await expect(preview.getByText("Not set", { exact: true })).toHaveCount(3);
  await expect(
    preview.getByRole("region", { name: "Saved HTTP limits", exact: true }),
  ).toContainText("7");
  for (const source of [
    "{",
    JSON.stringify({ ...fixture, schema: "contractor.engagement.v2" }),
    '{"schema":"contractor.engagement.v1","schema":"contractor.engagement.v1"}',
  ]) {
    await install(page, source);
    await page.goto(path);
    await expect(page.getByRole("alert")).toContainText(
      "invalid or unsupported",
    );
    await page.getByRole("tab", { name: "Source", exact: true }).click();
    await expect(
      page.getByRole("tabpanel", { name: "Source", exact: true }),
    ).toContainText(source);
  }
  await install(
    page,
    JSON.stringify(fixture).padEnd(MAXIMUM_PREVIEW_BYTES + 1, " "),
  );
  await page.goto(path);
  await expect(
    page.getByRole("button", { name: "Load preview", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Download this revision", exact: true }),
  ).toBeEnabled();
});

test("all route and nested pages remain reachable with stable keyboard selection", async ({
  page,
}) => {
  const origin = `https://${"a".repeat(63)}.${"b".repeat(63)}.test`;
  const route = { ...fixture.egress.routes[0]!, origin };
  const routes = Array.from({ length: 128 }, (_, i) => ({
    ...route,
    pathPrefixes:
      i === 0
        ? Array.from(
            { length: 128 },
            (_, p) => `/path-${String(p).padStart(3, "0")}`,
          )
        : [`/route-${String(i).padStart(3, "0")}`],
    addressRanges:
      i === 0
        ? Array.from({ length: 128 }, (_, n) => `192.0.2.${n}/32`)
        : route.addressRanges,
  }));
  await install(
    page,
    JSON.stringify({
      ...fixture,
      egress: {
        routes,
        exclusions: Array.from({ length: 128 }, (_, i) => ({
          ...route,
          pathPrefixes: [`/exclude-${String(i).padStart(3, "0")}`],
        })),
      },
      identities: [{ id: "anonymous", kind: "anonymous", origins: [origin] }],
    }),
  );
  const preview = await load(page);
  const nav = await browse(page, 1440);
  await expect(nav.getByRole("button")).toHaveCount(50);
  const paths = preview.getByRole("region", {
    name: "Path prefixes",
    exact: true,
  });
  await expect(paths.getByRole("listitem")).toHaveCount(50);
  const pathPages = preview.getByRole("navigation", {
    name: "Path prefixes pages",
    exact: true,
  });
  await pathPages.getByRole("button", { name: "Next", exact: true }).click();
  await pathPages.getByRole("button", { name: "Next", exact: true }).click();
  await expect(paths.getByRole("listitem")).toHaveCount(28);
  await expect(paths.getByText("/path-127", { exact: true })).toBeVisible();
  const networks = preview.getByRole("region", {
    name: "Address ranges",
    exact: true,
  });
  const networkPages = preview.getByRole("navigation", {
    name: "Address ranges pages",
    exact: true,
  });
  await networkPages.getByRole("button", { name: "Next", exact: true }).click();
  await networkPages.getByRole("button", { name: "Next", exact: true }).click();
  await expect(
    networks.getByText("192.0.2.127/32", { exact: true }),
  ).toBeVisible();
  const routePages = preview.getByRole("navigation", {
    name: "Egress route pages",
    exact: true,
  });
  for (let i = 0; i < 5; i++)
    await routePages.getByRole("button", { name: "Next", exact: true }).click();
  await expect(nav.getByRole("button")).toHaveCount(6);
  const target = nav.getByRole("button").last();
  await target.focus();
  const before = await page.evaluate(() => ({
    y: scrollY,
    x: document.querySelector(".ep-browser")!.getBoundingClientRect().x,
  }));
  await target.press("Enter");
  await expect(
    preview.getByRole("article", {
      name: "Selected egress route",
      exact: true,
    }),
  ).toContainText("/exclude-127");
  expect(await page.evaluate(() => scrollY)).toBeCloseTo(before.y, 0);
  expect((await page.locator(".ep-browser").boundingBox())!.x).toBeCloseTo(
    before.x,
    0,
  );
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await preview.getByLabel("Find a route", { exact: true }).fill("/route-127");
  await expect(
    preview.getByRole("article", {
      name: "Selected egress route",
      exact: true,
    }),
  ).toContainText("/route-127");
});
