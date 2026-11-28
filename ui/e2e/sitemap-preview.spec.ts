import { test, expect, type Page } from "@playwright/test";
import { createHash } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import fixture from "../src/test/fixtures/sitemap-preview.json" with { type: "json" };
import { artifactMetadata, project } from "../src/test/builders";
import {
  SITEMAP_MEDIA_TYPE,
  MAXIMUM_PREVIEW_BYTES,
} from "../src/api/artifacts";
import {
  mockAPI,
  fulfillJSON,
  fulfillContent,
  fulfillError,
  uiOrigin,
} from "./support/mock-api";

const path = "/projects/project_shop/artifacts/sitemap/map?revision=sitemap-r1";
const empty = {
  ...fixture,
  groups: [],
  sources: [],
  rules: [],
  markers: [],
  gaps: [],
};
async function install(page: Page, source = JSON.stringify(fixture)) {
  const writes: string[] = [];
  await mockAPI(page, async (route, request, url) => {
    if (request.method() !== "GET") {
      writes.push(request.method() + " " + url.pathname);
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
      ["/v1/projects", "/v1/audits", "/v1/findings", "/v1/reviews"].includes(
        url.pathname,
      )
    )
      return fulfillJSON(route, { items: [], page: { hasMore: false } });
    const metadata = artifactMetadata("sitemap", "map", {
      artifact: { namespace: "sitemap", name: "map", revision: "sitemap-r1" },
      mediaType: SITEMAP_MEDIA_TYPE,
      size: Buffer.byteLength(source),
      canUploadNewVersion: false,
    });
    if (url.pathname.endsWith("/sitemap/map/metadata"))
      return fulfillJSON(route, metadata);
    if (url.pathname.endsWith("/sitemap/map/versions"))
      return fulfillJSON(route, {
        items: [metadata],
        page: { hasMore: false },
      });
    if (url.pathname.endsWith("/sitemap/map/lineage"))
      return fulfillJSON(route, { items: [], page: { hasMore: false } });
    if (url.pathname.endsWith("/sitemap/map")) {
      expect(url.searchParams.get("revision")).toBe("sitemap-r1");
      return fulfillContent(route, source, SITEMAP_MEDIA_TYPE);
    }
    return fulfillError(route, 404, "not_found");
  });
  return writes;
}
async function load(page: Page) {
  await page.goto(path);
}
async function browse(page: Page, width: number) {
  if (width <= 820)
    await page.getByRole("button", { name: /Browse routes/ }).click();
  return page.getByRole("navigation", { name: "Sitemap routes", exact: true });
}

for (const theme of ["light", "dark", "black"])
  for (const width of [1440, 390]) {
    test(`saved sitemap ${theme} ${width}px`, async ({ page }, info) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (theme) => localStorage.setItem("contractor.theme", theme),
        theme,
      );
      const writes = await install(page);
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
      await load(page);
      const nav = await browse(page, width);
      await nav
        .getByRole("button", { name: "GET /v1/orders/{id} 2", exact: true })
        .click();
      const detail = page.locator(".sm-details");
      await expect(
        detail.getByRole("heading", {
          name: "Recorded observations",
          exact: true,
        }),
      ).toBeVisible();
      await page
        .getByRole("button", { name: "/v1/orders/42 200", exact: true })
        .click();
      await expect(
        detail.getByText("object-access", { exact: true }),
      ).toBeVisible();
      await expect(
        detail.getByText("caido · staging", { exact: true }),
      ).toBeVisible();
      await expect(
        detail.getByRole("link", { name: "details →", exact: true }),
      ).toHaveAttribute(
        "href",
        "/runs/run_sitemap_demo/artifacts/evidence/order-42?revision=order-r1",
      );
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(width);
      if (process.env.CONTRACTOR_SITEMAP_CAPTURE_DIR) {
        await mkdir(process.env.CONTRACTOR_SITEMAP_CAPTURE_DIR, {
          recursive: true,
        });
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({
          path: join(
            process.env.CONTRACTOR_SITEMAP_CAPTURE_DIR,
            `sitemap-${theme}-${width}.png`,
          ),
          fullPage: true,
        });
      }
      await page.getByLabel("Find a route", { exact: true }).fill("/v2/export");
      await expect(
        page.locator(".sm-details").getByText("Declared only", { exact: true }),
      ).toBeVisible();
      await page.getByLabel("Find a route", { exact: true }).fill("");
      await page
        .getByRole("combobox", { name: "Records", exact: true })
        .selectOption("suppressed");
      await expect(
        page.locator(".sm-details").getByText("Suppressed", { exact: true }),
      ).toBeVisible();
      await page
        .getByRole("combobox", { name: "Records", exact: true })
        .selectOption("");
      await page
        .getByRole("combobox", { name: "Method", exact: true })
        .selectOption("DELETE");
      await expect(page.locator(".sm-details .sm-method")).toHaveText("DELETE");
      await page.getByLabel("Find a route", { exact: true }).fill("no-match");
      await expect(
        page.getByText("No matching route", { exact: true }),
      ).toBeVisible();
      await page.getByRole("tab", { name: "Source", exact: true }).click();
      await expect(page.locator(".materials-preview-shell pre")).toContainText(
        '"project-sitemap@1"',
      );
      expect(writes).toEqual([]);
      expect(errors).toEqual([]);
      expect(external).toEqual([]);
    });
  }

test("empty, unsupported and oversized sitemap retain exact Source/download behavior", async ({
  page,
}) => {
  for (const [source, valid] of [
    [JSON.stringify(empty), true],
    [JSON.stringify({ ...empty, version: "project-sitemap@2" }), false],
    ['{"version":"project-sitemap@1","version":"project-sitemap@1"}', false],
  ] as const) {
    await install(page, source);
    await load(page);
    if (valid)
      await expect(
        page.getByText("No routes recorded", { exact: true }),
      ).toBeVisible();
    else
      await expect(page.getByRole("alert")).toContainText(
        "sitemap is invalid or unsupported",
      );
    await page.getByRole("tab", { name: "Source", exact: true }).click();
    await expect(page.locator(".materials-preview-shell pre")).toHaveText(
      source,
    );
  }
  await install(page, " ".repeat(MAXIMUM_PREVIEW_BYTES + 1));
  await page.goto(path);
  await expect(
    page.getByRole("button", { name: "Load preview", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Download this revision", exact: true }),
  ).toBeEnabled();
});

test("large saved route collection bounds mounted rows and keeps selection stable", async ({
  page,
}) => {
  const key = (...parts: string[]) =>
    createHash("sha256")
      .update(
        parts.map((part) => Buffer.byteLength(part) + ":" + part).join(""),
      )
      .digest("hex");
  const groups = Array.from({ length: 1200 }, (_, index) => {
    const route = `/route/${String(index).padStart(4, "0")}`;
    return {
      id: key("https://large.test", "GET", route),
      origin: "https://large.test",
      method: "GET",
      route,
      structural: [],
      members: [],
      declarations: [],
      omitted: 0,
    };
  });
  await install(page, JSON.stringify({ ...empty, groups }));
  await load(page);
  const nav = page.getByRole("navigation", {
    name: "Sitemap routes",
    exact: true,
  });
  await expect(nav.getByRole("button")).toHaveCount(50);
  await page
    .getByRole("navigation", { name: "Route pages", exact: true })
    .getByRole("button", { name: "Next", exact: true })
    .click();
  await expect(nav.getByRole("button")).toHaveCount(50);
  await expect(nav.getByRole("button").first()).toContainText("/route/0050");
  await page.evaluate(() => window.scrollTo(0, 0));
  const before = await page.evaluate(() => ({
    y: scrollY,
    x: document.querySelector(".sm-browser")!.getBoundingClientRect().x,
  }));
  await nav.getByRole("button").nth(1).focus();
  await nav.getByRole("button").nth(1).press("Enter");
  await expect(page.locator(".sm-route-head")).toContainText("/route/0051");
  expect(await page.evaluate(() => scrollY)).toBeCloseTo(before.y, 0);
  expect((await page.locator(".sm-browser").boundingBox())!.x).toBeCloseTo(
    before.x,
    0,
  );
  await page.getByLabel("Find a route", { exact: true }).fill("/route/1199");
  await expect(nav.getByRole("button")).toHaveCount(1);
  await expect(page.locator(".sm-route-head")).toContainText("/route/1199");
});

test("all 64 concrete members and the last of 128 notes remain reachable", async ({
  page,
}) => {
  const key = (...parts: string[]) =>
    createHash("sha256")
      .update(
        parts.map((part) => Buffer.byteLength(part) + ":" + part).join(""),
      )
      .digest("hex");
  const group = fixture.groups.find(
    (group) => group.route === "/v1/orders/{id}" && group.method === "GET",
  )!;
  const owner = fixture.sources.find(
    (source) => source.provenance.sourceId === "owner",
  )!;
  const members = Array.from({ length: 64 }, (_, index) => {
    const path = `/v1/orders/${index}`;
    return {
      id: key("member", group.origin, group.method, path),
      path,
      summaries: [],
      examples: [],
      evidence: [],
      annotations:
        index === 0
          ? Array.from({ length: 128 }, (_, note) => ({
              provenance: owner.provenance,
              note: `Retained note ${note}`,
              tags: [],
            }))
          : [],
      omitted: 0,
    };
  });
  await install(
    page,
    JSON.stringify({
      ...empty,
      groups: [{ ...group, members, declarations: [], omitted: 0 }],
      sources: [owner],
    }),
  );
  await load(page);
  await expect(page.locator(".sm-member")).toHaveCount(50);
  const notes = page.getByRole("navigation", {
    name: "Annotation pages",
    exact: true,
  });
  await expect(page.locator(".sm-annotation")).toHaveCount(50);
  await notes.getByRole("button", { name: "Next", exact: true }).click();
  await notes.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.locator(".sm-annotation")).toHaveCount(28);
  await expect(
    page.getByText("Retained note 127", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("navigation", { name: "Concrete path pages", exact: true })
    .getByRole("button", { name: "Next", exact: true })
    .click();
  await expect(page.locator(".sm-member")).toHaveCount(14);
  await page.locator(".sm-member").filter({ hasText: "/v1/orders/63" }).click();
  await expect(page.locator(".sm-section-head code")).toHaveText(
    "/v1/orders/63",
  );
  await expect(page.locator(".sm-annotation")).toHaveCount(0);
});

test("long saved origins and paths wrap at 390px without losing their values", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const origin = `https://${"a".repeat(63)}.${"b".repeat(63)}.test`;
  const route = `/api/${"resource".repeat(24)}`;
  const id = createHash("sha256")
    .update(
      [origin, "GET", route]
        .map((part) => Buffer.byteLength(part) + ":" + part)
        .join(""),
    )
    .digest("hex");
  await install(
    page,
    JSON.stringify({
      ...empty,
      groups: [
        {
          id,
          origin,
          method: "GET",
          route,
          structural: [],
          members: [],
          declarations: [],
          omitted: 0,
        },
      ],
    }),
  );
  await load(page);
  await expect(page.locator(".sm-route-head .sm-eyebrow")).toHaveText(origin);
  await expect(page.locator(".sm-route-head h3 code")).toHaveText(route);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  const nav = await browse(page, 390);
  await expect(nav.getByRole("button")).toContainText(route);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
});
