import { test, expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";
import fixtures from "../src/test/fixtures/material-format-examples.json" with { type: "json" };
import { artifactMetadata, project } from "../src/test/builders";
import {
  fulfillJSON,
  fulfillContent,
  fulfillError,
  mockAPI,
  uiOrigin,
} from "./support/mock-api";

async function install(page: Page, openapiSource?: string) {
  await mockAPI(page, async (route, request, url) => {
    if (request.method() !== "GET")
      return fulfillError(route, 403, "forbidden");
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
    const match =
      /^\/v1\/(?:projects\/project_shop\/|runs\/run_demo\/)?artifacts\/([^/]+)\/([^/]+)(?:\/(metadata|versions|lineage|archive(?:\/file)?))?$/.exec(
        url.pathname,
      );
    const file = fixtures.examples.find(
      (e) => e.namespace === match?.[1] && e.name === match?.[2],
    );
    if (!file) return fulfillError(route, 404, "not_found");
    const revision =
      file.versions.find(
        (v) => v.revision === url.searchParams.get("revision"),
      ) ?? file.versions[0]!;
    const meta = (v: typeof revision) =>
      artifactMetadata(file.namespace, file.name, {
        artifact: {
          namespace: file.namespace,
          name: file.name,
          revision: v.revision,
        },
        mediaType: file.mediaType,
        size:
          file.id === "openapi" && openapiSource !== undefined
            ? Buffer.byteLength(openapiSource)
            : file.sizeBytes,
        current: v.current,
        frozen: v.frozen,
        canUploadNewVersion: v.canUploadNewVersion,
        createdAt: v.createdAt,
      });
    if (match?.[3] === "metadata") return fulfillJSON(route, meta(revision));
    if (match?.[3] === "versions")
      return fulfillJSON(route, {
        items: file.versions.map(meta),
        page: { hasMore: false },
      });
    if (match?.[3] === "lineage")
      return fulfillJSON(route, { items: [], page: { hasMore: false } });
    const headers = { etag: `"${revision.revision}"` };
    if (match?.[3] === "archive" && file.files)
      return fulfillJSON(
        route,
        {
          artifact: meta(revision).artifact,
          entries: file.files.map((f) => ({
            path: f.path,
            kind: f.directory ? "directory" : "file",
            size: f.sizeBytes,
            previewable: !f.directory && f.text !== undefined,
          })),
        },
        200,
        headers,
      );
    if (match?.[3] === "archive/file" && file.files) {
      const f = file.files.find((f) => f.path === url.searchParams.get("path"));
      if (!f || f.text === undefined)
        return fulfillError(route, 404, "not_found");
      return fulfillJSON(
        route,
        {
          artifact: meta(revision).artifact,
          path: f.path,
          text: f.text,
          size: f.sizeBytes,
        },
        200,
        headers,
      );
    }
    if (url.searchParams.get("revision") !== revision.revision)
      return fulfillError(route, 412, "precondition_failed");
    return fulfillContent(
      route,
      file.payloadBase64
        ? Buffer.from(file.payloadBase64, "base64")
        : file.id === "openapi" && openapiSource !== undefined
          ? openapiSource
          : file.source,
      file.mediaType,
    );
  });
}
for (const theme of ["light", "dark", "black"])
  for (const width of [1440, 1100, 390, 320]) {
    test(`material reader ${theme} ${width}px`, async ({ page }, info) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript((theme) => {
        localStorage.setItem("contractor.theme", theme);
        Object.defineProperty(navigator, "clipboard", {
          configurable: true,
          value: {
            writeText: async (value: string) => {
              document.documentElement.dataset.testClipboard = value;
            },
          },
        });
      }, theme);
      await install(page);
      const errors: string[] = [],
        external: string[] = [];
      page.on("pageerror", (e) => errors.push(e.message));
      const origin = uiOrigin(info);
      page.on("request", (r) => {
        if (
          ![origin, "http://127.0.0.1:1", "http://127.0.0.3:8080"].includes(
            new URL(r.url()).origin,
          )
        )
          external.push(r.url());
      });
      for (const file of fixtures.examples) {
        const v = file.versions[0]!;
        await page.goto(
          `/projects/project_shop/artifacts/${file.namespace}/${file.name}?revision=${v.revision}`,
        );
        await expect(
          page.getByRole("heading", {
            level: 1,
            name: `${file.namespace}/${file.name}`,
          }),
        ).toBeVisible();
        await page
          .getByRole("button", {
            name: "Copy material identifier",
            exact: true,
          })
          .click();
        await expect(page.locator("html")).toHaveAttribute(
          "data-test-clipboard",
          `${file.namespace}/${file.name}@${v.revision}`,
        );
        const downloadButton = page.getByRole("button", {
          name: "Download this revision",
          exact: true,
        });
        await expect(downloadButton).toHaveText("");
        const save = page.waitForEvent("download");
        await downloadButton.click();
        const saved = await save;
        const raw = await readFile((await saved.path())!);
        expect(raw).toEqual(
          file.payloadBase64
            ? Buffer.from(file.payloadBase64, "base64")
            : Buffer.from(file.source),
        );
        if (file.id === "archive") {
          await page
            .getByRole("button", { name: "Browse files", exact: true })
            .click();
          await expect(
            page.getByRole("heading", { name: "Shop API source bundle" }),
          ).toBeVisible();
          if (width <= 820)
            await page.getByRole("button", { name: /Browse files ·/ }).click();
          const tree = page.getByRole("navigation", { name: "Archive files" });
          await tree.getByText("src/", { exact: true }).click();
          await tree.getByRole("button", { name: /orders.ts/ }).click();
          await expect(page.locator(".archive-file-content pre")).toContainText(
            "async function getOrder",
          );
        } else {
          await page
            .getByRole("button", { name: "Load preview", exact: true })
            .click();
          if (file.id === "source")
            await expect(
              page.locator(".materials-detail-reader pre"),
            ).toContainText("async function getOrder");
          if (file.id === "http")
            await expect(
              page.getByRole("tab", { name: "Request", exact: true }),
            ).toBeVisible();
          if (file.id === "report")
            await expect(
              page.getByRole("heading", {
                name: "Order access review",
                exact: true,
              }),
            ).toBeVisible();
          if (file.id === "json")
            await expect(
              page.locator(".materials-detail-reader pre"),
            ).toContainText("demo-customer");
          if (file.id === "openapi") {
            await expect(
              page
                .locator(".openapi-artifact-preview")
                .getByRole("heading", { name: "List orders", exact: true }),
            ).toBeVisible();
            const nav = page.getByRole("navigation", {
              name: "API reference",
              exact: true,
            });
            if (width > 1100) {
              const browse = page.getByRole("button", { name: /Browse API ·/ });
              await expect(browse).toHaveAttribute("aria-expanded", "true");
              await browse.click();
              await expect(nav).not.toBeVisible();
              await browse.click();
              await expect(nav).toBeVisible();
            }
            async function choose(name: string) {
              if (width <= 1100)
                await page
                  .getByRole("button", { name: /Browse API ·/ })
                  .click();
              await nav.getByRole("button", { name, exact: true }).click();
            }
            await choose("GET /orders/{orderId}");
            const renderer = page.locator(".openapi-artifact-preview");
            await expect(
              renderer.getByRole("heading", {
                name: "Get an order",
                exact: true,
              }),
            ).toBeVisible();
            await expect(
              renderer.getByRole("button", {
                name: "Copy link to orderId",
                exact: true,
              }),
            ).toBeVisible();
            await expect(
              renderer.getByRole("button", {
                name: "Copy link to customerId",
                exact: true,
              }),
            ).toBeVisible();
            await choose("Order");
            await expect(
              renderer.locator(
                '[id$="/models/material-selected-schema-content"]',
              ),
            ).toBeVisible();
            await expect(
              renderer.getByRole("heading", { name: "NewOrder", exact: true }),
            ).not.toBeVisible();
            await choose("GET /orders/{orderId}");
            await expect(
              renderer.getByRole("heading", {
                name: "Get an order",
                exact: true,
              }),
            ).toBeVisible();
          }
          if (file.id === "source") {
            await expect(page.locator(".materials-source-line")).toHaveCount(4);
            expect(
              await page.locator("pre.artifact-preview").textContent(),
            ).toBe(file.source);
          }
          if (file.id === "likec4")
            await expect(
              page.getByRole("combobox", { name: "View", exact: true }),
            ).toBeVisible();
        }
        const geometry = await page.evaluate(() => ({
          width: innerWidth,
          doc: document.documentElement.scrollWidth,
          reader: document
            .querySelector(".materials-detail-reader")!
            .getBoundingClientRect()
            .toJSON(),
          support: document
            .querySelector(".materials-detail-support")!
            .getBoundingClientRect()
            .toJSON(),
        }));
        expect(geometry.doc).toBeLessThanOrEqual(width);
        if (width <= 820) {
          expect(geometry.reader.width).toBeGreaterThan(width * 0.78);
          expect(geometry.support.y).toBeGreaterThanOrEqual(
            geometry.reader.bottom - 1,
          );
        } else {
          expect(geometry.support.x).toBeGreaterThan(geometry.reader.x);
          expect(geometry.reader.width).toBeGreaterThan(
            geometry.support.width * 1.8,
          );
        }
        if (["source", "openapi", "archive"].includes(file.id))
          await page.screenshot({
            path: info.outputPath(`${file.id}-${theme}-${width}.png`),
            fullPage: true,
          });
      }
      expect(errors).toEqual([]);
      expect(external).toEqual([]);
    });
  }

test("scope and revision retain context, exact identity and local reader state", async ({
  page,
}) => {
  await install(page);
  const file = fixtures.examples.find((f) => f.id === "source")!;
  for (const prefix of [
    "/artifacts",
    "/projects/project_shop/artifacts",
    "/runs/run_demo/artifacts",
    "/evals/workspaces/project_shop/artifacts",
  ]) {
    await page.goto(
      `${prefix}/${file.namespace}/${file.name}?revision=${file.versions[0]!.revision}`,
    );
    await page
      .getByRole("button", { name: "Load preview", exact: true })
      .click();
    await expect(page.locator("pre.artifact-preview")).toContainText(
      "async function getOrder",
    );
    await page.getByRole("button", { name: "Wrap lines", exact: true }).click();
    await expect(page.locator("pre.artifact-preview")).toHaveAttribute(
      "data-wrap",
      "false",
    );
    await page.getByRole("button", { name: "Versions", exact: true }).click();
    await page
      .getByRole("link", { name: new RegExp(file.versions[1]!.revision) })
      .click();
    await expect(
      page.getByText("Historical revision", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Load preview", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByText("Upload a new version", { exact: true }),
    ).toHaveCount(0);
  }
});

test("Swagger definitions and path-level parameters survive selection", async ({
  page,
}) => {
  await install(
    page,
    JSON.stringify({
      swagger: "2.0",
      info: { title: "Saved Swagger", version: "1" },
      paths: {
        "/orders/{id}": {
          parameters: [
            {
              name: "id",
              in: "path",
              required: true,
              type: "integer",
              description: "Shared path parameter",
            },
          ],
          get: {
            summary: "Read saved order",
            responses: {
              "200": {
                description: "Saved result",
                schema: { $ref: "#/definitions/Order" },
              },
            },
          },
          delete: {
            summary: "Delete declaration",
            responses: { "204": { description: "Saved declaration" } },
          },
        },
      },
      definitions: {
        Order: {
          type: "object",
          properties: { customerId: { type: "string" } },
        },
      },
    }),
  );
  await page.goto(
    "/projects/project_shop/artifacts/outputs/shop-api?revision=shop-api-r2",
  );
  await page.getByRole("button", { name: "Load preview", exact: true }).click();
  const renderer = page.locator(".openapi-artifact-preview");
  await expect(
    renderer.getByRole("heading", { name: "Read saved order", exact: true }),
  ).toBeVisible();
  await expect(
    renderer.getByText("Shared path parameter", { exact: true }),
  ).toBeVisible();
  await expect(
    renderer.getByRole("button", {
      name: "Copy link to customerId",
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByRole("navigation", { name: "API reference", exact: true })
    .getByRole("button", { name: "Order", exact: true })
    .click();
  await expect(
    renderer.locator('[id$="/models/material-selected-schema-content"]'),
  ).toBeVisible();
});

test("local references to other paths keep their actual Scalar targets", async ({
  page,
}) => {
  await install(
    page,
    JSON.stringify({
      openapi: "3.0.3",
      info: { title: "Path references", version: "1" },
      paths: {
        "/orders": { $ref: "#/paths/~1shared" },
        "/shared": {
          get: {
            summary: "Shared saved operation",
            responses: { "200": { description: "Reference target retained" } },
          },
        },
      },
    }),
  );
  await page.goto(
    "/projects/project_shop/artifacts/outputs/shop-api?revision=shop-api-r2",
  );
  await page.getByRole("button", { name: "Load preview", exact: true }).click();
  const renderer = page.locator(".openapi-artifact-preview");
  await expect(
    renderer.getByRole("heading", {
      name: "Shared saved operation",
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    renderer
      .getByText("Reference target retained", { exact: true })
      .filter({ visible: true })
      .first(),
  ).toBeVisible();
  const nav = page.getByRole("navigation", {
    name: "API reference",
    exact: true,
  });
  await nav.getByRole("button", { name: "GET /shared", exact: true }).click();
  await expect(
    renderer.getByRole("heading", {
      name: "Shared saved operation",
      exact: true,
    }),
  ).toBeVisible();
});

for (const theme of ["light", "dark", "black"]) {
  test(`OpenAPI selection keeps the renderer and outer scroll stable in ${theme}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.addInitScript(
      (theme) => localStorage.setItem("contractor.theme", theme),
      theme,
    );
    await install(page);
    const file = fixtures.examples.find((file) => file.id === "openapi")!;
    await page.goto(
      `/projects/project_shop/artifacts/${file.namespace}/${file.name}`,
    );
    await page
      .getByRole("button", { name: "Load preview", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "List orders", exact: true }),
    ).toBeVisible();
    const renderer = page.locator(".openapi-artifact-preview > div").last();
    await renderer.evaluate((el) =>
      el.setAttribute("data-reader-continuity", "kept"),
    );
    const before = await page.evaluate(() => ({
      y: scrollY,
      x: document
        .querySelector(".materials-api-browser")!
        .getBoundingClientRect().x,
      width: document
        .querySelector(".materials-api-browser")!
        .getBoundingClientRect().width,
    }));
    await page
      .getByRole("button", { name: "POST /orders", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Create an order", exact: true }),
    ).toBeVisible();
    await expect(renderer).toHaveAttribute("data-reader-continuity", "kept");
    expect(await page.evaluate(() => scrollY)).toBeCloseTo(before.y, 0);
    expect(
      (await page.locator(".materials-api-browser").boundingBox())!.x,
    ).toBeCloseTo(before.x, 0);
    await page
      .getByRole("button", { name: "GET /orders", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "List orders", exact: true }),
    ).toBeVisible();
    expect(await page.evaluate(() => scrollY)).toBeCloseTo(before.y, 0);
  });
}
