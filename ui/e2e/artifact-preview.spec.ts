import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const REVISION = "revision-1";

const artifacts = {
  markdown: {
    mediaType: "text/markdown",
    source: [
      "# Rendered report",
      "",
      "The **Markdown** renderer is ready.",
      "",
      "![Remote image](https://preview.invalid/tracker.png)",
      "",
      "<script>document.body.dataset.injected = 'true'</script>",
    ].join("\n"),
  },
  openapi: {
    mediaType: "application/yaml",
    source: [
      "openapi: 3.1.0",
      "info:",
      "  title: Preview API",
      "  version: 1.0.0",
      "paths:",
      "  /pets:",
      "    get:",
      "      summary: List pets",
      "      responses:",
      "        '200':",
      "          description: Pet list",
    ].join("\n"),
  },
  architecture: {
    mediaType: "text/vnd.likec4",
    source: [
      "specification {",
      "  element component",
      "  element user {",
      "    style { shape person }",
      "  }",
      "}",
      "model {",
      "  customer = user 'Customer'",
      "  system = component 'Contractor' {",
      "    api = component 'Contractor API'",
      "  }",
      "  customer -> system.api 'uses'",
      "}",
      "views {",
      "  view index {",
      "    title 'Contractor context'",
      "    include customer",
      "    include system",
      "    include system.api",
      "    autoLayout LeftRight",
      "  }",
      "}",
    ].join("\n"),
  },
} as const;

type ArtifactName = keyof typeof artifacts;

function apiHeaders(contentType: string): Record<string, string> {
  return {
    "cache-control": "no-store",
    "content-type": contentType,
    "x-contractor-api-version": API_VERSION,
  };
}

async function fulfillJson(route: Route, value: unknown): Promise<void> {
  await route.fulfill({
    body: JSON.stringify(value),
    headers: apiHeaders("application/json"),
    status: 200,
  });
}

async function installArtifactAPI(page: Page, uiOrigin: string): Promise<void> {
  await page.route("**/runtime-config.json", async (route) => {
    await route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: uiOrigin,
      },
    });
  });
  await page.route(`${uiOrigin}/v1/**`, async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/auth/session") {
      await fulfillJson(route, {
        principal: {
          userId: "user_preview",
          username: "preview",
          capabilities: ["user", "operations"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }

    const match =
      /^\/v1\/artifacts\/projects\/([^/]+)(?:\/(metadata|versions|lineage))?$/.exec(
        url.pathname,
      );
    const name = match?.[1] as ArtifactName | undefined;
    const endpoint = match?.[2];
    if (name === undefined || !(name in artifacts)) {
      await route.fulfill({ status: 404, body: "not found" });
      return;
    }
    const artifact = artifacts[name];
    const metadata = {
      artifact: { namespace: "projects", name, revision: REVISION },
      mediaType: artifact.mediaType,
      size: new TextEncoder().encode(artifact.source).byteLength,
      current: false,
      frozen: false,
      createdAt: "2026-09-02T09:00:00Z",
    };
    if (endpoint === "metadata") {
      await fulfillJson(route, metadata);
      return;
    }
    if (endpoint === "versions") {
      await fulfillJson(route, {
        items: [metadata],
        page: { hasMore: false },
      });
      return;
    }
    if (endpoint === "lineage") {
      await fulfillJson(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.searchParams.get("revision") !== REVISION) {
      await route.fulfill({ status: 412, body: "exact revision required" });
      return;
    }
    await route.fulfill({
      body: artifact.source,
      headers: {
        ...apiHeaders(artifact.mediaType),
        "content-length": String(metadata.size),
      },
      status: 200,
    });
  });
}

async function openPreview(page: Page, name: ArtifactName): Promise<void> {
  await page.goto(`/artifacts/projects/${name}?revision=${REVISION}`);
  await expect(
    page.getByRole("heading", { name: `projects/${name}` }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Load preview" }).click();
  await expect(page.getByRole("tab", { name: "Rendered" })).toBeVisible();
}

test("renders Markdown, OpenAPI and LikeC4 artifacts locally", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const uiOrigin = new URL(configuredBaseURL).origin;
  const externalRequests: string[] = [];
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin !== uiOrigin) {
      externalRequests.push(request.url());
    }
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await installArtifactAPI(page, uiOrigin);

  await openPreview(page, "markdown");
  await expect(
    page.getByRole("heading", { name: "Rendered report" }),
  ).toBeVisible();
  await expect(page.getByText("Image omitted: Remote image")).toBeVisible();
  await expect(page.locator(".markdown-artifact-preview img")).toHaveCount(0);
  await expect(page.locator(".markdown-artifact-preview script")).toHaveCount(
    0,
  );
  await page.getByRole("tab", { name: "Source" }).click();
  await expect(page.locator("pre.artifact-preview")).toContainText(
    "<script>document.body.dataset.injected",
  );

  await openPreview(page, "openapi");
  await expect(
    page.getByText("Preview API", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText("List pets", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText(/external references and requests are disabled/i),
  ).toBeVisible();
  await expect(
    page.locator('link[rel="stylesheet"][href*="/assets/openapi-"]'),
  ).toHaveCount(1);

  await openPreview(page, "architecture");
  await expect(page.getByLabel("View")).toBeVisible();
  await expect(page.locator(".likec4-artifact-canvas")).toBeVisible();
  await expect(
    page.locator(".likec4-artifact-canvas .react-flow.dark"),
  ).toBeVisible();
  await expect(
    page.getByText("Contractor API", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText("Customer", { exact: true }).first(),
  ).toBeVisible();

  expect(externalRequests).toEqual([]);
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});
