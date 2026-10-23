import { expect, test, type Page } from "@playwright/test";

import {
  fulfillContent,
  fulfillError,
  fulfillJSON,
  mockAPI,
  uiOrigin,
} from "./support/mock-api";

const REVISION = "revision-1";

// A report just under the 1 MiB preview limit (LIM-UI-12), in ASCII so that
// characters are bytes; each part carries a list, a table and a fence.
function largeReport(): { source: string; lastSection: number } {
  let source = "";
  for (let index = 0; ; index++) {
    const part = [
      `## Section ${index}`,
      "",
      "A paragraph with `code` and **bold** text.",
      "",
      "- one",
      "- two",
      "",
      "| Check | Result |",
      "| --- | --- |",
      `| ${index} | passed |`,
      "",
      "```text",
      `# not a heading ${index}`,
      "```",
      "",
      "",
    ].join("\n");
    if (source.length + part.length > 1024 * 1024 - 1024) {
      return { source, lastSection: index - 1 };
    }
    source += part;
  }
}

const LARGE_REPORT = largeReport();

// One table longer than a Markdown section may be (LIM-UI-16).
function longTable(): string {
  let table = "| Endpoint | Method |\n| --- | --- |\n";
  while (table.length <= 300 * 1024) {
    table += "| `/v1/items/{id}` | **GET** |\n";
  }
  return `# Inventory\n\n${table}\n## After the inventory\n\nDone.\n`;
}

const artifacts = {
  "http-evidence": {
    mediaType: "application/vnd.contractor.http-exchange+json",
    source: JSON.stringify({
      schema: "contractor.http.exchange.v1",
      capture: {
        source: "http",
        request_id: "1",
        request_tag: "r-browser-h1",
        attempts: [
          {
            request: {
              method: "POST",
              url:
                "https://target.example/a/very/long/request/path?query=" +
                "x".repeat(180),
              message: {
                kind: "structured",
                headers: [{ name: "Content-Type", value: "application/json" }],
                body_base64: Buffer.from('{"selected":"request"}').toString(
                  "base64",
                ),
              },
            },
            response: {
              status: 403,
              complete: true,
              message: {
                kind: "structured",
                headers: [{ name: "Content-Type", value: "text/html" }],
                body_base64: Buffer.from(
                  '<img src="https://preview.invalid/leak"><script>window.httpInjected = true</script>',
                ).toString("base64"),
              },
            },
          },
        ],
      },
      finding: {
        proposal_id: "proposal-browser",
        receipt_id: "receipt-browser",
      },
    }),
  },
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
  "large-report": {
    mediaType: "text/markdown",
    source: LARGE_REPORT.source,
  },
  "long-table": {
    mediaType: "text/markdown",
    source: longTable(),
  },
  "workspace-diff": {
    mediaType: "text/x-diff",
    source: [
      "--- a/src/config.ts",
      "+++ b/src/config.ts",
      "@@ -1,3 +1,4 @@",
      " export const ui = {",
      "-  colorDiff: false,",
      "+  colorDiff: true,",
      "+  lineNumbers: true,",
      " };",
    ].join("\n"),
  },
} as const;

type ArtifactName = keyof typeof artifacts;

async function installArtifactAPI(page: Page): Promise<void> {
  await mockAPI(page, async (route, _request, url) => {
    if (
      ["/v1/projects", "/v1/audits", "/v1/findings", "/v1/reviews"].includes(
        url.pathname,
      )
    ) {
      // The Inbox badge reads owner-wide lists on every page.
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }

    const match =
      /^\/v1\/artifacts\/projects\/([^/]+)(?:\/(metadata|versions|lineage))?$/.exec(
        url.pathname,
      );
    const name = match?.[1] as ArtifactName | undefined;
    const endpoint = match?.[2];
    if (name === undefined || !(name in artifacts)) {
      await fulfillError(route, 404, "not_found");
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
      await fulfillJSON(route, metadata);
      return;
    }
    if (endpoint === "versions") {
      await fulfillJSON(route, {
        items: [metadata],
        page: { hasMore: false },
      });
      return;
    }
    if (endpoint === "lineage") {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.searchParams.get("revision") !== REVISION) {
      await fulfillError(route, 412, "precondition_failed");
      return;
    }
    await fulfillContent(route, artifact.source, artifact.mediaType);
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

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test.describe(`${viewport.width}px previews`, () => {
    test.use({ viewport });

    test("renders Markdown, diff, OpenAPI and LikeC4 artifacts locally", async ({
      page,
    }, testInfo) => {
      const origin = uiOrigin(testInfo);
      const externalRequests: string[] = [];
      const consoleErrors: string[] = [];
      const pageErrors: string[] = [];
      page.on("request", (request) => {
        const url = new URL(request.url());
        if (url.origin !== origin) {
          externalRequests.push(request.url());
        }
      });
      page.on("console", (message) => {
        if (message.type() === "error") {
          consoleErrors.push(message.text());
        }
      });
      page.on("pageerror", (error) => pageErrors.push(error.message));
      await installArtifactAPI(page);

      await openPreview(page, "markdown");
      await expect(
        page.getByRole("heading", { name: "Rendered report" }),
      ).toBeVisible();
      await expect(page.getByText("Image omitted: Remote image")).toBeVisible();
      await expect(page.locator(".markdown-artifact-preview img")).toHaveCount(
        0,
      );
      await expect(
        page.locator(".markdown-artifact-preview script"),
      ).toHaveCount(0);
      await page.getByRole("tab", { name: "Source" }).click();
      await expect(page.locator("pre.artifact-preview")).toContainText(
        "<script>document.body.dataset.injected",
      );

      await openPreview(page, "workspace-diff");
      // Line classification is unit-tested (previews/diff.test.tsx); the
      // browser checks that the production chunk renders the diff in place.
      const diff = page.getByRole("region", { name: "Unified diff preview" });
      await expect(diff.getByText("1 file changed")).toBeVisible();
      await expect(diff.getByText("+2 additions")).toBeVisible();
      await expect(diff.getByText("−1 deletion")).toBeVisible();
      await expect(
        diff.getByText("@@ -1,3 +1,4 @@", { exact: true }),
      ).toBeVisible();
      await expect(
        diff.getByText("+  lineNumbers: true,", { exact: true }),
      ).toBeVisible();

      await openPreview(page, "openapi");
      await expect(
        page.getByText("Preview API", { exact: true }).first(),
      ).toBeVisible();
      await expect(
        page.getByRole("heading", { name: "List pets", exact: true }),
      ).toBeVisible();
      await expect(
        page.getByText(/external references and requests are disabled/i),
      ).toBeVisible();
      await expect(
        page.locator('link[rel="stylesheet"][href*="/assets/openapi-"]'),
      ).toHaveCount(1);

      await openPreview(page, "architecture");
      await expect(
        page.getByRole("combobox", { name: "View", exact: true }),
      ).toBeVisible();
      await expect(page.locator(".likec4-artifact-canvas")).toBeVisible();
      await expect(
        page.locator(".likec4-artifact-canvas .react-flow.light"),
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
  });
}

test("renders a 1 MiB Markdown report in sections and a long table as text", async ({
  page,
}) => {
  await installArtifactAPI(page);
  const preview = page.locator(".markdown-artifact-preview");

  await openPreview(page, "large-report");
  await expect(
    preview.getByRole("heading", { name: "Section 0", exact: true }),
  ).toBeVisible();
  await expect(
    preview.getByRole("heading", {
      name: `Section ${LARGE_REPORT.lastSection}`,
      exact: true,
    }),
  ).toBeAttached({ timeout: 60_000 });
  await expect(preview.getByRole("status")).toHaveCount(0);
  await expect(preview.locator("table")).toHaveCount(
    LARGE_REPORT.lastSection + 1,
  );
  // Fenced "# not a heading" lines stay code in every section.
  await expect(preview.locator("h1")).toHaveCount(0);

  await openPreview(page, "long-table");
  await expect(
    preview.getByText(/too long to render, so it is shown as text/),
  ).toBeVisible();
  await expect(preview.locator("table")).toHaveCount(0);
  await expect(
    preview.getByRole("heading", { name: "Inventory", exact: true }),
  ).toBeVisible();
  await expect(
    preview.getByRole("heading", { name: "After the inventory" }),
  ).toBeAttached();
});

for (const theme of ["light", "dark", "black"]) {
  test("HTTP evidence at 390px in " + theme, async ({ page }, testInfo) => {
    const origin = uiOrigin(testInfo);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    const external: string[] = [];
    page.on("request", (request) => {
      if (new URL(request.url()).origin !== origin)
        external.push(request.url());
    });
    await installArtifactAPI(page);
    await openPreview(page, "http-evidence");
    const request = page.getByRole("tab", { name: "Request", exact: true });
    await expect(request).toBeVisible();
    await expect(
      page.getByText('"selected": "request"', { exact: false }),
    ).toBeVisible();
    await request.focus();
    await page.keyboard.press("ArrowRight");
    await expect(
      page.getByRole("tab", { name: "Response", exact: true }),
    ).toBeFocused();
    await expect(page.getByText("HTTP 403 · Complete response")).toBeVisible();
    await expect(
      page.locator(".http-evidence pre").filter({ hasText: "<img" }),
    ).toBeVisible();
    await expect(
      page.locator(
        ".http-evidence img, .http-evidence script, .http-evidence iframe",
      ),
    ).toHaveCount(0);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    const download = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Download response body", exact: true })
      .click();
    expect((await download).suggestedFilename()).toBe("response-body.txt");
    const saved = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Download this revision", exact: true })
      .click();
    expect((await saved).suggestedFilename()).toMatch(/revision-1\.json$/);
    expect(external).toEqual([]);
  });
}
