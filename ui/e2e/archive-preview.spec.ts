import { expect, test } from "@playwright/test";

import packageMetadata from "../package.json" with { type: "json" };

for (const width of [1440, 390]) {
  test(`browse Skill and ZIP folders safely at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 1000 });
    const origin = new URL(String(testInfo.project.use.baseURL)).origin;
    const externalRequests: string[] = [];
    const pageErrors: string[] = [];
    const opened: string[] = [];
    page.on("request", (request) => {
      if (new URL(request.url()).origin !== origin)
        externalRequests.push(request.url());
    });
    page.on("pageerror", (error) => pageErrors.push(error.message));
    const files: Record<string, string> = {
      "package/SKILL.md":
        "# Package guide\n\nBrowse the reference files.\n\n![tracker](https://preview.invalid/tracker)\n\n<script>document.body.dataset.injected='yes'</script>",
      "package/assets/page.html":
        '<img src="https://preview.invalid/pixel" onerror="document.body.dataset.injected=1">',
      "package/references/example.py": 'print("Source only")',
    };
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
      const headers = {
        "content-type": "application/json",
        "x-contractor-api-version": "contractor.public.v1",
        etag: '"r1"',
      };
      const json = (body: unknown) => route.fulfill({ headers, json: body });
      if (url.pathname === "/v1/auth/session")
        return json({
          principal: {
            userId: "owner",
            username: "owner",
            capabilities: ["user", "operations"],
          },
          csrfToken: "a".repeat(43),
          idleExpiresAt: "2099-01-01T00:00:00Z",
          absoluteExpiresAt: "2099-01-02T00:00:00Z",
        });
      const namespace = url.pathname.includes("/skills/")
        ? "skills"
        : "sources";
      const artifact = { namespace, name: "package", revision: "r1" };
      const metadata = {
        artifact,
        mediaType:
          namespace === "skills"
            ? "application/vnd.contractor.agent-skill+zip"
            : "application/zip",
        size: 500000,
        current: false,
        frozen: true,
        createdAt: "2026-09-19T10:00:00Z",
      };
      if (url.pathname.endsWith("/metadata")) return json(metadata);
      if (url.pathname.endsWith("/versions"))
        return json({ items: [metadata], page: { hasMore: false } });
      if (url.pathname.endsWith("/lineage"))
        return json({ items: [], page: { hasMore: false } });
      if (url.searchParams.get("revision") !== "r1")
        return route.fulfill({ status: 400 });
      if (url.pathname.endsWith("/archive"))
        return json({
          artifact,
          entries: [
            ...["package", "package/assets", "package/references"].map(
              (path) => ({
                path,
                kind: "directory",
                size: 0,
                previewable: false,
              }),
            ),
            ...Object.entries(files).map(([path, text]) => ({
              path,
              kind: "file",
              size: text.length,
              previewable: true,
            })),
            {
              path: "package/assets/large.txt",
              kind: "file",
              size: 300000,
              previewable: false,
            },
          ],
        });
      if (url.pathname.endsWith("/archive/file")) {
        const path = url.searchParams.get("path")!;
        opened.push(path);
        const text = files[path]!;
        return json({ artifact, path, size: text.length, text });
      }
      return route.fulfill({ status: 404 });
    });
    for (const namespace of ["skills", "sources"]) {
      await page.goto(`/artifacts/${namespace}/package?revision=r1`);
      await page.getByRole("button", { name: "Browse files" }).click();
      await expect(
        page.getByRole("heading", { name: "Package guide" }),
      ).toBeVisible();
      const tree = page.getByRole("navigation", { name: "Archive files" });
      await tree.getByText("assets/", { exact: true }).click();
      await tree.getByRole("button", { name: /page.html/ }).click();
      await expect(page.locator(".archive-file-content pre")).toHaveText(
        files["package/assets/page.html"]!,
      );
      await expect(
        page.locator(
          ".archive-file-content img, .archive-file-content script, .archive-file-content iframe",
        ),
      ).toHaveCount(0);
      await tree.getByRole("button", { name: /large.txt/ }).click();
      await expect(
        page.getByText(/This file exceeds the preview limits/),
      ).toBeVisible();
      await tree.getByText("references/", { exact: true }).click();
      await tree.getByRole("button", { name: /example.py/ }).click();
      await expect(page.locator(".archive-file-content pre")).toHaveText(
        files["package/references/example.py"]!,
      );
      expect(
        await page.evaluate(() => document.body.dataset.injected),
      ).toBeUndefined();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      ).toBe(true);
    }
    expect(opened).not.toContain("package/assets/large.txt");
    expect(externalRequests).toEqual([]);
    expect(pageErrors).toEqual([]);
    await page.screenshot({
      path: testInfo.outputPath(`archive-${width}.png`),
      fullPage: true,
    });
  });
}
