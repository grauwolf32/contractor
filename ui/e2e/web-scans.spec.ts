import { expect, test } from "@playwright/test";
import { artifactWrite, project } from "../src/test/builders";
import {
  webScanCreated,
  webScanPack,
  webScanWorkflow,
} from "../src/test/web-scans";
import {
  authSession,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

for (const theme of ["light", "dark", "black"]) {
  test(`web scan widget selects and uploads owner rules at 390px in ${theme}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 900 });
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    let submitted: Record<string, unknown> | undefined;
    const uploads: string[] = [];
    await mockAPI(
      page,
      async (route, request, url) => {
        if (url.pathname === "/v1/workflows/nuclei-web/versions/1")
          return fulfillJSON(route, webScanWorkflow);
        if (url.pathname === "/v1/projects")
          return fulfillJSON(route, {
            items: [
              project("p1", {
                name: "Fixture Project",
                httpTarget: { url: "https://fixture.example/" },
              }),
            ],
            page: { hasMore: false },
          });
        if (url.pathname === "/v1/artifacts")
          return fulfillJSON(route, {
            items: [webScanPack],
            page: { hasMore: false },
          });
        if (request.method() === "PUT") {
          uploads.push(url.pathname);
          const ref = {
            namespace: "inputs",
            name: url.pathname.split("/").at(-1) ?? "",
            revision: "uploaded",
          };
          const metadata = {
            ...webScanPack,
            artifact: ref,
            size: request.postDataBuffer()?.length ?? 0,
          };
          return fulfillJSON(route, artifactWrite(metadata), 201, {
            etag: '"uploaded"',
          });
        }
        if (
          url.pathname === "/v1/projects/p1/runs" &&
          request.method() === "POST"
        ) {
          submitted = request.postDataJSON() as Record<string, unknown>;
          return fulfillJSON(route, webScanCreated("p1"), 202);
        }
        return fulfillEmptyPageOrNotFound(route);
      },
      { session: authSession("owner", ["user"]) },
    );
    await page.goto("/checks/web/new");
    await page.getByLabel("Save results in").selectOption("p1");
    await expect(page.getByLabel("Target URL")).toHaveValue(
      "https://fixture.example/",
    );
    await page.getByLabel("Rule source").selectOption("custom");
    await page
      .getByRole("button", { name: "Upload rulepack", exact: true })
      .click();
    const dialog = page.getByRole("dialog");
    await dialog.getByLabel("Drop a file here").setInputFiles({
      name: "my-rule.yaml",
      mimeType: "application/yaml",
      buffer: Buffer.from("id: fixture\n"),
    });
    await dialog.getByRole("button", { name: "Upload and select" }).click();
    await expect(dialog).toHaveCount(0);
    expect(uploads).toEqual(["/v1/artifacts/inputs/my-rule"]);
    await page.getByLabel("Rule IDs").fill("fixture");
    await expect(
      page.getByRole("button", { name: "Start scan", exact: true }),
    ).toBeEnabled();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: `/private/tmp/contractor-web-scan-${theme}.png`,
      fullPage: true,
    });
    await page.getByRole("button", { name: "Start scan", exact: true }).click();
    await expect
      .poll(() => submitted)
      .toEqual({
        workflow: "nuclei-web@1",
        parameters: {
          target: "https://fixture.example/",
          template_ids: "fixture",
        },
        artifacts: {
          rulepack: {
            namespace: "inputs",
            name: "my-rule",
            revision: "uploaded",
          },
        },
      });
  });
}
