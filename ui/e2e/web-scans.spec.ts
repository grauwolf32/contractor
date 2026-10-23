import { expect, test } from "@playwright/test";
import { failure, noContent } from "../src/test/responses";
import { artifactWrite, project } from "../src/test/builders";
import {
  scanSweep,
  sweepMember,
  webScanCreated,
  webScanPack,
  webScanWorkflow,
} from "../src/test/web-scans";
import {
  authSession,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  fulfillResponse,
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
    const uploadedMedia: string[] = [];
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
          uploadedMedia.push(request.headers()["content-type"] ?? "");
          const ref = {
            namespace: "inputs",
            name: url.pathname.split("/").at(-1) ?? "",
            revision: "uploaded",
          };
          const metadata = {
            ...webScanPack,
            mediaType: request.headers()["content-type"] ?? "",
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
      name: theme === "dark" ? "my-rule.zip" : "my-rule.yaml",
      mimeType: theme === "dark" ? "application/zip" : "application/yaml",
      buffer: Buffer.from("id: fixture\n"),
    });
    await dialog.getByRole("button", { name: "Upload and select" }).click();
    await expect(dialog).toHaveCount(0);
    expect(uploads).toEqual(["/v1/artifacts/inputs/my-rule"]);
    expect(uploadedMedia).toEqual([
      theme === "dark"
        ? webScanWorkflow.inputs.rulepack!.mediaTypes.find((type) =>
            type.endsWith("+zip"),
          )
        : "application/yaml",
    ]);
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

for (const theme of ["light", "dark", "black"]) {
  test(`rule sweep widget keeps results and confirms purge at 390px in ${theme}`, async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 900 });
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    let submitted: Record<string, unknown> | undefined;
    let deleted = false;
    const deletes: string[] = [];
    await mockAPI(
      page,
      async (route, request, url) => {
        if (url.pathname === "/v1/workflows/nuclei-web/versions/1")
          return fulfillJSON(route, webScanWorkflow);
        if (url.pathname === "/v1/projects")
          return fulfillJSON(route, {
            items: [
              project("p1", {
                name: "First Project",
                httpTarget: { url: "https://fixture.example/" },
              }),
              project("p2", { name: "Second Project" }),
            ],
            page: { hasMore: false },
          });
        if (url.pathname === "/v1/artifacts")
          return fulfillJSON(route, {
            items: [webScanPack],
            page: { hasMore: false },
          });
        if (url.pathname === "/v1/scan-sweeps" && request.method() === "POST") {
          submitted = request.postDataJSON() as Record<string, unknown>;
          return fulfillJSON(route, scanSweep(), 201);
        }
        if (url.pathname === "/v1/scan-sweeps")
          return fulfillJSON(route, {
            items: deleted ? [] : [scanSweep("completed")],
            page: { hasMore: false },
          });
        if (url.pathname.endsWith("/members"))
          return fulfillJSON(route, {
            items: [
              sweepMember(0, true),
              {
                ...sweepMember(1),
                state: "skipped",
                reason: "missing_http_target",
                settled: true,
              },
            ],
            page: { hasMore: false },
          });
        if (
          url.pathname === "/v1/scan-sweeps/sweep-one" &&
          request.method() === "DELETE"
        ) {
          deletes.push(url.pathname);
          deleted = true;
          return fulfillResponse(route, noContent());
        }
        if (url.pathname === "/v1/scan-sweeps/sweep-one")
          return deleted
            ? fulfillResponse(route, failure(404))
            : fulfillJSON(route, scanSweep("completed"));
        return fulfillEmptyPageOrNotFound(route);
      },
      { session: authSession("owner", ["user"]) },
    );
    await page.goto("/checks/sweeps/new");
    await page.getByRole("checkbox", { name: /First Project/ }).check();
    await page.getByRole("checkbox", { name: /Second Project/ }).check();
    await page.getByLabel("Rule source").selectOption("custom");
    await page
      .getByLabel("Rulepack from my library")
      .selectOption("rules/web-check@r1");
    await page.getByLabel("Rule IDs").fill("fixture");
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page
      .getByRole("button", { name: "Start sweep", exact: true })
      .click();
    await expect(page).toHaveURL(/\/checks\/sweeps\/sweep-one$/);
    expect(submitted).toEqual({
      workflow: "nuclei-web@1",
      projectIds: ["p1", "p2"],
      parameters: { template_ids: "fixture" },
      rulepack: webScanPack.artifact,
    });
    await expect(
      page.getByRole("link", { name: "Open scan and results" }),
    ).toHaveAttribute("href", "/runs/run-0");
    await expect(page.getByText("Run Failed", { exact: false })).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: `/private/tmp/contractor-rule-sweep-${theme}.png`,
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Delete sweep", exact: true })
      .click();
    expect(deletes).toEqual([]);
    await page
      .getByRole("alertdialog")
      .getByRole("button", { name: "Delete sweep", exact: true })
      .click();
    await expect(page).toHaveURL(/\/checks\/sweeps$/);
    expect(deletes).toEqual(["/v1/scan-sweeps/sweep-one"]);
  });
}
