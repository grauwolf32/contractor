import { expect, test } from "@playwright/test";
import { artifactMetadata, artifactWrite } from "../src/test/builders";
import {
  draftAudit,
  projectFixture,
  startResponse,
  webScanProfile,
  engagementMaterial,
} from "../src/routes/checks/start/test-fixtures";
import { page as pageResponse } from "../src/test/responses";
import {
  authSession,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

for (const theme of ["light", "dark", "black"]) {
  for (const width of [390, 1440]) {
    test(`Web scan Audit with custom project rules at ${width}px in ${theme}`, async ({
      page,
    }, testInfo) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (value) => localStorage.setItem("contractor.theme", value),
        theme,
      );
      const project = projectFixture({
        httpTarget: { url: "https://shop.test/" },
      });
      const rules = artifactMetadata("rules", "project-check", {
        mediaType: "application/yaml",
      });
      const draft = draftAudit(webScanProfile);
      let submitted:
        | {
            profile: { name: string };
            inputs: Record<string, unknown>;
            labels: Record<string, string>;
            runtimeLabels: string[];
          }
        | undefined;
      const writes: string[] = [];
      await mockAPI(
        page,
        async (route, request, url) => {
          if (
            url.pathname === "/v1/projects/project_shop" &&
            request.method() === "GET"
          )
            return fulfillJSON(route, project, 200, { etag: '"1"' });
          if (url.pathname === "/v1/audit-profiles")
            return fulfillJSON(route, pageResponse([webScanProfile]));
          if (url.pathname === "/v1/audit-profiles/web-scan/versions/1")
            return fulfillJSON(route, webScanProfile, 200, {
              etag: `"${webScanProfile.ref.digest}"`,
            });
          if (
            url.pathname === "/v1/projects/project_shop/artifacts" &&
            request.method() === "GET"
          )
            return fulfillJSON(
              route,
              pageResponse([rules, engagementMaterial]),
            );
          if (request.method() === "PUT") {
            writes.push(url.pathname);
            const metadata = artifactMetadata(
              "scan-inputs",
              url.pathname.split("/").at(-1)!,
              {
                mediaType: request.headers()["content-type"]!,
                size: request.postDataBuffer()!.length,
              },
            );
            return fulfillJSON(route, artifactWrite(metadata), 201, {
              etag: `"${metadata.artifact.revision}"`,
            });
          }
          if (
            url.pathname === "/v1/projects/project_shop/audits" &&
            request.method() === "POST"
          ) {
            submitted = request.postDataJSON() as typeof submitted;
            return fulfillJSON(route, draft, 201, { etag: '"1"' });
          }
          if (url.pathname.endsWith("/start") && request.method() === "POST")
            return fulfillJSON(route, startResponse(draft), 200, {
              etag: '"2"',
            });
          if (url.pathname === `/v1/audits/${draft.auditId}`)
            return fulfillJSON(route, draft, 200, { etag: '"1"' });
          return fulfillEmptyPageOrNotFound(route);
        },
        { session: authSession("owner", ["user"]) },
      );
      await page.goto("/checks/new?project=project_shop&type=web-scan");
      await expect(page.getByLabel("Target URL", { exact: false })).toHaveValue(
        "https://shop.test/",
      );
      await page.getByRole("radio", { name: /Custom Nuclei rules/ }).check();
      await expect(
        page.getByRole("button", { name: "Start check", exact: true }),
      ).toBeDisabled();
      await page
        .getByLabel("Project rulepack", { exact: false })
        .selectOption(
          `${rules.artifact.namespace}/${rules.artifact.name}@${rules.artifact.revision}`,
        );
      await expect(
        page.getByRole("button", { name: "Start check", exact: true }),
      ).toBeEnabled();
      await page.getByText("Advanced options", { exact: true }).click();
      await page.getByRole("button", { name: "+ cloud", exact: true }).click();
      await page.getByLabel("Label value 1").fill("aws");
      const options = page
        .locator("details.start-advanced")
        .filter({ hasText: "Labels, version and check limits." });
      await options.getByText("Technical details", { exact: true }).click();
      await page.getByLabel("Runtime configuration labels").fill("debug");
      await page
        .getByLabel("Time limit", { exact: true })
        .selectOption("604800");
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      ).toBe(true);
      await page.screenshot({
        path: testInfo.outputPath(`web-scan-${width}-${theme}.png`),
        fullPage: true,
      });
      await page
        .getByRole("button", { name: "Save as draft", exact: true })
        .click();
      await expect(page).toHaveURL(
        new RegExp(`/projects/project_shop/audits/${draft.auditId}$`),
      );
      expect(submitted?.profile.name).toBe("web-scan");
      expect(submitted?.labels).toEqual({ cloud: "aws" });
      expect(submitted?.runtimeLabels).toEqual(["debug"]);
      expect(submitted?.inputs.rulepack).toEqual(rules.artifact);
      expect(Object.keys(submitted?.inputs ?? {}).sort()).toEqual([
        "engagement",
        "rulepack",
        "target",
      ]);
      expect(writes).toHaveLength(1);
      expect(
        writes.every((path) =>
          path.startsWith("/v1/projects/project_shop/artifacts/"),
        ),
      ).toBe(true);
    });
  }
}
