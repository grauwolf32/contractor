import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { test, expect } from "@playwright/test";
import {
  assertionProfileFixture,
  profileFixture,
  projectFixture,
  sourceZip,
  openapiYaml,
} from "../src/routes/checks/start/test-fixtures";
import {
  fulfillJSON,
  fulfillEmptyPageOrNotFound,
  mockAPI,
} from "./support/mock-api";

for (const theme of ["light", "dark", "black"])
  for (const width of [1440, 390]) {
    test(`New check keeps its compact current-contract composer at ${width}px in ${theme}`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (t) => localStorage.setItem("contractor.theme", t),
        theme,
      );
      const profiles = [
        assertionProfileFixture(),
        profileFixture("owasp-top10-2025-source-risk"),
        profileFixture("owasp-asvs-5-0-l1-source-pilot"),
        profileFixture("openapi-operation-trace", {
          inputs: {
            source: { required: true, media_types: ["application/zip"] },
            openapi: { required: true, media_types: ["application/yaml"] },
          },
        }),
      ];
      const mutations: string[] = [];
      await mockAPI(page, async (route, request, url) => {
        if (request.method() !== "GET") mutations.push(url.pathname);
        if (url.pathname === "/v1/projects/project_shop")
          return fulfillJSON(route, projectFixture(), 200, {
            ETag: `"${projectFixture().revision}"`,
          });
        if (url.pathname === "/v1/audit-profiles")
          return fulfillJSON(route, {
            items: profiles,
            page: { hasMore: false },
          });
        const profile = profiles.find(
          (p) => url.pathname === `/v1/audit-profiles/${p.ref.name}/versions/1`,
        );
        if (profile)
          return fulfillJSON(route, profile, 200, {
            ETag: `"${profile.ref.digest}"`,
          });
        if (url.pathname === "/v1/projects/project_shop/artifacts")
          return fulfillJSON(route, {
            items: [sourceZip, openapiYaml],
            page: { hasMore: false },
          });
        return fulfillEmptyPageOrNotFound(route);
      });
      await page.goto("/checks/new?project=project_shop&type=source-checklist");
      await expect(
        page.getByRole("heading", {
          name: "Custom source checklist",
          exact: true,
        }),
      ).toBeVisible();
      await expect(
        page.getByRole("heading", { name: "Scope", exact: true }),
      ).toHaveCount(0);
      await page
        .getByRole("button", { name: "Write a checklist", exact: true })
        .click();
      await expect(
        page.getByRole("textbox", { name: "Check 1", exact: true }),
      ).toHaveValue("");
      await expect(
        page.getByRole("textbox", { name: "Check 2", exact: true }),
      ).toHaveCount(0);
      await page
        .getByRole("textbox", { name: "Check 1", exact: true })
        .fill("A user can read a different project's export");
      await page
        .getByRole("textbox", { name: "Why check these?", exact: true })
        .fill("Exports contain private data");
      await page
        .getByRole("button", { name: "Add check", exact: true })
        .click();
      await expect(
        page.getByRole("button", {
          name: "Save and attach checklist",
          exact: true,
        }),
      ).toBeDisabled();
      await page
        .getByRole("button", { name: "Remove check 2", exact: true })
        .click();
      await expect(
        page.getByRole("button", {
          name: "Save and attach checklist",
          exact: true,
        }),
      ).toBeEnabled();
      await expect(
        page.getByRole("button", { name: "Start check", exact: true }),
      ).toBeDisabled();
      expect(mutations).toEqual([]);
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(width);
      if (process.env.CONTRACTOR_UI_POLISH_CAPTURE_DIR) {
        await mkdir(process.env.CONTRACTOR_UI_POLISH_CAPTURE_DIR, {
          recursive: true,
        });
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({
          path: join(
            process.env.CONTRACTOR_UI_POLISH_CAPTURE_DIR,
            `new-check-${theme}-${width}.png`,
          ),
          fullPage: true,
        });
      }
    });
  }
