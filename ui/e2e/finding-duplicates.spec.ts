import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "@playwright/test";

import { audit, project } from "../src/test/builders";
import {
  correlationAudit,
  correlationFinding,
  CORRELATION_PROJECT,
  members,
  observation,
  candidates,
  provenance,
} from "../src/test/finding-correlation";
import { ownerListResponse } from "../src/test/owner-lists";
import {
  fulfillJSON,
  fulfillEmptyPageOrNotFound,
  mockAPI,
} from "./support/mock-api";

for (const width of [390, 1440]) {
  for (const theme of ["light", "dark", "black"]) {
    test(`retained duplicates at ${width}px in ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (theme) => localStorage.setItem("contractor.theme", theme),
        theme,
      );
      const left = correlationFinding;
      const right = observation("right", {
        title:
          "Authorization check with a very long endpoint and independent analyst result",
        state: "confirmed",
        analystVerdict: "true_positive",
        analystSeverity: "low",
      });
      const imported = observation("imported", {
        origin: { ...right.origin, receiptId: left.firstProposal.receiptId },
      });
      const data = {
        projects: [project(CORRELATION_PROJECT)],
        audits: [correlationAudit],
        findings: [left],
      };
      const reads: string[] = [];
      await mockAPI(page, async (route, request, url) => {
        const owned = ownerListResponse(url, data);
        if (owned !== undefined) return fulfillJSON(route, await owned.json());
        const path = url.pathname;
        if (path === "/v1/projects")
          return fulfillJSON(route, {
            items: data.projects,
            page: { hasMore: false },
          });
        if (path === `/v1/projects/${CORRELATION_PROJECT}`)
          return fulfillJSON(route, data.projects[0], 200, { ETag: '"1"' });
        if (path === "/v1/audits/audit-left")
          return fulfillJSON(route, correlationAudit, 200, { ETag: '"1"' });
        if (path === "/v1/audits/audit-right")
          return fulfillJSON(
            route,
            audit(CORRELATION_PROJECT, "audit-right", "completed"),
            200,
            { ETag: '"1"' },
          );
        if (path === "/v1/audits/audit-left/findings/finding-left")
          return fulfillJSON(route, left, 200, { ETag: '"1"' });
        if (path === "/v1/audits/audit-left/reviews")
          return fulfillJSON(route, {
            auditRevision: 1,
            asOf: left.updatedAt,
            total: 0,
            items: [],
            page: { hasMore: false },
          });
        if (path.includes("/finding-correlations/")) {
          reads.push(request.method());
          return fulfillJSON(
            route,
            url.searchParams.has("cursor")
              ? members([imported])
              : members([right], true),
          );
        }
        if (path.endsWith("/provenance"))
          return fulfillJSON(route, provenance());
        if (path.endsWith("/duplicate-candidates"))
          return fulfillJSON(
            route,
            candidates([], !url.searchParams.has("cursor")),
          );
        return fulfillEmptyPageOrNotFound(route);
      });
      await page.goto("/issues/audit-left/finding-left");
      const section = page.locator(".issues-duplicates");
      await expect(section.locator("summary").first()).toHaveText("Duplicates");
      expect(reads).toEqual([]);
      await section.locator("summary").first().focus();
      await page.keyboard.press("Enter");
      await expect(
        section.getByText(right.title, { exact: true }),
      ).toBeVisible();
      await section
        .getByRole("button", { name: "Load more observations" })
        .click();
      await expect(
        section.getByText(/Same source proposal as this finding/),
      ).toBeVisible();
      await expect(section.getByText(/complete list/)).toBeVisible();
      await section.getByText("Check context", { exact: true }).first().click();
      await expect(
        section.getByText(/Coverage has not been compared/),
      ).toBeVisible();
      await section.getByText("Possible matches", { exact: true }).click();
      await expect(
        section.getByText(/reached its search-work limit/),
      ).toBeVisible();
      await section.getByRole("button", { name: "Continue searching" }).click();
      await expect(section.getByText(/search complete/)).toBeVisible();
      await expect(
        section.getByRole("link", { name: "Evidence" }).first(),
      ).toHaveAttribute("href", "/issues/audit-right/right?tab=evidence");
      expect(reads.every((method) => method === "GET")).toBe(true);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await expect(page.getByText(/The order endpoint may read/)).toBeVisible();
      await section.locator("summary").first().focus();
      await page.screenshot({
        path: join(tmpdir(), `finding-duplicates-${theme}-${width}.png`),
        fullPage: true,
      });
    });
  }
}
