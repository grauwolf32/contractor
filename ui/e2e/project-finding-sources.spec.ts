import { expect, test } from "@playwright/test";
import {
  finding,
  findingSource,
  findingSourcePreview,
  project,
} from "../src/test/builders";
import { ownerListResponse } from "../src/test/owner-lists";
import {
  fulfillJSON,
  fulfillEmptyPageOrNotFound,
  mockAPI,
} from "./support/mock-api";

for (const width of [390, 1440]) {
  for (const theme of ["light", "dark", "black"]) {
    test(`canonical Finding retains independent sources at ${width}px in ${theme}`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (theme) => localStorage.setItem("contractor.theme", theme),
        theme,
      );
      const current = finding(
        "project_shared",
        "finding_shared",
        {},
        { title: "One retained problem" },
      );
      const first = findingSource(current);
      const second = findingSource(
        current,
        {
          receiptId: "source_second",
          proposalId: "proposal_second",
          origin: {
            ...first.origin,
            runId: "run_second",
            audit: {
              auditId: "audit_second",
              executionId: "execution_second",
              taskId: "task_second",
            },
            workflow: { ...first.origin.workflow, name: "independent-check" },
          },
        },
        { description: "Independent result from the second check" },
      );
      const data = {
        projects: [project(current.projectId)],
        audits: [],
        findings: [current],
      };
      const reads: string[] = [];
      await mockAPI(page, async (route, request, url) => {
        expect(request.method()).toBe("GET");
        const owner = ownerListResponse(url, data);
        if (owner !== undefined) return fulfillJSON(route, await owner.json());
        if (url.pathname === "/v1/projects")
          return fulfillJSON(route, {
            items: data.projects,
            page: { hasMore: false },
          });
        if (url.pathname === `/v1/projects/${current.projectId}`)
          return fulfillJSON(route, data.projects[0], 200, { ETag: '"1"' });
        if (
          url.pathname ===
          `/v1/projects/${current.projectId}/findings/${current.findingId}`
        )
          return fulfillJSON(route, current, 200, { ETag: '"1"' });
        if (
          url.pathname ===
            `/v1/projects/${current.projectId}/finding-sources` &&
          url.searchParams.get("findingId") === current.findingId
        ) {
          reads.push(url.toString());
          return fulfillJSON(route, {
            items: [
              findingSourcePreview(
                url.searchParams.has("cursor") ? second : first,
              ),
            ],
            projectRevision: 1,
            page: url.searchParams.has("cursor")
              ? { hasMore: false }
              : { hasMore: true, nextCursor: "source-page-2" },
          });
        }
        const source = [first, second].find(
          (source) =>
            url.pathname ===
            `/v1/projects/${current.projectId}/finding-sources/${source.receiptId}`,
        );
        if (source !== undefined) return fulfillJSON(route, source);
        if (url.pathname.endsWith("/history"))
          return fulfillJSON(route, {
            items: [],
            projectRevision: 1,
            page: { hasMore: false },
          });
        return fulfillEmptyPageOrNotFound(route);
      });
      await page.goto(`/issues/${current.projectId}/${current.findingId}`);
      await expect(
        page.getByRole("heading", { name: current.title, exact: true }),
      ).toBeVisible();
      expect(reads).toEqual([]);
      await page.getByRole("tab", { name: "History" }).click();
      const sources = page.getByRole("region", { name: "Independent sources" });
      await expect(
        sources.getByRole("button", { name: "More sources" }),
      ).toBeVisible();
      expect(reads).toHaveLength(1);
      await sources.getByRole("button", { name: "More sources" }).click();
      await sources.getByRole("button", { name: /independent-check/ }).click();
      await expect(
        sources.getByText("Independent result from the second check", {
          exact: true,
        }),
      ).toBeVisible();
      await expect(
        sources.getByText("Original check: audit_second", { exact: true }),
      ).toBeVisible();
      await expect(
        sources.getByRole("region", { name: "Your decision" }),
      ).toHaveCount(0);
      expect(reads).toHaveLength(2);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await expect(
        page
          .getByRole("region", { name: "Review", exact: true })
          .getByText("Not assessed", { exact: true })
          .first(),
      ).toBeVisible();
    });
  }
}
