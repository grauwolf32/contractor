import { expect, test, type Page } from "@playwright/test";
import { findingSourcePreview } from "../src/test/builders";
import { issueReviewFixtures } from "../src/test/issue-review-fixtures";
import { ownerListResponse } from "../src/test/owner-lists";
import { page as listPage } from "../src/test/responses";
import {
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillError,
  fulfillJSON,
  fulfillResponse,
  mockAPI,
} from "./support/mock-api";

type Fixtures = ReturnType<typeof issueReviewFixtures>;
async function install(page: Page, data: Fixtures) {
  const payloads: string[] = [],
    mutations: string[] = [];
  await mockAPI(page, async (route, request, url) => {
    const path = url.pathname;
    if (request.method() !== "GET") {
      mutations.push(path);
      return fulfillError(route, 403, "forbidden", "Read-only design examples");
    }
    if (path === "/v1/projects")
      return fulfillJSON(route, listPage(data.projects));
    const owned = ownerListResponse(url, data);
    if (owned !== undefined) return fulfillResponse(route, owned);
    const project = data.projects.find(
      (value) => path === `/v1/projects/${value.projectId}`,
    );
    if (project !== undefined)
      return fulfillJSON(route, project, 200, { ETag: '"1"' });
    const audit = data.audits.find(
      (value) => path === `/v1/audits/${value.auditId}`,
    );
    if (audit !== undefined)
      return fulfillJSON(route, audit, 200, { ETag: '"1"' });
    const finding = data.findings.find(
      (value) =>
        path === `/v1/projects/${value.projectId}/findings/${value.findingId}`,
    );
    if (finding !== undefined)
      return fulfillJSON(route, finding, 200, { ETag: '"1"' });
    const check = data.audits.find(
      (value) => path === `/v1/audits/${value.auditId}/reviews`,
    );
    if (check !== undefined) {
      const items = data.reviews.filter(
        (value) =>
          value.auditId === check.auditId &&
          (!url.searchParams.has("state") ||
            value.state === url.searchParams.get("state")),
      );
      return fulfillJSON(route, {
        ...listPage(items),
        total: items.length,
        auditRevision: check.revision,
        asOf: check.updatedAt,
      });
    }

    const source = data.sources.find(
      (value) =>
        path ===
        `/v1/projects/${value.projectId}/finding-sources/${value.receiptId}`,
    );
    if (source !== undefined) return fulfillJSON(route, source);
    const sourceFinding = data.findings.find(
      (value) =>
        path ===
        `/v1/projects/${value.projectId}/findings/${value.findingId}/sources`,
    );
    if (sourceFinding !== undefined)
      return fulfillJSON(route, {
        items: data.sources
          .filter((value) => value.findingId === sourceFinding.findingId)
          .map(findingSourcePreview),
        projectRevision: 1,
        page: { hasMore: false },
      });
    if (path.endsWith("/history"))
      return fulfillJSON(route, {
        items: [],
        projectRevision: 1,
        page: { hasMore: false },
      });
    const evidence = data.evidence.find((value) => {
      const ref = value.metadata.artifact;
      return (
        (path ===
          `/v1/projects/${value.projectId}/artifacts/${ref.namespace}/${ref.name}` ||
          path ===
            `/v1/projects/${value.projectId}/artifacts/${ref.namespace}/${ref.name}/metadata`) &&
        url.searchParams.get("revision") === ref.revision
      );
    });
    if (evidence !== undefined) {
      if (path.endsWith("/metadata"))
        return fulfillJSON(route, evidence.metadata);
      payloads.push(url.toString());
      return fulfillContent(route, evidence.text, evidence.metadata.mediaType);
    }
    return fulfillEmptyPageOrNotFound(route);
  });
  return { payloads, mutations };
}
function findingPath(finding: Fixtures["findings"][number]) {
  return `/issues/${finding.projectId}/${finding.findingId}?state=all`;
}

for (const theme of ["light", "dark", "black"]) {
  for (const width of [1440, 1100, 390, 320]) {
    test(`Issue review composition: ${theme}, ${width}px, six finding kinds`, async ({
      page,
    }, info) => {
      const data = issueReviewFixtures();
      const { payloads, mutations } = await install(page, data);
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (value) => localStorage.setItem("contractor.theme", value),
        theme,
      );
      for (const finding of data.findings) {
        await page.goto(findingPath(finding));
        const review = page.getByRole("region", {
          name: "Review",
          exact: true,
        });
        await expect(
          review.getByRole("heading", {
            name: finding.title,
            exact: true,
          }),
        ).toBeVisible();
        await expect(
          review.getByLabel(/(?:Possible issue|Issue) \d of 6/),
        ).toBeVisible();
        await expect(review.getByText(/^Verification:/)).toBeVisible();
        if (finding.claim?.subject?.kind === "file") {
          await expect(review.locator(".issues-subject-kind")).toHaveText(
            "FILE",
          );
        }
        expect(
          await page.evaluate(() => document.documentElement.scrollWidth),
        ).toBeLessThanOrEqual(width);
        const status = review.locator(".issues-identity-status");
        expect(await status.innerText()).toMatch(
          /Verification:.*\n(?:AI|Analyst)/,
        );
        if (finding.analystVerdict !== null)
          await expect(
            review.getByRole("region", { name: "Recorded decision" }),
          ).toBeVisible();
        else
          await expect(
            review.getByRole("button", {
              name: "Mark as duplicate",
              exact: true,
            }),
          ).toBeVisible();
        const impact = review.getByRole("region", {
          name: "If real, the impact",
        });
        const verification = review.getByRole("region", {
          name: "Verification",
          exact: true,
        });
        expect((await impact.boundingBox())!.y).toBeLessThan(
          (await verification.boundingBox())!.y,
        );
        for (const file of await review
          .locator(".issues-evidence-file:has(.issues-evidence-toolbar)")
          .all()) {
          const toolbar = await file
            .locator(".issues-evidence-toolbar")
            .boundingBox();
          const details = file.locator(".issues-evidence-footer a");
          await expect(details).toHaveText("details →");
          expect((await details.boundingBox())!.y).toBeGreaterThanOrEqual(
            toolbar!.y + toolbar!.height,
          );
        }
        if (width <= 390) {
          const header = review.locator(".issues-detail-bar");
          expect(
            await header.evaluate((node) => getComputedStyle(node).position),
          ).toBe("static");
          expect(
            await review
              .locator(".ui-detail-pane-footer")
              .evaluate((node) => getComputedStyle(node).position),
          ).toBe("static");
          const prev = await review
            .getByRole("button", { name: "Previous possible issue" })
            .boundingBox();
          const next = await review
            .getByRole("button", { name: "Next possible issue" })
            .boundingBox();
          const count = await review
            .getByLabel(/(?:Possible issue|Issue) \d of 6/)
            .boundingBox();
          const link = await review
            .getByRole("link", { name: "View in Project" })
            .boundingBox();
          expect(Math.abs(prev!.y - next!.y)).toBeLessThan(2);
          expect(count!.x).toBeGreaterThanOrEqual(prev!.x + prev!.width);
          expect(count!.x + count!.width).toBeLessThanOrEqual(next!.x);
          expect(link!.y).toBeGreaterThanOrEqual(prev!.y + prev!.height);
          expect((await header.boundingBox())!.height).toBeLessThan(350);
        }
        if (finding.findingId === "issue-product-search-sql") {
          await page.screenshot({
            path: info.outputPath(`issue-${theme}-${width}.png`),
            fullPage: true,
          });
        }
      }
      expect(payloads).toHaveLength(4);
      for (const url of payloads) {
        const file = data.evidence.find((value) =>
          url.includes(
            `/artifacts/${value.metadata.artifact.namespace}/${value.metadata.artifact.name}?revision=${value.metadata.artifact.revision}`,
          ),
        );
        expect(file?.metadata.mediaType).toBe("text/plain");
        expect(file!.metadata.size).toBeLessThanOrEqual(64 * 1024);
      }
      expect(mutations).toEqual([]);
      expect(errors).toEqual([]);
    });
  }
}

test("sort/search preserve a draft; explicit evidence preview/download use the exact retained revision", async ({
  page,
}) => {
  const data = issueReviewFixtures();
  const sql = data.findings.find(
    (value) => value.findingId === "issue-product-search-sql",
  )!;
  const { payloads, mutations } = await install(page, data);
  await page.goto(findingPath(sql));
  const review = page.getByRole("region", { name: "Review", exact: true });
  await expect(
    review.getByRole("button", { name: "Mark as duplicate", exact: true }),
  ).toBeEnabled();
  await review
    .getByRole("button", { name: "Not an issue", exact: true })
    .click();
  const reason = review.getByRole("textbox", { name: "Why", exact: true });
  await reason.fill("Keep this unsaved analyst rationale.");
  await expect(
    review.getByRole("button", { name: "Next item", exact: true }),
  ).toHaveCount(0);

  await page.getByLabel("Sort issues", { exact: true }).selectOption("title");
  await expect(reason).toHaveValue("Keep this unsaved analyst rationale.");
  await expect(page).toHaveURL(/sort=title/);
  await page.getByRole("searchbox", { name: "Search issues" }).fill("SQL");
  await expect(page.locator(".issues-list-nav .ui-row")).toHaveCount(1);
  await expect(reason).toHaveValue("Keep this unsaved analyst rationale.");
  const file = review.locator(".issues-evidence-file").first();
  await expect(
    file.getByRole("button", { name: "Hide preview", exact: true }),
  ).toBeVisible();
  const evidence = data.evidence.find(
    (value) => value.metadata.artifact.name === "search-route",
  )!;
  await expect(file.getByText(evidence.text, { exact: true })).toBeVisible();
  expect(payloads.filter((url) => url.includes("/search-route?"))).toHaveLength(
    1,
  );
  expect(payloads.find((url) => url.includes("/search-route?"))).toContain(
    "/projects/project_shop/artifacts/evidence/search-route?revision=search-route-r1",
  );
  const downloaded = page.waitForEvent("download");
  await file
    .getByRole("button", { name: "Download search-route", exact: true })
    .click();
  expect((await downloaded).suggestedFilename()).toContain("search-route");
  expect(mutations).toEqual([]);
});

test("mobile review uses four verdicts and a full-width rationale/submit, with no redundant Next action", async ({
  page,
}) => {
  const data = issueReviewFixtures();
  const sql = data.findings.find(
    (value) => value.findingId === "issue-product-search-sql",
  )!;
  await install(page, data);
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto(findingPath(sql));
    const review = page.getByRole("region", { name: "Review", exact: true });
    const bar = review.getByRole("region", {
      name: "Your decision",
      exact: true,
    });
    const actions = bar.locator(".ui-decision-actions > button");
    await expect(actions).toHaveCount(4);
    await expect(
      review.getByRole("button", { name: "Next item", exact: true }),
    ).toHaveCount(0);
    const boxes = await actions.evaluateAll((nodes) =>
      nodes.map((node) => {
        const box = node.getBoundingClientRect();
        return { x: box.x, y: box.y, width: box.width, height: box.height };
      }),
    );
    expect(Math.abs(boxes[0]!.y - boxes[1]!.y)).toBeLessThan(2);
    expect(Math.abs(boxes[2]!.y - boxes[3]!.y)).toBeLessThan(2);
    expect(boxes[2]!.y).toBeGreaterThan(boxes[0]!.y);
    await bar
      .getByRole("button", { name: "Not an issue", exact: true })
      .click();
    const field = bar.getByRole("textbox", { name: "Why", exact: true });
    await expect(field).toBeFocused();
    await field.fill("The saved source confirms the check.");
    const record = bar.getByRole("button", {
      name: "Record decision",
      exact: true,
    });
    await expect(record).toBeEnabled();
    const fieldBox = await field.boundingBox(),
      recordBox = await record.boundingBox();
    expect(fieldBox!.height).toBeGreaterThanOrEqual(72);
    expect(recordBox!.y).toBeGreaterThanOrEqual(fieldBox!.y + fieldBox!.height);
    expect(Math.abs(recordBox!.width - fieldBox!.width)).toBeLessThan(2);
    await expect(
      page.getByRole("button", { name: "Next possible issue", exact: true }),
    ).toBeEnabled();
  }
});
