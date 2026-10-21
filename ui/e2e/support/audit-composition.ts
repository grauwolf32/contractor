import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import type { Page, TestInfo } from "@playwright/test";

import type { Audit } from "../../src/api/audits";
import {
  compositionItems,
  compositionRows,
  COMPOSITION_AUDIT,
  COMPOSITION_PROJECT,
} from "../../src/test/audit-composition";
import { NOW, project, review, workspace } from "../../src/test/builders";
import {
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./mock-api";

export async function mockComposition(
  page: Page,
  current: () => Audit,
  start?: () => void,
  scanner: "sqlmap" | "nuclei" = "sqlmap",
) {
  await mockAPI(page, async (route, request, url) => {
    const value = current();
    const path = url.pathname;
    const inRound = value.phase === "rounds";
    if (path === `/v1/projects/${COMPOSITION_PROJECT}`)
      return fulfillJSON(
        route,
        project(COMPOSITION_PROJECT, { name: "Shop API" }),
        200,
        { etag: '"1"' },
      );
    if (path === `/v1/audits/${COMPOSITION_AUDIT}`)
      return fulfillJSON(route, value, 200, { etag: `"${value.revision}"` });
    if (
      path === `/v1/audits/${COMPOSITION_AUDIT}/start` &&
      request.method() === "POST"
    ) {
      start?.();
      return fulfillJSON(route, { audit: current(), items: [] }, 200, {
        etag: `"${current().revision}"`,
      });
    }
    if (
      path === "/v1/audits" ||
      path === `/v1/projects/${COMPOSITION_PROJECT}/audits`
    )
      return fulfillJSON(route, { items: [value], page: { hasMore: false } });
    if (path === `/v1/audits/${COMPOSITION_AUDIT}/coverage`)
      return fulfillJSON(route, {
        items: inRound ? compositionRows(scanner) : [],
        page: { hasMore: false },
        auditRevision: value.revision,
      });
    if (path === `/v1/audits/${COMPOSITION_AUDIT}/items`)
      return fulfillJSON(route, {
        items: inRound ? compositionItems() : [],
        page: { hasMore: false },
      });
    const pending = review(
      COMPOSITION_AUDIT,
      "review_composition",
      "active-check-approval",
      { subjectId: "item_1" },
    );
    if (path === `/v1/audits/${COMPOSITION_AUDIT}/reviews/review_composition`)
      return fulfillJSON(route, pending, 200, { etag: '"1"' });
    if (path === `/v1/audits/${COMPOSITION_AUDIT}/reviews`)
      return fulfillJSON(route, {
        items: inRound ? [pending] : [],
        page: { hasMore: false },
        auditRevision: value.revision,
        asOf: NOW,
        total: inRound ? 1 : 0,
      });
    if (path === `/v1/audits/${COMPOSITION_AUDIT}/workspace`)
      return fulfillJSON(
        route,
        workspace(
          value,
          inRound
            ? { totalChecks: 3, gaps: 1, unchecked: 2, pendingReviews: 1 }
            : {},
        ),
      );
    if (path === `/v1/audits/${COMPOSITION_AUDIT}/findings`)
      return fulfillJSON(route, {
        items: [],
        page: { hasMore: false },
        auditRevision: value.revision,
        asOf: NOW,
        total: 0,
      });
    if (path === `/v1/audits/${COMPOSITION_AUDIT}/events`)
      return fulfillJSON(route, {
        items: [],
        page: { hasMore: false },
        total: 0,
        throughSequence: 0,
      });
    if (path === `/v1/audits/${COMPOSITION_AUDIT}/artifacts`)
      return fulfillContent(
        route,
        url.searchParams.has("member")
          ? '{"observations":[{"message":"Investigate this scanner observation"}]}'
          : '{"openapi":"3.1.0","paths":{"/orders":{}}}',
        "application/json",
      );
    await fulfillEmptyPageOrNotFound(route);
  });
}

export async function compositionScreenshot(
  page: Page,
  testInfo: TestInfo,
  name: string,
) {
  const directory = process.env.CONTRACTOR_V62_SCREENSHOT_DIR;
  if (directory !== undefined) await mkdir(directory, { recursive: true });
  const path =
    directory === undefined ? testInfo.outputPath(name) : join(directory, name);
  await page.screenshot({ path, fullPage: true, animations: "disabled" });
  await testInfo.attach(name, { path, contentType: "image/png" });
}
