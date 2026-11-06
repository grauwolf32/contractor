import { expect, test } from "@playwright/test";
import { audit, project } from "../src/test/builders";
import {
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

for (const theme of ["light", "dark", "black"]) {
  test(`assertion editing keeps the original denominator at 390px in ${theme}`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.addInitScript(
      (theme) => localStorage.setItem("contractor.theme", theme),
      theme,
    );
    const current = audit("project_assertions", "audit_assertions", "paused");
    current.collectionPending = true;
    let sourceRevision = 1;
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await mockAPI(page, async (route, request, url) => {
      if (url.pathname === `/v1/projects/${current.projectId}`)
        return fulfillJSON(route, project(current.projectId), 200, {
          ETag: '"1"',
        });
      if (url.pathname === `/v1/audits/${current.auditId}`)
        return fulfillJSON(route, current, 200, {
          ETag: `"${current.revision}"`,
        });
      if (url.pathname.endsWith("/hypotheses"))
        return fulfillJSON(route, {
          checking_workflows: ["inspect"],
          fixed_denominator: 2,
          pending_fixed_count: 1,
          items: [
            {
              document: {
                schema: "contractor.audit.hypothesis.v1",
                id: "hypothesis-original",
                revision: sourceRevision,
                statement:
                  sourceRevision === 1
                    ? "Only the selected owner can access the selected record"
                    : "Another author's updated assertion",
                scope: {
                  conditions: "The selected fixture and actor only",
                  required_evidence: "Exact inspected source",
                },
                grounds: "An original checklist requirement",
                origin: { kind: "checklist", audit_id: current.auditId },
              },
              source_key: "original",
              fixed: true,
              ordinal: 1,
            },
          ],
          page: { hasMore: false },
        });
      return fulfillEmptyPageOrNotFound(route);
    });
    await page.goto(`/projects/${current.projectId}/audits/${current.auditId}`);
    await page.getByRole("link", { name: "Hypotheses", exact: true }).click();
    await expect(
      page.getByText(/2 requirements in the original checklist/),
    ).toBeVisible();
    await expect(
      page.getByText(/1 original requirements have no accepted assertion/),
    ).toBeVisible();
    await page.getByRole("button", { name: "Edit assertion" }).click();
    await expect(page.getByRole("textbox", { name: /^Assertion/ })).toHaveValue(
      "Only the selected owner can access the selected record",
    );
    await expect(
      page.getByRole("button", { name: "Save and check" }),
    ).toBeVisible();
    await page
      .getByRole("textbox", { name: /^Assertion/ })
      .fill("My unsaved assertion");
    sourceRevision = 2;
    current.revision++;
    await expect(page.getByText(/Your draft is preserved/)).toBeVisible();
    await expect(page.getByRole("textbox", { name: /^Assertion/ })).toHaveValue(
      "My unsaved assertion",
    );
    await expect(
      page.getByRole("button", { name: "Save assertion" }),
    ).toBeDisabled();
    await expect(
      page.getByRole("button", { name: "Save and check" }),
    ).toBeDisabled();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    expect(errors).toEqual([]);
    await page.screenshot({
      path: testInfo.outputPath(`assertions-${theme}-390.png`),
      fullPage: true,
    });
    await page.getByRole("button", { name: "Edit assertion" }).click();
    await expect(page.getByRole("textbox", { name: /^Assertion/ })).toHaveValue(
      "Another author's updated assertion",
    );
    await expect(
      page.getByRole("button", { name: "Save assertion" }),
    ).toBeEnabled();
  });
}
