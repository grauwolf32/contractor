import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, type Page, type Download } from "@playwright/test";
import type { components } from "../../src/api/generated/public";

type Schemas = components["schemas"];
async function downloadedJSON(
  download: Download,
): Promise<Record<string, unknown>> {
  const path = await download.path();
  expect(path).not.toBeNull();
  return JSON.parse(await readFile(path!, "utf8")) as Record<string, unknown>;
}

/** The UI owns selection/options; the real Server owns grouping and saved bytes. */
export async function exerciseFindingExports(
  page: Page,
  audit: Schemas["Audit"],
  findings: Schemas["AuditFinding"][],
) {
  await page.goto(
    `/projects/${audit.projectId}/audits/${audit.auditId}/findings`,
  );
  const panel = page.locator("details.issues-export");
  await panel.locator("summary").first().click();
  await expect(
    page.getByRole("checkbox", {
      name: /Select Embedded private key for export/,
    }),
  ).toHaveCount(2);
  await panel.getByRole("button", { name: "Select this page" }).click();
  await expect(
    panel.getByRole("checkbox", { name: "Include duplicates" }),
  ).not.toBeChecked();
  const publish = async () => {
    const response = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        response.url().endsWith(`/v1/audits/${audit.auditId}/findings/exports`),
    );
    await panel.getByRole("button", { name: "Create SARIF export" }).click();
    const value = await response;
    expect(value.status(), await value.text()).toBe(201);
    return (await value.json()) as Schemas["PublishedFindingExport"];
  };
  const first = await publish();
  expect(first.selectedCount).toBe(2);
  expect(first.resultCount).toBe(1);
  const downloadSARIF = async (count: number) => {
    const waiting = page.waitForEvent("download");
    await panel.getByRole("button", { name: "Download SARIF" }).click();
    const download = await waiting;
    expect(download.suggestedFilename()).toBe(`audit-${audit.auditId}.sarif`);
    const document = await downloadedJSON(download);
    expect(document.version).toBe("2.1.0");
    const runs = document.runs as Array<{
      tool: { driver: Record<string, unknown> };
      results: Array<{
        properties: { contractor: { selected_finding_ids: string[] } };
      }>;
      properties: {
        contractor: { audit_id: string; include_duplicates: boolean };
      };
    }>;
    expect(runs).toHaveLength(1);
    expect(runs[0]!.tool.driver).not.toHaveProperty("supportedTaxonomies");
    expect(runs[0]!.properties.contractor.audit_id).toBe(audit.auditId);
    expect(runs[0]!.results).toHaveLength(count);
    expect(
      runs[0]!.results
        .flatMap((r) => r.properties.contractor.selected_finding_ids)
        .sort(),
    ).toEqual(findings.map((f) => f.findingId).sort());
  };
  await downloadSARIF(1);
  await panel.getByRole("checkbox", { name: "Include duplicates" }).check();
  const expanded = await publish();
  expect(expanded.exportId).not.toBe(first.exportId);
  expect(expanded.selectedCount).toBe(2);
  expect(expanded.resultCount).toBe(2);
  await downloadSARIF(2);
  await panel.getByText("Technical details", { exact: true }).click();
  const waiting = page.waitForEvent("download");
  await panel.getByRole("button", { name: "Download snapshot" }).click();
  const snapshot = await downloadedJSON(await waiting);
  expect(snapshot.audit_id).toBe(audit.auditId);
  expect(
    (snapshot.results as Array<{ finding_id: string }>)
      .map((r) => r.finding_id)
      .sort(),
  ).toEqual(findings.map((f) => f.findingId).sort());
  expect(snapshot.request).toMatchObject({ includeDuplicates: true });

  const original = await page.evaluate(
    () => localStorage.getItem("contractor.theme") ?? "system",
  );
  const changeTheme = async (theme: string) => {
    await page
      .getByRole("navigation", { name: "Primary navigation" })
      .getByRole("button", { name: "Account", exact: true })
      .click();
    await page
      .getByRole("radio", {
        name: theme[0]!.toUpperCase() + theme.slice(1),
        exact: true,
      })
      .check();
    await page.keyboard.press("Escape");
  };
  for (const theme of ["light", "dark", "black"]) {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await changeTheme(theme);
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await expect(
        panel.getByRole("button", { name: "Download SARIF" }),
      ).toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await panel.screenshot({
        path: join(tmpdir(), `finding-export-${theme}-${width}.png`),
      });
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await changeTheme(original);
}
