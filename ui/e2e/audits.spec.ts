import { mkdir } from "node:fs/promises";
import path from "node:path";
import { expect, test } from "@playwright/test";
import {
  commonAudit,
  COMMON_AUDIT,
  COMMON_PROJECT,
  mockCommonTasks,
} from "./support/audit-composition";

for (const width of [390, 1440]) {
  for (const theme of ["light", "dark", "black"]) {
    test(`common Audit tasks and exact retained results at ${width}px in ${theme}`, async ({
      page,
    }, info) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (value) => localStorage.setItem("contractor.theme", value),
        theme,
      );
      await mockCommonTasks(page, commonAudit);
      await page.goto(`/projects/${COMMON_PROJECT}/audits/${COMMON_AUDIT}`);
      await expect(
        page.getByRole("heading", {
          name: "Inspect the selected exact material",
        }),
      ).toBeVisible();
      await expect(page.getByText("Paused", { exact: true })).toBeVisible();
      await page.getByRole("button", { name: "Read saved result" }).click();
      await expect(page.getByText(/Exact saved observation/)).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Download complete Run result" }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "Project possible issues" }),
      ).toHaveAttribute("href", `/projects/${COMMON_PROJECT}/issues`);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      ).toBe(true);
      const evidence = process.env.CONTRACTOR_AUDIT_SCREENSHOT_DIR;
      if (evidence) await mkdir(evidence, { recursive: true });
      const screenshot = (name: string) =>
        evidence ? path.join(evidence, name) : info.outputPath(name);
      await page.screenshot({
        path: screenshot(`common-audit-${theme}-${width}.png`),
        fullPage: true,
        animations: "disabled",
      });
      await page.getByRole("button", { name: "Next", exact: true }).click();
      await expect(
        page.getByRole("heading", {
          name: "Inspect the second exact material",
        }),
      ).toBeVisible();
      await expect(
        page.getByRole("heading", {
          name: "Inspect the selected exact material",
        }),
      ).toHaveCount(0);
      await page.getByRole("button", { name: "Previous", exact: true }).click();
      await expect(
        page.getByRole("heading", {
          name: "Inspect the selected exact material",
        }),
      ).toBeVisible();
    });
  }
}
