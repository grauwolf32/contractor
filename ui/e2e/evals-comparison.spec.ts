import { expect, test } from "@playwright/test";
import { installEvalFixture } from "./evals-fixture";

for (const width of [390, 1280]) {
  test(`comparison charts, pair evidence and owner review at ${width}px`, async ({
    page,
  }, info) => {
    await page.setViewportSize({ width, height: 900 });
    await installEvalFixture(page, { prepared: true });
    await page.goto("/evals/experiments/experiment-1/overview");
    const quality = page.getByRole("heading", {
      name: "Quality A/B",
      exact: true,
    });
    const progress = page.getByRole("heading", {
      name: "Execution progress",
      exact: true,
    });
    const overviewChart = page.getByLabel("Overview chart", { exact: true });
    await expect(quality).toBeVisible();
    if (width === 390) {
      await expect(progress).not.toBeVisible();
      await overviewChart.focus();
      await overviewChart.press("ArrowDown");
      await overviewChart.press("Enter");
      await expect(overviewChart).toHaveValue("progress");
      await expect(progress).toBeVisible();
      await expect(quality).not.toBeVisible();
    } else {
      await expect(overviewChart).not.toBeVisible();
      await expect(progress).toBeVisible();
    }
    await page.screenshot({
      path: info.outputPath(`overview-${width}.png`),
      fullPage: true,
    });
    await page.goto("/evals/experiments/experiment-1/comparison?filter=all");
    await expect(
      page.getByRole("heading", { name: "Token distribution", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("cell", { name: "125", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("cell", { name: "130", exact: true }),
    ).toBeVisible();
    await page.getByText("Show bin data table", { exact: true }).click();
    await expect(
      page.getByRole("table", { name: "Shared bins", exact: true }),
    ).toBeVisible();
    await page
      .getByLabel("Chart view", { exact: true })
      .selectOption("pair-deltas");
    await expect(
      page.getByRole("heading", { name: "Case differences", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("link", { name: "unsafe-query / sample 1", exact: true })
      .click();
    await expect(
      page.getByRole("heading", {
        name: "Attributed records and evidence",
        exact: true,
      }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Review A", exact: true }).click();
    await page
      .getByLabel("Decision for evidence-review", { exact: true })
      .selectOption("pass");
    await page
      .getByLabel("Reason for evidence-review", { exact: true })
      .fill("Verified exact browser evidence");
    await page
      .getByRole("button", { name: "Save and select assessment", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Review exact result", exact: true }),
    ).toHaveCount(0);
    await page.getByRole("link", { name: "← Comparison", exact: true }).click();
    await expect(page).toHaveURL(/filter=all/);
    await expect
      .poll(() =>
        page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      )
      .toBe(true);
    await page.screenshot({
      path: info.outputPath(`comparison-${width}.png`),
      fullPage: true,
    });
  });
}

test("external Audit keeps inspection and exact execution navigation without dispatch controls", async ({
  page,
}) => {
  await installEvalFixture(page, {
    prepared: true,
    external: true,
    audit: true,
  });
  await page.goto("/evals/experiments/experiment-1/comparison?filter=all");
  await expect(page.getByText(/Externally controlled/)).toBeVisible();
  for (const label of ["Start", "Pause", "Resume", "Cancel", "Duplicate"])
    await expect(
      page.getByRole("button", { name: label, exact: true }),
    ).toHaveCount(0);
  await page
    .getByRole("link", { name: "unsafe-query / sample 1", exact: true })
    .click();
  await expect(
    page.getByRole("link", { name: "audit audit-1", exact: true }).first(),
  ).toHaveAttribute("href", "/projects/member-project/audits/audit-1");
});
