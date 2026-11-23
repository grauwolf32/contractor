import { expect, test } from "@playwright/test";

test.skip(
  process.env.CONTRACTOR_SCAN_BROWSER !== "1",
  "Requires the real scanner stack",
);
test("real scanner Findings expose retained evidence and their source Run", async ({
  page,
}) => {
  const project = process.env.CONTRACTOR_SCAN_FINDING_PROJECT!;
  const run = process.env.CONTRACTOR_SCAN_FINDING_RUN!;
  await page.goto(`/projects/${project}/findings`);
  await page
    .getByLabel("Username")
    .fill(process.env.CONTRACTOR_UI_E2E_USERNAME!);
  await page
    .getByLabel("Password")
    .fill(process.env.CONTRACTOR_UI_E2E_PASSWORD!);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(
    page.getByText("Exposed Git configuration", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText("SQL injection reported by SQLMap", { exact: true }).first(),
  ).toBeVisible();
  await page
    .getByText("Exposed Git configuration", { exact: true })
    .first()
    .click();
  await page.getByRole("tab", { name: /^Evidence/ }).click();
  await expect(
    page
      .getByRole("heading", { name: "Retained evidence", exact: true })
      .first(),
  ).toBeVisible();
  await page.getByRole("tab", { name: "History", exact: true }).click();
  await page.getByRole("button", { name: /^project-nuclei-scan/ }).click();
  await expect(
    page.getByRole("link", { name: run, exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: run, exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${run}$`));
  await page.goto(`/runs/${run}/artifacts/scanner/report`);
  await expect(
    page.getByRole("heading", { name: "scanner/report", exact: true }).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Load preview", exact: true }).click();
  await expect(
    page.getByRole("region", { name: "Scanner execution", exact: true }),
  ).toBeVisible();
  const audit = process.env.CONTRACTOR_SCAN_FINDING_AUDIT!;
  await page.goto(`/projects/${project}/audits/${audit}/work`);
  await expect(
    page
      .getByRole("region", { name: "Finding results", exact: true })
      .getByRole("link"),
  ).toHaveCount(1);
  await expect(
    page.getByRole("button", {
      name: "Download complete Run result",
      exact: true,
    }),
  ).toHaveCount(0);
  const empty = process.env.CONTRACTOR_SCAN_EMPTY_AUDIT!;
  await page.goto(`/projects/${project}/audits/${empty}/work`);
  await expect(
    page.getByText("No Findings recorded for this check.", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Download complete Run result",
      exact: true,
    }),
  ).toHaveCount(0);
});
