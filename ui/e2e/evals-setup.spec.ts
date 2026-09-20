import { expect, test } from "@playwright/test";
import { installEvalFixture } from "./evals-fixture";

for (const width of [390, 1280]) {
  for (const kind of ["workflow", "audit"]) {
    test(`native ${kind} setup, reload and explicit controls at ${width}px`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 900 });
      const fixture = await installEvalFixture(page);
      await page.goto("/evals/new?project=evaluation-1");
      await page
        .getByLabel("Experiment name", { exact: true })
        .fill(`Browser ${kind}`);
      await page
        .getByLabel("Execution kind", { exact: true })
        .selectOption(kind);
      const label = kind === "audit" ? "AuditProfile" : "Workflow";
      const prefix = kind === "audit" ? "audit" : "trace";
      await page
        .getByLabel(`A ${label} family`, { exact: true })
        .selectOption(`${prefix}-a`);
      await page
        .getByLabel(`B ${label} family`, { exact: true })
        .selectOption(`${prefix}-b`);
      await expect(
        page.getByLabel("A exact version", { exact: true }),
      ).toHaveValue(`${prefix}-a@2`);
      await page
        .getByRole("button", { name: "Next step", exact: true })
        .click();
      await page
        .getByLabel("Dataset revision", { exact: true })
        .selectOption("r1");
      await page
        .getByRole("button", { name: "Select all 2 cases", exact: true })
        .click();
      await page
        .getByRole("button", { name: "Next step", exact: true })
        .click();
      await page
        .getByRole("button", { name: "Add assessment check", exact: true })
        .click();
      await page.getByLabel("Repetitions", { exact: true }).fill("2");
      await page
        .getByRole("button", { name: "Next step", exact: true })
        .click();
      await page
        .getByRole("button", { name: "Save draft", exact: true })
        .click();
      await expect(page).toHaveURL(/experiment-1\/setup/);
      await page.reload();
      await expect(
        page.getByLabel("Experiment name", { exact: true }),
      ).toHaveValue(`Browser ${kind}`);
      expect(
        fixture.state.requests.filter((r) => r.path.endsWith("/commands")),
      ).toHaveLength(0);
      await page.getByRole("button", { name: "Prepare", exact: true }).click();
      await expect(
        page.getByText("Verified preparation", { exact: true }),
      ).toBeVisible();
      expect(
        fixture.state.requests
          .filter((r) => r.path.endsWith("/commands"))
          .map((r) => (r.body as { kind: string }).kind),
      ).toEqual(["prepare"]);
      await page.getByRole("button", { name: "Start", exact: true }).click();
      await page
        .getByRole("button", { name: "Confirm start", exact: true })
        .click();
      await expect(
        page.getByRole("button", { name: "Pause", exact: true }),
      ).toBeEnabled();
      await page.getByRole("button", { name: "Cancel", exact: true }).click();
      await page.keyboard.press("Escape");
      await expect(
        page.getByRole("button", { name: "Cancel", exact: true }),
      ).toBeFocused();
      expect(
        fixture.state.requests.some(
          (r) => (r.body as { kind?: string } | undefined)?.kind === "cancel",
        ),
      ).toBe(false);
      await expect
        .poll(() =>
          page.evaluate(
            () => document.documentElement.scrollWidth <= window.innerWidth,
          ),
        )
        .toBe(true);
    });
  }
}

test("lost Start response survives a real page reload", async ({ page }) => {
  const fixture = await installEvalFixture(page, { prepared: true });
  fixture.state.lostCommand = true;
  await page.goto("/evals/experiments/experiment-1/setup");
  await page.getByRole("button", { name: "Start", exact: true }).click();
  await page
    .getByRole("button", { name: "Confirm start", exact: true })
    .click();
  await expect(
    page.getByText("Public API is unavailable", { exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByText("Start: completed.", { exact: true }),
  ).toBeVisible();
  const requests = fixture.state.requests.filter((r) =>
    r.path.endsWith("/commands"),
  );
  expect(requests).toHaveLength(2);
  expect(requests[1]).toMatchObject({
    key: requests[0]!.key,
    etag: requests[0]!.etag,
    body: requests[0]!.body,
  });
});

test("dataset authoring keeps private review material separate from visible cases and browser storage", async ({
  page,
}) => {
  const fixture = await installEvalFixture(page);
  await page.goto("/evals/datasets?project=evaluation-1");
  await page
    .getByRole("button", { name: "Create or import dataset", exact: true })
    .click();
  await page.getByLabel("Dataset ID", { exact: true }).fill("browser-cases");
  await page.getByLabel("Dataset name", { exact: true }).fill("Browser cases");
  await page.getByLabel("Case ID", { exact: true }).fill("case-one");
  await page
    .getByLabel("Visible task objective", { exact: true })
    .fill("Inspect the retained source");
  await page.getByText("Private human review rubrics", { exact: true }).click();
  await page
    .getByRole("button", { name: "Add human rubric", exact: true })
    .click();
  await page.getByLabel("Review check ID", { exact: true }).fill("review");
  await page.getByLabel("Rubric revision", { exact: true }).fill("r1");
  await page
    .getByLabel("Private rubric", { exact: true })
    .fill("PRIVATE_BROWSER_SENTINEL");
  await page
    .getByRole("button", { name: "Save dataset revision", exact: true })
    .click();
  await expect(page.getByText("Browser cases", { exact: true })).toBeVisible();
  const body = fixture.state.requests.find(
    (r) => r.method === "POST" && r.path.endsWith("/eval-datasets"),
  )!.body as { cases: unknown; privateChecks: unknown };
  expect(JSON.stringify(body.cases)).not.toContain("PRIVATE_BROWSER_SENTINEL");
  expect(JSON.stringify(body.privateChecks)).toContain(
    "PRIVATE_BROWSER_SENTINEL",
  );
  expect(await page.evaluate(() => JSON.stringify(localStorage))).not.toContain(
    "PRIVATE_BROWSER_SENTINEL",
  );
});
