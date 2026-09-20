import { expect, test } from "@playwright/test";
import { installEvalFixture } from "./evals-fixture";

for (const width of [390, 1280]) {
  test(`Evals layout and authoring at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    const fixture = await installEvalFixture(page);
    fixture.state.experiment.draft!.variants.reverse();
    fixture.state.experiment.diagnostics = [
      {
        code: "eval_pin_mismatch",
        field: "modelPolicy",
        recovery: "edit_draft",
      },
    ];
    await page.goto("/evals");
    await expect(
      page.getByRole("heading", { name: "Experiments", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: "New experiment", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByLabel("Lifecycle", { exact: true }),
    ).not.toBeVisible();
    await page.getByText("Filter experiments", { exact: true }).click();
    await page.getByLabel("Lifecycle", { exact: true }).selectOption("draft");
    await expect(page).toHaveURL(/state=draft/);
    await page
      .getByRole("button", { name: "Clear filters", exact: true })
      .click();
    await page.screenshot({
      path: info.outputPath(`list-${width}.png`),
      fullPage: true,
    });
    await page.goto("/evals/experiments/experiment-1/setup");
    await expect(
      page.getByLabel("Experiment name", { exact: true }),
    ).toBeVisible();
    await expect(page.getByRole("alert")).toContainText(
      "Preparation needs attention",
    );
    await expect(
      page.getByLabel("A exact version", { exact: true }),
    ).toHaveValue("trace-a@1");
    const sections = page.getByLabel("Experiment section", { exact: true });
    if (width === 390) {
      await expect(sections).toBeVisible();
      await expect(
        page.getByRole("navigation", {
          name: "Experiment sections",
          exact: true,
        }),
      ).not.toBeVisible();
    } else {
      await expect(sections).not.toBeVisible();
    }
    await page.screenshot({
      path: info.outputPath(`setup-${width}.png`),
      fullPage: true,
    });
    await expect
      .poll(() =>
        page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      )
      .toBe(true);
    await expect(
      page.getByRole("button", { name: "Save draft", exact: true }),
    ).toBeDisabled();
    await page.getByText("Execution settings", { exact: true }).first().click();
    const labels = page.getByLabel("A runtime labels", { exact: true });
    await labels.fill("");
    await labels.pressSequentially("linux, gpu");
    await expect(labels).toHaveValue("linux, gpu");
    await expect(
      page.getByRole("button", { name: "Prepare", exact: true }),
    ).toBeDisabled();
    await page.getByRole("button", { name: "Save draft", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Prepare", exact: true }),
    ).toBeEnabled();
    expect(
      fixture.state.experiment.draft!.variants.find((v) => v.id === "a")!
        .runtimeLabels,
    ).toEqual(["linux", "gpu"]);
    if (width === 390) {
      await sections.selectOption("/evals/experiments/experiment-1/overview");
    } else {
      await page.getByRole("link", { name: "Overview", exact: true }).click();
    }
    await expect(page).toHaveURL(/experiment-1\/overview$/);
  });
}
