import { expect, test } from "@playwright/test";
import {
  RESEARCH_AUDIT,
  RESEARCH_PROJECT,
} from "../src/test/audit-research-fixtures";
import { installResearchAPI } from "./support/research-api";

const root = `/projects/${RESEARCH_PROJECT}/audits/${RESEARCH_AUDIT}`;
for (const theme of ["light", "dark", "black"])
  test(`timeline status stays a pill when results expand in ${theme}`, async ({
    page,
  }) => {
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    await installResearchAPI(page);
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(root);
      const row = page.locator(".research-timeline > li").filter({
        has: page.getByRole("heading", {
          name: "A customer can read another customer’s order",
          exact: true,
        }),
      });
      const badge = row.getByText("Collected", { exact: true });
      await expect(badge).toBeVisible();
      const before = (await badge.boundingBox())!;
      expect(before.height).toBeLessThan(32);
      await row
        .getByRole("button", { name: "Read saved result", exact: true })
        .click();
      await expect(
        row.getByRole("heading", { name: "Recorded result", exact: true }),
      ).toBeVisible();
      const after = (await badge.boundingBox())!;
      expect(after.height).toBeCloseTo(before.height, 0);
    }
  });

for (const theme of ["light", "dark", "black"])
  test(`combined Checks research views in ${theme}`, async ({ page }, info) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    const fixture = await installResearchAPI(page);
    for (const width of [1440, 1100, 390, 320])
      for (const section of [
        "overview",
        "work",
        "hypotheses",
        "findings",
        "materials",
        "activity",
      ]) {
        await page.setViewportSize({ width, height: 1000 });
        await page.goto(root + (section === "overview" ? "" : "/" + section));
        await expect(
          page.getByRole("heading", { name: "Source checklist", exact: true }),
        ).toBeVisible();
        await expect(
          page.getByRole("navigation", { name: "Check sections" }),
        ).toBeVisible();
        if (section === "overview")
          await expect(
            page.getByText("8 accepted tasks", { exact: true }),
          ).toBeVisible();
        if (section === "work")
          await expect(
            page.getByRole("button", {
              name: "Read saved result",
              exact: true,
            }),
          ).toBeVisible();
        if (section === "hypotheses") {
          await expect(
            page.getByRole("button", {
              name: /Product search input can change/,
            }),
          ).toBeVisible();
          const readable = await page
            .getByRole("button", { name: /Product search input can change/ })
            .evaluate((button) => {
              const box = button.getBoundingClientRect();
              const range = document.createRange();
              range.selectNodeContents(button.querySelector("strong")!);
              const label = range.getBoundingClientRect();
              return (
                label.height > 0 &&
                label.bottom <= box.bottom &&
                label.top >= box.top &&
                label.right <= box.right + 1
              );
            });
          expect(readable).toBe(true);
        }
        if (section === "materials")
          await expect(
            page.getByRole("heading", {
              name: "order-api-observations",
              exact: true,
            }),
          ).toBeVisible();
        if (section === "activity")
          await expect(
            page.getByText("Task settled", { exact: true }),
          ).toBeVisible();
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBe(true);
        if (
          (width === 1440 || width === 390) &&
          ["overview", "work", "hypotheses", "materials"].includes(section)
        )
          await page.screenshot({
            path: info.outputPath(`${section}-${width}-${theme}.png`),
            fullPage: true,
          });
      }
    expect(
      fixture.calls.filter((c) => c.path.endsWith("/artifacts")),
    ).toHaveLength(0);
    expect(errors).toEqual([]);
  });

test("exact hypothesis result, retained grounds and source Run deletion", async ({
  page,
}) => {
  await installResearchAPI(page, "deleted-run");
  await page.goto(root + "/hypotheses");
  await page
    .getByRole("button", { name: /Product search input can change/ })
    .click();
  await page
    .getByRole("button", { name: "Read saved result", exact: true })
    .click();
  await expect(page.getByText("Refuted", { exact: true })).toBeVisible();
  await page.getByText("Run attempts", { exact: true }).click();
  await expect(page.getByText(/source Run deleted/)).toBeVisible();
  await page.getByRole("link", { name: /order-api-observations/ }).click();
  await expect(
    page.getByRole("heading", { name: "order-api-observations", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Preview", exact: true })
    .first()
    .click();
  await expect(
    page
      .getByLabel("Rendered", { exact: true })
      .getByText(/Order lookup and invoice exports accept/),
  ).toBeVisible();
});

test("ordinary work retries the original request after a lost acknowledgement", async ({
  page,
}) => {
  const fixture = await installResearchAPI(page, "ambiguous");
  await page.goto(root + "/work");
  await page.getByRole("button", { name: "Add work", exact: true }).click();
  await page
    .getByLabel("What should this work investigate?")
    .fill("Explore invoice access");
  await page.getByRole("combobox", { name: /^Method/ }).selectOption("inspect");
  await page
    .getByLabel("Material for source")
    .selectOption("material/order-api-observations");
  await page.getByRole("button", { name: "Accept work", exact: true }).click();
  await expect(
    page.getByLabel("What should this work investigate?"),
  ).toBeDisabled();
  await page
    .getByRole("button", { name: "Retry accepting work", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const writes = fixture.calls.filter(
    (c) => c.method === "POST" && c.path.endsWith("/tasks"),
  );
  expect(writes).toHaveLength(2);
  expect(writes[0]?.key).toBe(writes[1]?.key);
  expect(writes[0]?.body).toEqual(writes[1]?.body);
});
