import { expect, test, type Page } from "@playwright/test";
import { screenshotPolishFixture } from "../src/test/screenshot-polish-fixture";
import { fulfillContent, fulfillResponse, mockAPI } from "./support/mock-api";

async function install(
  page: Page,
  options: Parameters<typeof screenshotPolishFixture>[0] = {},
) {
  const fixture = screenshotPolishFixture(options);
  await mockAPI(page, async (route, request, url) => {
    const response = fixture.respond(
      new Request(request.url(), { method: request.method() }),
      url,
    );
    if (response.headers.has("content-length")) {
      return fulfillContent(
        route,
        await response.text(),
        response.headers.get("content-type")!,
        response.status,
      );
    }
    return fulfillResponse(route, response);
  });
  return fixture;
}
async function noOverflow(page: Page, width: number) {
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(width);
}

for (const theme of ["light", "dark", "black"]) {
  for (const width of [1440, 390]) {
    test(`Screenshot regressions: ${theme}, ${width}px`, async ({
      page,
    }, info) => {
      const fixture = await install(page);
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (value) => localStorage.setItem("contractor.theme", value),
        theme,
      );
      await page.goto(fixture.paths.issue);
      const decision = page.getByRole("region", {
        name: "Assess possible issue",
      });
      await expect(
        decision.getByRole("button", { name: "Change decision" }),
      ).toBeVisible();
      await expect(
        decision.getByRole("button", { name: "Confirm issue" }),
      ).toHaveCount(0);
      expect((await decision.boundingBox())!.height).toBeLessThan(90);
      await expect(
        page.getByText(
          "The synthetic review fixture records an export containing a row owned by another project.",
          { exact: true },
        ),
      ).toBeVisible();
      await noOverflow(page, width);
      await page.screenshot({
        path: info.outputPath(`issue-${theme}-${width}.png`),
        fullPage: true,
      });
      await decision.getByRole("button", { name: "Change decision" }).click();
      await decision.getByRole("button", { name: "Confirm issue" }).click();
      await expect(
        decision.getByRole("textbox", { name: "Why" }),
      ).toBeFocused();
      expect(
        await page
          .locator(".ui-detail-pane-footer")
          .evaluate((e) => getComputedStyle(e).position),
      ).toBe("static");
      await noOverflow(page, width);

      await page.goto(fixture.paths.overview);
      const workspace = page.locator(".projects-workspace");
      const header = workspace.locator(".ui-detail-header");
      await expect(
        header.getByRole("heading", { name: fixture.project.name }),
      ).toBeVisible();
      await expect(
        workspace.getByRole("button", { name: /^Refresh/ }),
      ).toHaveCount(1);
      await expect(
        workspace.getByText("No recorded cost", { exact: true }),
      ).toBeVisible();
      await expect(
        workspace
          .getByText("Result labels could not be loaded", { exact: true })
          .first(),
      ).toBeVisible();
      await expect(
        workspace.getByRole("button", { name: "Copy report revision" }),
      ).toBeVisible();
      const geometry = await header.boundingBox();
      await noOverflow(page, width);
      await page.screenshot({
        path: info.outputPath(`project-${theme}-${width}.png`),
        fullPage: true,
      });
      const result = workspace.getByRole("article", {
        name: "report",
        exact: true,
      });
      await result
        .getByRole("button", { name: "Preview", exact: true })
        .click();
      await expect(
        result.getByRole("heading", {
          name: "Invoice API review",
          exact: true,
        }),
      ).toBeVisible();
      await workspace
        .getByRole("link", { name: "Materials", exact: true })
        .click();
      await expect(
        workspace.getByRole("region", { name: "Materials", exact: true }),
      ).toBeVisible();
      const after = await header.boundingBox();
      expect(after!.height).toBe(geometry!.height);
      expect(after!.width).toBe(geometry!.width);
      await expect(
        workspace.getByRole("link", {
          name: "Preview outputs/template_manifest",
        }),
      ).toHaveAttribute("href", /\?revision=/);
      await expect(
        workspace.getByPlaceholder("all namespaces"),
      ).not.toBeVisible();
      await workspace.getByText("Filter by namespace", { exact: true }).click();
      await expect(workspace.getByPlaceholder("all namespaces")).toBeVisible();
      await noOverflow(page, width);
      await page.screenshot({
        path: info.outputPath(`materials-${theme}-${width}.png`),
        fullPage: true,
      });

      await page.goto(fixture.paths.run);
      const attempts = page.locator("#run-attempts");
      if ((await attempts.getAttribute("open")) === null)
        await attempts.locator(":scope > summary").click();
      const attempt = page.locator("#attempt-stage_review_failed");
      if ((await attempt.getAttribute("open")) === null)
        await attempt.locator(":scope > summary").click();
      await expect(
        page.getByText("Planner telemetry export failed", { exact: true }),
      ).toBeVisible();
      const diagnostics = page
        .getByRole("heading", { name: "Attempt diagnostics", exact: true })
        .locator("..");
      const calls = page
        .getByRole("heading", { name: "Tool calls", exact: true })
        .locator("..");
      expect((await diagnostics.boundingBox())!.width).toBe(
        (await calls.boundingBox())!.width,
      );
      await calls.getByText("Call details", { exact: true }).click();
      await expect(
        calls.getByRole("button", { name: "Copy call reference" }),
      ).toBeVisible();
      await expect(calls.getByTitle(fixture.callReference)).toBeVisible();
      await noOverflow(page, width);
      await page.screenshot({
        path: info.outputPath(`diagnostics-${theme}-${width}.png`),
        fullPage: true,
      });
    });
  }
}

test("Incomplete Inbox opens a loaded issue and report without Continue", async ({
  page,
}) => {
  const fixture = await install(page, { partialInbox: true });
  await page.goto("/");
  await expect(
    page.locator(".inbox-full-issue .issues-workspace-title"),
  ).toBeVisible();
  await expect(
    page.getByText(/Some projects, checks or Runs could not be read/),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Continue", exact: true }),
  ).toHaveCount(0);
  const selected = page.locator(".inbox-sections a[aria-current='true']");
  const title = await page.locator(".issues-workspace-title").innerText();
  await expect(selected).toContainText(title);
  await page.goto(fixture.paths.report);
  await expect(
    page.getByRole("heading", { name: "Invoice API review", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Continue", exact: true }),
  ).toHaveCount(0);
});
