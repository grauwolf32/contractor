import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { test, expect } from "@playwright/test";
import { artifactMetadata, project } from "../src/test/builders";
import {
  fulfillJSON,
  fulfillEmptyPageOrNotFound,
  mockAPI,
} from "./support/mock-api";
for (const theme of ["light", "dark", "black"])
  for (const width of [1440, 390]) {
    test(`Project retains compact exact long material links at ${width}px in ${theme}`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (t) => localStorage.setItem("contractor.theme", t),
        theme,
      );
      const file = artifactMetadata("scan-inputs", "target." + "a".repeat(64), {
        mediaType: "application/json",
        artifact: {
          namespace: "scan-inputs",
          name: "target." + "a".repeat(64),
          revision: "rev_" + "b".repeat(32),
        },
      });
      const owner = project("project_shop");
      await mockAPI(page, async (route, _request, url) => {
        if (url.pathname === "/v1/projects/project_shop")
          return fulfillJSON(route, owner, 200, {
            ETag: `"${owner.revision}"`,
          });
        if (url.pathname === "/v1/projects")
          return fulfillJSON(route, {
            items: [owner],
            page: { hasMore: false },
          });
        if (url.pathname === "/v1/projects/project_shop/findings")
          return fulfillJSON(route, {
            items: [],
            projectRevision: 1,
            page: { hasMore: false },
          });
        if (url.pathname === "/v1/projects/project_shop/artifacts")
          return fulfillJSON(route, {
            items: url.searchParams.has("namespace") ? [] : [file],
            page: { hasMore: false },
          });
        return fulfillEmptyPageOrNotFound(route);
      });
      await page.goto("/projects/project_shop");
      const row = page.locator(".projects-material-item");
      await expect(row).toHaveCount(1);
      await expect(row).toHaveAttribute(
        "title",
        `scan-inputs/${file.artifact.name}@${file.artifact.revision}`,
      );
      expect((await row.innerText()).length).toBeLessThan(150);
      expect((await row.boundingBox())!.height).toBeLessThan(130);
      await page.getByText("Timeline", { exact: true }).click();
      const exact = page.getByRole("link", {
        name: `scan-inputs/${file.artifact.name}`,
        exact: true,
      });
      await expect(exact).toBeVisible();
      await expect(exact).toHaveAttribute(
        "href",
        `/projects/project_shop/artifacts/scan-inputs/${file.artifact.name}`,
      );
      expect((await exact.innerText()).length).toBeLessThan(49);
      await expect(
        page.getByRole("checkbox", { name: "Materials", exact: true }),
      ).toBeChecked();
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(width);
      if (process.env.CONTRACTOR_UI_POLISH_CAPTURE_DIR) {
        await mkdir(process.env.CONTRACTOR_UI_POLISH_CAPTURE_DIR, {
          recursive: true,
        });
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({
          path: join(
            process.env.CONTRACTOR_UI_POLISH_CAPTURE_DIR,
            `project-${theme}-${width}.png`,
          ),
          fullPage: true,
        });
      }
    });
  }

test("Project context selection keeps its scroll while the next detail loads", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const projects = Array.from({ length: 40 }, (_, i) =>
    project(`project_${i}`, { name: `Project ${String(i).padStart(2, "0")}` }),
  );
  let release: () => void = () => {};
  const delayed = new Promise<void>((resolve) => {
    release = resolve;
  });
  await mockAPI(page, async (route, _request, url) => {
    if (url.pathname === "/v1/projects")
      return fulfillJSON(route, { items: projects, page: { hasMore: false } });
    const owner = projects.find(
      (p) => url.pathname === `/v1/projects/${p.projectId}`,
    );
    if (owner) {
      if (owner.projectId === "project_20") await delayed;
      return fulfillJSON(route, owner, 200, { ETag: `"${owner.revision}"` });
    }
    return fulfillEmptyPageOrNotFound(route);
  });
  await page.goto("/projects/project_0");
  await expect(
    page.getByRole("heading", { name: "Project 00", exact: true }),
  ).toBeVisible();
  const next = page.getByRole("link", { name: "Project 20", exact: true });
  await next.scrollIntoViewIfNeeded();
  const pane = page.locator(".ui-panes-list");
  const before = await pane.evaluate((el) => ({
    top: el.scrollTop,
    x: el.getBoundingClientRect().x,
    width: el.getBoundingClientRect().width,
    outer: scrollY,
  }));
  try {
    await next.click();
    await expect(next).toHaveAttribute("aria-current", "true");
    const during = await pane.evaluate((el) => ({
      top: el.scrollTop,
      x: el.getBoundingClientRect().x,
      width: el.getBoundingClientRect().width,
      outer: scrollY,
    }));
    expect(during).toEqual(before);
  } finally {
    release();
  }
  await expect(
    page.getByRole("heading", { name: "Project 20", exact: true }),
  ).toBeVisible();
  expect(await pane.evaluate((el) => el.scrollTop)).toBe(before.top);
});
