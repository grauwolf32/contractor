import { expect, test } from "@playwright/test";
import { costResearchFixture } from "../src/test/llm-cost-fixtures";
import {
  RESEARCH_AUDIT,
  RESEARCH_PROJECT,
} from "../src/test/audit-research-fixtures";
import {
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

for (const theme of ["light", "dark", "black"])
  for (const width of [390, 1440]) {
    test(`LLM cost across Check Project and Run · ${theme} · ${width}`, async ({
      page,
    }) => {
      const fixture = costResearchFixture();
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (theme) => localStorage.setItem("contractor.theme", theme),
        theme,
      );
      await mockAPI(page, async (route, request, url) => {
        const response = await fixture.respond(
          new Request(url, { method: request.method() }),
        );
        if (!response) return fulfillEmptyPageOrNotFound(route);
        const headers: Record<string, string> = {};
        response.headers.forEach((value, key) => {
          headers[key] = value;
        });
        await fulfillJSON(
          route,
          await response.json(),
          response.status,
          headers,
        );
      });
      const root = `/projects/${RESEARCH_PROJECT}/audits/${RESEARCH_AUDIT}`;
      await page.goto(root);
      let card = page.getByRole("region", {
        name: "Check LLM cost",
        exact: true,
      });
      await expect(card.getByText("$12.84", { exact: true })).toBeVisible();
      await card.getByRole("button", { name: "Details", exact: true }).click();
      await card.getByRole("button", { name: "By model", exact: true }).click();
      await expect(
        card.getByText("openrouter/demo-worker", { exact: true }),
      ).toBeVisible();
      await expect
        .poll(() =>
          page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        )
        .toBe(true);
      await page.getByRole("link", { name: /^Work/ }).last().click();
      await page.getByText("Run attempts", { exact: true }).click();
      await expect(page.locator(".research-attempt-cost")).toContainText(
        "<$0.01",
      );
      await page.goto(`/projects/${RESEARCH_PROJECT}`);
      card = page.getByRole("region", {
        name: "Project LLM cost",
        exact: true,
      });
      await expect(card.getByText("$48.76", { exact: true })).toBeVisible();
      await expect(card.getByText("$37.70", { exact: true })).toBeVisible();
      await expect(card.getByText("$11.06", { exact: true })).toBeVisible();
      await page.goto("/runs/run_1");
      card = page.getByRole("region", { name: "Run LLM cost", exact: true });
      await expect(card.getByText("<$0.01", { exact: true })).toBeVisible();
      await expect
        .poll(() =>
          page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        )
        .toBe(true);
      expect(errors).toEqual([]);
    });
  }
