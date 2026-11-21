import { expect, test } from "@playwright/test";
import { projectFactsFixture } from "../src/test/project-facts-fixtures";
import {
  RESEARCH_AUDIT,
  RESEARCH_PROJECT,
  researchFixture,
} from "../src/test/audit-research-fixtures";
import {
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillResponse,
  mockAPI,
} from "./support/mock-api";

for (const width of [390, 1440]) {
  for (const theme of ["light", "dark", "black"]) {
    test(`Project facts retain canonical sources at ${width}px in ${theme}`, async ({
      page,
    }, info) => {
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (value) => localStorage.setItem("contractor.theme", value),
        theme,
      );
      const data = projectFactsFixture();
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await mockAPI(page, async (route, request, url) => {
        expect(request.method()).toBe("GET");
        const response = data.respond(new Request(url));
        if (
          response &&
          ((url.pathname.includes("/artifacts/") &&
            !url.pathname.endsWith("/metadata")) ||
            url.pathname.endsWith("/artifacts"))
        )
          return fulfillContent(
            route,
            await response.text(),
            response.headers.get("content-type")!,
          );
        return response
          ? fulfillResponse(route, response)
          : fulfillEmptyPageOrNotFound(route);
      });
      await page.goto(`/projects/${data.owner.projectId}/facts`);
      const nav = page.getByRole("navigation", { name: "Project sections" });
      await expect(
        nav.getByRole("link", { name: "Facts", exact: true }),
      ).toHaveAttribute("aria-current", "page");
      const facts = page.getByRole("list", { name: "Facts", exact: true });
      await expect(facts.getByRole("link")).toHaveCount(1);
      expect(data.listReads).toEqual([0, 50, 100, 150, 200]);
      await page.getByRole("button", { name: "Load more facts" }).click();
      await expect(facts.getByRole("link")).toHaveCount(2);
      expect(data.listReads).toHaveLength(10);
      expect(data.payloadReads).toEqual([]);
      await facts
        .getByRole("link", { name: /Reported nginx deployment/ })
        .click();
      await expect(
        page.getByRole("heading", { name: data.fact.title, exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("link", { name: "View in Project" }),
      ).toHaveAttribute("href", `/projects/${data.owner.projectId}/facts`);
      expect(data.sourceReads).toEqual([]);
      expect(data.payloadReads).toEqual([]);
      const evidence = page.getByRole("tab", { name: /^Evidence/ });
      await evidence.click();
      const file = page.locator(".issues-evidence-file").first();
      await expect(
        file.getByRole("button", { name: "Preview", exact: true }),
      ).toBeVisible();
      if (theme === "black" && width === 390) data.setUnavailable(true);
      await file.getByRole("button", { name: "Preview", exact: true }).click();
      if (theme === "black" && width === 390) {
        await expect(
          file.getByText(/Original memory is unavailable/),
        ).toBeVisible();
        expect(data.payloadReads).toEqual([]);
        data.setUnavailable(false);
        await file.getByRole("button", { name: "Retry", exact: true }).click();
      }
      await expect(file.getByText(/Statements from memories/)).toBeVisible();
      await expect(file.locator("pre")).toContainText("service_banner");
      await expect(file.locator("pre")).toContainText("version not verified");
      expect(data.payloadReads).toHaveLength(1);
      expect(data.payloadReads[0]).toContain("revision=memory-r1");
      expect(await page.evaluate(() => "memoryExecuted" in window)).toBe(false);
      await page.getByRole("tab", { name: "History", exact: true }).click();
      const sources = page.getByRole("region", { name: "Independent sources" });
      await expect(
        sources.getByRole("heading", {
          name: data.first.document.title,
          exact: true,
        }),
      ).toBeVisible();
      await sources.getByRole("button", { name: "More sources" }).click();
      await sources.getByRole("button", { name: /independent-memory/ }).click();
      await expect(
        sources.getByRole("heading", {
          name: "Another title over the same memory",
          exact: true,
        }),
      ).toBeVisible();
      await expect(
        sources.getByText(/without a Task or hypothesis/),
      ).toBeVisible();
      expect(data.sourceReads).toHaveLength(2);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await page.screenshot({
        path: info.outputPath(`facts-${theme}-${width}.png`),
        fullPage: true,
      });
      data.assess();
      await page.goto(`/projects/${data.owner.projectId}/facts`);
      await expect(
        page
          .getByRole("list", { name: "Facts" })
          .getByRole("link", { name: /Reported nginx deployment/ }),
      ).toHaveCount(0);
      await expect(
        page.getByText("No facts in the loaded records"),
      ).toBeVisible();
      expect(data.fact.findingId).toBe("fact_nginx");
      expect(errors).toEqual([]);
    });
  }
}

test("a hypothesis opens its pinned selected grounds separately from check results", async ({
  page,
}) => {
  const data = researchFixture();
  const selected = data.hypotheses[0]!.document.sources![0]!;
  const payloads: string[] = [];
  await mockAPI(page, async (route, request, url) => {
    expect(request.method()).toBe("GET");
    const response = await data.respond(new Request(url));
    if (
      response &&
      ((url.pathname.includes("/artifacts/") &&
        !url.pathname.endsWith("/metadata")) ||
        url.pathname.endsWith("/artifacts"))
    ) {
      payloads.push(url.toString());
      return fulfillContent(
        route,
        await response.text(),
        response.headers.get("content-type")!,
      );
    }
    return response
      ? fulfillResponse(route, response)
      : fulfillEmptyPageOrNotFound(route);
  });
  await page.goto(
    `/projects/${RESEARCH_PROJECT}/audits/${RESEARCH_AUDIT}/hypotheses`,
  );
  await page
    .getByText(data.hypotheses[0]!.document.statement, { exact: true })
    .first()
    .click();
  await expect(
    page.getByRole("heading", { name: "Grounds · exact materials" }),
  ).toBeVisible();
  const link = page
    .getByRole("link", { name: new RegExp(selected.ref.name) })
    .first();
  await expect(link).toHaveAttribute(
    "href",
    new RegExp(encodeURIComponent(selected.ref.revision)),
  );
  await link.click();
  expect(payloads).toEqual([]);
  const material = page.locator(".research-material[data-selected]");
  await expect(
    material.getByRole("heading", { name: selected.ref.name, exact: true }),
  ).toBeVisible();
  await material.getByRole("button", { name: "Preview", exact: true }).click();
  await expect(page.getByText(/Order API observations/).first()).toBeVisible();
  expect(payloads).toHaveLength(1);
  expect(new URL(payloads[0]!).searchParams.get("revision")).toBe(
    selected.ref.revision,
  );
});
