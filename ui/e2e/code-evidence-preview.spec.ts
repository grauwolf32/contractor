import { expect, test, type Page } from "@playwright/test";
import proof from "../../api/testdata/finding-trace-evidence.json" with { type: "json" };
import { ARTIFACT_STORAGE_TIMESTAMPS } from "../src/test/builders";
import {
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

const source = JSON.stringify(proof);
const ref = {
  namespace: "evidence",
  name: "trace-proof",
  revision: "proof-r1",
};

async function install(page: Page, text = source) {
  const payloads: string[] = [];
  await mockAPI(page, async (route, request, url) => {
    const base = `/v1/artifacts/${ref.namespace}/${ref.name}`;
    if (!url.pathname.startsWith(base))
      return fulfillEmptyPageOrNotFound(route);
    expect(request.method()).toBe("GET");
    expect(url.searchParams.get("revision")).toBe(ref.revision);
    if (url.pathname.endsWith("/metadata"))
      return fulfillJSON(route, {
        canUploadNewVersion: false,
        storageTimestamps: ARTIFACT_STORAGE_TIMESTAMPS,
        artifact: ref,
        mediaType: "application/json",
        size: new TextEncoder().encode(text).length,
        current: false,
        frozen: false,
        createdAt: "2026-11-28T08:00:00Z",
      });
    if (url.pathname.endsWith("/versions") || url.pathname.endsWith("/lineage"))
      return fulfillJSON(route, { items: [], page: { hasMore: false } });
    payloads.push(url.toString());
    return fulfillContent(route, text, "application/json");
  });
  return payloads;
}

for (const theme of ["light", "dark", "black"]) {
  for (const width of [1440, 390]) {
    test(`saved code evidence ${theme} ${width}px`, async ({ page }, info) => {
      const payloads = await install(page);
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.setViewportSize({ width, height: 1000 });
      await page.addInitScript(
        (value) => localStorage.setItem("contractor.theme", value),
        theme,
      );
      await page.goto(
        `/artifacts/${ref.namespace}/${ref.name}?revision=${ref.revision}`,
      );
      await page.getByRole("button", { name: "Load preview" }).click();
      const preview = page.getByRole("region", { name: "Code evidence" });
      await expect(preview).toBeVisible();
      await expect(
        preview.getByText("Line 24 selected · saved workspace"),
      ).toBeVisible();
      await expect(preview.getByText("Line 22", { exact: true })).toBeVisible();
      const code = preview.getByLabel("Saved code", { exact: true });
      await expect(code).toHaveText(proof.workspace.text);
      await expect(code.locator('[data-line="24"]')).toHaveAttribute(
        "data-selected",
        "true",
      );
      await expect(code.locator(".hljs-keyword").first()).toBeVisible();
      await preview.getByRole("button", { name: "Wrap lines" }).focus();
      await page.keyboard.press("Enter");
      await expect(code).toHaveAttribute("data-wrap", "false");
      await expect(code).toHaveText(proof.workspace.text);
      await preview.getByRole("button", { name: "Wrap lines" }).click();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await page.screenshot({
        path: info.outputPath(`code-evidence-${theme}-${width}.png`),
        fullPage: true,
      });
      await page.getByRole("tab", { name: "Rendered", exact: true }).focus();
      await page.keyboard.press("ArrowRight");
      const raw = page.getByRole("tabpanel", { name: "Source", exact: true });
      await expect(raw).toBeVisible();
      await expect(raw.locator("pre")).toHaveText(source);
      expect(payloads).toHaveLength(1);
      expect(errors).toEqual([]);
    });
  }
}

test("invalid trace evidence retains the exact JSON source", async ({
  page,
}) => {
  const text = JSON.stringify({
    ...proof,
    workspace: { ...proof.workspace, snippetEndLine: 99 },
  });
  await install(page, text);
  await page.goto(
    `/artifacts/${ref.namespace}/${ref.name}?revision=${ref.revision}`,
  );
  await page.getByRole("button", { name: "Load preview" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "code evidence is invalid or unsupported",
  );
  await page.getByRole("tab", { name: "Source", exact: true }).click();
  await expect(
    page.getByRole("tabpanel", { name: "Source", exact: true }).locator("pre"),
  ).toHaveText(text);
});
