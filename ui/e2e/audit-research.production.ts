import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test } from "@playwright/test";
import type { components } from "../src/api/generated/public";

type Schemas = components["schemas"];
function required(name: string) {
  const value = process.env[name];
  if (!value) throw new Error(name + " is required");
  return value;
}

test("research continuation, exact grounds and Project publication through real APIs", async ({
  page,
}) => {
  const api = required("CONTRACTOR_UI_E2E_API_URL");
  const project = required("CONTRACTOR_RESEARCH_PROJECT");
  const audit = required("CONTRACTOR_RESEARCH_AUDIT");
  const directory = required("CONTRACTOR_AUDIT_RESEARCH_EVIDENCE_DIR");
  const root = `/projects/${project}/audits/${audit}`;
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await mkdir(directory, { recursive: true });
  await page.goto(root);
  await page
    .getByLabel("Username")
    .fill(required("CONTRACTOR_UI_E2E_USERNAME"));
  await page
    .getByLabel("Password")
    .fill(required("CONTRACTOR_UI_E2E_PASSWORD"));
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Account", exact: true }),
  ).toBeVisible();
  await page.goto(root + "/work");
  await page
    .getByRole("button", { name: "Read saved result", exact: true })
    .click();
  await expect(page.getByText("Refuted", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "Continue check", exact: true })
    .click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Continue check", exact: true })
    .click();
  await expect
    .poll(
      async () => {
        const response = await page.request.get(api + "/v1/audits/" + audit);
        expect(response.status()).toBe(200);
        return ((await response.json()) as Schemas["Audit"]).state;
      },
      { timeout: 180000 },
    )
    .toBe("completed");
  await page.goto(root + "/hypotheses");
  const response = await page.request.get(
    api + `/v1/audits/${audit}/hypotheses?limit=50`,
  );
  expect(response.status()).toBe(200);
  const hypotheses = ((await response.json()) as Schemas["AuditHypothesisPage"])
    .items;
  const generated = hypotheses.find(
    (h) => h.document.origin.kind === "run" && h.document.sources.length === 3,
  )!;
  expect(generated).toBeDefined();
  await page.goto(root + `/hypotheses?hypothesis=${generated.document.id}`);
  await expect(
    page
      .getByRole("button", {
        name: new RegExp("Generated " + generated.document.statement),
      })
      .getByText("Generated", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Read saved result", exact: true })
    .click();
  await expect(page.getByText("Inconclusive", { exact: true })).toBeVisible();
  await expect(
    page.getByText("selected fixture and actor", { exact: true }),
  ).toBeVisible();
  const ground = generated.document.sources[0]!;
  await page.getByRole("link", { name: new RegExp(ground.ref.name) }).click();
  await expect(
    page.getByRole("heading", { name: ground.ref.name, exact: true }),
  ).toBeVisible();
  const selectedMaterial = page.getByRole("article").filter({
    has: page.getByRole("heading", { name: ground.ref.name, exact: true }),
  });
  await selectedMaterial
    .getByRole("button", { name: "Publish to Project", exact: true })
    .click();
  await page
    .getByRole("dialog")
    .getByLabel("Material name")
    .fill("research-browser-ground");
  await page
    .getByRole("button", { name: "Publish exact revision", exact: true })
    .click();
  await expect(
    page.getByText("Material published.", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close", exact: true }).click();
  const findings = await page.request.get(
    api + `/v1/projects/${project}/findings?limit=50`,
  );
  expect(findings.status()).toBe(200);
  const issues = ((await findings.json()) as Schemas["ProjectFindingPage"])
    .items;
  expect(issues).toHaveLength(1);
  await page.goto(`/issues/${project}/${issues[0]!.findingId}`);
  await expect(
    page.getByRole("heading", { name: issues[0]!.title, exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "History", exact: true }).click();
  const sources = page.getByRole("region", { name: "Independent sources" });
  await expect(sources.getByText(/^Run run_/)).toBeVisible();
  await page.getByRole("tab", { name: /^Evidence/ }).click();
  await expect(
    page.getByRole("heading", { name: /[Ee]vidence/ }).first(),
  ).toBeVisible();
  for (const theme of ["light", "dark", "black"]) {
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      for (const section of [
        "",
        "/work",
        `/hypotheses?hypothesis=${generated.document.id}`,
        "/materials",
        "/activity",
      ]) {
        await page.goto(root + section);
        await expect(
          page.getByRole("navigation", { name: "Check sections" }),
        ).toBeVisible();
        expect(
          await page.evaluate(
            () => document.documentElement.scrollWidth <= innerWidth,
          ),
        ).toBe(true);
      }
      await page.screenshot({
        path: path.join(directory, `research-${theme}-${width}.png`),
        fullPage: true,
      });
    }
  }
  expect(errors).toEqual([]);
  await writeFile(
    path.join(directory, "journey.json"),
    JSON.stringify({
      passed: true,
      audit,
      project,
      generatedHypothesis: generated.document.id,
      exactGrounds: generated.document.sources,
      finding: issues[0]!.findingId,
    }),
  );
});
