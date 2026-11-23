import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
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
  if (process.env.CONTRACTOR_RESEARCH_SCENARIO === "interrupted") {
    await inspectInterruptedCheck(page);
    return;
  }
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
    (h) =>
      h.document.origin.kind === "run" &&
      (h.document.sources?.length ?? 0) === 3,
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
  const ground = generated.document.sources![0]!;
  await page.getByRole("link", { name: new RegExp(ground.ref.name) }).click();
  const materialsResponse = await page.request.get(
    api + `/v1/audits/${audit}/materials?limit=50`,
  );
  expect(materialsResponse.status()).toBe(200);
  const materials = (
    (await materialsResponse.json()) as Schemas["AuditMaterialPage"]
  ).items;
  const receipts = materials.filter(
    ({ material }) =>
      material.ref.namespace === ground.ref.namespace &&
      material.ref.name === ground.ref.name &&
      material.ref.revision === ground.ref.revision &&
      material.digest === ground.digest,
  );
  expect(receipts.length).toBeGreaterThan(0);
  const selectedMaterials = page.locator('article[data-selected="true"]');
  await expect(selectedMaterials).toHaveCount(receipts.length);
  // Multiple receipts may retain the same exact bytes. Select a concrete receipt.
  for (const material of await selectedMaterials.all()) {
    await material
      .getByRole("button", { name: "details →", exact: true })
      .click();
  }
  const selectedMaterial = selectedMaterials.filter({
    has: page.getByText(receipts[0]!.key, { exact: true }),
  });
  await expect(selectedMaterial).toHaveCount(1);
  await expect(
    selectedMaterial.getByRole("heading", {
      name: ground.ref.name,
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    selectedMaterial.getByText(
      `${ground.ref.namespace}/${ground.ref.name}@${ground.ref.revision}`,
      { exact: true },
    ),
  ).toBeVisible();
  await expect(
    selectedMaterial.getByText(ground.digest, { exact: true }),
  ).toBeVisible();
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

async function inspectInterruptedCheck(page: Page) {
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
  const response = await page.request.get(
    api + `/v1/audits/${audit}/tasks?limit=50`,
  );
  expect(response.status()).toBe(200);
  const tasks = (await response.json()) as Schemas["AuditTaskPage"];
  expect(tasks.page.hasMore).toBe(false);
  const model = tasks.items.find((task) => task.result)!;
  const missing = tasks.items.find((task) => task.stop)!;
  expect(model).toBeDefined();
  expect(missing.stop?.author).toBe("system");
  expect(missing.stop?.meaning).toBe("no_accepted_conclusion");
  expect(missing.stop!.evidence.length).toBeGreaterThan(0);
  await page.goto(root + `/work?task=${model.document.task_id}`);
  await page
    .getByRole("button", { name: "Read saved result", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Recorded result" }),
  ).toBeVisible();
  await expect(
    page.getByText("Latest accepted correction B", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Inconclusive", { exact: true })).toBeVisible();
  await page.getByText("Run attempts", { exact: true }).click();
  const attempt = page.getByRole("listitem", { name: "Attempt 1" });
  await expect(attempt).toContainText("source Run deleted");
  await expect(attempt).toContainText(/failed|cancelled/);
  await attempt
    .getByRole("button", { name: "Read saved result", exact: true })
    .click();
  await expect(
    attempt.getByText("Latest accepted correction B", { exact: true }),
  ).toBeVisible();
  for (const theme of ["light", "dark", "black"]) {
    await page.addInitScript(
      (value) => localStorage.setItem("contractor.theme", value),
      theme,
    );
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(root + `/work?task=${missing.document.task_id}`);
      const record = page.getByRole("region", {
        name: "System interruption",
      });
      await expect(
        record.getByRole("heading", { name: "No accepted conclusion" }),
      ).toBeVisible();
      await expect(
        record.getByText("System record", { exact: true }),
      ).toBeVisible();
      await expect(
        record.getByText(
          new RegExp(
            `${missing.stop!.evidence.length} retained source references`,
          ),
        ),
      ).toBeVisible();
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await page.screenshot({
        path: path.join(directory, `partial-${theme}-${width}.png`),
        fullPage: true,
      });
    }
  }
  await page
    .getByRole("region", { name: "System interruption" })
    .getByRole("link", { name: "Inspect retained evidence and materials →" })
    .click();
  await expect(page).toHaveURL(root + "/materials");
  await expect(
    page.getByRole("navigation", { name: "Check sections" }),
  ).toBeVisible();
  const findingsResponse = await page.request.get(
    api + `/v1/projects/${project}/findings?limit=50`,
  );
  expect(findingsResponse.status()).toBe(200);
  const independent = (
    (await findingsResponse.json()) as Schemas["ProjectFindingPage"]
  ).items;
  expect(independent).toHaveLength(1);
  expect(independent[0]!.title).toBe(
    "Incidental query concatenation observation",
  );
  await page.goto(`/issues/${project}/${independent[0]!.findingId}`);
  await expect(
    page.getByRole("heading", { name: independent[0]!.title, exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
  await writeFile(
    path.join(directory, "partial-journey.json"),
    JSON.stringify({
      passed: true,
      audit,
      project,
      modelTask: model.document.task_id,
      missingTask: missing.document.task_id,
      sourceRunDeleted: true,
      evidence: missing.stop!.evidence,
    }),
  );
}
