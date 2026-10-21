import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test } from "@playwright/test";
import type { components } from "../src/api/generated/public";

type Audit = components["schemas"]["Audit"];
type Ref = components["schemas"]["ExactArtifactRef"];
function required(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

// Real HTTPS session, Server, PostgreSQL, Scheduler, Runtime and scanners.
// API calls seed ordinary Project materials and inspect durable state; every
// check start, pause, resume, approval and evidence preview uses the browser.
test("source preparation, approval, Nuclei and retained evidence survive Run deletion", async ({
  page,
}) => {
  const api = required("CONTRACTOR_UI_E2E_API_URL");
  const origin = required("CONTRACTOR_UI_E2E_BASE_URL");
  const directory = required("CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR");
  await mkdir(directory, { recursive: true });
  await page.goto("/projects");
  await expect(page.getByRole("region", { name: "Sign in" })).toBeVisible();
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
  const session = await page.request.get(`${api}/v1/auth/session`);
  expect(session.status()).toBe(200);
  const { csrfToken } = (await session.json()) as { csrfToken: string };
  const headers = () => ({
    "X-CSRF-Token": csrfToken,
    Origin: origin,
    "Idempotency-Key": randomUUID(),
  });
  const getAudit = async (id: string): Promise<Audit> => {
    for (let attempt = 0; attempt < 5; attempt++) {
      const response = await page.request.get(`${api}/v1/audits/${id}`);
      // A concurrent projection revision is fenced with 409. Only this read
      // is retried; mutations retain their own UI request identity.
      if (response.status() === 409 && attempt < 4) {
        expect(((await response.json()) as { code: string }).code).toBe(
          "conflict",
        );
        continue;
      }
      expect(response.status()).toBe(200);
      return (await response.json()) as Audit;
    }
    throw new Error("Audit projection did not settle within five reads");
  };
  const capture = async (name: string) =>
    page.screenshot({
      path: path.join(directory, `${name}.png`),
      fullPage: true,
      animations: "disabled",
    });
  const control = async (command: string) => {
    const response = await page.request.post(
      `${required("CONTRACTOR_COMPOSITION_CONTROL")}/${command}`,
      {
        headers: {
          Authorization: `Bearer ${required("CONTRACTOR_COMPOSITION_CONTROL_TOKEN")}`,
        },
      },
    );
    expect(response.status()).toBe(204);
  };
  const seed = async (source: Buffer, name: string) => {
    const created = await page.request.post(`${api}/v1/projects`, {
      headers: headers(),
      data: { kind: "project", name, activeChecks: "approval-required" },
    });
    expect(created.status()).toBe(201);
    const { projectId } = (await created.json()) as { projectId: string };
    const upload = async (
      name: string,
      mediaType: string,
      data: Buffer,
    ): Promise<Ref> => {
      const response = await page.request.put(
        `${api}/v1/projects/${projectId}/artifacts/fixture/${name}`,
        {
          headers: {
            ...headers(),
            "If-None-Match": "*",
            "Content-Type": mediaType,
          },
          data,
        },
      );
      expect(response.status()).toBe(201);
      return ((await response.json()) as { artifact: Ref }).artifact;
    };
    const sourceRef = await upload("source", "application/zip", source);
    const settingsRef = await upload(
      "settings",
      "application/json",
      Buffer.from(
        JSON.stringify({
          schema: "contractor.audit.openapi-scan-settings.v1",
          scanner: "nuclei",
          server: required("CONTRACTOR_COMPOSITION_TARGET"),
          operations: {
            "#/paths/~1widgets~1{widget_id}/get": {
              parameters: { "path:widget_id": "7", "query:q": "Milo" },
            },
          },
        }),
      ),
    );
    return { projectId, sourceRef, settingsRef };
  };
  const start = async (projectId: string): Promise<string> => {
    await page.goto(
      `/checks/new?project=${projectId}&type=source-openapi-nuclei-fixture`,
    );
    await page
      .getByLabel("Your objective, in your own words")
      .fill(
        "Generate the API spec from these local sources, then check the fixed loopback URL.",
      );
    await expect(
      page.getByRole("button", { name: "Start check", exact: true }),
    ).toBeEnabled();
    await page
      .getByRole("button", { name: "Start check", exact: true })
      .click();
    await expect(page).toHaveURL(/\/audits\/audit_[A-Za-z0-9_-]+/);
    const id = new URL(page.url()).pathname.match(
      /\/audits\/(audit_[A-Za-z0-9_-]+)/,
    )?.[1];
    if (!id) throw new Error("Started check URL contains no Audit ID");
    return id;
  };
  const seeded = await seed(
    await readFile(required("CONTRACTOR_COMPOSITION_SOURCE")),
    "Source composition browser",
  );
  const id = await start(seeded.projectId);
  await expect(
    page.getByRole("heading", { name: "Preparing materials", exact: true }),
  ).toBeVisible();
  expect((await getAudit(id)).currentRoundId).toBeUndefined();
  await capture("preparing");
  await page
    .getByRole("button", { name: "Pause new work", exact: true })
    .click();
  await expect.poll(async () => (await getAudit(id)).state).toBe("paused");
  await page
    .getByRole("button", { name: "Continue check", exact: true })
    .click();
  await page
    .getByRole("dialog", { name: "Continue check" })
    .getByRole("button", { name: "Continue check", exact: true })
    .click();
  await control("release-preparation");
  await expect
    .poll(
      async () => {
        const current = await getAudit(id);
        if (current.state === "failed" || current.state === "cancelled") {
          throw new Error(
            `Preparation stopped: ${current.state} in ${current.phase}`,
          );
        }
        return current.phase;
      },
      { timeout: 300_000 },
    )
    .toBe("inventory");
  await expect(
    page.getByRole("heading", { name: "Building the worklist", exact: true }),
  ).toBeVisible({ timeout: 300_000 });
  const inventoryPending = await getAudit(id);
  expect(inventoryPending.phase).toBe("inventory");
  expect(inventoryPending.currentRoundId).toBeUndefined();
  expect(inventoryPending.generatedInventory).toBeUndefined();
  await capture("inventory-pending");
  await control("accept-inventory");
  await expect(
    page.getByRole("heading", { name: "Endpoint checks", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Approve", exact: true }),
  ).toBeVisible();
  await capture("approval");
  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Why", exact: true })
    .fill(
      "Only the controlled local target; URL template coverage acknowledged.",
    );
  await page
    .getByRole("button", { name: "Record decision", exact: true })
    .click();
  await expect
    .poll(async () => (await getAudit(id)).state, { timeout: 180_000 })
    .toBe("completed");
  await expect(
    page.getByText("Scan completed", { exact: true }).first(),
  ).toBeVisible();
  const audit = await getAudit(id);
  expect(audit.inputs.source?.ref).toEqual(seeded.sourceRef);
  expect(audit.inputs.settings?.ref).toEqual(seeded.settingsRef);
  expect(audit.generatedInventory?.itemCount).toBe(1);
  const role = audit.preparation?.roles["generate-api"];
  expect(role?.status).toBe("accepted");
  expect(audit.generatedInventory?.sources.source).toEqual(
    role?.outputs.api?.artifact,
  );
  await capture("overview-completed");
  await page
    .getByRole("link", { name: "GET /widgets/{widget_id}", exact: true })
    .click();
  const scan = page.getByRole("region", { name: "Scan inputs and evidence" });
  await expect(scan).toContainText("Nuclei · URL template check");
  await expect(scan).toContainText("does not replay the OpenAPI method");
  const preview = async (label: string, content: RegExp) => {
    await page.getByRole("button", { name: label, exact: true }).click();
    await expect(
      page.getByRole("dialog", { name: label, exact: true }),
    ).toContainText(content);
    await page.keyboard.press("Escape");
    await expect(
      page.getByRole("button", { name: label, exact: true }),
    ).toBeFocused();
  };
  await preview("API spec", /widgets/);
  await preview("Scanner report", /nuclei/);
  await capture("endpoint-completed");
  const itemsResponse = await page.request.get(`${api}/v1/audits/${id}/items`);
  expect(itemsResponse.status()).toBe(200);
  const items = (await itemsResponse.json()) as {
    items: Array<{ attempts: Array<{ runId?: string }> }>;
  };
  const runIds = [role?.runId, items.items[0]?.attempts[0]?.runId];
  expect(runIds.every(Boolean)).toBe(true);
  for (const runId of runIds) {
    await expect
      .poll(
        async () => {
          const response = await page.request.get(`${api}/v1/runs/${runId}`);
          expect(response.status()).toBe(200);
          return ((await response.json()) as { deletable: boolean }).deletable;
        },
        { timeout: 30_000 },
      )
      .toBe(true);
    const deleted = await page.request.delete(`${api}/v1/runs/${runId}`, {
      headers: headers(),
    });
    expect(deleted.status()).toBe(204);
  }
  await page.reload();
  await preview("API spec", /widgets/);
  await preview("Scanner report", /nuclei/);
  await capture("retained-after-run-deletion");
  const broken = await seed(
    Buffer.from("invalid ZIP fixture"),
    "Preparation failure browser",
  );
  const failedId = await start(broken.projectId);
  await expect
    .poll(async () => (await getAudit(failedId)).state, { timeout: 180_000 })
    .toBe("failed");
  await expect(
    page.getByRole("heading", { name: "Preparation stopped", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "API spec", exact: true }),
  ).toHaveCount(0);
  const failed = await getAudit(failedId);
  expect(failed.currentRoundId).toBeUndefined();
  expect(failed.generatedInventory).toBeUndefined();
  await capture("preparation-failed");
  await writeFile(
    path.join(directory, "journey.json"),
    JSON.stringify(
      {
        schemaVersion: 1,
        auditId: id,
        failedAuditId: failedId,
        projectId: seeded.projectId,
        originalInputs: audit.inputs,
        generatedInventory: audit.generatedInventory,
        retainedAfterDeletingRuns: runIds,
        failedPhase: failed.phase,
        browserAssertions: [
          "setup",
          "preparing",
          "pause-resume",
          "inventory-pending",
          "approval",
          "nuclei-url",
          "exact-provenance",
          "retained-evidence",
          "failed-preparation",
        ],
        passed: true,
      },
      null,
      2,
    ),
  );
});
