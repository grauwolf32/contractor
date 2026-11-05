import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test } from "@playwright/test";
import type { components } from "../src/api/generated/public";

type Audit = components["schemas"]["Audit"];
type Task = components["schemas"]["AuditTask"];
type Ref = components["schemas"]["ExactArtifactRef"];
function required(name: string) {
  const v = process.env[name];
  if (!v) throw new Error(name + " is required");
  return v;
}

test("common preparation and retained scanner result survive Run deletion", async ({
  page,
}) => {
  const api = required("CONTRACTOR_UI_E2E_API_URL"),
    origin = required("CONTRACTOR_UI_E2E_BASE_URL");
  const directory = required("CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR");
  await mkdir(directory, { recursive: true });
  await page.goto("/projects");
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
  const session = await page.request.get(api + "/v1/auth/session");
  expect(session.status()).toBe(200);
  const { csrfToken } = (await session.json()) as { csrfToken: string };
  const headers = () => ({
    "X-CSRF-Token": csrfToken,
    Origin: origin,
    "Idempotency-Key": randomUUID(),
  });
  const project = await page.request.post(api + "/v1/projects", {
    headers: headers(),
    data: {
      kind: "project",
      name: "Common browser composition",
      activeChecks: "automatic",
    },
  });
  expect(project.status()).toBe(201);
  const { projectId } = (await project.json()) as { projectId: string };
  async function upload(name: string, content: Buffer): Promise<Ref> {
    const response = await page.request.put(
      api + `/v1/projects/${projectId}/artifacts/fixture/${name}`,
      {
        headers: {
          ...headers(),
          "If-None-Match": "*",
          "Content-Type": "application/json",
        },
        data: content,
      },
    );
    expect(response.status()).toBe(201);
    return ((await response.json()) as { artifact: Ref }).artifact;
  }
  const openapi = await upload(
    "openapi",
    await readFile(required("CONTRACTOR_COMPOSITION_OPENAPI")),
  );
  const settings = await upload(
    "settings",
    Buffer.from(
      JSON.stringify({
        schema: "contractor.audit.openapi-scan-settings.v1",
        scanner: "nuclei",
        server: required("CONTRACTOR_COMPOSITION_TARGET") + "/api",
        operations: {
          "#/paths/~1pets~1{id}/post": {
            parameters: { "path:id": 7, "query:search": "Milo" },
          },
        },
      }),
    ),
  );
  await page.goto(`/checks/new?project=${projectId}&type=openapi-nuclei-scan`);
  await page
    .getByLabel("Your objective, in your own words")
    .fill("Inspect the exact controlled API target.");
  for (const [label, name] of [
    ["API spec", "openapi"],
    ["Scan settings", "settings"],
  ]) {
    const select = page.getByRole("combobox", {
      name: "Material for " + label,
    });
    await select.selectOption({
      label: (await select.getByRole("option").allTextContents()).find((text) =>
        text.startsWith("fixture/" + name + "@"),
      )!,
    });
  }
  await page.getByRole("button", { name: "Start check", exact: true }).click();
  await expect(page).toHaveURL(/\/audits\/audit_[A-Za-z0-9_-]+/);
  const auditId = new URL(page.url()).pathname.split("/").at(-1)!;
  async function getAudit(): Promise<Audit> {
    const response = await page.request.get(api + "/v1/audits/" + auditId);
    expect(response.status()).toBe(200);
    return (await response.json()) as Audit;
  }
  await expect
    .poll(async () => (await getAudit()).state, { timeout: 180000 })
    .toBe("completed");
  const audit = await getAudit();
  expect(audit.taskCount).toBe(2);
  expect(audit.submittedRunCount).toBe(2);
  expect(audit.inputs.openapi?.ref).toEqual(openapi);
  expect(audit.inputs.settings?.ref).toEqual(settings);
  const tasksResponse = await page.request.get(
    api + `/v1/audits/${auditId}/tasks?limit=64`,
  );
  expect(tasksResponse.status()).toBe(200);
  const { items: tasks } = (await tasksResponse.json()) as { items: Task[] };
  expect(tasks).toHaveLength(2);
  expect(
    tasks.every(
      (task) =>
        task.state === "settled" &&
        task.result !== undefined &&
        task.gap_code === undefined,
    ),
  ).toBe(true);
  const scan = tasks.find((task) => task.document.workflow === "scan")!;
  expect(scan.document.hypothesis).toBeUndefined();
  const events = await page.request.get(
    api + `/v1/audits/${auditId}/events?limit=100`,
  );
  expect(events.status()).toBe(200);
  const eventPage =
    (await events.json()) as components["schemas"]["AuditEventPage"];
  const runIds = [
    ...new Set(
      eventPage.items.flatMap((event) =>
        event.summary.runId === undefined ? [] : [event.summary.runId],
      ),
    ),
  ];
  expect(runIds).toHaveLength(2);
  for (const runId of runIds) {
    await expect
      .poll(
        async () => {
          const response = await page.request.get(api + "/v1/runs/" + runId);
          expect(response.status()).toBe(200);
          return ((await response.json()) as { deletable: boolean }).deletable;
        },
        { timeout: 30000 },
      )
      .toBe(true);
    const deleted = await page.request.delete(api + "/v1/runs/" + runId, {
      headers: headers(),
    });
    expect(deleted.status()).toBe(204);
  }
  await page.reload();
  const card = page
    .getByRole("heading", { name: scan.document.objective, exact: true })
    .locator("..");
  await card.getByRole("button", { name: "Read saved result" }).click();
  await expect(
    card.getByText(/contractor.scan.operation-report.v1/),
  ).toBeVisible();
  await expect(
    card.getByRole("button", { name: "Download complete Run result" }),
  ).toBeVisible();
  await page.screenshot({
    path: path.join(directory, "common-retained-after-run-deletion.png"),
    fullPage: true,
    animations: "disabled",
  });
  await writeFile(
    path.join(directory, "journey.json"),
    JSON.stringify({
      passed: true,
      auditId,
      taskCount: 2,
      runIds,
      retainedAfterRunDeletion: true,
    }),
  );
});
