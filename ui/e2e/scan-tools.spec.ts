import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import type { components } from "../src/api/generated/public";

type ExactArtifactRef = components["schemas"]["ExactArtifactRef"];
type RunStatus = components["schemas"]["RunStatus"];
interface ScanReport {
  schemaVersion: number;
  tool: string;
  inputDigest: string;
  inputArtifacts: Record<string, ExactArtifactRef>;
  observation: {
    status: string;
    errorCode: string | null;
    exitCode: number;
    wordlistArtifact: ExactArtifactRef;
    wordlistEntries: number;
    payloadsAttempted: number;
    requestErrors: number;
    scanComplete: boolean;
    resultsTruncated: boolean;
    results: Array<{ status: number }>;
  };
}
interface BrowserEvidence {
  schemaVersion: 1;
  workflow: string;
  target: string;
  wordlistDigest: string;
  projectId?: string;
  uploadedArtifact?: ExactArtifactRef;
  submittedArtifact?: ExactArtifactRef;
  runId?: string;
  runInput?: ExactArtifactRef;
  runState?: string;
  reportArtifact?: ExactArtifactRef;
  technicalOutcome?: string;
  observation?: ScanReport["observation"];
  screenshots: Record<string, string>;
  passed: boolean;
}
const enabled = process.env.CONTRACTOR_SCAN_BROWSER === "1";
test.skip(!enabled, "CONTRACTOR_SCAN_BROWSER=1 requires the real scan stack");
function requiredEnvironment(name: string): string {
  const value = process.env[name];
  if (value === undefined || value === "")
    throw new Error(`${name} is required`);
  return value;
}
function exactName(ref: ExactArtifactRef): string {
  return `${ref.namespace}/${ref.name}@${ref.revision}`;
}
async function captureScreenshot(
  page: Page,
  evidence: BrowserEvidence,
  directory: string,
  name: string,
): Promise<void> {
  const filename = `${name}.png`;
  await page.screenshot({
    path: path.join(directory, filename),
    fullPage: true,
  });
  evidence.screenshots[name] = filename;
}
// The Go harness provisions real Server, Runtime, ffuf and a loopback fixture.
// All mutations use the visible UI; there are no route mocks or seeded inputs.
test("uploads a Project wordlist, pins its revision and opens a real ffuf report", async ({
  page,
  context,
}) => {
  const apiURL = requiredEnvironment("CONTRACTOR_UI_E2E_API_URL");
  const target = `${requiredEnvironment("CONTRACTOR_SCAN_TARGET_URL").replace(/\/$/, "")}/FUZZ`;
  const directory = requiredEnvironment("CONTRACTOR_SCAN_EVIDENCE_DIR");
  const payload = Buffer.from("health\nmissing\n", "utf8");
  const evidence: BrowserEvidence = {
    schemaVersion: 1,
    workflow: "ffuf-wordlist@1",
    target,
    wordlistDigest: `sha256:${createHash("sha256").update(payload).digest("hex")}`,
    screenshots: {},
    passed: false,
  };
  await mkdir(directory, { recursive: true });
  try {
    await page.goto("/projects");
    await expect(page.getByRole("region", { name: "Sign in" })).toBeVisible();
    await page
      .getByLabel("Username")
      .fill(requiredEnvironment("CONTRACTOR_UI_E2E_USERNAME"));
    await page
      .getByLabel("Password")
      .fill(requiredEnvironment("CONTRACTOR_UI_E2E_PASSWORD"));
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible();
    await page.getByRole("button", { name: "New Project" }).first().click();
    const projectForm = page.locator("form.project-create-form");
    await projectForm
      .getByLabel("Name", { exact: true })
      .fill("Scan browser fixture");
    await projectForm
      .getByLabel("Description", { exact: true })
      .fill("Controlled local ffuf upload and report journey");
    await projectForm.getByRole("button", { name: "Create Project" }).click();
    await expect(page).toHaveURL(/\/projects\/project_[A-Za-z0-9_-]+$/);
    const projectId = new URL(page.url()).pathname.split("/").at(-1)!;
    evidence.projectId = projectId;
    await page
      .getByRole("navigation", { name: "Project sections" })
      .getByRole("link", { name: "Artifacts", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Add artifact", exact: true })
      .click();
    await page.getByRole("button", { name: "Other", exact: true }).click();
    const upload = page.getByRole("dialog").filter({
      has: page.getByRole("heading", { name: "Other", exact: true }),
    });
    await expect(upload).toBeVisible();
    await upload
      .getByRole("combobox", { name: "File format", exact: true })
      .selectOption("text/vnd.contractor.wordlist");
    await upload.getByLabel("Drop a file here").setInputFiles({
      name: "paths.txt",
      mimeType: "text/plain",
      buffer: payload,
    });
    await upload.getByLabel("Namespace", { exact: true }).fill("lists");
    await upload.getByLabel("Name", { exact: true }).fill("paths");
    await expect(upload.getByLabel("Media type", { exact: true })).toHaveValue(
      "text/vnd.contractor.wordlist",
    );
    await captureScreenshot(page, evidence, directory, "wordlist-upload");
    const stored = page.waitForResponse(
      (response) =>
        response.request().method() === "PUT" &&
        new URL(response.url()).pathname ===
          `/v1/projects/${projectId}/artifacts/lists/paths`,
    );
    await upload.getByRole("button", { name: "Create binding" }).click();
    const storedResponse = await stored;
    expect(storedResponse.ok(), await storedResponse.text()).toBe(true);
    expect(storedResponse.request().postDataBuffer()).toEqual(payload);
    const written =
      (await storedResponse.json()) as components["schemas"]["ArtifactWriteResponse"];
    expect(written.mediaType).toBe("text/vnd.contractor.wordlist");
    expect(written.artifact).toMatchObject({
      namespace: "lists",
      name: "paths",
    });
    expect(written.artifact.revision).not.toBe("");
    evidence.uploadedArtifact = written.artifact;
    await expect(upload).toBeHidden();
    await page.getByRole("link", { name: "lists/paths", exact: true }).click();
    await page
      .getByRole("button", { name: "Load preview", exact: true })
      .click();
    await expect(page.locator("pre.artifact-preview")).toHaveText(
      payload.toString(),
    );
    await captureScreenshot(page, evidence, directory, "wordlist-preview");
    await page.goto(`/projects/${projectId}/workflows`);
    await page
      .getByRole("button", { name: "All workflows", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Configure ffuf-wordlist@1", exact: true })
      .click();
    const setup = page.getByRole("dialog").filter({
      has: page.getByRole("heading", { name: "Configure Run", exact: true }),
    });
    await expect(setup.locator(".workflow-drawer-heading code")).toHaveText(
      "ffuf-wordlist@1",
    );
    await setup.locator('[name="parameter-target"]').fill(target);
    const selected = exactName(written.artifact);
    await setup.locator('[name="artifact-wordlist"]').selectOption(selected);
    await expect(setup.locator('[name="artifact-wordlist"]')).toHaveValue(
      selected,
    );
    const inputReview = setup.getByRole("region", {
      name: "Exact input review for wordlist",
    });
    await expect(inputReview).toContainText(`ProjectScope · ${projectId}`);
    const confirm = setup.getByRole("button", {
      name: "Confirm exact input for wordlist",
    });
    if (await confirm.isVisible()) await confirm.click();
    await expect(inputReview).toContainText("Confirmed");
    await expect(inputReview).toContainText(selected);
    await captureScreenshot(page, evidence, directory, "selected-input");
    const created = page.waitForResponse(
      (response) =>
        response.request().method() === "POST" &&
        new URL(response.url()).pathname === `/v1/projects/${projectId}/runs`,
    );
    await setup
      .getByRole("button", { name: "Start Project Workflow Run", exact: true })
      .click();
    const createdResponse = await created;
    expect(createdResponse.status(), await createdResponse.text()).toBe(202);
    const submitted = createdResponse.request().postDataJSON() as {
      workflow: string;
      parameters: Record<string, string>;
      artifacts: Record<string, ExactArtifactRef>;
    };
    expect(submitted.workflow).toBe("ffuf-wordlist@1");
    expect(submitted.parameters).toEqual({ target });
    expect(submitted.artifacts.wordlist).toEqual(written.artifact);
    evidence.submittedArtifact = submitted.artifacts.wordlist!;
    await expect(page).toHaveURL(/\/runs\/run_[A-Za-z0-9_-]+$/);
    const runId = new URL(page.url()).pathname.split("/").at(-1)!;
    evidence.runId = runId;
    await expect(page.locator(".run-triage")).toHaveClass(
      /run-triage-(?:succeeded|failed|cancelled)(?:\s|$)/,
      { timeout: 180_000 },
    );
    // Read-only API evidence checks the terminal Run displayed by the UI.
    const statusResponse = await context.request.get(
      `${apiURL}/v1/runs/${runId}`,
    );
    expect(statusResponse.status()).toBe(200);
    const run = (await statusResponse.json()) as RunStatus;
    evidence.runState = run.state;
    expect(run.state, await page.locator(".run-triage").innerText()).toBe(
      "succeeded",
    );
    expect(run.projectId).toBe(projectId);
    expect(run.workflow).toBe("ffuf-wordlist@1");
    const runInput = run.inputs?.wordlist;
    const reportArtifact = run.outputs.report;
    expect(runInput).toMatchObject({ namespace: "inputs", name: "wordlist" });
    if (runInput === undefined || reportArtifact === undefined)
      throw new Error("Completed scan is missing exact input/report refs");
    expect(runInput.revision).not.toBe("");
    expect(reportArtifact.revision).not.toBe("");
    evidence.runInput = runInput;
    evidence.reportArtifact = reportArtifact;
    const output = page.locator(".run-result-card").filter({
      has: page.getByRole("heading", { name: "report", exact: true }),
    });
    await output
      .getByRole("button", { name: "Preview result", exact: true })
      .click();
    const summary = output.getByRole("region", { name: "Scanner execution" });
    await expect(summary).toBeVisible();
    await expect(summary.getByText("Completed", { exact: true })).toBeVisible();
    await expect(summary).toContainText(
      "does not establish that the target is free of vulnerabilities",
    );
    const source = output.locator("pre.artifact-preview");
    await expect(source).toBeVisible();
    const report = JSON.parse((await source.textContent()) ?? "") as ScanReport;
    expect(report.schemaVersion).toBe(1);
    expect(report.tool).toBe("scan_ffuf");
    expect(report.inputDigest).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(report.inputArtifacts.wordlist).toEqual(runInput);
    expect(report.observation.wordlistArtifact).toEqual(runInput);
    expect(report.observation).toMatchObject({
      status: "completed",
      errorCode: null,
      exitCode: 0,
      wordlistEntries: 2,
      payloadsAttempted: 2,
      requestErrors: 0,
      scanComplete: true,
      resultsTruncated: false,
    });
    expect(
      report.observation.results.map((item) => item.status).sort(),
    ).toEqual([200, 404]);
    evidence.technicalOutcome = "Completed";
    evidence.observation = report.observation;
    await captureScreenshot(page, evidence, directory, "scan-report");
    evidence.passed = true;
  } finally {
    await writeFile(
      path.join(directory, "browser-evidence.json"),
      `${JSON.stringify(evidence, null, 2)}\n`,
      { mode: 0o600 },
    );
  }
});
