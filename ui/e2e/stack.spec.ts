import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  expect,
  request as playwrightRequest,
  test,
  type APIRequestContext,
  type BrowserContext,
  type Locator,
  type Page,
  type Request,
  type Response,
  type WebSocket,
} from "@playwright/test";

interface BrowserEvidence {
  requests: Array<{ method: string; url: string; bodyBase64?: string }>;
  responses: Array<{ status: number; url: string; bodyBase64?: string }>;
  websocketURLs: string[];
  websocketFrames: Array<{ direction: "sent" | "received"; payload: string }>;
  console: string[];
  pageText: string;
  localStorage: Record<string, string>;
  sessionStorage: Record<string, string>;
  indexedDatabases: string[];
  cookies: unknown[];
}

const enabled = process.env.CONTRACTOR_UI_STACK === "1";
test.skip(!enabled, "CONTRACTOR_UI_STACK=1 is required for the real stack");

function requiredEnvironment(name: string): string {
  const value = process.env[name];
  if (value === undefined || value === "") {
    throw new Error(`${name} is required`);
  }
  return value;
}

function boundedBase64(value: Buffer, maximum = 64 * 1024): string {
  return value.subarray(0, maximum).toString("base64");
}

function textPayload(value: string | Buffer): string {
  return typeof value === "string"
    ? value.slice(0, 64 * 1024)
    : boundedBase64(value);
}

async function bounded<T>(
  operation: Promise<T>,
  timeoutMilliseconds: number,
  label: string,
): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(
          () => reject(new Error(`${label} exceeded ${timeoutMilliseconds}ms`)),
          timeoutMilliseconds,
        );
      }),
    ]);
  } finally {
    if (timeout !== undefined) {
      clearTimeout(timeout);
    }
  }
}

function installEvidenceCapture(
  page: Page,
  apiOrigin: string,
  evidence: BrowserEvidence,
): () => Promise<void> {
  const pending = new Set<Promise<void>>();
  const track = (operation: Promise<void>) => {
    pending.add(operation);
    void operation.finally(() => pending.delete(operation));
  };
  const captureRequest = (request: Request) => {
    if (!request.url().startsWith(`${apiOrigin}/v1/`)) {
      return;
    }
    const body = request.postDataBuffer();
    evidence.requests.push({
      method: request.method(),
      url: request.url(),
      ...(body === null ? {} : { bodyBase64: boundedBase64(body) }),
    });
  };
  const captureResponse = async (response: Response) => {
    if (!response.url().startsWith(`${apiOrigin}/v1/`)) {
      return;
    }
    let bodyBase64: string | undefined;
    try {
      bodyBase64 = boundedBase64(
        await bounded(response.body(), 2_000, "response body capture"),
      );
    } catch {
      // A navigation or download may dispose the response before inspection.
    }
    evidence.responses.push({
      status: response.status(),
      url: response.url(),
      ...(bodyBase64 === undefined ? {} : { bodyBase64 }),
    });
  };
  const captureSocket = (socket: WebSocket) => {
    if (!socket.url().startsWith(apiOrigin.replace(/^http/, "ws"))) {
      return;
    }
    evidence.websocketURLs.push(socket.url());
    socket.on("framesent", ({ payload }) =>
      evidence.websocketFrames.push({
        direction: "sent",
        payload: textPayload(payload),
      }),
    );
    socket.on("framereceived", ({ payload }) =>
      evidence.websocketFrames.push({
        direction: "received",
        payload: textPayload(payload),
      }),
    );
  };
  page.on("request", captureRequest);
  page.on("response", (response) => track(captureResponse(response)));
  page.on("websocket", captureSocket);
  page.on("console", (message) =>
    evidence.console.push(message.text().slice(0, 4096)),
  );
  return async () => {
    await Promise.all([...pending]);
  };
}

async function selectOptionContaining(
  select: Locator,
  expected: string,
): Promise<void> {
  const option = select.locator("option").filter({ hasText: expected }).first();
  await expect(option).toHaveCount(1);
  const value = await option.getAttribute("value");
  if (value === null || value === "") {
    throw new Error(`option containing ${expected} has no value`);
  }
  await select.selectOption(value);
}

async function openDetails(details: Locator): Promise<void> {
  if (
    !(await details.evaluate((element: HTMLDetailsElement) => element.open))
  ) {
    await details.locator(":scope > summary").click();
  }
}

function stageAttempt(page: Page, stage: string): Locator {
  return page.locator("details.run-attempt").filter({
    has: page.locator(":scope > summary").getByText(stage, { exact: true }),
  });
}

async function invokeControl(
  client: APIRequestContext,
  controlURL: string,
  token: string,
  action: string,
): Promise<void> {
  const response = await client.post(`${controlURL}/${action}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  expect(response.status(), await response.text()).toBe(204);
}

async function login(page: Page, username: string, password: string) {
  await page.getByLabel("Username").fill(username);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("button", { name: "Sign out" })).toBeVisible();
}

async function uploadArtifact(
  page: Page,
  input: {
    name: string;
    mediaType: string;
    path?: string;
    payload?: Buffer;
  },
) {
  await page.locator("details.artifact-create-disclosure > summary").click();
  const form = page.locator("form.artifact-form");
  await form.getByLabel("Namespace", { exact: true }).fill("projects");
  await form.getByLabel("Name", { exact: true }).fill(input.name);
  await form.getByLabel("Media type", { exact: true }).fill(input.mediaType);
  if (input.path !== undefined) {
    await form.getByLabel("Drop a file here").setInputFiles(input.path);
  } else {
    await form.getByLabel("Drop a file here").setInputFiles({
      name: `${input.name}.txt`,
      mimeType: input.mediaType,
      buffer: input.payload ?? Buffer.from(""),
    });
  }
  await form.getByRole("button", { name: "Create binding" }).click();
  await expect(
    page.getByRole("link", {
      name: `projects/${input.name}`,
      exact: true,
    }),
  ).toBeVisible();
}

async function uploadProjectArtifact(
  page: Page,
  shortcut: "Sources" | "OpenAPI",
  input: {
    namespace: string;
    name: string;
    mediaType: string;
    path?: string;
    payload?: Buffer;
  },
) {
  await page.getByRole("button", { name: shortcut, exact: true }).click();
  const dialog = page.getByRole("dialog", { name: shortcut });
  if (input.path !== undefined) {
    await dialog.getByLabel("Drop a file here").setInputFiles(input.path);
  } else {
    await dialog.getByLabel("Drop a file here").setInputFiles({
      name: `${input.name}.yaml`,
      mimeType: input.mediaType,
      buffer: input.payload ?? Buffer.from(""),
    });
  }
  await dialog.getByLabel("Namespace", { exact: true }).fill(input.namespace);
  await dialog.getByLabel("Name", { exact: true }).fill(input.name);
  await dialog.getByLabel("Media type", { exact: true }).fill(input.mediaType);
  await dialog.getByRole("button", { name: "Create binding" }).click();
  await expect(
    page.getByRole("link", {
      name: `${input.namespace}/${input.name}`,
      exact: true,
    }),
  ).toBeVisible();
}

async function startRun(
  page: Page,
  workflow: string,
  parameters: Record<string, string>,
  artifacts: Record<string, string>,
  metadataLabels: Record<string, string> = {},
): Promise<string> {
  await page.goto(`/workflows/${workflow.replace("@", "/")}`);
  await expect(page.getByRole("heading", { name: workflow })).toBeVisible();
  for (const [name, value] of Object.entries(parameters)) {
    const input = page.locator(`input[name="parameter-${name}"]`);
    if (await input.isDisabled()) {
      await page.getByLabel(`Include optional ${name}`).check();
    }
    await input.fill(value);
  }
  for (const [name, binding] of Object.entries(artifacts)) {
    await selectOptionContaining(
      page.locator(`select[name="artifact-${name}"]`),
      binding,
    );
  }
  let metadataIndex = 0;
  if (Object.keys(metadataLabels).length > 0) {
    await openDetails(
      page.locator("details.run-draft-disclosure").filter({
        has: page.getByText("Run metadata", { exact: true }),
      }),
    );
  }
  for (const [key, value] of Object.entries(metadataLabels)) {
    metadataIndex += 1;
    await page.getByRole("button", { name: "Add metadata label" }).click();
    await page.getByLabel(`Run metadata label key ${metadataIndex}`).fill(key);
    await page
      .getByLabel(`Run metadata label value ${metadataIndex}`)
      .fill(value);
  }
  await page.getByRole("button", { name: "Start Workflow Run" }).click();
  await expect(page).toHaveURL(/\/runs\/run_[A-Za-z0-9_-]+$/);
  return new URL(page.url()).pathname.split("/").at(-1)!;
}

async function waitForUI(client: APIRequestContext, baseURL: string) {
  await expect
    .poll(
      async () => {
        try {
          return (await client.get(baseURL)).status();
        } catch {
          return 0;
        }
      },
      { timeout: 20_000 },
    )
    .toBe(200);
}

async function storageEvidence(
  context: BrowserContext,
  page: Page,
): Promise<
  Pick<
    BrowserEvidence,
    | "pageText"
    | "localStorage"
    | "sessionStorage"
    | "indexedDatabases"
    | "cookies"
  >
> {
  const storage = await bounded(
    page.evaluate(async () => {
      const local: Record<string, string> = {};
      const session: Record<string, string> = {};
      for (let index = 0; index < localStorage.length; index += 1) {
        const key = localStorage.key(index);
        if (key !== null) local[key] = localStorage.getItem(key) ?? "";
      }
      for (let index = 0; index < sessionStorage.length; index += 1) {
        const key = sessionStorage.key(index);
        if (key !== null) session[key] = sessionStorage.getItem(key) ?? "";
      }
      const databases =
        typeof indexedDB.databases === "function"
          ? (await indexedDB.databases())
              .map((database) => database.name)
              .filter((name): name is string => name !== undefined)
          : [];
      return {
        pageText: document.body.textContent ?? "",
        localStorage: local,
        sessionStorage: session,
        indexedDatabases: databases,
      };
    }),
    10_000,
    "browser storage capture",
  );
  return { ...storage, cookies: await context.cookies() };
}

test("operates the real single-VM stack without crossing secret boundaries", async ({
  page,
  context,
  request,
}) => {
  const baseURL = requiredEnvironment("CONTRACTOR_UI_E2E_BASE_URL");
  const apiURL = requiredEnvironment("CONTRACTOR_UI_E2E_API_URL");
  const apiDirectURL = requiredEnvironment("CONTRACTOR_UI_E2E_API_DIRECT_URL");
  const uiDirectURL = requiredEnvironment("CONTRACTOR_UI_E2E_UI_DIRECT_URL");
  const controlURL = requiredEnvironment("CONTRACTOR_UI_E2E_CONTROL_URL");
  const controlToken = requiredEnvironment("CONTRACTOR_UI_E2E_CONTROL_TOKEN");
  const username = requiredEnvironment("CONTRACTOR_UI_E2E_USERNAME");
  const password = requiredEnvironment("CONTRACTOR_UI_E2E_PASSWORD");
  const sourceArchive = requiredEnvironment("CONTRACTOR_UI_E2E_SOURCE_ARCHIVE");
  const evidencePath = requiredEnvironment("CONTRACTOR_UI_E2E_EVIDENCE_PATH");
  const screenshotPath = requiredEnvironment(
    "CONTRACTOR_UI_E2E_SCREENSHOT_PATH",
  );
  await mkdir(path.dirname(evidencePath), { recursive: true });

  const evidence: BrowserEvidence = {
    requests: [],
    responses: [],
    websocketURLs: [],
    websocketFrames: [],
    console: [],
    pageText: "",
    localStorage: {},
    sessionStorage: {},
    indexedDatabases: [],
    cookies: [],
  };
  const flushCapture = installEvidenceCapture(page, apiURL, evidence);

  await page.goto("/workflows");
  await expect(page.getByRole("region", { name: "Sign in" })).toBeVisible();
  await login(page, username, password);

  const apiCookies = await context.cookies([apiURL]);
  expect(apiCookies).toHaveLength(1);
  expect(apiCookies[0]).toMatchObject({
    httpOnly: true,
    secure: true,
    sameSite: "Lax",
  });
  expect(await context.cookies([baseURL])).toHaveLength(0);
  await page.reload();
  await expect(page.getByRole("heading", { name: "Workflows" })).toBeVisible();

  const preflight = await request.fetch(`${apiDirectURL}/v1/runs`, {
    method: "OPTIONS",
    headers: {
      Origin: baseURL,
      "Access-Control-Request-Method": "POST",
      "Access-Control-Request-Headers":
        "content-type, idempotency-key, x-csrf-token",
    },
  });
  expect(preflight.status()).toBe(204);
  expect(preflight.headers()["access-control-allow-origin"]).toBe(baseURL);
  expect(preflight.headers()["access-control-allow-credentials"]).toBe("true");

  const runsBefore = await page.evaluate(async (origin) => {
    const response = await fetch(`${origin}/v1/runs`, {
      credentials: "include",
    });
    return (await response.json()) as { items: unknown[] };
  }, apiURL);
  const missingCSRF = await page.evaluate(async (origin) => {
    const response = await fetch(`${origin}/v1/runs`, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": "ui-stack-missing-csrf",
      },
      body: JSON.stringify({ workflow: "artifact-copy@1" }),
    });
    return response.status;
  }, apiURL);
  expect(missingCSRF).toBe(403);
  const runsAfter = await page.evaluate(async (origin) => {
    const response = await fetch(`${origin}/v1/runs`, {
      credentials: "include",
    });
    return (await response.json()) as { items: unknown[] };
  }, apiURL);
  expect(runsAfter.items).toHaveLength(runsBefore.items.length);

  const hostile = await playwrightRequest.newContext({
    ignoreHTTPSErrors: true,
    extraHTTPHeaders: { Origin: "http://127.0.0.99:6553" },
  });
  try {
    const rejected = await hostile.post(
      `${apiDirectURL}/v1/configurations/model-policies`,
      {
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": "ui-stack-hostile-origin",
        },
        data: {
          name: "must-not-exist",
          version: "1",
          modelPolicy: { model: "forbidden" },
        },
      },
    );
    expect(rejected.status()).toBe(403);
  } finally {
    await hostile.dispose();
  }

  await page.goto("/artifacts");
  await uploadArtifact(page, {
    name: "ui-stack-text",
    mediaType: "text/plain",
    payload: Buffer.from("browser-driven streamline input\n"),
  });

  const streamlineRunID = await startRun(
    page,
    "streamline-copy@1",
    { mode: "streamline-strict" },
    { source: "ui-stack-text" },
    {
      purpose: "eval",
      "eval.name": "ui-stack-smoke",
      "eval.id": "ui-stack-eval-01",
      "eval.leg": "a",
    },
  );
  await expect(
    page.getByRole("heading", { name: /STREAMLINE_E2E_GLOBAL/ }),
  ).toBeVisible();
  await expect(page.getByText("STREAMLINE_WORKER_TASK")).toBeVisible();
  await expect(page.getByText("current", { exact: true })).toBeVisible();
  await expect(page.getByText(/Live events: live/)).toBeVisible();

  const rejectedCancel = await page.evaluate(
    async ({ origin, runId }) => {
      const response = await fetch(`${origin}/v1/runs/${runId}/cancel`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: "must not be accepted without CSRF" }),
      });
      return response.status;
    },
    { origin: apiURL, runId: streamlineRunID },
  );
  expect(rejectedCancel).toBe(403);
  const runAfterRejectedCancel = await page.evaluate(
    async ({ origin, runId }) => {
      const response = await fetch(`${origin}/v1/runs/${runId}`, {
        credentials: "include",
      });
      return (await response.json()) as {
        state: string;
        cancellation?: unknown;
      };
    },
    { origin: apiURL, runId: streamlineRunID },
  );
  expect(runAfterRejectedCancel).toMatchObject({ state: "running" });
  expect(runAfterRejectedCancel.cancellation).toBeUndefined();

  await page.getByRole("link", { name: "Operations" }).click();
  await expect(page.getByText(/Operations events: live/)).toBeVisible();
  await page.getByRole("link", { name: "Runtime Agents", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Runtime Agents" }),
  ).toBeVisible();
  await expect(
    page.locator(".operations-library table tbody tr").first(),
  ).toBeVisible();
  await page.getByRole("link", { name: "Allocations" }).click();
  const allocation = page.locator("details.allocation-card").first();
  await expect(allocation).toBeVisible();
  await allocation.locator("summary").click();
  await expect(allocation.getByText("Safe aggregate metrics")).toBeVisible();
  await expect(
    page.getByRole("button", { name: /force|release|reassign/i }),
  ).toHaveCount(0);

  const socketsBeforeReconnect = evidence.websocketURLs.length;
  await invokeControl(request, controlURL, controlToken, "interrupt-events");
  await expect(page.getByText(/Operations events: reconnecting/)).toBeVisible({
    timeout: 15_000,
  });
  await expect
    .poll(() => evidence.websocketURLs.length, { timeout: 20_000 })
    .toBeGreaterThan(socketsBeforeReconnect);
  await expect(page.getByText(/Operations events: live/)).toBeVisible({
    timeout: 20_000,
  });

  await invokeControl(request, controlURL, controlToken, "restart-ui");
  await waitForUI(request, uiDirectURL);
  await page.reload();
  await expect(page.getByRole("heading", { name: "Operations" })).toBeVisible();
  await invokeControl(request, controlURL, controlToken, "release-worker");

  await page.goto(`/runs/${streamlineRunID}`);
  await expect(
    page.locator(".run-triage").getByText("succeeded", { exact: true }),
  ).toBeVisible({
    timeout: 45_000,
  });
  const streamlineMetadata = page.locator(".run-metadata-label-panel");
  await expect(streamlineMetadata).toContainText("eval.id:ui-stack-eval-01");
  await expect(streamlineMetadata).toContainText("eval.leg:a");
  await expect(streamlineMetadata).toContainText(
    "Labels are fixed at creation",
  );
  await expect(page.locator(".run-runtime-configuration")).toContainText(
    "Runtime infrastructure configuration",
  );
  await page.goto("/runs?view=completed");
  await openDetails(
    page.locator("details.run-label-filters").filter({
      has: page.getByText("Metadata & eval filters", { exact: true }),
    }),
  );
  await page.getByLabel("Eval ID").fill("ui-stack-eval-01");
  await page.getByLabel("Eval leg").fill("a");
  await page.getByRole("button", { name: "Apply eval filters" }).click();
  await expect(page.getByRole("link", { name: streamlineRunID })).toBeVisible();
  await expect(page).toHaveURL(/label=eval.id%3Dui-stack-eval-01/);
  await expect(page).toHaveURL(/label=eval.leg%3Da/);
  await page.goto(`/runs/${streamlineRunID}`);
  const streamlineAttempt = stageAttempt(page, "copy");
  await expect(streamlineAttempt).toHaveCount(1);
  await openDetails(streamlineAttempt);
  await expect(streamlineAttempt.getByText("Reports complete")).toBeVisible();
  await expect(
    streamlineAttempt.getByRole("heading", {
      name: "Allocation resource observations",
    }),
  ).toBeVisible();
  const runResource = streamlineAttempt.locator(
    "article.allocation-resource-card",
  );
  await expect(runResource).toHaveCount(1);
  await expect(
    runResource.getByText("requested", { exact: true }),
  ).toBeVisible();
  await expect(runResource.getByText(/Runtime process scope/)).toBeVisible();

  await page.goto("/operations/performance");
  await expect(
    page.getByRole("heading", { name: "Server performance" }),
  ).toBeVisible();
  await expect(
    page.getByText("Performance collection is disabled."),
  ).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Server process" }),
  ).toBeVisible();

  await page.goto("/operations/allocations/completed");
  await page.getByLabel("Exact Run ID (optional)").fill(streamlineRunID);
  await page.getByRole("button", { name: "Apply filter" }).click();
  const retainedResource = page
    .locator("article.allocation-resource-card")
    .filter({ has: page.getByRole("link", { name: streamlineRunID }) });
  await expect(retainedResource).toHaveCount(1);
  await expect(
    retainedResource.getByText("available", { exact: true }),
  ).toBeVisible();
  await expect(
    retainedResource.getByText("requested", { exact: true }),
  ).toBeVisible();
  await expect(
    retainedResource.getByText(/Runtime process scope/),
  ).toBeVisible();

  await page.goto(`/runs/${streamlineRunID}`);
  const streamlineOutput = page
    .locator(".run-result-card")
    .filter({ hasText: "outputs/result@" });
  await streamlineOutput
    .getByRole("link", { name: /^Open outputs\/result@/ })
    .click();
  const streamlineDownloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download exact revision" }).click();
  const streamlineDownload = await streamlineDownloadPromise;
  expect((await readFile(await streamlineDownload.path())).toString()).toBe(
    "browser-driven streamline input\n",
  );

  await page.goto("/projects");
  await page.getByRole("button", { name: "New Project" }).first().click();
  const projectForm = page.locator("form.project-create-form");
  await projectForm
    .getByLabel("Name", { exact: true })
    .fill("UI Stack Workspace");
  await projectForm
    .getByLabel("Description", { exact: true })
    .fill("Production browser Project workflow fixture");
  await projectForm.getByRole("button", { name: "Create Project" }).click();
  await expect(page).toHaveURL(/\/projects\/project_[A-Za-z0-9_-]+$/);
  await expect(
    page.getByRole("heading", { name: "UI Stack Workspace" }),
  ).toBeVisible();
  const projectURL = page.url();

  await uploadProjectArtifact(page, "Sources", {
    namespace: "sources",
    name: "ui-stack-source",
    mediaType: "application/zip",
    path: sourceArchive,
  });
  await uploadProjectArtifact(page, "OpenAPI", {
    namespace: "openapi",
    name: "ui-stack-openapi-seed",
    mediaType: "application/yaml",
    payload: Buffer.from(
      "openapi: 3.0.3\ninfo:\n  title: UI Stack\n  version: 1.0.0\npaths: {}\n",
    ),
  });

  await page
    .getByRole("button", { name: "Run openapi-from-workspace@5" })
    .click();
  const workflowDialog = page.getByRole("dialog", {
    name: "openapi-from-workspace",
  });
  await expect(
    workflowDialog.locator(".project-dialog-heading code"),
  ).toHaveText("openapi-from-workspace@5");
  await workflowDialog
    .getByRole("button", { name: "Confirm exact input for source" })
    .click();
  await workflowDialog
    .getByRole("button", { name: "Confirm exact input for existing_openapi" })
    .click();
  await workflowDialog.getByLabel("Include optional objective").check();
  await workflowDialog
    .locator('input[name="parameter-objective"]')
    .fill("Model the browser fixture API and trust boundary");
  await workflowDialog
    .getByRole("button", { name: "Start Project Workflow Run" })
    .click();
  await expect(page).toHaveURL(/\/runs\/run_[A-Za-z0-9_-]+$/);
  const openAPIRunID = new URL(page.url()).pathname.split("/").at(-1)!;
  await page.getByRole("link", { name: "Runs", exact: true }).click();
  await expect(page.getByRole("link", { name: openAPIRunID })).toBeVisible();
  await page.goto(`/runs/${openAPIRunID}`);
  try {
    await expect(
      page.locator(".run-triage").getByText("succeeded", { exact: true }),
    ).toBeVisible({
      timeout: 180_000,
    });
  } catch (error) {
    const diagnostics = await page.evaluate(
      async ({ origin, runId }) =>
        await Promise.all(
          [
            `/v1/runs/${runId}`,
            "/v1/operations/snapshot",
            "/v1/operations/runtime-agent-principals?limit=50",
          ].map(async (path) =>
            (
              await fetch(`${origin}${path}`, {
                credentials: "include",
              })
            ).json(),
          ),
        ),
      { origin: apiURL, runId: openAPIRunID },
    );
    throw new Error(
      `${String(error)}\nOpenAPI diagnostics: ${JSON.stringify(diagnostics)}`,
      { cause: error },
    );
  }
  await expect(
    page.locator(".run-metadata").getByText("openapi-from-workspace@5", {
      exact: true,
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  const validationAttempt = stageAttempt(page, "openapi_validate");
  await expect(validationAttempt).toHaveCount(1);
  await openDetails(validationAttempt);
  await expect(
    validationAttempt
      .locator(":scope > summary")
      .getByText("openapi_validate", { exact: true }),
  ).toBeVisible();
  const openAPIOutput = page
    .locator(".run-result-card")
    .filter({ hasText: "outputs/openapi@" });
  await openAPIOutput
    .getByRole("link", { name: /^Open outputs\/openapi@/ })
    .click();
  const openAPIDownloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download exact revision" }).click();
  const openAPIDownload = await openAPIDownloadPromise;
  const openAPIBytes = await readFile(await openAPIDownload.path());
  expect(openAPIBytes.toString()).toContain("/widgets/{widget_id}");
  expect(openAPIRunID).toMatch(/^run_/);

  await page.goto(projectURL);
  await expect(
    page.getByRole("link", { name: "outputs/openapi", exact: true }),
  ).toBeVisible();
  await openDetails(page.locator("details.project-all-workflows"));
  await expect(
    page.getByRole("button", { name: "Run again openapi-from-workspace@5" }),
  ).toBeVisible();

  await page.goto("/operations/configurations");
  const workerRow = page
    .getByRole("row")
    .filter({ has: page.getByText("worker@1", { exact: true }) });
  await workerRow.getByRole("link", { name: "Inspect / clone" }).click();
  await page.getByLabel("New immutable version").fill("ui-stack-1");
  await page.getByRole("button", { name: "Publish immutable version" }).click();
  await expect(page.getByText("Published worker@ui-stack-1")).toBeVisible();

  await page.goto("/operations/configurations");
  await expect(page.getByText("must-not-exist@1", { exact: true })).toHaveCount(
    0,
  );
  await page.getByRole("link", { name: "Credentials" }).click();
  const credentialForm = page.locator("form.credential-create-form");
  await credentialForm.getByLabel("Credential ID").fill("ui-stack-key");
  await credentialForm
    .getByLabel("Safe label (optional)")
    .fill("Browser E2E key");
  await selectOptionContaining(
    credentialForm.getByLabel("Exact managed LLM Gateway"),
    "local-litellm@1",
  );
  await credentialForm.getByLabel(/Maximum spend/).fill("3.5");
  await credentialForm.getByLabel(/Budget reset/).fill("1d");
  await credentialForm.getByLabel(/TPM limit/).fill("1000");
  await credentialForm.getByLabel(/RPM limit/).fill("20");
  await credentialForm.getByLabel(/Parallel requests/).fill("2");
  await credentialForm
    .getByLabel(/^worker@1 ·/)
    .first()
    .check();
  await credentialForm
    .getByRole("button", { name: "Create active credential" })
    .click();
  await expect(
    page.getByRole("heading", { name: "ui-stack-key" }),
  ).toBeVisible();
  await expect(page.getByText("3.5", { exact: true })).toBeVisible();
  await invokeControl(
    request,
    controlURL,
    controlToken,
    "check-credential-storage",
  );
  const rejectedCredentialDelete = await page.evaluate(async (origin) => {
    const response = await fetch(
      `${origin}/v1/operations/credentials/ui-stack-key`,
      {
        method: "DELETE",
        credentials: "include",
        headers: { "Idempotency-Key": "ui-stack-delete-without-csrf" },
      },
    );
    return response.status;
  }, apiURL);
  expect(rejectedCredentialDelete).toBe(403);
  const credentialAfterRejectedDelete = await page.evaluate(async (origin) => {
    return (
      await fetch(`${origin}/v1/operations/credentials/ui-stack-key`, {
        credentials: "include",
      })
    ).status;
  }, apiURL);
  expect(credentialAfterRejectedDelete).toBe(200);
  await page.getByLabel(/I understand/).check();
  await page
    .getByRole("button", { name: "Delete from LiteLLM and Contractor" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Active credentials" }),
  ).toBeVisible();
  await expect(page.getByText("ui-stack-key", { exact: true })).toHaveCount(0);

  await page.goto("/operations/runtime-configs");
  await expect(
    page.getByRole("heading", { name: "RuntimeConfig versions" }),
  ).toBeVisible();
  const runtimeCredentialForm = page.locator("form.runtime-credential-form");
  await runtimeCredentialForm
    .getByLabel("Runtime credential ID")
    .fill("ui-stack-otel");
  await runtimeCredentialForm
    .getByLabel(/Header value/)
    .fill("Bearer browser-write-only-value");
  await runtimeCredentialForm
    .getByRole("button", { name: "Create active Runtime credential" })
    .click();
  await expect(
    runtimeCredentialForm.getByText(/Created safe metadata/),
  ).toBeVisible();
  await expect(runtimeCredentialForm.getByLabel(/Header value/)).toHaveValue(
    "",
  );
  await expect(page.getByText("Bearer browser-write-only-value")).toHaveCount(
    0,
  );

  await page.getByLabel("RuntimeConfig name").fill("ui-stack-debug");
  const workerTelemetry = page.getByRole("group", {
    name: /Worker telemetry/,
  });
  await workerTelemetry
    .getByRole("checkbox", { name: /^Worker telemetry/ })
    .check();
  await workerTelemetry
    .getByLabel("OTLP traces endpoint")
    .fill("http://127.0.0.1:9/v1/traces");
  await workerTelemetry
    .getByLabel("Runtime credential ID (optional)")
    .fill("ui-stack-otel");
  await page
    .getByRole("button", { name: "Publish immutable RuntimeConfig" })
    .click();
  await expect(page.getByText(/Published ui-stack-debug@1/)).toBeVisible();
  const runtimeLabelForm = page.locator("form.runtime-label-create");
  await runtimeLabelForm.getByLabel("New label").fill("ui-stack-debug");
  await selectOptionContaining(
    runtimeLabelForm.getByLabel("Exact RuntimeConfig"),
    "ui-stack-debug@1",
  );
  await runtimeLabelForm
    .getByRole("button", { name: "Create binding" })
    .click();
  await expect(
    page
      .locator("article.runtime-binding-card")
      .filter({ hasText: "ui-stack-debug" }),
  ).toBeVisible();

  await page.goto("/operations/runtime-agents");
  const principal = page.locator("article.runtime-principal-card").first();
  await expect(principal).toBeVisible();
  await principal.getByLabel(/ui-stack-debug/).check();
  await principal
    .getByRole("button", { name: "Replace labels with current revision" })
    .click();
  await expect(principal.getByLabel(/ui-stack-debug/)).toBeChecked();
  await expect(principal.getByText(/future allocations/)).toBeVisible();

  await page.goto("/operations");
  const generationBefore = await page
    .locator(".operations-snapshot-record code")
    .first()
    .textContent();
  await invokeControl(request, controlURL, controlToken, "restart-server");
  await page.reload();
  await expect(page.getByRole("region", { name: "Sign in" })).toBeVisible();
  await login(page, username, password);
  await page.goto("/operations");
  const generationAfter = await page
    .locator(".operations-snapshot-record code")
    .first()
    .textContent();
  expect(generationAfter).not.toBe(generationBefore);

  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByRole("region", { name: "Sign in" })).toBeVisible();
  expect(await context.cookies([apiURL])).toHaveLength(0);

  const incompatible = await context.newPage();
  await incompatible.route("**/runtime-config.json", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        uiVersion: "0.1.0",
        supportedApiVersions: ["contractor.public.v2"],
        apiBaseUrl: apiURL,
      }),
    });
  });
  await incompatible.goto(baseURL);
  await expect(
    incompatible.getByRole("heading", {
      name: "UI runtime configuration is invalid",
    }),
  ).toBeVisible();
  await incompatible.close();

  await page.screenshot({ path: screenshotPath, fullPage: true });
  Object.assign(evidence, await storageEvidence(context, page));
  await flushCapture();
  await writeFile(evidencePath, JSON.stringify(evidence, null, 2), {
    mode: 0o600,
  });
});
