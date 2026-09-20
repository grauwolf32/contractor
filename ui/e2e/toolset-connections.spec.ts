import { expect, test, type Page } from "@playwright/test";

// All requests reach the real public Server. Fault injection drops a successful
// response only after route.fetch() commits the production mutation.
test.use({ trace: "off" });
const secret = "V61_LOCAL_MCP_SECRET_CANARY";
function required(name: string): string {
  const value = process.env[name];
  if (!value)
    throw new Error(`${name} is required for the toolset release gate`);
  return value;
}
async function login(page: Page) {
  await page.goto("/catalog/workflows/toolset-shared/1");
  await page.getByLabel("Username").fill("admin");
  await page.getByLabel("Password").fill("contractor e2e local password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByRole("button", { name: "Configure Run" }).click();
}
async function safeBrowserState(page: Page) {
  const state = await page.evaluate(async () => ({
    local: { ...localStorage },
    session: { ...sessionStorage },
    databases: await indexedDB.databases(),
    url: location.href,
  }));
  expect(JSON.stringify(state)).not.toContain(secret);
  expect(await page.locator("body").innerText()).not.toContain(secret);
}
for (const step of ["credential", "configuration", "label"] as const) {
  test(`toolset production recovers lost ${step} reply and launches a shared connection`, async ({
    page,
  }) => {
    const endpoint = required("CONTRACTOR_TOOLSET_MCP_URL");
    const label = `browser-${step}`;
    const path =
      step === "credential"
        ? "/v1/operations/runtime-credentials"
        : step === "configuration"
          ? "/v1/operations/runtime-configs"
          : `/v1/operations/runtime-labels/${label}`;
    const keys: string[] = [];
    let lost = false;
    await page.route(`**${path}`, async (route) => {
      if (route.request().method() === "GET") return route.continue();
      keys.push(route.request().headers()["idempotency-key"]!);
      const reply = await route.fetch();
      expect(reply.status()).toBe(201);
      if (!lost) {
        lost = true;
        await route.abort("failed");
      } else {
        expect(reply.headers()["idempotency-replayed"]).toBe("true");
        await route.fulfill({ response: reply });
      }
    });
    await login(page);
    await expect(page.getByText("Inherited from default")).toBeVisible();
    await page
      .getByRole("button", { name: "Create or resume MCP connection" })
      .click();
    const dialog = page.getByRole("dialog", { name: "Create MCP connection" });
    await expect(dialog.getByLabel("Connection name")).toBeFocused();
    await dialog.getByLabel("Connection name").fill(label);
    await dialog.getByLabel("MCP endpoint").fill(endpoint);
    await dialog
      .getByRole("combobox", { name: /^Authentication/ })
      .selectOption("new");
    await dialog.getByLabel("New MCP credential ID").fill(`${label}-secret`);
    await dialog
      .getByLabel("Header value · write only")
      .fill(`Bearer ${secret}`);
    await dialog
      .getByRole("button", { name: "Save and select connection" })
      .click();
    await expect(dialog.getByText(/Retry resumes at/)).toBeVisible();
    await safeBrowserState(page);
    if (step === "credential") {
      await expect(dialog.getByLabel("Header value · write only")).toHaveValue(
        "",
      );
      await dialog
        .getByLabel("Header value · write only")
        .fill(`Bearer ${secret}`);
    }
    if (step === "label") {
      await dialog
        .getByRole("button", { name: "Close MCP connection form" })
        .click();
      await page.getByRole("button", { name: "Close Run setup" }).click();
      await page.getByRole("button", { name: "Configure Run" }).click();
      await page
        .getByRole("button", { name: "Create or resume MCP connection" })
        .click();
      await expect(
        dialog.getByText(new RegExp(`Configuration saved: ${label}@1`)),
      ).toBeVisible();
      await expect(dialog.getByLabel("MCP endpoint")).toHaveCount(0);
    }
    await dialog
      .getByRole("button", { name: "Save and select connection" })
      .click();
    await expect(dialog).toHaveCount(0);
    expect(keys).toHaveLength(2);
    expect(keys[0]).toBe(keys[1]);
    await expect(page.getByText(`Configured by ${label}`)).toBeVisible();
    await expect(
      page.getByRole("checkbox", { name: new RegExp(label) }),
    ).toBeChecked();
    await safeBrowserState(page);
    const accepted = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/v1/runs" &&
        response.request().method() === "POST",
    );
    await page.getByRole("button", { name: "Start Workflow Run" }).click();
    const reply = await accepted;
    expect(reply.status()).toBe(202);
    expect(reply.request().postDataJSON()).toMatchObject({
      workflow: "toolset-shared@1",
      runtimeLabels: [label],
    });
    expect(JSON.stringify(reply.request().postDataJSON())).not.toContain(
      endpoint,
    );
    const run = (await reply.json()) as { runId: string };
    await expect
      .poll(
        async () =>
          (
            (await (
              await page.request.get(`/v1/runs/${run.runId}`)
            ).json()) as { state: string }
          ).state,
        { timeout: 120_000 },
      )
      .toBe("succeeded");
    await safeBrowserState(page);
  });
}

test("toolset production displays conflicting mixed profiles and reviews changed Repeat bindings", async ({
  page,
}) => {
  const sourceRun = required("CONTRACTOR_TOOLSET_SOURCE_RUN");
  await login(page);
  await page.getByRole("checkbox", { name: /^knowledge / }).check();
  await expect(
    page.getByText("Changes: worker.llmGateway, worker.toolsets").first(),
  ).toBeVisible();
  await page.getByRole("checkbox", { name: /^same-knowledge / }).check();
  await expect(
    page.getByText("Conflicting profiles: knowledge, same-knowledge"),
  ).toBeVisible();
  await page.getByRole("button", { name: "Start Workflow Run" }).click();
  await expect(
    page.getByText(/Resolve missing, cleared, unavailable or conflicting/),
  ).toBeVisible();
  await page.getByRole("checkbox", { name: /^same-knowledge / }).uncheck();
  await expect(page.getByText("Configured by knowledge")).toBeVisible();
  await page.getByRole("button", { name: "Close Run setup" }).click();
  await page.goto(`/runs/${sourceRun}`);
  await page.getByRole("button", { name: "Configure another Run" }).click();
  await expect(page.getByText("runtime_binding_changed").first()).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start Workflow Run" }),
  ).toBeDisabled();
  await page
    .getByLabel(/I reviewed the retained inputs, labels and execution settings/)
    .check();
  const accepted = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === "/v1/runs" &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Start Workflow Run" }).click();
  const reply = await accepted;
  expect(reply.status()).toBe(202);
  const run = (await reply.json()) as {
    runId: string;
    runtimeConfiguration: { labels: { config: { version: string } }[] };
  };
  expect(run.runId).not.toBe(sourceRun);
  expect(run.runtimeConfiguration.labels[0]!.config.version).toBe("2");
  await expect
    .poll(
      async () =>
        (
          (await (await page.request.get(`/v1/runs/${run.runId}`)).json()) as {
            state: string;
          }
        ).state,
      { timeout: 120_000 },
    )
    .toBe("succeeded");
  await safeBrowserState(page);
});
