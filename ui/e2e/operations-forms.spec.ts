import { expect, test, type Page } from "@playwright/test";

const now = "2026-09-19T10:00:00Z";
const config = (name: string, digit: string) => ({
  ref: { name, version: "1", digest: `sha256:${digit.repeat(64)}` },
  document: {
    apiVersion: "contractor/v1alpha1",
    kind: "RuntimeConfig",
    metadata: { name, version: "1" },
    spec: {},
  },
  builtIn: false,
  createdBy: "operator",
  createdAt: now,
});
const base = config("base", "1"),
  proposed = config("proposed", "2"),
  concurrent = config("concurrent", "3");

async function installOperations(page: Page, authorized = true) {
  const origin = new URL(test.info().project.use.baseURL as string).origin;
  let binding = {
    label: "debug",
    config: base.ref,
    revision: "1",
    createdBy: "operator",
    updatedBy: "operator",
    createdAt: now,
    updatedAt: now,
  };
  const writes: { path: string; ifMatch: string | undefined; body: unknown }[] =
    [];
  let bindingAttempts = 0;
  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: ["contractor.public.v1"],
        apiBaseUrl: origin,
      },
    }),
  );
  await page.route(`${origin}/v1/**`, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const reply = (value: unknown, status = 200, headers = {}) =>
      route.fulfill({
        status,
        json: value,
        headers: {
          "X-Contractor-API-Version": "contractor.public.v1",
          ...headers,
        },
      });
    if (path === "/v1/auth/session")
      return reply({
        principal: {
          userId: "operator",
          username: "operator",
          capabilities: authorized ? ["user", "operations"] : ["user"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
    if (request.method() !== "GET")
      writes.push({
        path,
        ifMatch: request.headers()["if-match"],
        body: request.postDataJSON(),
      });
    if (path === "/v1/operations/snapshot")
      return reply({
        cursor: { generation: "operations-form-fixture", revision: "3" },
        runtimeAgents: [
          {
            instanceId: "idle-runtime",
            softwareVersion: "1.0.0",
            supportedRuntimes: ["adk@1"],
            supportedToolsets: [],
            supportedSandboxProfiles: ["local-workdir@1"],
            supportedRuntimeAdapters: [],
            observedState: "idle",
            slotState: "idle",
          },
        ],
        allocations: [],
      });
    if (path === "/v1/operations/runtime-configs") {
      if (request.method() === "POST")
        return reply(
          {
            code: "conflict",
            message:
              "This immutable version already exists. Choose a new version.",
            retryable: false,
            requestId: "publish-conflict",
          },
          409,
        );
      return reply({
        items: [base, proposed, concurrent],
        page: { hasMore: false },
      });
    }
    if (path === "/v1/operations/runtime-labels")
      return reply({ items: [binding], page: { hasMore: false } });
    if (
      path === "/v1/operations/runtime-labels/debug" &&
      request.method() === "PUT"
    ) {
      bindingAttempts++;
      if (bindingAttempts === 1) {
        binding = { ...binding, config: concurrent.ref, revision: "2" };
        return reply(
          {
            code: "precondition_failed",
            message: "Runtime label revision changed",
            retryable: false,
            requestId: "binding-conflict",
          },
          412,
        );
      }
      expect(request.headers()["if-match"]).toBe('"2"');
      expect(request.postDataJSON()).toEqual({ config: proposed.ref });
      binding = { ...binding, config: proposed.ref, revision: "3" };
      return reply(binding, 200, { ETag: '"3"' });
    }
    if (path === "/v1/operations/runtime-credentials") {
      if (request.method() === "POST")
        return reply(
          {
            code: "forbidden",
            message: "Credential creation denied by Server",
            retryable: false,
            requestId: "credential-denied",
          },
          403,
        );
      return reply({ items: [], page: { hasMore: false } });
    }
    if (path === "/v1/configurations/llm-gateways")
      return reply({ items: [], page: { hasMore: false } });
    return reply(
      {
        code: "not_found",
        message: `Fixture does not implement ${path}`,
        retryable: false,
        requestId: "fixture-missing",
      },
      404,
    );
  });
  return writes;
}

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test(`explicit Operations forms preserve proposals and clear secrets at ${viewport.width}px`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    const writes = await installOperations(page);
    await page.goto("/operations");
    await expect(
      page.getByRole("heading", { name: "Execution readiness" }),
    ).toBeVisible();
    await expect(
      page.getByText(/Idle slots do not establish compatible capacity/),
    ).toBeVisible();
    await expect(
      page.getByText("operations-form-fixture", { exact: true }),
    ).not.toBeVisible();
    await page
      .getByText("Diagnostics: snapshot and live connection", { exact: true })
      .click();
    await expect(
      page.getByText("operations-form-fixture", { exact: true }),
    ).toBeVisible();
    await page
      .getByRole("link", { name: "Runtime configuration →", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "RuntimeConfig versions" }),
    ).toBeVisible();
    await expect(page.getByRole("dialog")).toHaveCount(0);
    await expect(page.getByLabel("RuntimeConfig name")).toHaveCount(0);

    const publish = page.getByRole("button", {
      name: "Publish RuntimeConfig",
      exact: true,
    });
    await publish.focus();
    await page.keyboard.press("Enter");
    let dialog = page.getByRole("dialog", {
      name: "Publish RuntimeConfig",
      exact: true,
    });
    await expect(
      dialog.getByLabel("RuntimeConfig name", { exact: true }),
    ).toBeFocused();
    await dialog
      .getByLabel("RuntimeConfig name", { exact: true })
      .fill("proposed");
    await dialog.getByRole("checkbox", { name: /^Worker telemetry/ }).check();
    await dialog
      .getByRole("group", { name: "Worker telemetry" })
      .getByLabel("OTLP traces endpoint", { exact: true })
      .fill("http://collector.test/v1/traces");
    await dialog
      .getByRole("button", { name: "Publish immutable RuntimeConfig" })
      .click();
    await expect(
      dialog.getByText(/This immutable version already exists/),
    ).toBeVisible();
    await expect(
      dialog.getByLabel("RuntimeConfig name", { exact: true }),
    ).toHaveValue("proposed");
    await expect(
      dialog.getByText("Proposed new version: proposed@1"),
    ).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(publish).toBeFocused();

    await page
      .getByRole("button", { name: "Manage bindings for base@1" })
      .click();
    dialog = page.getByRole("dialog", { name: "Runtime label bindings" });
    const proposal = dialog.getByRole("combobox", {
      name: "RuntimeConfig for debug",
    });
    await proposal.selectOption(`proposed@1:${proposed.ref.digest}`);
    await expect(
      dialog.getByText(
        /Already prepared allocations keep their pinned settings/,
      ),
    ).toBeVisible();
    await dialog
      .getByRole("button", { name: "Rebind with current revision" })
      .click();
    await expect(
      dialog.getByText("Binding changed in another view."),
    ).toBeVisible();
    await expect(
      dialog.getByRole("button", { name: "Rebind with current revision" }),
    ).toBeDisabled();
    await dialog
      .getByRole("button", { name: "Reload authoritative binding" })
      .click();
    await expect(dialog.getByText("Current binding revision 2")).toBeVisible();
    await expect(proposal).toHaveValue(`proposed@1:${proposed.ref.digest}`);
    await expect(
      dialog.getByText("concurrent@1", { exact: true }).first(),
    ).toBeVisible();
    expect(
      writes.filter((write) => write.path.endsWith("/debug")),
    ).toHaveLength(1);
    await dialog
      .getByRole("button", { name: "Rebind with current revision" })
      .click();
    await expect(dialog.getByText("Current binding revision 3")).toBeVisible();
    await page.keyboard.press("Escape");

    const credential = page.getByRole("button", {
      name: "Add Runtime credential",
    });
    await credential.focus();
    await page.keyboard.press("Enter");
    dialog = page.getByRole("dialog", { name: "Create Runtime credential" });
    await expect(
      dialog.getByLabel("Runtime credential ID", { exact: true }),
    ).toBeFocused();
    await dialog
      .getByLabel("Runtime credential ID", { exact: true })
      .fill("review-only");
    await expect(dialog.getByLabel(/Header value/)).toHaveAttribute(
      "type",
      "password",
    );
    await dialog.getByLabel(/Header value/).fill("fixture-secret-discarded");
    await dialog
      .getByRole("button", { name: "Create active Runtime credential" })
      .click();
    await expect(
      dialog.getByText("Credential creation denied by Server"),
    ).toBeVisible();
    await expect(dialog.getByLabel(/Header value/)).toHaveValue("");
    await dialog.getByLabel(/Header value/).fill("fixture-secret-close");
    await page.keyboard.press("Escape");
    await expect(credential).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(
      page.getByRole("dialog").getByLabel(/Header value/),
    ).toHaveValue("");
    await page.keyboard.press("Escape");
    expect(writes).toHaveLength(4);
    expect(
      writes
        .filter((write) => write.path.endsWith("/debug"))
        .map((write) => write.ifMatch),
    ).toEqual(['"1"', '"2"']);
    await expect(page.locator("body")).not.toContainText("fixture-secret");
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  });
}

test("Operations forms are unavailable without the actual session capability", async ({
  page,
}) => {
  const writes = await installOperations(page, false);
  await page.goto("/runs/configuration");
  await expect(
    page.getByText(
      "Operations capability is required to manage Runtime configurations.",
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Publish RuntimeConfig" }),
  ).toHaveCount(0);
  await page.goto("/operations");
  await expect(
    page.getByText(/not authorized to observe or manage Operations/),
  ).toBeVisible();
  expect(writes).toEqual([]);
});
