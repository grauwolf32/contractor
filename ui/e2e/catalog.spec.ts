import { expect, test } from "@playwright/test";

test("Catalog discovers exact Workflows and Agent usage while preserving legacy routes", async ({
  page,
}, testInfo) => {
  const origin = new URL(String(testInfo.project.use.baseURL)).origin;
  const digest = `sha256:${"1".repeat(64)}`;
  const secondDigest = `sha256:${"2".repeat(64)}`;
  const prompt =
    "# Evidence analyst\n\nRead the supplied evidence before drawing conclusions.\n\n- Cite the relevant files.\n- State uncertainty explicitly.";
  const agent = (version: "1" | "2") => {
    const exactDigest = version === "1" ? digest : secondDigest;
    return {
      ref: {
        kind: "agent-templates",
        name: "evidence-analyst",
        version,
        digest: exactDigest,
      },
      source: "operator",
      body: {
        description: "Investigates source evidence and explains findings.",
        runtime: "adk@1",
        instructions: {
          ref: "instructions/evidence-analyst.md",
          digest: exactDigest,
        },
        modelPolicy: { policyId: "worker", version: "1", digest },
        toolsets: [
          {
            ref: "run-artifacts@1",
            tools: ["read_artifact", "write_artifact"],
          },
        ],
        skills: [{ namespace: "skills", name: "research" }],
        sandboxProfile: "local-workdir@1",
      },
    };
  };
  const workflow = {
    ref: { name: "openapi-from-source", version: "1" },
    presentation: {
      displayName: "OpenAPI source analysis",
      description: "Build an API contract from one exact source artifact.",
    },
    entryStage: "analyze",
    parameters: { objective: { required: false } },
    inputs: {
      source: { required: true, mediaTypes: ["application/zip"] },
    },
    outputs: {
      openapi: { required: true, mediaTypes: ["application/yaml"] },
    },
  };

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
    const url = new URL(route.request().url());
    const path = url.pathname;
    const version = path.includes("/versions/2") ? "2" : "1";
    const exactAgent = agent(version);
    let value: unknown = { items: [], page: { hasMore: false } };
    if (path === "/v1/auth/session")
      value = {
        principal: {
          userId: "owner",
          username: "owner",
          capabilities: ["user"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      };
    else if (path.endsWith("/workflow-bindings"))
      value = {
        items: [
          {
            workflow: workflow.ref,
            stage: "analyze",
            logicalWorker: version === "1" ? "researcher" : "reviewer",
          },
        ],
        page: { hasMore: false },
      };
    else if (path.endsWith("/instructions"))
      value = {
        template: {
          templateId: exactAgent.ref.name,
          version,
          digest: exactAgent.ref.digest,
        },
        instructions: {
          ...exactAgent.body.instructions,
          text: version === "1" ? prompt : "# Evidence analyst v2",
        },
      };
    else if (
      path.startsWith("/v1/configurations/agent-templates/") &&
      path.includes("/versions/")
    )
      value = exactAgent;
    else if (path === "/v1/configurations/agent-templates") {
      const items = [agent("1"), agent("2")];
      const q = (url.searchParams.get("q") ?? "").toLowerCase();
      value = {
        items:
          q === "" ? items : items.filter((item) => item.ref.name.includes(q)),
        page: { hasMore: false },
      };
    } else if (path === "/v1/workflows") {
      const q = (url.searchParams.get("q") ?? "").toLowerCase();
      value = {
        items:
          q === "" ||
          `${workflow.ref.name} ${workflow.presentation.displayName} ${workflow.presentation.description}`
            .toLowerCase()
            .includes(q)
            ? [workflow]
            : [],
        page: { hasMore: false },
      };
    } else if (path === "/v1/workflows/openapi-from-source/versions/1")
      value = { ...workflow, stages: {} };
    await route.fulfill({
      json: value,
      headers: { "X-Contractor-API-Version": "contractor.public.v1" },
    });
  });

  await page.goto("/workflows?keep=yes#section");
  await expect(page).toHaveURL(`${origin}/catalog/workflows?keep=yes#section`);
  await expect(
    page.getByRole("heading", { name: "OpenAPI source analysis" }),
  ).toBeVisible();
  await expect(
    page.getByText("Build an API contract from one exact source artifact."),
  ).toBeVisible();
  await expect(page.getByText("openapi-from-source@1")).toBeVisible();
  await page.getByLabel("Search workflows").fill("API contract");
  await expect(page).toHaveURL(/q=API\+contract/);
  await expect(
    page.getByRole("heading", { name: "OpenAPI source analysis" }),
  ).toBeVisible();

  await page.getByRole("link", { name: "Agents", exact: true }).click();
  await page.getByLabel("Search agents").fill("evidence-analyst");
  await expect(page).toHaveURL(/q=evidence-analyst/);
  await page.locator('a[href="/catalog/agents/evidence-analyst/1"]').click();
  await expect(
    page.getByRole("heading", { name: "Evidence analyst", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Where used" }).getByText("researcher"),
  ).toBeVisible();
  await page.getByLabel("Published version").selectOption("2");
  await expect(page).toHaveURL(/\/catalog\/agents\/evidence-analyst\/2$/);
  await expect(
    page.getByRole("heading", { name: "Evidence analyst v2", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Where used" }).getByText("reviewer"),
  ).toBeVisible();
  await page
    .getByRole("region", { name: "Where used" })
    .getByRole("link", { name: "openapi-from-source@1" })
    .click();
  await expect(
    page.getByRole("heading", { name: "OpenAPI source analysis" }),
  ).toBeVisible();
  await page.getByRole("link", { name: /evidence-analyst@2/ }).click();
  await expect(page.getByLabel("Published version")).toHaveValue("2");

  await page.getByRole("button", { name: "Source", exact: true }).click();
  await expect(page.locator(".catalog-prompt-source")).toHaveText(
    "# Evidence analyst v2",
  );
  await page.getByRole("button", { name: "Preview", exact: true }).click();
  await expect(
    page.getByRole("link", { name: "skills/research" }),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("catalog-agent-desktop.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", { name: "Evidence analyst v2", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("catalog-agent-mobile.png"),
    fullPage: true,
  });
  await page.goto("/skills?keep=yes#section");
  await expect(page).toHaveURL(`${origin}/catalog/skills?keep=yes#section`);
  await expect(
    page.getByRole("heading", { name: "Skills", exact: true }),
  ).toBeVisible();
});
