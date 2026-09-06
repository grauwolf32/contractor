import { expect, test } from "@playwright/test";

test("Catalog exposes agent base prompts and preserves legacy routes", async ({
  page,
}, testInfo) => {
  const origin = new URL(String(testInfo.project.use.baseURL)).origin;
  const digest = `sha256:${"1".repeat(64)}`;
  const prompt =
    "# Evidence analyst\n\nRead the supplied evidence before drawing conclusions.\n\n- Cite the relevant files.\n- State uncertainty explicitly.";
  const agent = {
    ref: {
      kind: "agent-templates",
      name: "evidence-analyst",
      version: "1",
      digest,
    },
    source: "operator",
    body: {
      description: "Investigates source evidence and explains findings.",
      runtime: "adk@1",
      instructions: { ref: "instructions/evidence-analyst.md", digest },
      modelPolicy: { policyId: "worker", version: "1", digest },
      toolsets: [
        { ref: "run-artifacts@1", tools: ["read_artifact", "write_artifact"] },
      ],
      skills: [{ namespace: "skills", name: "research" }],
      sandboxProfile: "local-workdir@1",
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
    const path = new URL(route.request().url()).pathname;
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
    else if (path.endsWith("/instructions"))
      value = {
        template: { templateId: agent.ref.name, version: "1", digest },
        instructions: { ...agent.body.instructions, text: prompt },
      };
    else if (path.includes("/versions/")) value = agent;
    else if (path.endsWith("/agent-templates"))
      value = { items: [agent], page: { hasMore: false } };
    await route.fulfill({
      json: value,
      headers: { "X-Contractor-API-Version": "contractor.public.v1" },
    });
  });
  await page.goto("/workflows?keep=yes#section");
  await expect(page).toHaveURL(`${origin}/catalog/workflows?keep=yes#section`);
  await page.getByRole("link", { name: "Agents", exact: true }).click();
  await page
    .getByRole("link", { name: /evidence-analyst.*View prompt/ })
    .click();
  await expect(
    page.getByRole("heading", { name: "Evidence analyst", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Source", exact: true }).click();
  await expect(page.locator(".catalog-prompt-source")).toHaveText(prompt);
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
    page.getByRole("heading", { name: "Evidence analyst", exact: true }),
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
