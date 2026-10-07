import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "yaml";
import { expect, test } from "@playwright/test";
import packageMetadata from "../package.json" with { type: "json" };
import { presetFixture } from "../src/test/audit-presets-fixture";

const configRoot = resolve(process.cwd(), "../configs");
for (const width of [1440, 390, 320]) {
  test(`studio imports, edits and exports authored YAML at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    const origin = new URL(String(testInfo.project.use.baseURL)).origin;
    const errors: string[] = [],
      mutations: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.route("**/runtime-config.json", (route) =>
      route.fulfill({
        json: {
          uiVersion: packageMetadata.version,
          supportedApiVersions: ["contractor.public.v1"],
          apiBaseUrl: origin,
        },
      }),
    );
    await page.route(`${origin}/v1/**`, async (route) => {
      if (route.request().method() !== "GET")
        mutations.push(route.request().method());
      await route.fulfill({
        headers: { "x-contractor-api-version": "contractor.public.v1" },
        json:
          new URL(route.request().url()).pathname === "/v1/auth/session"
            ? {
                principal: {
                  userId: "owner",
                  username: "owner",
                  capabilities: ["user"],
                },
                csrfToken: "a".repeat(43),
                idleExpiresAt: "2099-01-01T00:00:00Z",
                absoluteExpiresAt: "2099-01-02T00:00:00Z",
              }
            : { items: [], page: { hasMore: false } },
      });
    });
    await page.goto("/catalog/studio");
    await expect(
      page.getByRole("heading", { name: "Node Studio", exact: true }),
    ).toBeVisible();
    const original = readFileSync(
      resolve(configRoot, "workflows/openapi_from_workspace_v7_memory.yaml"),
      "utf8",
    );
    await page
      .getByRole("button", { name: "Import YAML", exact: true })
      .click();
    await page.getByLabel("YAML file").setInputFiles({
      name: "workflow.yaml",
      mimeType: "application/yaml",
      buffer: Buffer.from(`# Keep this bundle comment\n${original}`),
    });
    await page.getByRole("button", { name: "Load draft", exact: true }).click();
    await expect(
      page.getByText("openapi-from-workspace@7", { exact: true }),
    ).toBeVisible();
    await page
      .locator(".studio-node-select")
      .filter({ hasText: "dependency_discovery" })
      .click();
    const planner = page.getByLabel("Planner", { exact: true });
    await planner.fill("router@1");
    await planner.press("Tab");
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    await expect(planner).toHaveValue("passthrough@1");
    await page.getByRole("button", { name: "Redo", exact: true }).click();
    await expect(planner).toHaveValue("router@1");
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(
      page.getByText("No errors in local checks.", { exact: true }),
    ).toBeVisible();
    const downloadPromise = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export YAML", exact: true })
      .click();
    const download = await downloadPromise;
    const exported = readFileSync((await download.path())!, "utf8");
    expect(exported).toContain("# Keep this bundle comment");
    const authored = parse(original),
      result = parse(exported);
    authored.spec.stages.dependency_discovery.planner = "router@1";
    expect(result).toEqual(authored);
    // Apply syntax-error feedback without losing pending text, then revert.
    await page.getByRole("button", { name: "YAML", exact: true }).click();
    await page.getByLabel("Authored YAML").fill("kind: Workflow\nspec: [");
    await page.getByRole("button", { name: "Apply YAML", exact: true }).click();
    await expect(page.getByLabel("Authored YAML")).toHaveValue(
      "kind: Workflow\nspec: [",
    );
    await page
      .getByRole("button", { name: "Revert YAML edits", exact: true })
      .click();
    // Both other document kinds use the same lossless import/export path.
    for (const file of [
      "audit-profiles/openapi_operation_trace.yaml",
      "agent-templates/workspace_openapi_builder_v3_memory.yaml",
    ]) {
      const source = readFileSync(resolve(configRoot, file), "utf8");
      await page
        .getByRole("button", { name: "Import YAML", exact: true })
        .click();
      await page.getByLabel("Paste YAML").fill(source);
      await page
        .getByRole("button", { name: "Load draft", exact: true })
        .click();
      await page.getByRole("button", { name: "Validate", exact: true }).click();
      await expect(
        page.getByText("No errors in local checks.", { exact: true }),
      ).toBeVisible();
      const promise = page.waitForEvent("download");
      await page
        .getByRole("button", { name: "Export YAML", exact: true })
        .click();
      expect(readFileSync((await (await promise).path())!, "utf8")).toBe(
        source,
      );
    }
    for (const theme of ["light", "dark", "black"]) {
      await page.evaluate((theme) => {
        document.documentElement.dataset.theme = theme;
      }, theme);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await page.screenshot({
        path: testInfo.outputPath(`studio-${theme}-${width}.png`),
      });
    }
    expect(
      await page.evaluate(() =>
        Object.keys(localStorage).filter(
          (key) => key !== "contractor.ui.theme",
        ),
      ),
    ).toEqual([]);
    expect(await page.evaluate(() => sessionStorage.length)).toBe(0);
    expect(mutations).toEqual([]);
    expect(errors).toEqual([]);
  });
}

test("studio live fences revisions, preserves paged items on errors and keeps its design draft", async ({
  page,
}, testInfo) => {
  const origin = new URL(String(testInfo.project.use.baseURL)).origin;
  const date = "2026-10-07T10:00:00Z",
    revision = 7;
  const audit = {
    auditId: "audit_studio",
    projectId: "project_studio",
    profile: { ...presetFixture.ref },
    inputs: {},
    scope: {},
    runtimeLabels: [],
    state: "completed",
    phase: "rounds",
    currentRoundId: `round_${"a".repeat(64)}`,
    revision,
    eventSequence: revision,
    dispatchState: "closed",
    holdState: "released",
    limits: {
      maxRounds: 1,
      batchSize: 1,
      maxItemsPerRound: 64,
      maxItemsTotal: 64,
      maxSubmittedRuns: 64,
      maxItemRunAttempts: 2,
      maxEvidenceBytes: 1048576,
    },
    reservedRunCount: 0,
    submittedRunCount: 0,
    outstandingRunCount: 0,
    retainedEvidenceBytes: 0,
    createdAt: date,
    updatedAt: date,
  };
  const items = Array.from({ length: 51 }, (_, i) => ({
    itemId: `item_${i}`,
    roundId: `round_${"a".repeat(64)}`,
    itemKey: `Operation ${i + 1}`,
    state: "settled",
    finalDisposition: i === 0 ? "execution-failed" : "accepted-result",
  }));
  let failItems = true,
    wrongRevision = false;
  const mutations: string[] = [],
    errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: packageMetadata.version,
        supportedApiVersions: ["contractor.public.v1"],
        apiBaseUrl: origin,
      },
    }),
  );
  await page.route(`${origin}/v1/**`, async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() !== "GET")
      mutations.push(route.request().method());
    let json: unknown = {
      items: [],
      page: { hasMore: false },
      total: 0,
      auditRevision: revision,
      asOf: date,
    };
    let etag = `"${revision}"`;
    if (url.pathname === "/v1/auth/session")
      json = {
        principal: {
          userId: "owner",
          username: "owner",
          capabilities: ["user"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      };
    else if (url.pathname === "/v1/audits")
      json = { items: [audit], page: { hasMore: false } };
    else if (url.pathname === "/v1/audits/audit_studio") json = audit;
    else if (url.pathname.endsWith("/workspace"))
      json = {
        auditId: audit.auditId,
        auditRevision: wrongRevision ? revision + 1 : revision,
        asOf: date,
        roundId: `round_${"a".repeat(64)}`,
        executionState: "completed",
        outstandingRuns: 0,
        totalChecks: 51,
        completedChecks: 51,
        issues: 0,
        gaps: 1,
        unchecked: 0,
        findings: 0,
        unreviewedFindings: 0,
        pendingReviews: 0,
      };
    else if (url.pathname.startsWith("/v1/audit-profiles/")) {
      json = presetFixture;
      etag = `"${presetFixture.ref.digest}"`;
    } else if (url.pathname.endsWith("/events"))
      json = {
        items: [
          {
            auditId: audit.auditId,
            sequence: 1,
            kind: "audit.created",
            entityId: audit.auditId,
            summary: {},
            createdAt: date,
          },
        ],
        total: 1,
        throughSequence: 1,
        page: { hasMore: false },
      };
    else if (url.pathname.endsWith("/items")) {
      if (url.searchParams.has("cursor") && failItems) {
        await route.fulfill({
          status: 500,
          json: {
            code: "internal_error",
            message: "Unavailable",
            requestId: "studio-request",
          },
          headers: { "x-contractor-api-version": "contractor.public.v1" },
        });
        return;
      }
      json = {
        items: url.searchParams.has("cursor")
          ? items.slice(50)
          : items.slice(0, 50),
        page: url.searchParams.has("cursor")
          ? { hasMore: false }
          : { hasMore: true, nextCursor: "older" },
      };
    }
    await route.fulfill({
      json,
      headers: {
        "x-contractor-api-version": "contractor.public.v1",
        ETag: etag,
      },
    });
  });
  await page.goto("/catalog/studio?audit=audit_studio");
  await expect(page.locator(".studio-matrix a")).toHaveCount(50);
  await expect(
    page.locator('.studio-matrix a[data-attention="true"]'),
  ).toHaveCount(1);
  await page.getByRole("button", { name: "Load more items" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "More items could not be loaded",
  );
  await expect(page.locator(".studio-matrix a")).toHaveCount(50);
  failItems = false;
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.locator(".studio-matrix a")).toHaveCount(51);
  wrongRevision = true;
  await page.getByRole("button", { name: "Refresh live view" }).click();
  await expect(page.getByRole("alert")).toContainText("check changed");
  await expect(page.locator(".studio-matrix a")).toHaveCount(51);
  wrongRevision = false;
  await page.getByRole("button", { name: "Refresh live view" }).click();
  await expect(page.getByRole("alert")).toHaveCount(0);
  audit.profile.digest = `sha256:${"f".repeat(64)}`;
  await page.getByRole("button", { name: "Refresh live view" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "catalog profile has a different digest",
  );
  await expect(page.locator(".studio-matrix a")).toHaveCount(51);
  await expect(page.getByText("source-check@3", { exact: false })).toHaveCount(
    0,
  );
  audit.profile.digest = presetFixture.ref.digest;
  await page.getByRole("button", { name: "Refresh live view" }).click();
  await expect(page.getByRole("alert")).toHaveCount(0);
  await page.getByRole("button", { name: "Design", exact: true }).click();
  await page.getByLabel("Objective").fill("Keep this local objective");
  await page.getByLabel("Objective").press("Tab");
  await page.getByRole("button", { name: "Live", exact: true }).click();
  await expect(page.locator(".studio-matrix a")).toHaveCount(51);
  for (const width of [1440, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
  await page.getByRole("button", { name: "Design", exact: true }).click();
  if (page.viewportSize()!.width <= 800)
    await page.getByRole("button", { name: "Properties", exact: true }).click();
  await expect(page.getByLabel("Objective")).toHaveValue(
    "Keep this local objective",
  );
  expect(mutations).toEqual([]);
  expect(errors).toEqual([]);
});
