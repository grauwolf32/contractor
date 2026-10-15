import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parse } from "yaml";
import { expect, test } from "@playwright/test";
import { presetFixture } from "../src/test/audit-presets-fixture";
import { audit as auditBuilder, auditItem } from "../src/test/builders";
import {
  authSession,
  fulfillEmptyPageOrNotFound,
  fulfillError,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

// Fixture copies of authored definitions (tests never read configs/).
const fixtureRoot = resolve(process.cwd(), "src/routes/studio/testdata");
for (const width of [1440, 390, 320]) {
  test(`studio advanced forms preserve authored contracts at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    const errors: string[] = [],
      mutations: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await mockAPI(
      page,
      async (route, request, url) => {
        if (request.method() !== "GET") mutations.push(request.method());
        const kind = url.pathname.split("/").at(-1)!;
        if (kind !== "llm-gateways") return fulfillEmptyPageOrNotFound(route);
        return fulfillJSON(route, {
          items: [
            {
              ref: {
                kind,
                name: "local-litellm",
                version: "1",
                digest: `sha256:${"a".repeat(64)}`,
              },
              source: "operator",
              body: {
                protocol: "openai-compatible@1",
                url: "http://127.0.0.1:4000",
              },
            },
          ],
          page: { hasMore: false },
        });
      },
      { session: authSession("owner", ["user"]) },
    );
    await page.goto("/catalog/studio");
    const load = async (source: string) => {
      await page
        .getByRole("button", { name: "Import YAML", exact: true })
        .click();
      await page.getByLabel("Paste YAML").fill(source);
      await page
        .getByRole("button", { name: "Load draft", exact: true })
        .click();
    };
    const select = async (title: string) => {
      if (width <= 800)
        await page.getByRole("button", { name: "Graph", exact: true }).click();
      await page
        .locator(".studio-node-select")
        .filter({ hasText: title })
        .click();
    };
    const exported = async () => {
      const promise = page.waitForEvent("download");
      await page
        .getByRole("button", { name: "Export YAML", exact: true })
        .click();
      return readFileSync((await (await promise).path())!, "utf8");
    };
    const original = readFileSync(
      resolve(fixtureRoot, "openapi_from_workspace_v7_memory.yaml"),
      "utf8",
    );
    const expected = parse(original),
      stage = expected.spec.stages.dependency_discovery;
    await load(`# Advanced form note\n${original}`);
    await select("dependency_discovery");
    await page.getByLabel("Source 1 target").fill("project");
    await page.getByLabel("Source 1 target").press("Tab");
    stage.context.workspace.sources[0].target = "project";
    await page
      .getByRole("button", { name: "Remove workspace export", exact: true })
      .click();
    await page
      .getByRole("alertdialog")
      .getByRole("button", { name: "Remove block", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Add workspace export", exact: true })
      .click();
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "Add workspace export", exact: true }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Redo", exact: true }).click();
    await expect(
      page.getByRole("combobox", { name: "State output", exact: true }),
    ).toHaveValue("workspace_state");
    await page.getByText("Stage execution overrides", { exact: true }).click();
    await page
      .getByRole("button", {
        name: "Choose stage agent analyst gateway from catalog",
        exact: true,
      })
      .click();
    await page
      .getByRole("button", { name: "Use local-litellm@1", exact: true })
      .click();
    expected.spec.executionConfig.stages = {
      dependency_discovery: {
        agents: { analyst: { llmGateway: "local-litellm@1" } },
      },
    };
    await select("Execution defaults");
    await page
      .getByLabel("Default workers model policy", { exact: true })
      .fill("worker@3");
    await page
      .getByLabel("Default workers model policy", { exact: true })
      .press("Tab");
    expected.spec.executionConfig.workers.modelPolicy = "worker@3";
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
        animations: "disabled",
        path: testInfo.outputPath(`studio-routing-${theme}-${width}.png`),
      });
    }
    const source = await exported();
    expect(source).toContain("# Advanced form note");
    expect(parse(source)).toEqual(expected);
    const toolSource = readFileSync(
        resolve(fixtureRoot, "audit_sqlmap_scan.yaml"),
        "utf8",
      ),
      toolExpected = parse(toolSource);
    await load(toolSource);
    await select("Tool execution");
    await page.getByLabel("level literal type").selectOption("boolean");
    toolExpected.spec.execution.arguments.level.value = false;
    await page.getByLabel("risk source").selectOption("parameter");
    await page.getByLabel("risk binding name").fill("risk_level");
    await page.getByLabel("risk binding name").press("Tab");
    toolExpected.spec.execution.arguments.risk = {
      source: "parameter",
      name: "risk_level",
    };
    await page.getByRole("button", { name: "Validate", exact: true }).click();
    await expect(
      page.getByText("No errors in local checks.", { exact: true }),
    ).toBeVisible();
    const result = await exported();
    expect(result).toContain("# SQLMap's lowest supported test level and risk");
    expect(parse(result)).toEqual(toolExpected);
    expect(
      await page.evaluate(() =>
        Object.keys(localStorage).filter((key) => key !== "contractor.theme"),
      ),
    ).toEqual([]);
    expect(await page.evaluate(() => sessionStorage.length)).toBe(0);
    expect(errors).toEqual([]);
    expect(mutations).toEqual([]);
  });
}
for (const width of [1440, 390, 320]) {
  test(`studio imports, edits and exports authored YAML at ${width}px`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    const errors: string[] = [],
      mutations: string[] = [];
    let failCatalogPage = true;
    page.on("pageerror", (error) => errors.push(error.message));
    await mockAPI(
      page,
      async (route, request, url) => {
        if (request.method() !== "GET") mutations.push(request.method());
        if (url.pathname === "/v1/configurations/agent-templates") {
          expect(url.searchParams.get("limit")).toBe("50");
          const template = (version: string) => ({
            ref: {
              kind: "agent-templates",
              name: "workspace_source_graph_analyst",
              version,
              digest: `sha256:${"a".repeat(64)}`,
            },
            source: "operator",
            body: {
              description: "Source graph analyst",
              runtime: "adk@1",
              instructions: {
                ref: "instructions/source-graph-analyst.md",
                digest: `sha256:${"a".repeat(64)}`,
              },
              modelPolicy: {
                policyId: "worker",
                version: "3",
                digest: `sha256:${"a".repeat(64)}`,
              },
              sandboxProfile: "local-workdir@1",
              toolsets: [],
            },
          });
          if (url.searchParams.has("cursor")) {
            expect(url.searchParams.get("q")).toBe("source graph");
            if (failCatalogPage) {
              failCatalogPage = false;
              await fulfillError(
                route,
                500,
                "internal_error",
                "Retry this catalog page",
              );
            } else
              await fulfillJSON(route, {
                items: [template("3"), template("4")],
                page: { hasMore: false },
              });
          } else
            await fulfillJSON(route, {
              items: [template("3")],
              page: { hasMore: true, nextCursor: "catalog-page-two" },
            });
          return;
        }
        if (url.pathname === "/v1/workflows") {
          await fulfillJSON(route, {
            items: [
              {
                ref: { name: "audit-openapi-operation-trace", version: "2" },
                entryStage: "check",
                inputs: {},
                outputs: {},
                parameters: {},
              },
            ],
            page: { hasMore: false },
          });
          return;
        }
        if (url.pathname === "/v1/configurations/model-policies") {
          await fulfillJSON(route, {
            items: [
              {
                ref: {
                  kind: "model-policies",
                  name: "worker",
                  version: "3",
                  digest: `sha256:${"a".repeat(64)}`,
                },
                source: "operator",
                body: { model: "test-model", contextWindowTokens: 10000 },
              },
            ],
            page: { hasMore: false },
          });
          return;
        }
        await fulfillEmptyPageOrNotFound(route);
      },
      { session: authSession("owner", ["user"]) },
    );
    await page.goto("/catalog/studio");
    await expect(
      page.getByRole("heading", { name: "Node Studio", exact: true }),
    ).toBeVisible();
    const original = readFileSync(
      resolve(fixtureRoot, "openapi_from_workspace_v7_memory.yaml"),
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
    const move = page.getByRole("button", {
      name: "Move dependency_discovery",
      exact: true,
    });
    const node = move.locator("..");
    const left = await node.evaluate(
      (node) => (node as HTMLElement).style.left,
    );
    await move.press("ArrowRight");
    await expect(node).not.toHaveCSS("left", left);
    await page
      .getByRole("button", { name: "Auto arrange", exact: true })
      .click();
    await expect(node).toHaveCSS("left", left);
    expect(
      await page.locator(".studio-wires path").evaluateAll((paths) =>
        paths.every((path) => {
          const bounds = (path as SVGGraphicsElement).getBBox();
          const svg = (path as SVGElement).ownerSVGElement!;
          return (
            bounds.x >= 0 &&
            bounds.y >= 0 &&
            bounds.x + bounds.width <= svg.width.baseVal.value &&
            bounds.y + bounds.height <= svg.height.baseVal.value
          );
        }),
      ),
    ).toBe(true);
    await expect(
      page.getByRole("button", { name: "Undo", exact: true }),
    ).toBeDisabled();
    await page.getByRole("button", { name: "Fit graph", exact: true }).click();
    expect(
      await page.locator(".studio-world").evaluate((world) => {
        const bounds = world.getBoundingClientRect();
        const viewport = document.querySelector(".studio-viewport")!;
        return (
          bounds.width <= viewport.clientWidth + 1 &&
          bounds.height <= viewport.clientHeight + 1
        );
      }),
    ).toBe(true);
    await page
      .getByRole("button", { name: "Auto arrange", exact: true })
      .click();
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
    await page
      .getByRole("button", {
        name: "Choose template from catalog",
        exact: true,
      })
      .click();
    await expect(
      page.getByLabel("Search catalog", { exact: true }),
    ).toBeFocused();
    await page
      .getByLabel("Search catalog", { exact: true })
      .fill("source graph");
    await page.getByRole("button", { name: "Search", exact: true }).click();
    await page
      .getByRole("button", { name: "Load more versions", exact: true })
      .click();
    await expect(page.getByRole("alert")).toContainText(
      "More versions could not be loaded",
    );
    await expect(
      page.getByRole("button", {
        name: "Use workspace_source_graph_analyst@3",
        exact: true,
      }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Retry catalog", exact: true })
      .click();
    await expect(
      page.getByRole("button", {
        name: "Use workspace_source_graph_analyst@4",
        exact: true,
      }),
    ).toBeVisible();
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
        path: testInfo.outputPath(`studio-catalog-${theme}-${width}.png`),
      });
    }
    await page
      .getByRole("button", {
        name: "Use workspace_source_graph_analyst@4",
        exact: true,
      })
      .click();
    await expect(page.getByLabel("Template", { exact: true })).toHaveValue(
      "workspace_source_graph_analyst@4",
    );
    await page.getByRole("button", { name: "Undo", exact: true }).click();
    await expect(page.getByLabel("Template", { exact: true })).toHaveValue(
      "workspace_source_graph_analyst@3",
    );
    await page.getByRole("button", { name: "Redo", exact: true }).click();
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
    authored.spec.stages.dependency_discovery.agents.analyst.template =
      "workspace_source_graph_analyst@4";
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
      "openapi_operation_trace.yaml",
      "workspace_openapi_builder_v3_memory.yaml",
    ]) {
      const source = readFileSync(resolve(fixtureRoot, file), "utf8");
      const expected = parse(source);
      await page
        .getByRole("button", { name: "Import YAML", exact: true })
        .click();
      await page.getByLabel("Paste YAML").fill(source);
      await page
        .getByRole("button", { name: "Load draft", exact: true })
        .click();
      if (expected.kind === "AuditProfile") {
        await page
          .locator(".studio-node-select")
          .filter({ hasText: "trace" })
          .click();
        await page
          .getByRole("button", {
            name: "Choose workflow from catalog",
            exact: true,
          })
          .click();
        await page.getByRole("button", { name: "Cancel", exact: true }).click();
        await expect(page.getByLabel("Workflow", { exact: true })).toHaveValue(
          "audit-openapi-operation-trace@1",
        );
        await page
          .getByRole("button", {
            name: "Choose workflow from catalog",
            exact: true,
          })
          .click();
        await page
          .getByRole("button", {
            name: "Use audit-openapi-operation-trace@2",
            exact: true,
          })
          .click();
        expected.spec.workflows.trace.ref = "audit-openapi-operation-trace@2";
      } else {
        await page
          .locator(".studio-node-select")
          .filter({ hasText: "Model policy" })
          .click();
        await page
          .getByRole("button", {
            name: "Choose model policy from catalog",
            exact: true,
          })
          .click();
        await page
          .getByRole("button", { name: "Use worker@3", exact: true })
          .click();
        expected.spec.modelPolicy = "worker@3";
      }
      await page.getByRole("button", { name: "Validate", exact: true }).click();
      await expect(
        page.getByText("No errors in local checks.", { exact: true }),
      ).toBeVisible();
      const promise = page.waitForEvent("download");
      await page
        .getByRole("button", { name: "Export YAML", exact: true })
        .click();
      expect(
        parse(readFileSync((await (await promise).path())!, "utf8")),
      ).toEqual(expected);
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
        Object.keys(localStorage).filter((key) => key !== "contractor.theme"),
      ),
    ).toEqual([]);
    expect(await page.evaluate(() => sessionStorage.length)).toBe(0);
    expect(mutations).toEqual([]);
    expect(errors).toEqual([]);
  });
}

test("studio live fences revisions, preserves paged items on errors and keeps its design draft", async ({
  page,
}) => {
  const date = "2026-10-07T10:00:00Z",
    revision = 7;
  const audit = auditBuilder("project_studio", "audit_studio", "completed", {
    profile: { ...presetFixture.ref },
    currentRoundId: `round_${"a".repeat(64)}`,
    revision,
    eventSequence: revision,
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
    createdAt: date,
    updatedAt: date,
  });
  const items = Array.from({ length: 51 }, (_, i) =>
    auditItem(`item_${i}`, i + 1, "operation", {
      roundId: `round_${"a".repeat(64)}`,
      itemKey: `Operation ${i + 1}`,
      state: "settled",
      finalDisposition: i === 0 ? "execution-failed" : "accepted-result",
    }),
  );
  let failItems = true,
    wrongRevision = false;
  const mutations: string[] = [],
    errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await mockAPI(
    page,
    async (route, request, url) => {
      if (request.method() !== "GET") mutations.push(request.method());
      // Findings and reviews pages carry the Audit revision they were read at.
      let json: unknown = /^\/v1\/audits\/[^/]+\/(findings|reviews)$/.test(
        url.pathname,
      )
        ? {
            items: [],
            page: { hasMore: false },
            total: 0,
            auditRevision: revision,
            asOf: date,
          }
        : { items: [], page: { hasMore: false } };
      let etag = `"${revision}"`;
      if (url.pathname === "/v1/audits")
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
          await fulfillError(route, 500, "internal_error", "Unavailable");
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
      await fulfillJSON(route, json, 200, { ETag: etag });
    },
    { session: authSession("owner", ["user"]) },
  );
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
  // A refresh that fails keeps the last loaded view under the shared
  // stale-data warning.
  const stale = page
    .getByRole("status")
    .filter({ hasText: "Could not refresh; showing the last loaded data." });
  wrongRevision = true;
  await page.getByRole("button", { name: "Refresh live view" }).click();
  await expect(stale).toContainText("check changed");
  await expect(page.locator(".studio-matrix a")).toHaveCount(51);
  wrongRevision = false;
  await page.getByRole("button", { name: "Refresh live view" }).click();
  await expect(stale).toHaveCount(0);
  audit.profile.digest = `sha256:${"f".repeat(64)}`;
  await page.getByRole("button", { name: "Refresh live view" }).click();
  // A pinned profile digest the catalog does not hold is an error, not
  // stale data.
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
