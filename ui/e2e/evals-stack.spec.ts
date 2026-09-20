import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import type { EvalExperiment } from "../src/api/evals";

const enabled = process.env.CONTRACTOR_UI_EVAL_STACK === "1";
test.skip(!enabled, "The disposable managed Evals process stack is required");
const directory = process.env.CONTRACTOR_EVAL_FIXTURE_DIR ?? "";
const apiURL = process.env.CONTRACTOR_UI_E2E_API_URL ?? "";

test.afterEach(async ({ page }, info) => {
  if (info.status !== info.expectedStatus) {
    console.log(
      "Managed Evals failure page:",
      await page.locator("body").innerText(),
    );
  }
});

async function get<T>(page: Page, resource: string): Promise<T> {
  return page.evaluate(
    async ({ origin, resource }) => {
      const response = await fetch(origin + resource, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(`GET ${resource}: ${response.status}`);
      return response.json();
    },
    { origin: apiURL, resource },
  );
}

async function signIn(page: Page) {
  await page.goto("/evals");
  await authenticate(page);
}

async function authenticate(page: Page) {
  await page
    .getByLabel("Username")
    .fill(process.env.CONTRACTOR_UI_E2E_USERNAME!);
  await page
    .getByLabel("Password")
    .fill(process.env.CONTRACTOR_UI_E2E_PASSWORD!);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByLabel("Username", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("main")).toBeVisible();
}

async function authorWorkflowCases(page: Page) {
  await page.getByLabel("Dataset ID", { exact: true }).fill("release-workflow");
  await page
    .getByLabel("Dataset name", { exact: true })
    .fill("Release workflow");
  for (const [index, identifier] of ["case-one", "case-two"].entries()) {
    if (index) {
      await page.getByRole("button", { name: "Add case", exact: true }).click();
      await page.getByText("Case 2: New case", { exact: true }).click();
    }
    const editor = page.locator(".eval-case").nth(index);
    await editor.getByLabel("Case ID", { exact: true }).fill(identifier);
    await editor
      .getByLabel("Visible task objective", { exact: true })
      .fill("Inspect retained fixture");
    await editor.getByLabel("Input role", { exact: true }).fill("source");
    await editor
      .getByRole("button", { name: "Choose exact input", exact: true })
      .click();
    await editor
      .getByRole("listitem")
      .filter({ hasText: "eval-fixtures/plain-source" })
      .getByRole("button", { name: "Use input", exact: true })
      .click();
    await editor.getByLabel("Output role", { exact: true }).fill("report");
    await editor
      .getByLabel("Output media types", { exact: true })
      .fill("text/plain");
    await editor
      .getByRole("button", { name: "Add output role", exact: true })
      .click();
  }
  await page.getByText("Private human review rubrics", { exact: true }).click();
  await page
    .getByRole("button", { name: "Add human rubric", exact: true })
    .click();
  await page.getByLabel("Review check ID", { exact: true }).fill("review");
  await page.getByLabel("Rubric revision", { exact: true }).fill("r1");
  await page
    .getByLabel("Private rubric", { exact: true })
    .fill("PRIVATE_MANAGED_EVAL_RELEASE_TRUTH");
}

test("real native setup, restart, exact evidence and review; independent external inspection", async ({
  page,
  request,
}, info) => {
  test.setTimeout(720_000);
  await signIn(page);
  if (process.env.CONTRACTOR_EVAL_EXTERNAL_ONLY === "1") {
    const evidence = JSON.parse(
      await readFile(path.join(directory, "external-evidence.json"), "utf8"),
    ) as {
      experiments: Record<string, { experimentId: string }>;
    };
    for (const [kind, experiment] of Object.entries(evidence.experiments)) {
      for (const width of [390, 1280]) {
        await page.setViewportSize({ width, height: 900 });
        await page.goto(
          `/evals/experiments/${experiment.experimentId}/comparison?filter=all`,
        );
        await expect(page.getByText(/Externally controlled/)).toBeVisible();
        await expect(
          page.getByRole("heading", {
            name: "Token distribution",
            exact: true,
          }),
        ).toBeVisible();
        await expect(
          page.getByRole("button", { name: "Start", exact: true }),
        ).toHaveCount(0);
        await expect(
          page.getByRole("link", { name: "case-one / sample 1", exact: true }),
        ).toBeVisible();
        await page.screenshot({
          path: info.outputPath(`external-${kind}-${width}.png`),
          fullPage: true,
        });
      }
    }
    return;
  }
  const fixture = JSON.parse(
    await readFile(path.join(directory, "fixture.json"), "utf8"),
  ) as {
    projectId: string;
    datasets: Record<string, { file: string }>;
  };
  const evidence: Record<string, unknown> = {};
  for (const kind of ["workflow", "audit"] as const) {
    await page.setViewportSize({
      width: kind === "workflow" ? 390 : 1280,
      height: 900,
    });
    await page.goto(`/evals/datasets?project=${fixture.projectId}`);
    await page
      .getByRole("button", { name: "Create or import dataset", exact: true })
      .click();
    if (kind === "workflow") await authorWorkflowCases(page);
    else {
      await page
        .getByLabel("Import dataset JSON", { exact: true })
        .setInputFiles(fixture.datasets[kind]!.file);
      await page
        .getByRole("checkbox", { name: /Include private assessment rubrics/ })
        .check();
    }
    await page
      .getByRole("button", { name: "Save dataset revision", exact: true })
      .click();
    await expect(
      page.getByText(`Release ${kind}`, { exact: true }),
    ).toBeVisible();
    const datasets = await get<{
      items: Array<{ datasetId: string; revision: string }>;
    }>(page, `/v1/projects/${fixture.projectId}/eval-datasets`);
    const dataset = datasets.items.find(
      (row) => row.datasetId === `release-${kind}`,
    )!;
    await page.goto(`/evals/new?project=${fixture.projectId}`);
    await page
      .getByLabel("Experiment name", { exact: true })
      .fill(`Native release ${kind}`);
    await page.getByLabel("Execution kind", { exact: true }).selectOption(kind);
    for (const arm of ["A", "B"]) {
      await page
        .getByLabel(
          `${arm} ${kind === "workflow" ? "Workflow" : "AuditProfile"} family`,
          { exact: true },
        )
        .selectOption(
          `eval-${kind === "workflow" ? "copy" : "audit"}-${arm.toLowerCase()}`,
        );
      if (kind === "workflow") {
        const variant = page.locator(`.eval-arm-${arm.toLowerCase()}`);
        await variant
          .getByText("Input/output mapping and parameters", { exact: true })
          .click();
        await variant
          .getByRole("button", {
            name: `Add ${arm.toLowerCase()} output mapping`,
            exact: true,
          })
          .click();
        await variant
          .getByLabel(`${arm} output mapping name 1`, { exact: true })
          .fill("report");
        await variant
          .getByLabel(`${arm} output mapping value 1`, { exact: true })
          .fill("result");
      }
    }
    await page.getByRole("button", { name: "Next step", exact: true }).click();
    await page
      .getByLabel("Dataset revision", { exact: true })
      .selectOption(dataset.revision);
    await page
      .getByRole("button", { name: "Select all 2 cases", exact: true })
      .click();
    await page.getByRole("button", { name: "Next step", exact: true }).click();
    await page
      .getByRole("button", { name: "Add assessment check", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Add assessment check", exact: true })
      .click();
    const human = page.locator(".eval-check").nth(1);
    await human.getByLabel("Check ID", { exact: true }).fill("review");
    await human
      .getByLabel("Evaluator", { exact: true })
      .selectOption("human-review@1");
    await human
      .getByLabel("Pinned rubric revision", { exact: true })
      .fill("r1");
    await page.getByLabel("Repetitions", { exact: true }).fill("2");
    await page.getByRole("button", { name: "Next step", exact: true }).click();
    await page.getByRole("button", { name: "Save draft", exact: true }).click();
    await expect(page).toHaveURL(/\/experiments\/[^/]+\/setup/);
    const experimentId = new URL(page.url()).pathname.split("/")[3]!;
    const experimentPath = `/v1/eval-experiments/${experimentId}`;
    await page.reload();
    await page.getByRole("button", { name: "Prepare", exact: true }).click();
    await expect(
      page.getByText("Verified preparation", { exact: true }),
    ).toBeVisible({ timeout: 60_000 });
    const prepared = await get<EvalExperiment>(page, experimentPath);
    expect(prepared.expectedMembers).toBe(8);
    expect(prepared.state).toBe("ready");
    let dropped = false;
    await page.route(`**${experimentPath}/commands`, async (route) => {
      if (!dropped && route.request().postDataJSON()?.kind === "start") {
        const response = await route.fetch({
          url: route
            .request()
            .url()
            .replace(apiURL, process.env.CONTRACTOR_UI_E2E_API_DIRECT_URL!),
        });
        expect(response.status()).toBe(202);
        dropped = true;
        await route.abort("connectionclosed");
      } else await route.continue();
    });
    await page.getByRole("button", { name: "Start", exact: true }).click();
    await page
      .getByRole("button", { name: "Confirm start", exact: true })
      .click();
    await expect(
      page.getByText("Public API is unavailable", { exact: true }),
    ).toBeVisible();
    await expect
      .poll(
        async () => (await get<EvalExperiment>(page, experimentPath)).startedAt,
        { timeout: 30_000 },
      )
      .not.toBeNull();
    const beforeRestart = await get<EvalExperiment>(page, experimentPath);
    const restarted = await request.post(
      `${process.env.CONTRACTOR_UI_E2E_CONTROL_URL}/restart-server`,
      {
        headers: {
          Authorization: `Bearer ${process.env.CONTRACTOR_UI_E2E_CONTROL_TOKEN}`,
        },
      },
    );
    expect(restarted.status(), await restarted.text()).toBe(204);
    await page.reload();
    await authenticate(page);
    await expect
      .poll(
        async () => (await get<EvalExperiment>(page, experimentPath)).state,
        { timeout: 420_000, intervals: [500, 1000, 2000] },
      )
      .toBe("finished");
    const finished = await get<EvalExperiment>(page, experimentPath);
    expect(finished.planSha256).toBe(prepared.planSha256);
    expect(finished.startedAt).toBe(beforeRestart.startedAt);
    expect(finished.deadlineAt).toBe(beforeRestart.deadlineAt);
    await page.goto(`/evals/experiments/${experimentId}/comparison?filter=all`);
    await page
      .getByRole("link", { name: "case-one / sample 1", exact: true })
      .click();
    await page.getByRole("button", { name: "Review A", exact: true }).click();
    await page
      .getByLabel("Decision for review", { exact: true })
      .selectOption("pass");
    await page
      .getByLabel("Reason for review", { exact: true })
      .fill("Verified deterministic process evidence");
    await page
      .getByRole("button", { name: "Save and select assessment", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Review exact result", exact: true }),
    ).toHaveCount(0);
    await page.getByRole("link", { name: "← Comparison", exact: true }).click();
    await expect(page).toHaveURL(/filter=all/);
    for (const width of [390, 1280]) {
      await page.setViewportSize({ width, height: 900 });
      await expect
        .poll(() =>
          page.evaluate(
            () => document.documentElement.scrollWidth <= window.innerWidth,
          ),
        )
        .toBe(true);
      await page.screenshot({
        path: info.outputPath(`native-${kind}-${width}.png`),
        fullPage: true,
      });
    }
    const report = await get(page, experimentPath + "/report");
    expect(JSON.stringify(report)).not.toContain(
      "PRIVATE_MANAGED_EVAL_RELEASE_TRUTH",
    );
    evidence[kind] = {
      experimentId,
      expectedMembers: 8,
      planSha256: finished.planSha256,
      report,
    };
    await page.unroute(`**${experimentPath}/commands`);
  }
  await writeFile(
    path.join(directory, "native-evidence.json"),
    JSON.stringify(evidence),
  );
});
