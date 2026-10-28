import { ARTIFACT_STORAGE_TIMESTAMPS } from "../src/test/builders";
import { expect, test, type Page } from "@playwright/test";

import { project } from "../src/test/builders";
import {
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./support/mock-api";

const EVAL_ID = "evaluation-browser";

async function installAPI(page: Page): Promise<void> {
  const evaluation = project(EVAL_ID, {
    kind: "evaluation",
    name: "Browser evaluation",
    description: "Evaluation route fixture",
    createdAt: "2026-09-05T08:00:00Z",
    updatedAt: "2026-09-05T08:01:00Z",
  });
  await mockAPI(page, async (route, _request, url) => {
    if (url.pathname === "/v1/projects") {
      await fulfillJSON(route, {
        items:
          url.searchParams.get("kind") === "evaluation" ? [evaluation] : [],
        page: { hasMore: false },
      });
      return;
    }
    if (url.pathname === `/v1/projects/${EVAL_ID}`) {
      await fulfillJSON(route, evaluation, 200, { etag: '"1"' });
      return;
    }
    if (url.pathname === `/v1/projects/${EVAL_ID}/artifacts`) {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.pathname === `/v1/projects/${EVAL_ID}/runs`) {
      await fulfillJSON(route, {
        items: [
          {
            runId: "run-browser-leg-a",
            projectId: EVAL_ID,
            workflow: "openapi-from-workspace@5",
            state: "succeeded",
            deletable: true,
            labels: {
              purpose: "eval",
              "eval.name": "browser-regression",
              "eval.id": "eval-browser-01",
              "eval.leg": "a",
              "eval.case": "routing",
              "eval.sample": "1",
            },
            createdAt: "2026-09-05T08:10:00Z",
            updatedAt: "2026-09-05T08:12:00Z",
            finishedAt: "2026-09-05T08:12:00Z",
          },
        ],
        page: { hasMore: false },
      });
      return;
    }
    if (url.pathname === "/v1/workflows") {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.pathname === "/v1/artifacts") {
      await fulfillJSON(route, {
        items:
          url.searchParams.get("namespace") === "skills"
            ? [
                {
                  storageTimestamps: ARTIFACT_STORAGE_TIMESTAMPS,
                  artifact: {
                    namespace: "skills",
                    name: "architecture-review",
                    revision: "skill-r1",
                  },
                  mediaType: "application/vnd.contractor.agent-skill+zip",
                  size: 128,
                  current: true,
                  frozen: false,
                  createdAt: "2026-09-05T08:00:00Z",
                },
              ]
            : [],
        page: { hasMore: false },
      });
      return;
    }
    await fulfillEmptyPageOrNotFound(route);
  });
}

test("Evals and Skills stay separate Project/UserScope UI projections", async ({
  page,
}) => {
  await installAPI(page);

  await page.goto("/evals/workspaces");
  await expect(
    page.getByRole("heading", { name: "Evaluation workspaces", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Browser evaluation" }).click();
  await expect(page.getByRole("heading", { name: "Eval Runs" })).toBeVisible();
  await expect(page.getByText("eval-browser-01")).toBeVisible();
  await expect(
    page.getByRole("link", { name: "run-browser-leg-a" }),
  ).toBeVisible();
  // Evaluation workspaces keep history and a read-only artifact list only.
  const sections = page.getByRole("navigation", { name: "Project sections" });
  await expect(sections.getByRole("link")).toHaveText([
    "Experiments",
    "Datasets",
    "Runs",
    "Artifacts",
    "Workspace settings",
  ]);
  await expect(page.getByRole("heading", { name: "Artifacts" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Add material" })).toHaveCount(
    0,
  );
  await expect(page.getByText("Format matches")).toHaveCount(0);

  await page.getByRole("link", { name: "Library", exact: true }).click();
  await page.getByRole("link", { name: "Skills", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Skills" })).toBeVisible();
  const skill = page.getByRole("link", { name: "architecture-review" });
  await expect(skill).toHaveAttribute(
    "href",
    "/artifacts/skills/architecture-review",
  );
});

for (const width of [390, 1280]) {
  for (const theme of ["light", "dark", "black"]) {
    test(`current evaluation workspace at ${width}px in ${theme}`, async ({
      page,
    }, info) => {
      await page.setViewportSize({ width, height: 900 });
      await page.addInitScript(
        (value) => localStorage.setItem("contractor.theme", value),
        theme,
      );
      await installAPI(page);
      await page.goto("/evals/workspaces");
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
      await expect(
        page.getByRole("heading", {
          name: "Evaluation workspaces",
          exact: true,
        }),
      ).toBeVisible();
      await page
        .getByRole("button", { name: "New workspace", exact: true })
        .click();
      const dialog = page.getByRole("dialog", {
        name: "New evaluation workspace",
      });
      await expect(dialog.getByLabel("Name", { exact: true })).toHaveAttribute(
        "maxlength",
        "160",
      );
      await expect(
        dialog.getByRole("button", { name: "Create workspace" }),
      ).toBeVisible();
      await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
      await expect(dialog).toHaveCount(0);
      await page
        .getByRole("link", { name: "Browser evaluation", exact: true })
        .click();
      await expect(page).toHaveURL(`/evals/workspaces/${EVAL_ID}`);
      await page.reload();
      const sections = page.getByRole("navigation", {
        name: "Project sections",
      });
      await expect(
        sections.getByRole("link", { name: "Experiments", exact: true }),
      ).toHaveAttribute("href", `/evals?projectId=${EVAL_ID}`);
      await expect(
        sections.getByRole("link", { name: "Datasets", exact: true }),
      ).toHaveAttribute("href", `/evals/datasets?project=${EVAL_ID}`);
      await expect(
        page.getByRole("link", { name: "run-browser-leg-a" }),
      ).toBeVisible();
      await page
        .getByText("Workspace settings", { exact: true })
        .last()
        .click();
      await expect(
        page.getByRole("heading", { name: "Live target", exact: true }),
      ).toBeVisible();
      await expect
        .poll(() =>
          page.evaluate(
            () => document.documentElement.scrollWidth <= window.innerWidth,
          ),
        )
        .toBe(true);
      await page.screenshot({
        path: info.outputPath(`workspace-${width}-${theme}.png`),
        fullPage: true,
      });
    });
  }
}
