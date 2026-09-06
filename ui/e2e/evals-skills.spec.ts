import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const EVAL_ID = "evaluation-browser";

function apiHeaders(): Record<string, string> {
  return {
    "cache-control": "no-store",
    "content-type": "application/json",
    "x-contractor-api-version": API_VERSION,
  };
}

async function fulfillJSON(route: Route, value: unknown): Promise<void> {
  await route.fulfill({
    body: JSON.stringify(value),
    headers: apiHeaders(),
    status: 200,
  });
}

async function installAPI(page: Page, origin: string): Promise<void> {
  const evaluation = {
    projectId: EVAL_ID,
    kind: "evaluation",
    name: "Browser evaluation",
    description: "Evaluation route fixture",
    lifecycle: "active",
    revision: "1",
    createdAt: "2026-09-05T08:00:00Z",
    updatedAt: "2026-09-05T08:01:00Z",
  };
  await page.route("**/runtime-config.json", async (route) => {
    await route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: origin,
      },
    });
  });
  await page.route(`${origin}/v1/**`, async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/v1/auth/session") {
      await fulfillJSON(route, {
        principal: {
          userId: "user_browser",
          username: "browser",
          capabilities: ["user", "operations"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (url.pathname === "/v1/projects") {
      await fulfillJSON(route, {
        items:
          url.searchParams.get("kind") === "evaluation" ? [evaluation] : [],
        page: { hasMore: false },
      });
      return;
    }
    if (url.pathname === `/v1/projects/${EVAL_ID}`) {
      await route.fulfill({
        body: JSON.stringify(evaluation),
        headers: { ...apiHeaders(), etag: '"1"' },
        status: 200,
      });
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
    await route.fulfill({ status: 404, body: "not found" });
  });
}

test("Evals and Skills stay separate Project/UserScope UI projections", async ({
  page,
}, testInfo) => {
  const baseURL = testInfo.project.use.baseURL;
  if (typeof baseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const origin = new URL(baseURL).origin;
  await installAPI(page, origin);

  await page.goto("/evals");
  await expect(page.getByRole("heading", { name: "Evals" })).toBeVisible();
  await page.getByRole("link", { name: "Browser evaluation" }).click();
  await expect(page.getByRole("heading", { name: "Eval Runs" })).toBeVisible();
  await expect(page.getByText("eval-browser-01")).toBeVisible();
  await expect(
    page.getByRole("link", { name: "run-browser-leg-a" }),
  ).toBeVisible();

  await page.getByRole("link", { name: "Catalog", exact: true }).click();
  await page.getByRole("link", { name: "Skills", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Skills" })).toBeVisible();
  const skill = page.getByRole("link", { name: "architecture-review" });
  await expect(skill).toHaveAttribute(
    "href",
    "/artifacts/skills/architecture-review",
  );
  await expect(
    page.getByText(/never become Project-owned copies/i),
  ).toBeVisible();
});
