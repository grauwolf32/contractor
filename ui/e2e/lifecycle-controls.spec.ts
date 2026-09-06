import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const CSRF_TOKEN = "l".repeat(43);
const PROJECT_ID = "project-lifecycle-browser";

function responseHeaders(
  extra: Record<string, string> = {},
): Record<string, string> {
  return {
    "cache-control": "no-store",
    "content-type": "application/json",
    "x-contractor-api-version": API_VERSION,
    ...extra,
  };
}

async function fulfillJSON(
  route: Route,
  value: unknown,
  status = 200,
  extraHeaders: Record<string, string> = {},
): Promise<void> {
  await route.fulfill({
    body: JSON.stringify(value),
    headers: responseHeaders(extraHeaders),
    status,
  });
}

async function installLifecycleAPI(page: Page, uiOrigin: string) {
  let queuePaused = false;
  let queueRevision = 0;
  let runDeleted = false;
  let projectDeleting = false;
  let projectDeleted = false;
  const writes: Array<{
    method: string;
    path: string;
    headers: Record<string, string>;
  }> = [];

  const activeProject = {
    projectId: PROJECT_ID,
    kind: "project",
    name: "Lifecycle browser",
    description: "Lifecycle release fixture",
    lifecycle: "active",
    revision: "1",
    createdAt: "2026-09-05T08:00:00Z",
    updatedAt: "2026-09-05T08:00:00Z",
  };
  const deletingProject = {
    ...activeProject,
    lifecycle: "deleting",
    revision: "2",
    updatedAt: "2026-09-05T08:05:00Z",
    deletion: {
      phase: "draining",
      requestedAt: "2026-09-05T08:05:00Z",
    },
  };

  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: uiOrigin,
      },
    }),
  );
  await page.route(`${uiOrigin}/v1/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/v1/auth/session") {
      await fulfillJSON(route, {
        principal: {
          userId: "user-lifecycle-browser",
          username: "lifecycle",
          capabilities: ["user", "operations"],
        },
        csrfToken: CSRF_TOKEN,
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (url.pathname === "/v1/queue/control") {
      if (request.method() === "PUT") {
        writes.push({
          method: request.method(),
          path: url.pathname,
          headers: request.headers(),
        });
        queuePaused = (request.postDataJSON() as { paused: boolean }).paused;
        queueRevision += 1;
      }
      await fulfillJSON(
        route,
        {
          paused: queuePaused,
          revision: String(queueRevision),
          ...(queueRevision === 0 ? {} : { updatedAt: "2026-09-05T08:02:00Z" }),
        },
        200,
        { etag: `"${queueRevision}"` },
      );
      return;
    }
    if (url.pathname === "/v1/queue") {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.pathname === "/v1/runs" && request.method() === "GET") {
      await fulfillJSON(route, {
        items: runDeleted
          ? []
          : [
              {
                runId: "run-lifecycle-browser",
                workflow: "lifecycle-fixture@1",
                state: "succeeded",
                deletable: true,
                labels: {},
                createdAt: "2026-09-05T07:00:00Z",
                updatedAt: "2026-09-05T07:01:00Z",
                finishedAt: "2026-09-05T07:01:00Z",
              },
            ],
        page: { hasMore: false },
      });
      return;
    }
    if (
      url.pathname === "/v1/runs/run-lifecycle-browser" &&
      request.method() === "DELETE"
    ) {
      writes.push({
        method: request.method(),
        path: url.pathname,
        headers: request.headers(),
      });
      runDeleted = true;
      await route.fulfill({ status: 204, headers: responseHeaders() });
      return;
    }
    if (url.pathname === "/v1/projects" && request.method() === "GET") {
      await fulfillJSON(route, {
        items: projectDeleted
          ? []
          : [projectDeleting ? deletingProject : activeProject],
        page: { hasMore: false },
      });
      return;
    }
    if (url.pathname === `/v1/projects/${PROJECT_ID}`) {
      if (request.method() === "DELETE") {
        writes.push({
          method: request.method(),
          path: url.pathname,
          headers: request.headers(),
        });
        projectDeleting = true;
        await fulfillJSON(route, deletingProject, 202, { etag: '"2"' });
        return;
      }
      if (projectDeleted) {
        await fulfillJSON(
          route,
          {
            code: "not_found",
            message: "resource was not found",
            retryable: false,
            requestId: "request-lifecycle-project-deleted",
          },
          404,
        );
        return;
      }
      const project = projectDeleting ? deletingProject : activeProject;
      await fulfillJSON(route, project, 200, { etag: `"${project.revision}"` });
      return;
    }
    if (
      url.pathname === `/v1/projects/${PROJECT_ID}/artifacts` ||
      url.pathname === `/v1/projects/${PROJECT_ID}/runs`
    ) {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (url.pathname === "/v1/workflows") {
      await fulfillJSON(route, { items: [], page: { hasMore: false } });
      return;
    }
    await fulfillJSON(
      route,
      {
        code: "not_found",
        message: "not found",
        retryable: false,
        requestId: "request-lifecycle-unexpected",
      },
      404,
    );
  });

  return {
    completeProjectDeletion(): void {
      projectDeleted = true;
    },
    writes,
  };
}

test("independently served UI completes lifecycle controls without manual reload", async ({
  page,
}, testInfo) => {
  const configuredBaseURL = testInfo.project.use.baseURL;
  if (typeof configuredBaseURL !== "string") {
    throw new Error("Playwright baseURL is required");
  }
  const fixture = await installLifecycleAPI(
    page,
    new URL(configuredBaseURL).origin,
  );

  await page.goto("/runs");
  await page.getByRole("button", { name: "Pause queue" }).click();
  await expect(page.getByText("Admission paused")).toBeVisible();
  await page.getByRole("button", { name: "Resume queue" }).click();
  await expect(page.getByRole("button", { name: "Pause queue" })).toBeVisible();

  await page.getByRole("link", { name: /Completed/ }).click();
  await page
    .getByRole("button", { name: "Delete Run run-lifecycle-browser" })
    .click();
  await page.getByRole("button", { name: "Delete Run", exact: true }).click();
  await expect(
    page.getByRole("link", { name: "run-lifecycle-browser" }),
  ).toHaveCount(0);

  await page.goto(`/projects/${PROJECT_ID}`);
  await page.getByRole("button", { name: "Delete Project" }).click();
  const dialog = page.getByRole("alertdialog", {
    name: "Delete Lifecycle browser?",
  });
  await dialog
    .getByLabel("Type Lifecycle browser to confirm")
    .fill("Lifecycle browser");
  await dialog.getByRole("button", { name: "Delete Project" }).click();
  await expect(
    page.getByRole("heading", { name: "Waiting for Runtime release" }),
  ).toBeVisible();

  fixture.completeProjectDeletion();
  await expect(page).toHaveURL(/\/projects$/);
  await expect(page.getByRole("heading", { name: "Projects" })).toBeVisible();

  expect(fixture.writes.map(({ method, path }) => `${method} ${path}`)).toEqual(
    [
      "PUT /v1/queue/control",
      "PUT /v1/queue/control",
      "DELETE /v1/runs/run-lifecycle-browser",
      `DELETE /v1/projects/${PROJECT_ID}`,
    ],
  );
  for (const write of fixture.writes) {
    expect(write.headers["x-csrf-token"]).toBe(CSRF_TOKEN);
  }
  expect(fixture.writes.at(-1)?.headers["if-match"]).toBe('"1"');
});
