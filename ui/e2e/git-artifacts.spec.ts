import { expect, test, type Page, type Route } from "@playwright/test";

import { project } from "../src/test/builders";
import {
  stageContract,
  workflowResource,
  workflowSummary,
} from "../src/test/builders-execution";
import {
  authSession,
  fulfillEmpty,
  fulfillEmptyPageOrNotFound,
  fulfillError,
  fulfillJSON,
  mockAPI,
  uiOrigin,
  UNANSWERED,
} from "./support/mock-api";

// Key tests deliberately avoid traces/screenshots of request bodies or inputs.
test.use({ trace: "off", screenshot: "off" });
const commit = "a".repeat(40);
const timestamp = "2026-09-06T12:00:00Z";
const workflow = workflowResource("git-review", "1", {
  entryStage: "inspect",
  parameters: { note: { required: true } },
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
    context: { required: false, mediaTypes: ["text/plain"] },
  },
  stages: { inspect: stageContract() },
});

async function fixture(
  page: Page,
  origin: string,
  options: {
    failImport?: boolean;
    pendingImport?: boolean;
    operations?: boolean;
  } = {},
) {
  const writes: Array<{
    path: string;
    headers: Record<string, string>;
    body: unknown;
  }> = [];
  let configured = false;
  const metadata = [
    {
      artifact: {
        namespace: "sources",
        name: "previous",
        revision: "rev-previous",
      },
      mediaType: "application/zip",
      size: 123,
      current: true,
      frozen: false,
      createdAt: timestamp,
    },
    {
      artifact: {
        namespace: "notes",
        name: "context",
        revision: "rev-context",
      },
      mediaType: "text/plain",
      size: 12,
      current: true,
      frozen: false,
      createdAt: timestamp,
    },
  ];
  const reply = (route: Route, value: unknown, status = 200) =>
    status === 204
      ? fulfillEmpty(route, 204, { etag: '"1"' })
      : fulfillJSON(route, value, status, { etag: '"1"' });
  await mockAPI(
    page,
    async (route, request, url) => {
      const path = url.pathname;
      if (path === "/v1/settings/git-key") {
        if (request.method() === "PUT") {
          if (request.postDataJSON().privateKey === "bad")
            return fulfillError(
              route,
              400,
              "git_key_invalid",
              "Unsupported key format",
            );
          configured = true;
        }
        if (request.method() === "DELETE") {
          configured = false;
          return reply(route, undefined, 204);
        }
        return reply(
          route,
          configured
            ? {
                configured: true,
                fingerprint: "SHA256:fixture-public",
                keyType: "ssh-ed25519",
                updatedAt: timestamp,
              }
            : { configured: false },
        );
      }
      if (path === "/v1/operations/snapshot")
        return reply(route, {
          cursor: { generation: "git-settings", revision: "1" },
          runtimeAgents: [],
          allocations: [],
        });
      if (path === "/v1/operations/settings/scheduler")
        return reply(route, {
          maxConcurrentRuns: 1,
          revision: "1",
          updatedAt: timestamp,
        });
      if (path.endsWith("/git-import")) {
        writes.push({
          path,
          headers: request.headers(),
          body: request.postDataJSON(),
        });
        if (options.pendingImport) return UNANSWERED;
        if (options.failImport)
          return fulfillError(
            route,
            422,
            "git_host_untrusted",
            "Git SSH host trust is unavailable or verification failed",
          );
        const parts = path.split("/");
        const name = parts.at(-2)!;
        const namespace = parts.at(-3)!;
        const gitSource = {
          repositoryUrl: "https://example.test:443/repo.git",
          requestedRef: request.postDataJSON().ref ?? null,
          resolvedCommit: commit,
          importedAt: timestamp,
        };
        const item = {
          artifact: { namespace, name, revision: "rev-imported" },
          mediaType: "application/zip",
          size: 123,
          current: true,
          frozen: false,
          createdAt: timestamp,
          gitSource,
        };
        metadata.push(item);
        return reply(
          route,
          {
            artifact: item.artifact,
            mediaType: item.mediaType,
            size: item.size,
            gitSource,
          },
          201,
        );
      }
      if (path.endsWith("/metadata")) {
        const parts = path.split("/");
        const item = metadata.find(
          (item) =>
            item.artifact.name === parts.at(-2) &&
            item.artifact.namespace === parts.at(-3),
        );
        return item === undefined
          ? fulfillError(route, 404, "not_found")
          : reply(route, item);
      }
      if (path.endsWith("/artifacts"))
        return reply(route, { items: metadata, page: { hasMore: false } });
      if (path === "/v1/projects/git-project")
        return reply(route, project("git-project", { name: "Git project" }));
      if (path === "/v1/workflows/git-review/versions/1")
        return reply(route, workflow);
      if (path === "/v1/workflows")
        return reply(route, {
          items: [workflowSummary(workflow)],
          page: { hasMore: false },
        });
      if (request.method() === "POST")
        writes.push({
          path,
          headers: request.headers(),
          body: request.postDataJSON(),
        });
      return fulfillEmptyPageOrNotFound(route);
    },
    {
      apiOrigin: origin,
      session: authSession(
        "owner",
        options.operations ? ["user", "operations"] : ["user"],
      ),
    },
  );
  return writes;
}

test("Git key settings live under Operations and retain no browser secret", async ({
  page,
}, info) => {
  await fixture(page, uiOrigin(info), {
    operations: true,
  });
  await page.goto("/operations/settings");
  const input = page.getByLabel("SSH private key");
  await input.fill("fixture-private-key-canary");
  await page.getByRole("button", { name: "Save Git key" }).click();
  await expect(page.getByText("Git SSH key saved.")).toBeVisible();
  await expect(input).toHaveValue("");
  await input.fill("bad");
  await page.getByRole("button", { name: "Replace Git key" }).click();
  await expect(page.getByText("Unsupported key format")).toBeVisible();
  await expect(page.getByText("SHA256:fixture-public")).toBeVisible();
  await page.getByRole("button", { name: "Remove Git key" }).click();
  await page
    .getByRole("alertdialog", { name: "Remove Git key?" })
    .getByRole("button", { name: "Remove Git key" })
    .click();
  await expect(input).toHaveValue("");
  await expect(page.getByText("No Git SSH key configured.")).toBeVisible();
  expect(
    await page.evaluate(() =>
      JSON.stringify({
        local: { ...localStorage },
        session: { ...sessionStorage },
      }),
    ),
  ).not.toContain("fixture-private-key-canary");
});

for (const project of [false, true]) {
  test(`Git import preserves the ${project ? "Project" : "standalone"} Workflow draft and exact input`, async ({
    page,
  }, info) => {
    const writes = await fixture(page, uiOrigin(info));
    if (project) {
      await page.goto("/projects/git-project/workflows");
      await page
        .getByRole("button", { name: "Configure git-review@1" })
        .click();
    } else
      await page.goto("/catalog/workflows/git-review/1#workflow-run-setup");
    await page.locator('[name="parameter-note"]').fill("keep this draft");
    await page
      .locator('[name="artifact-context"]')
      .selectOption("notes/context@rev-context");
    await page.getByRole("button", { name: "Import Git for source" }).click();
    const dialog = page.getByRole("dialog", { name: "Import Git repository" });
    await expect(dialog.getByLabel("Repository URL")).toBeFocused();
    await dialog
      .getByLabel("Repository URL")
      .fill("https://example.test/repo.git");
    await dialog.getByLabel("Branch or tag (optional)").fill("release");
    await dialog.getByRole("button", { name: "Import snapshot" }).click();
    await expect(dialog).not.toBeVisible();
    await expect(page.locator('[name="artifact-source"]')).toHaveValue(
      "sources/source@rev-imported",
    );
    await expect(page.locator('[name="artifact-context"]')).toHaveValue(
      "notes/context@rev-context",
    );
    await expect(page.locator('[name="parameter-note"]')).toHaveValue(
      "keep this draft",
    );
    await expect(page.getByText(`Commit ${commit}`)).toBeVisible();
    expect(writes).toHaveLength(1);
    expect(writes[0]?.path).toBe(
      project
        ? "/v1/projects/git-project/artifacts/sources/source/git-import"
        : "/v1/artifacts/sources/source/git-import",
    );
    expect(writes[0]?.headers["if-none-match"]).toBe("*");
    expect(writes[0]?.body).toEqual({
      repositoryUrl: "https://example.test/repo.git",
      ref: "release",
    });
  });
}

test("Project import validates ASCII names, reports host errors and restores focus", async ({
  page,
}, info) => {
  const writes = await fixture(page, uiOrigin(info), { failImport: true });
  await page.goto("/projects/git-project/artifacts?add=artifact");
  const trigger = page.getByRole("button", {
    name: "Import Git repository",
    exact: true,
  });
  await trigger.click();
  const dialog = page.getByRole("dialog", { name: "Import Git repository" });
  await dialog.getByLabel("Repository URL").fill("git@example.test:repo.git");
  await dialog.getByLabel("Name", { exact: true }).fill("bad name");
  await dialog.getByRole("button", { name: "Import snapshot" }).click();
  await expect(dialog.getByRole("alert")).toContainText(
    "Spaces are not allowed",
  );
  expect(writes).toHaveLength(0);
  await dialog.getByLabel("Name", { exact: true }).fill("source");
  await dialog.getByRole("button", { name: "Import snapshot" }).click();
  await expect(dialog.getByRole("alert")).toContainText("host trust");
  expect(writes).toHaveLength(1);
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(trigger).toBeFocused();
});

test("cancelling a pending Git import keeps the Project Workflow form open", async ({
  page,
}, info) => {
  const writes = await fixture(page, uiOrigin(info), { pendingImport: true });
  await page.goto("/projects/git-project/workflows");
  await page.getByRole("button", { name: "Configure git-review@1" }).click();
  await page.locator('[name="parameter-note"]').fill("preserved");
  await page.getByRole("button", { name: "Import Git for source" }).click();
  const dialog = page.getByRole("dialog", { name: "Import Git repository" });
  await page.setViewportSize({ width: 390, height: 844 });
  await dialog
    .getByLabel("Repository URL")
    .fill("https://example.test/repo.git");
  await dialog.getByRole("button", { name: "Import snapshot" }).click();
  await expect.poll(() => writes.length).toBe(1);
  expect(
    await dialog.evaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    ),
  ).toBe(true);
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(
    page.getByRole("dialog", { name: "Configure Run" }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("dialog", { name: "Configure Run" })
      .locator(".workflow-drawer-heading code"),
  ).toHaveText("git-review@1");
  await expect(page.locator('[name="parameter-note"]')).toHaveValue(
    "preserved",
  );
  expect(writes).toHaveLength(1);
});
