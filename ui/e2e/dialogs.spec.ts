import { expect, test, type Page, type Route } from "@playwright/test";

const API_VERSION = "contractor.public.v1";
const PROJECT_ID = "project_dialog_fixture";
const WORKFLOW_NAME = "dialog-workflow";
const WORKFLOW_VERSION = "1";
const SOURCE = {
  artifact: {
    namespace: "sources",
    name: "dialog-source",
    revision: "revision-dialog-source",
  },
  mediaType: "application/zip",
  size: 3,
  current: true,
  frozen: false,
  createdAt: "2026-09-06T10:00:00Z",
};
const WORKFLOW = {
  ref: { name: WORKFLOW_NAME, version: WORKFLOW_VERSION },
  entryStage: "inspect",
  parameters: {},
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
  },
  outputs: {},
  stages: {},
};

function responseHeaders(origin?: string): Record<string, string> {
  return {
    "cache-control": "no-store",
    "content-type": "application/json",
    "x-contractor-api-version": API_VERSION,
    ...(origin === undefined
      ? {}
      : {
          "access-control-allow-origin": origin,
          "access-control-allow-credentials": "true",
          "access-control-expose-headers":
            "X-Contractor-API-Version, ETag, X-Request-ID",
        }),
  };
}

async function fulfill(
  route: Route,
  value: unknown,
  status = 200,
  extraHeaders: Record<string, string> = {},
) {
  const origin = route.request().headers().origin;
  await route.fulfill({
    status,
    body: JSON.stringify(value),
    headers: { ...responseHeaders(origin), ...extraHeaders },
  });
}

async function installDialogFixture(page: Page, apiOrigin: string) {
  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: "0.1.0",
        supportedApiVersions: [API_VERSION],
        apiBaseUrl: apiOrigin,
      },
    }),
  );
  await page.route(`${apiOrigin}/v1/**`, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "access-control-allow-origin": request.headers().origin ?? "*",
          "access-control-allow-credentials": "true",
          "access-control-allow-methods": "GET, POST, PUT, PATCH, OPTIONS",
          "access-control-allow-headers":
            "content-type, idempotency-key, if-match, if-none-match, x-csrf-token",
        },
      });
      return;
    }
    if (path === "/v1/auth/session") {
      await fulfill(route, {
        principal: {
          userId: "user_dialog",
          username: "dialog-user",
          capabilities: ["user", "operations"],
        },
        csrfToken: "a".repeat(43),
        idleExpiresAt: "2099-01-01T00:00:00Z",
        absoluteExpiresAt: "2099-01-02T00:00:00Z",
      });
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}`) {
      await fulfill(
        route,
        {
          projectId: PROJECT_ID,
          kind: "project",
          name: "Dialog workspace",
          description: "Nested dialog browser fixture",
          lifecycle: "active",
          revision: "1",
          createdAt: "2026-09-06T10:00:00Z",
          updatedAt: "2026-09-06T10:00:00Z",
        },
        200,
        { etag: '"1"' },
      );
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}/artifacts`) {
      await fulfill(route, { items: [SOURCE], page: { hasMore: false } });
      return;
    }
    if (path === `/v1/projects/${PROJECT_ID}/runs`) {
      await fulfill(route, { items: [], page: { hasMore: false } });
      return;
    }
    if (path === "/v1/workflows") {
      await fulfill(route, { items: [WORKFLOW], page: { hasMore: false } });
      return;
    }
    if (
      path === `/v1/workflows/${WORKFLOW_NAME}/versions/${WORKFLOW_VERSION}`
    ) {
      await fulfill(route, WORKFLOW);
      return;
    }
    await fulfill(
      route,
      {
        code: "not_found",
        message: `No fixture for ${request.method()} ${path}`,
        retryable: false,
        requestId: "request-dialog-fixture",
      },
      404,
    );
  });
}

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test.describe(`${viewport.width}px journeys`, () => {
    test.use({ viewport });

    test("nested Project and Git dialogs isolate focus, forms and Escape", async ({
      page,
    }, testInfo) => {
      const configuredBaseURL = testInfo.project.use.baseURL;
      if (typeof configuredBaseURL !== "string") {
        throw new Error("Playwright baseURL is required");
      }
      const apiOrigin =
        process.env.CONTRACTOR_UI_E2E_API_URL ?? "http://127.0.0.3:8080";
      await installDialogFixture(page, apiOrigin);
      const mutations: string[] = [];
      page.on("request", (request) => {
        if (
          request.url().startsWith(`${apiOrigin}/v1/`) &&
          ["POST", "PUT", "PATCH", "DELETE"].includes(request.method())
        )
          mutations.push(request.url());
      });

      await page.goto(`/projects/${PROJECT_ID}/workflows`);
      const launcher = page.getByRole("button", {
        name: `Configure ${WORKFLOW_NAME}@${WORKFLOW_VERSION}`,
      });
      await launcher.focus();
      await page.keyboard.press("Enter");

      const parent = page.getByRole("dialog", {
        name: "Configure Run",
      });
      await expect(parent).toBeVisible();
      await expect(parent.locator(".workflow-drawer-heading code")).toHaveText(
        `${WORKFLOW_NAME}@${WORKFLOW_VERSION}`,
      );
      await expect(
        parent.getByRole("button", { name: "Close Run setup" }),
      ).toBeFocused();
      await expect(page.locator("#root")).toHaveAttribute("inert", "");
      await expect(
        parent.getByRole("combobox", { name: /source required/ }),
      ).toHaveValue("sources/dialog-source@revision-dialog-source");

      const childTrigger = parent.getByRole("button", {
        name: "Import Git for source",
      });
      // Traverse the parent form using only the keyboard after opening it.
      for (
        let index = 0;
        index < 40 &&
        !(await childTrigger.evaluate(
          (node) => node === document.activeElement,
        ));
        index += 1
      ) {
        await page.keyboard.press("Tab");
      }
      await expect(childTrigger).toBeFocused();
      await page.keyboard.press("Enter");
      const child = page.getByRole("dialog", { name: "Import Git repository" });
      await expect(child).toBeVisible();
      await expect(child.getByLabel("Repository URL")).toBeFocused();
      await expect(
        page.locator("[data-contractor-dialog-layer]").first(),
      ).toHaveAttribute("inert", "");

      await page
        .getByLabel("Repository URL")
        .fill("https://example.test/source.git");
      // Tab and reverse Tab must remain in the topmost layer.
      for (let index = 0; index < 16; index += 1) {
        await page.keyboard.press(index < 8 ? "Tab" : "Shift+Tab");
        expect(
          await child.evaluate((node) => node.contains(document.activeElement)),
        ).toBe(true);
      }
      await page.keyboard.press("Escape");
      await expect(child).toBeHidden();
      await expect(parent).toBeVisible();
      await expect(childTrigger).toBeFocused();
      await expect(
        parent.getByRole("combobox", { name: /source required/ }),
      ).toHaveValue("sources/dialog-source@revision-dialog-source");

      await page.keyboard.press("Escape");
      await expect(parent).toBeHidden();
      await expect(launcher).toBeFocused();
      await expect(page.locator("#root")).not.toHaveAttribute("inert", "");
      await expect
        .poll(() => page.evaluate(() => document.body.style.overflow))
        .toBe("");
      expect(mutations).toEqual([]);
    });
  });
}
