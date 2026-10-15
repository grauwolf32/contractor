import { expect, test, type Page } from "@playwright/test";

import { project } from "../src/test/builders";
import {
  stageContract,
  workflowResource,
  workflowSummary,
} from "../src/test/builders-execution";
import {
  authSession,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
  mockedAPIOrigin,
} from "./support/mock-api";

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
const WORKFLOW = workflowResource(WORKFLOW_NAME, WORKFLOW_VERSION, {
  entryStage: "inspect",
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
  },
  stages: { inspect: stageContract() },
});

async function installDialogFixture(page: Page, apiOrigin: string) {
  await mockAPI(
    page,
    async (route, request, url) => {
      const path = url.pathname;
      if (path === `/v1/projects/${PROJECT_ID}`) {
        await fulfillJSON(
          route,
          project(PROJECT_ID, {
            name: "Dialog workspace",
            description: "Nested dialog browser fixture",
          }),
          200,
          { etag: '"1"' },
        );
        return;
      }
      if (path === `/v1/projects/${PROJECT_ID}/artifacts`) {
        await fulfillJSON(route, { items: [SOURCE], page: { hasMore: false } });
        return;
      }
      if (path === `/v1/projects/${PROJECT_ID}/runs`) {
        await fulfillJSON(route, { items: [], page: { hasMore: false } });
        return;
      }
      if (path === "/v1/workflows") {
        await fulfillJSON(route, {
          items: [workflowSummary(WORKFLOW)],
          page: { hasMore: false },
        });
        return;
      }
      if (
        path === `/v1/workflows/${WORKFLOW_NAME}/versions/${WORKFLOW_VERSION}`
      ) {
        await fulfillJSON(route, WORKFLOW);
        return;
      }
      await fulfillEmptyPageOrNotFound(
        route,
        `No fixture for ${request.method()} ${path}`,
      );
    },
    { apiOrigin, session: authSession("dialog-user") },
  );
}

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test.describe(`${viewport.width}px journeys`, () => {
    test.use({ viewport });

    test("nested Project and Git dialogs isolate focus, forms and Escape", async ({
      page,
    }) => {
      const apiOrigin = mockedAPIOrigin();
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
