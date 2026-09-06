import { expect, test, type Route } from "@playwright/test";

const apiVersion = "contractor.public.v1";

async function fulfillJSON(
  route: Route,
  value: unknown,
  status = 200,
  headers: Record<string, string> = {},
) {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: { "X-Contractor-API-Version": apiVersion, ...headers },
    body: JSON.stringify(value),
  });
}

test("operator CAS-updates Scheduler settings and recovers a stale edit", async ({
  page,
}) => {
  const origin = new URL(test.info().project.use.baseURL as string).origin;
  const csrf = "a".repeat(43);
  let maximum = 1;
  let revision = 1;
  const puts: Array<{
    ifMatch: string | null;
    csrf: string | null;
    body: unknown;
  }> = [];

  await page.route("**/runtime-config.json", (route) =>
    fulfillJSON(route, {
      uiVersion: "0.1.0",
      supportedApiVersions: [apiVersion],
      apiBaseUrl: origin,
    }),
  );
  await page.route(`${origin}/v1/**`, async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/v1/auth/session") {
      await fulfillJSON(route, {
        principal: {
          userId: "operator",
          username: "operator",
          capabilities: ["user", "operations"],
        },
        csrfToken: csrf,
        idleExpiresAt: "2026-09-06T20:00:00Z",
        absoluteExpiresAt: "2026-09-07T12:00:00Z",
      });
      return;
    }
    if (path === "/v1/operations/snapshot") {
      await fulfillJSON(route, {
        cursor: {
          generation: "scheduler-settings-browser",
          revision: "1",
        },
        runtimeAgents: [],
        allocations: [],
      });
      return;
    }
    if (path === "/v1/settings/git-key") {
      await fulfillJSON(route, { configured: false });
      return;
    }
    if (path === "/v1/operations/settings/scheduler") {
      if (request.method() === "GET") {
        await fulfillJSON(
          route,
          {
            maxConcurrentRuns: maximum,
            revision: String(revision),
            updatedAt: `2026-09-06T01:00:0${revision}Z`,
          },
          200,
          { ETag: `"${revision}"`, "Cache-Control": "no-store" },
        );
        return;
      }
      const body = request.postDataJSON() as { maxConcurrentRuns?: number };
      const ifMatch = await request.headerValue("If-Match");
      puts.push({
        ifMatch,
        csrf: await request.headerValue("X-CSRF-Token"),
        body,
      });
      if (ifMatch !== `"${revision}"`) {
        await fulfillJSON(
          route,
          {
            code: "precondition_failed",
            message: "resource revision precondition failed",
            retryable: false,
            requestId: "request-stale-browser",
          },
          412,
        );
        return;
      }
      maximum = body.maxConcurrentRuns ?? maximum;
      revision += 1;
      await fulfillJSON(
        route,
        {
          maxConcurrentRuns: maximum,
          revision: String(revision),
          updatedAt: `2026-09-06T01:00:0${revision}Z`,
        },
        200,
        { ETag: `"${revision}"`, "Cache-Control": "no-store" },
      );
      return;
    }
    await fulfillJSON(
      route,
      {
        code: "not_found",
        message: "not found",
        retryable: false,
        requestId: "request-browser-missing",
      },
      404,
    );
  });

  await page.goto("/operations/settings");
  const input = page.getByLabel("Maximum concurrent Workflow Runs");
  await expect(input).toHaveValue("1");
  await input.fill("2");
  await page.getByRole("button", { name: "Save scheduling limit" }).click();
  await expect(page.getByText("Saved maximum 2 at revision 2.")).toBeVisible();
  expect(puts[0]).toEqual({
    ifMatch: '"1"',
    csrf,
    body: { maxConcurrentRuns: 2 },
  });

  await input.fill("4");
  maximum = 5;
  revision = 3;
  await page.getByRole("button", { name: "Save scheduling limit" }).click();
  await expect(
    page.getByText(/Another operator|changed on the Server/),
  ).toBeVisible();
  await expect(input).toHaveValue("4");
  await expect(page.getByText("5", { exact: true })).toBeVisible();
  expect(puts[1]).toMatchObject({ ifMatch: '"2"', csrf });
  await expect(
    page.getByRole("button", { name: "Save scheduling limit" }),
  ).toBeDisabled();

  await page.getByRole("button", { name: "Reset to saved value" }).click();
  await expect(input).toHaveValue("5");
  await expect(
    page.getByText(/Another operator|changed on the Server/),
  ).toHaveCount(0);
});
