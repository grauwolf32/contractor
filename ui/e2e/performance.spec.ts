import { expect, test, type Route } from "@playwright/test";

const apiVersion = "contractor.public.v1";

async function fulfillJSON(route: Route, value: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: { "X-Contractor-API-Version": apiVersion },
    body: JSON.stringify(value),
  });
}

function freshness(
  observedAt: string,
  intervalSeconds: 15 | 60 | 300 = 15,
  status: "ok" | "partial" | "unavailable" = "ok",
) {
  const endedAt = observedAt;
  return {
    status,
    ...(status === "unavailable" ? { reason: "read_failed" } : {}),
    ...(status === "unavailable" ? {} : { observedAt }),
    lastAttemptAt: observedAt,
    intervalSeconds,
    coverage: {
      startedAt: new Date(
        Date.parse(observedAt) - intervalSeconds * 1_000,
      ).toISOString(),
      endedAt,
      durationSeconds: intervalSeconds,
      expectedSamples: 1,
      observedSamples: status === "unavailable" ? 0 : 1,
    },
  };
}

function sample(generation: string, observedAt: string, cpuCores?: number) {
  return {
    version: 1,
    generation,
    observedAt,
    process: {
      freshness: freshness(
        observedAt,
        15,
        cpuCores === undefined ? "unavailable" : "ok",
      ),
      ...(cpuCores === undefined ? {} : { cpuCores, rssBytes: 104_857_600 }),
    },
  };
}

function snapshot(
  enabled: boolean,
  observedAt: string,
  current?: ReturnType<typeof sample>,
) {
  return {
    enabled,
    generation: current?.generation ?? "performance-disabled-generation",
    observedAt,
    sampleIntervalSeconds: 15,
    databaseIntervalSeconds: 60,
    databaseSizeIntervalSeconds: 300,
    ...(current === undefined ? {} : { current }),
    diagnostics: {
      skippedSamples: 0,
      rejectedSamples: 0,
      skippedMinutes: 0,
      droppedMinutes: 0,
      pendingMinutes: 0,
    },
  };
}

test("renders disabled stale and discontinuous performance observations truthfully", async ({
  page,
}) => {
  const origin = new URL(test.info().project.use.baseURL as string).origin;
  const csrf = "a".repeat(43);
  let enabled = false;
  let registryReads = 0;

  await page.route("**/runtime-config.json", (route) =>
    fulfillJSON(route, {
      uiVersion: "0.1.0",
      supportedApiVersions: [apiVersion],
      apiBaseUrl: origin,
    }),
  );
  await page.route(`${origin}/v1/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/v1/auth/session") {
      await fulfillJSON(route, {
        principal: {
          userId: "operator",
          username: "operator",
          capabilities: ["user", "operations"],
        },
        csrfToken: csrf,
        idleExpiresAt: "2099-09-06T20:00:00Z",
        absoluteExpiresAt: "2099-09-07T12:00:00Z",
      });
      return;
    }
    if (url.pathname === "/v1/operations/snapshot") {
      registryReads += 1;
      await fulfillJSON(route, { code: "unexpected_registry_read" }, 500);
      return;
    }
    if (url.pathname === "/v1/operations/performance") {
      const readAt = new Date().toISOString();
      const staleAt = new Date(Date.now() - 90_000).toISOString();
      await fulfillJSON(
        route,
        enabled
          ? snapshot(true, readAt, sample("performance-current", staleAt, 1.5))
          : snapshot(false, readAt),
      );
      return;
    }
    if (url.pathname === "/v1/operations/performance/history") {
      const from = url.searchParams.get("from");
      const to = url.searchParams.get("to");
      const step = url.searchParams.get("step");
      if (from === null || to === null || step === null) {
        await fulfillJSON(route, { code: "invalid_query" }, 400);
        return;
      }
      const end = Date.parse(to);
      const points = enabled
        ? [
            {
              kind: "sample",
              ...sample(
                "performance-before-restart",
                new Date(end - 45_000).toISOString(),
                1,
              ),
            },
            {
              kind: "sample",
              ...sample(
                "performance-before-restart",
                new Date(end - 30_000).toISOString(),
              ),
            },
            {
              kind: "sample",
              ...sample(
                "performance-after-restart",
                new Date(end - 15_000).toISOString(),
                2,
              ),
            },
          ]
        : [];
      await fulfillJSON(route, { from, to, step, points });
      return;
    }
    await fulfillJSON(
      route,
      {
        code: "not_found",
        message: "not found",
        retryable: false,
        requestId: "request-performance-browser",
      },
      404,
    );
  });

  await page.goto("/operations/performance");
  await expect(
    page.getByText("Performance collection is disabled."),
  ).toBeVisible();
  await expect(page.getByText("No observations in this range.")).toHaveCount(3);

  enabled = true;
  await page.reload();
  const serverProcess = page
    .locator("article.performance-metric-card")
    .filter({ has: page.getByRole("heading", { name: "Server process" }) });
  await expect(serverProcess.getByText("stale", { exact: true })).toBeVisible();
  await expect(
    page.getByText(/This range crosses 2 Server generations/),
  ).toBeVisible();
  const cpuChart = page.getByRole("img", { name: "CPU usage" });
  await expect(cpuChart).toBeVisible();
  await expect(cpuChart.locator(".performance-chart-line")).toHaveCount(2);
  await expect(page.getByLabel("CPU usage numeric summary")).toContainText(
    "Observed points2",
  );
  expect(registryReads).toBe(0);
});
