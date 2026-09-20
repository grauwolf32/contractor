import type { Page } from "@playwright/test";
import packageMetadata from "../package.json" with { type: "json" };
import { createEvalFixture, EVAL_API_VERSION } from "../src/test/evals-fixture";

export async function installEvalFixture(
  page: Page,
  options: Parameters<typeof createEvalFixture>[0] = {},
) {
  const fixture = createEvalFixture(options);
  await page.route("**/runtime-config.json", (route) =>
    route.fulfill({
      json: {
        uiVersion: packageMetadata.version,
        supportedApiVersions: [EVAL_API_VERSION],
        apiBaseUrl: new URL(route.request().url()).origin,
      },
    }),
  );
  await page.route("**/v1/**", async (route) => {
    const incoming = route.request();
    try {
      const response = await fixture.handle(
        new Request(incoming.url(), {
          method: incoming.method(),
          headers: incoming.headers(),
          ...(incoming.postData() ? { body: incoming.postData()! } : {}),
        }),
      );
      await route.fulfill({
        status: response.status,
        headers: response.headers,
        body: JSON.stringify(response.body),
      });
    } catch (cause) {
      if (
        cause instanceof TypeError &&
        cause.message.startsWith("Simulated lost")
      ) {
        await route.abort("connectionclosed");
        return;
      }
      throw cause;
    }
  });
  return fixture;
}
