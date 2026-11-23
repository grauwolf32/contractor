import type { Page } from "@playwright/test";

import { createEvalFixture } from "../../src/test/evals-fixture";
import { fulfillJSON, routeRuntimeConfig } from "./mock-api";

/**
 * Serves the managed Evals fixture of the unit tests (src/test) to the
 * browser, every answer checked against the public contract.
 */
export async function installEvalFixture(
  page: Page,
  options: Parameters<typeof createEvalFixture>[0] = {},
) {
  const fixture = createEvalFixture(options);
  await routeRuntimeConfig(page);
  await page.route("**/v1/**", async (route) => {
    const incoming = route.request();
    try {
      const postData = incoming.postData();
      const response = await fixture.handle(
        new Request(incoming.url(), {
          method: incoming.method(),
          headers: incoming.headers(),
          ...(postData ? { body: postData } : {}),
        }),
      );
      const headers = { ...response.headers };
      delete headers["Content-Type"];
      await fulfillJSON(route, response.body, response.status, headers);
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
