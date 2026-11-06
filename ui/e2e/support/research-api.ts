import type { Page } from "@playwright/test";
import { researchFixture } from "../../src/test/audit-research-fixtures";
import {
  fulfillContent,
  fulfillEmptyPageOrNotFound,
  fulfillJSON,
  mockAPI,
} from "./mock-api";

export async function installResearchAPI(page: Page, state = "normal") {
  const fixture = researchFixture(state);
  await mockAPI(page, async (route, request, url) => {
    const body = request.postData();
    const response = await fixture.respond(
      new Request(url, {
        method: request.method(),
        headers: request.headers(),
        ...(body === null ? {} : { body }),
      }),
    );
    if (!response) return fulfillEmptyPageOrNotFound(route);
    const headers: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      headers[key] = value;
    });
    if (url.pathname.endsWith("/artifacts") && response.status === 200)
      return fulfillContent(
        route,
        await response.text(),
        headers["content-type"]!,
        200,
        headers,
      );
    return fulfillJSON(route, await response.json(), response.status, headers);
  });
  return fixture;
}
