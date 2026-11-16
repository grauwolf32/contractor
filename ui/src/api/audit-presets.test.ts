import { expect, it } from "vitest";
import { listAuditPresets } from "./audit-presets";
import { presetFixture } from "../test/audit-presets-fixture";
import { fakeServer } from "../test/fake-server";
import { json, page } from "../test/responses";
import { testAPI } from "../test/render";

it("refuses an incomplete check type inventory instead of defaulting to a partial version family", async () => {
  const server = fakeServer({
    "GET /v1/audit-profiles": ({ url }) => {
      const batch = Number(url.searchParams.get("cursor") ?? "0");
      return json(
        page(
          Array.from({ length: 50 }, (_, offset) => ({
            ...presetFixture,
            ref: {
              ...presetFixture.ref,
              name: `profile-${batch * 50 + offset}`,
            },
          })),
          String(batch + 1),
        ),
      );
    },
  });
  const { api } = testAPI(server);
  await expect(
    listAuditPresets(api, new AbortController().signal),
  ).rejects.toThrow("incomplete");
  expect(
    server.requests.filter(
      (r) => new URL(r.url).pathname === "/v1/audit-profiles",
    ),
  ).toHaveLength(5);
});
