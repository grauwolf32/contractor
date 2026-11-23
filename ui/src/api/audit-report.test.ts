import { describe, expect, it, vi } from "vitest";
import { TestPublicAPI as PublicAPI } from "../test/shared-config";
import * as build from "../test/builders";
import { getAuditReport } from "./audits";
const config = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};
function api(value: unknown) {
  return new PublicAPI(
    config,
    vi.fn(
      async () =>
        new Response(JSON.stringify(value), {
          status: 200,
          headers: {
            "content-type": "application/json",
            "x-contractor-api-version": "contractor.public.v1",
          },
        }),
    ),
  );
}
describe("Audit report API", () => {
  it("accepts a complete retained report without a review", async () => {
    await expect(
      getAuditReport(
        api(build.report("audit_report", "ready")),
        "audit_report",
      ),
    ).resolves.toMatchObject({ status: "ready" });
  });
  it("accepts normal absence after completion", async () => {
    await expect(
      getAuditReport(api({ status: "unavailable" }), "audit_report"),
    ).resolves.toEqual({ status: "unavailable" });
  });
  it("rejects the removed proposed report state", async () => {
    await expect(
      getAuditReport(api({ status: "proposed" }), "audit_report"),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
  });
});
