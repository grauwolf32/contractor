import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { standardFixture } from "../test/audit-presets-fixture";
import { getAuditStandard } from "./audit-presets";
import { PublicAPI } from "./client";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function standardAPI(scheme: string, version: string) {
  const fetch = vi.fn(
    async () =>
      new Response(
        JSON.stringify({
          standard: { ...standardFixture, reference: { scheme, version } },
        }),
        {
          headers: {
            "content-type": "application/json",
            "X-Contractor-API-Version": "contractor.public.v1",
          },
        },
      ),
  );
  return { api: new PublicAPI(runtimeConfig, fetch), fetch };
}

describe("Audit standard API", () => {
  it.each([
    ["owasp.asvs", "5.0.0+errata-1"],
    ["review-standard", "2026_rc.1"],
  ])("requests OpenAPI-valid identity %s@%s", async (scheme, version) => {
    const { api, fetch } = standardAPI(scheme, version);
    await expect(getAuditStandard(api, scheme, version)).resolves.toMatchObject(
      { reference: { scheme, version } },
    );
    expect(fetch).toHaveBeenCalledOnce();
  });

  it.each([
    ["OWASP", "1"],
    ["review_standard", "1"],
    ["1standard", "1"],
    ["review-standard", "+1"],
    ["review-standard", "1/2"],
  ])("rejects OpenAPI-invalid identity %s@%s", async (scheme, version) => {
    const { api, fetch } = standardAPI(scheme, version);
    await expect(getAuditStandard(api, scheme, version)).rejects.toThrow(
      "Audit standard identity is invalid",
    );
    expect(fetch).not.toHaveBeenCalled();
  });
});
