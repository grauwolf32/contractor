import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import { getAuditReport } from "./audits";
import { PublicAPI } from "./client";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

describe("Audit report API", () => {
  it("accepts the proposed state returned during exact human review", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              status: "proposed",
              machine: { certification: false },
              summary: "Awaiting owner acceptance.",
            }),
            {
              status: 200,
              headers: {
                "content-type": "application/json",
                "x-contractor-api-version": "contractor.public.v1",
              },
            },
          ),
      ),
    );

    await expect(getAuditReport(api, "audit_proposed")).resolves.toMatchObject({
      status: "proposed",
      machine: { certification: false },
    });
  });
});
