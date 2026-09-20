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
    const summary = "Awaiting owner acceptance.";
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              status: "proposed",
              machine: { certification: false },
              summary,
              summaryArtifact: {
                ref: {
                  namespace: "audit-proposed",
                  name: "report.md",
                  revision: "report-r1",
                },
                digest: `sha256:${"1".repeat(64)}`,
                mediaType: "text/markdown",
                sizeBytes: summary.length,
              },
              review: {
                requestId: "review-report",
                auditId: "audit_proposed",
                subjectKind: "audit-report",
                subjectId: "audit_proposed",
                kind: "report-acceptance",
                subjectRevision: 1,
                subjectDigest: `sha256:${"2".repeat(64)}`,
                state: "pending",
                revision: 1,
                requestedActions: ["approve", "reject"],
                createdAt: "2026-09-20T10:00:00Z",
                updatedAt: "2026-09-20T10:00:00Z",
              },
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
