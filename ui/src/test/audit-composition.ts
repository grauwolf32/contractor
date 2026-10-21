import type { Audit, AuditCoverageRow, AuditItem } from "../api/audits";
import {
  audit,
  auditInput,
  auditItem,
  conforming,
  DIGEST,
  NOW,
} from "./builders";

export const COMPOSITION_PROJECT = "project_composition";
export const COMPOSITION_AUDIT = "audit_composition";
export const COMPOSITION_ROOT = `/projects/${COMPOSITION_PROJECT}/audits/${COMPOSITION_AUDIT}`;

const source = auditInput("service-source");
const settings = auditInput("scan-settings", {
  mediaType: "application/json",
  sizeBytes: 128,
});
const openapi = auditInput("prepared-openapi", {
  ref: { namespace: "audit-composition", name: "openapi", revision: "api-r2" },
  mediaType: "application/json",
  sizeBytes: 128,
});
const worklist = auditInput("worklist", { mediaType: "application/zip" });
const snapshot = auditInput("initial-inventory", {
  mediaType: "application/json",
});
const result = auditInput("scan-result", {
  ref: {
    namespace: "audit-composition",
    name: "scan-result",
    revision: "result-r1",
  },
});

export function compositionAudit(
  stage: "draft" | "preparing" | "inventory" | "rounds" | "failed",
): Audit {
  const accepted = stage === "inventory" || stage === "rounds";
  return audit(
    COMPOSITION_PROJECT,
    COMPOSITION_AUDIT,
    stage === "draft" ? "draft" : stage === "failed" ? "failed" : "active",
    {
      profile: {
        name: "source-openapi-sqlmap-scan",
        version: "1",
        digest: DIGEST,
      },
      inputs: { source, settings },
      scope: {
        objective: "Inspect the shop API request checks",
        target: "http://shop.test",
      },
      revision:
        stage === "draft"
          ? 1
          : stage === "preparing"
            ? 2
            : stage === "inventory"
              ? 3
              : stage === "rounds"
                ? 4
                : 3,
      phase:
        stage === "draft"
          ? "not-started"
          : stage === "failed"
            ? "preparing"
            : stage,
      currentRoundId: stage === "rounds" ? "round_1" : undefined,
      baseline:
        stage === "draft"
          ? undefined
          : {
              activeChecks: "approval-required",
              inputs: { source, settings },
              scope: {},
              runtimeLabels: [],
              skills: [],
              standards: [],
              runtimeConfig: {
                default: {
                  label: "default",
                  explicit: false,
                  bindingRevision: 1,
                  config: {
                    name: "runtime-test",
                    version: "1",
                    digest: DIGEST,
                  },
                },
                labels: [],
              },
            },
      preparation:
        stage === "draft"
          ? undefined
          : {
              roles: {
                "generate-api": {
                  status: accepted
                    ? "accepted"
                    : stage === "failed"
                      ? "failed"
                      : "running",
                  attempts: 1,
                  maxAttempts: 2,
                  executionId: "prepare-exec",
                  runId: "prepare-run",
                  outputs: accepted
                    ? {
                        api: {
                          artifact: openapi,
                          executionId: "prepare-exec",
                          runId: "prepare-run",
                          workflowOutput: "openapi",
                        },
                      }
                    : {},
                },
              },
            },
      generatedInventory:
        stage !== "rounds"
          ? undefined
          : {
              snapshot,
              worklist,
              sources: { source: openapi, settings },
              itemCount: 3,
              sourceContentDigest: DIGEST,
              canonicalInventoryDigest: DIGEST,
              gaps: [],
            },
      stopReason:
        stage !== "failed"
          ? undefined
          : {
              code: "preparation_failed",
              message: "Generated API spec failed validation.",
            },
      outstandingRunCount: stage === "preparing" ? 1 : 0,
    },
  );
}

export function compositionRows(
  scanner: "sqlmap" | "nuclei" = "sqlmap",
): AuditCoverageRow[] {
  const method =
    scanner === "sqlmap" ? "sqlmap-request-scan" : "nuclei-url-template-scan";
  return ["/orders", "/users/{id}", "/health"].map((path, ordinal) =>
    conforming("AuditCoverageRow", {
      roundId: "round_1",
      itemId: `item_${ordinal}`,
      ordinal,
      itemKey: `scan-${"a".repeat(63)}${ordinal}`,
      subjectKey: `scan-${"a".repeat(63)}${ordinal}`,
      coverage: {
        status: ordinal === 0 ? "inconclusive" : "not-tested",
        requested: [method],
        completed: ordinal === 0 ? [method] : [],
        gaps: [],
      },
      ...(ordinal === 0 ? { result } : {}),
      details: {
        objective: `GET ${path}`,
        methods: [method],
        taskDocument: { schema: "contractor.audit.item-task.v1" },
        scan: {
          scanner,
          operation: `#/paths/${path.replaceAll("~", "~0").replaceAll("/", "~1")}/get`,
          source: openapi,
          settings,
          preparationDigest: DIGEST,
          ...(scanner === "sqlmap"
            ? { requestDigest: DIGEST }
            : { targetUrl: `http://shop.test${path.replace("{id}", "1")}` }),
          runnable: true,
          testParameters: scanner === "sqlmap" ? ["q"] : [],
          gaps: [],
        },
        ...(ordinal === 0
          ? {
              resultSummary:
                "Scan completed; the observation still requires verification.",
            }
          : {}),
        evidence:
          ordinal === 0
            ? [
                {
                  id: "scanner-report",
                  kind: "scanner-report",
                  summary: "Saved scanner output",
                  contentMemberId: "scanner-report",
                },
              ]
            : [],
      },
      updatedAt: NOW,
    }),
  );
}

export function compositionItems(): AuditItem[] {
  return compositionRows().map((row, ordinal) =>
    auditItem(row.itemId, ordinal, "openapi-scan", {
      itemKey: row.itemKey,
      subjectKey: row.subjectKey,
      approvalKind: ordinal === 1 ? "active-check-approval" : "none",
      state:
        ordinal === 0 ? "settled" : ordinal === 1 ? "awaiting_review" : "ready",
      ...(ordinal === 0
        ? {
            acceptedResult: result,
            lastExecutionItemId: "scan-execution-item",
            finalDisposition: "accepted-result",
          }
        : {}),
    }),
  );
}
