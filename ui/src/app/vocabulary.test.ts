import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import type {
  Audit,
  AuditFindingSeverity,
  AuditFindingState,
  AuditItem,
  AuditProfile,
  AuditState,
} from "../api/audits";
import { STATUS_TONES } from "./status-tone";
import {
  capitalize,
  CHECK_STATE_LABELS,
  checkItemKind,
  checkStateLabel,
  COVERAGE_STATUS_LABELS,
  coverageStatusLabel,
  FINDING_STATE_LABELS,
  findingStateLabel,
  itemCount,
  itemNoun,
  REPORT_STATUS_LABELS,
  reportStatusLabel,
  REVIEW_ACTION_LABELS,
  REVIEW_KIND_LABELS,
  REVIEW_STATE_LABELS,
  reviewActionLabel,
  reviewKindLabel,
  reviewStateLabel,
  SEVERITY_LABELS,
  severityLabel,
  TERMS,
  VERDICT_LABELS,
  verdictLabel,
  type AuditCoverageStatus,
  type AuditReviewKind,
  type ItemKind,
  type VocabularyLabel,
} from "./vocabulary";

// The generated types come from this document; reading its enums keeps the
// tables complete at run time as well as at compile time.
const openapi = parse(
  readFileSync("../api/openapi/contractor-public-v1.yaml", "utf8"),
) as {
  components: {
    schemas: Record<
      string,
      { enum?: string[]; properties?: Record<string, { enum?: string[] }> }
    >;
  };
};

function enumOf(schema: string, property?: string): string[] {
  const definition = openapi.components.schemas[schema];
  const values =
    property === undefined
      ? definition?.enum
      : definition?.properties?.[property]?.enum;
  if (values === undefined || values.length === 0)
    throw new Error(`OpenAPI has no enum at ${schema} ${property ?? ""}`);
  return values;
}

function expectLabel(entry: VocabularyLabel): void {
  expect(entry.label.trim()).not.toBe("");
  expect(STATUS_TONES).toContain(entry.tone);
}

describe("vocabulary", () => {
  it.each<[string, Readonly<Record<string, VocabularyLabel>>, string[]]>([
    ["AuditState", CHECK_STATE_LABELS, enumOf("AuditState")],
    [
      "AuditCoverageStatus",
      COVERAGE_STATUS_LABELS,
      enumOf("AuditCoverageStatus"),
    ],
    ["AuditFindingState", FINDING_STATE_LABELS, enumOf("AuditFindingState")],
    [
      "AuditAnalystVerdict",
      VERDICT_LABELS,
      [...enumOf("AuditAnalystVerdict"), "unreviewed"],
    ],
    ["AuditReviewAction", REVIEW_ACTION_LABELS, enumOf("AuditReviewAction")],
    ["AuditReviewState", REVIEW_STATE_LABELS, enumOf("AuditReviewState")],
    ["AuditReportStatus", REPORT_STATUS_LABELS, enumOf("AuditReportStatus")],
  ])("labels every %s value with a tone", (_schema, table, values) => {
    expect(Object.keys(table).sort()).toEqual([...values].sort());
    for (const value of values) expectLabel(table[value]!);
  });

  it("labels every review kind and severity", () => {
    const kinds = enumOf("AuditReviewRequest", "kind");
    expect(Object.keys(REVIEW_KIND_LABELS).sort()).toEqual([...kinds].sort());
    for (const kind of kinds)
      expect(reviewKindLabel(kind as AuditReviewKind).trim()).not.toBe("");
    const severities = enumOf("AuditFindingSeverity");
    expect(Object.keys(SEVERITY_LABELS).sort()).toEqual([...severities].sort());
    for (const severity of severities)
      expect(severityLabel(severity as AuditFindingSeverity)).toBe(
        capitalize(severity),
      );
  });

  it("looks labels up through the functions", () => {
    expect(checkStateLabel("waiting_review")).toBe(
      CHECK_STATE_LABELS.waiting_review,
    );
    expect(coverageStatusLabel("traced-partial")).toBe(
      COVERAGE_STATUS_LABELS["traced-partial"],
    );
    expect(findingStateLabel("needs-evidence")).toBe(
      FINDING_STATE_LABELS["needs-evidence"],
    );
    expect(reviewStateLabel("pending")).toEqual({
      label: "Waiting for you",
      tone: "review",
    });
  });

  it("matches the contract tables", () => {
    const states: Record<AuditState, [string, string]> = {
      draft: ["Draft", "idle"],
      active: ["Running", "progress"],
      waiting_review: ["Waiting for you", "review"],
      paused: ["Paused", "warning"],
      finalizing: ["Finishing", "progress"],
      cancelling: ["Stopping", "warning"],
      completed: ["Finished", "done"],
      cancelled: ["Stopped", "neutral"],
      failed: ["Failed", "blocked"],
      deleting: ["Deleting", "neutral"],
    };
    for (const [state, [label, tone]] of Object.entries(states))
      expect(checkStateLabel(state as AuditState)).toEqual({ label, tone });

    const coverage: Record<AuditCoverageStatus, [string, string]> = {
      "not-tested": ["Not checked yet", "idle"],
      inconclusive: ["Inconclusive", "warning"],
      satisfied: ["Met", "done"],
      violated: ["Issue found", "blocked"],
      "not-applicable": ["Not applicable", "neutral"],
      blocked: ["Blocked", "blocked"],
      excluded: ["Excluded", "neutral"],
      "traced-complete": ["Fully traced", "done"],
      "traced-partial": ["Partially traced", "partial"],
      unmapped: ["Unmapped", "neutral"],
    };
    for (const [status, [label, tone]] of Object.entries(coverage))
      expect(coverageStatusLabel(status as AuditCoverageStatus)).toEqual({
        label,
        tone,
      });

    const findings: Record<AuditFindingState, [string, string]> = {
      proposed: ["Needs review", "review"],
      confirmed: ["Confirmed", "success"],
      rejected: ["Not an issue", "neutral"],
      duplicate: ["Duplicate", "neutral"],
      "needs-evidence": ["Needs evidence", "warning"],
    };
    for (const [state, [label, tone]] of Object.entries(findings))
      expect(findingStateLabel(state as AuditFindingState)).toEqual({
        label,
        tone,
      });
  });

  it("names verdicts, actions, review kinds and severities", () => {
    expect(verdictLabel("true_positive").label).toBe("Confirmed");
    expect(verdictLabel("false_positive").label).toBe("Not an issue");
    expect(verdictLabel("needs_evidence").label).toBe("Needs evidence");
    expect(verdictLabel("duplicate").label).toBe("Duplicate");
    expect(verdictLabel("reopen").label).toBe("Reopened");
    expect(verdictLabel("unreviewed").label).toBe("Not reviewed");
    expect(verdictLabel(undefined).label).toBe("Not reviewed");
    expect(verdictLabel(null).label).toBe("Not reviewed");

    expect(reviewActionLabel("approve").label).toBe("Approved");
    expect(reviewActionLabel("reject").label).toBe("Rejected");
    expect(reviewActionLabel("not_applicable").label).toBe("Not applicable");

    expect(reviewKindLabel("finding-triage")).toBe("Possible issue");
    expect(reviewKindLabel("active-check-approval")).toBe(
      "Active test approval",
    );
    expect(reviewKindLabel("requirement-applicability")).toBe(
      "Requirement applicability",
    );
    expect(reviewKindLabel("report-acceptance")).toBe("Report acceptance");

    expect(severityLabel("informational")).toBe("Informational");
    expect(severityLabel("critical")).toBe("Critical");
    expect(severityLabel("")).toBe("Not set");
    expect(severityLabel(undefined)).toBe("Not set");
    expect(severityLabel(null)).toBe("Not set");
  });

  it("never reads a proposed report as accepted", () => {
    expect(reportStatusLabel("proposed").label).not.toMatch(/ready|accepted/i);
    expect(reportStatusLabel("ready").label).toBe("Ready");
  });

  it("keeps unknown Server values readable", () => {
    expect(checkStateLabel("on_hold" as AuditState)).toEqual({
      label: "On hold",
      tone: "neutral",
    });
    expect(findingStateLabel("toString" as AuditFindingState)).toEqual({
      label: "ToString",
      tone: "neutral",
    });
    expect(reviewKindLabel("budget-approval" as AuditReviewKind)).toBe(
      "Budget approval",
    );
    expect(severityLabel("severe" as AuditFindingSeverity)).toBe("Severe");
  });

  it("shares term spelling", () => {
    expect(TERMS.possibleIssue).toBe("possible issue");
    expect(TERMS.checkType).toBe("check type");
    expect(TERMS.material).toBe("material");
    expect(TERMS.file).toBe("file");
    expect(TERMS.inbox).toBe("Inbox");
    expect(TERMS.library).toBe("Library");
    expect(capitalize(TERMS.possibleIssues)).toBe("Possible issues");
  });

  it.each<[ItemKind, number, string]>([
    ["endpoint", 1, "endpoint"],
    ["endpoint", 0, "endpoints"],
    ["endpoint", 5, "endpoints"],
    ["requirement", 1, "requirement"],
    ["requirement", 94, "requirements"],
    ["scenario", 1, "scenario"],
    ["scenario", 2, "scenarios"],
    ["item", 1, "item"],
    ["item", 3, "items"],
  ])("names %s for a count of %i", (kind, count, noun) => {
    expect(itemNoun(kind, count)).toBe(noun);
  });

  it("counts items", () => {
    expect(itemCount("endpoint", 1)).toBe("1 endpoint");
    expect(itemCount("requirement", 1204)).toBe("1,204 requirements");
  });
});

describe("checkItemKind", () => {
  const digest = `sha256:${"1".repeat(64)}`;

  function profile(
    implementation: AuditProfile["inventory"]["implementation"],
    schemes: string[] = [],
  ): Pick<AuditProfile, "inventory" | "standards"> {
    return {
      inventory:
        implementation === "standard-mappings@1"
          ? { implementation, itemWorkflowRole: "check" }
          : {
              implementation,
              source: { source: "audit-input", name: "source" },
              itemWorkflowRole: "check",
            },
      standards: schemes.map((scheme) => ({ scheme, version: "1" })),
    };
  }

  function item(
    kind: string,
    scheme?: string,
  ): Pick<AuditItem, "kind" | "origin"> {
    return {
      kind,
      origin: {
        schema: "contractor.audit.item-origin.v1",
        sourceRef: { namespace: "sources", name: "inventory" },
        sourceContentDigest: digest,
        sourceMediaType: "application/json",
        canonicalInventoryDigest: digest,
        entryKey: "entry",
        ...(scheme === undefined
          ? {}
          : {
              standard: {
                scheme,
                version: "1",
                mappingKey: "mapping",
                entryIds: ["A01"],
                evidenceContract: { id: "contract", version: "1" },
              },
            }),
      },
    };
  }

  function check(
    name: string,
    baseline?: { schemes?: string[]; selection?: boolean },
  ): Pick<Audit, "profile" | "baseline"> {
    const exact = {
      ref: { namespace: "inputs", name: "source", revision: "r1" },
      digest,
    };
    return {
      profile: { name, version: "1", digest },
      ...(baseline === undefined
        ? {}
        : {
            baseline: {
              inputs: {},
              scope: {},
              runtimeLabels: [],
              runtimeConfig: {
                default: {
                  label: "default",
                  explicit: false,
                  bindingRevision: 1,
                  config: { name: "runtime", version: "1", digest },
                },
                labels: [],
              },
              skills: [],
              standards: (baseline.schemes ?? []).map((scheme) => ({
                reference: { scheme, version: "1" },
                title: scheme,
                source: { name: scheme, url: "https://example.test/" },
                license: {
                  id: "CC-BY-SA-4.0" as const,
                  url: "https://example.test/license",
                  attribution: "OWASP",
                  disclosure: "full" as const,
                },
                catalog: {
                  artifact: exact.ref,
                  digest,
                  mediaType:
                    "application/vnd.contractor.audit-standard+zip" as const,
                  sizeBytes: 1,
                },
                retained: {
                  artifact: exact.ref,
                  digest,
                  mediaType:
                    "application/vnd.contractor.audit-standard+zip" as const,
                  sizeBytes: 1,
                },
              })),
              ...(baseline.selection === true
                ? {
                    inventory: {
                      sourceContentDigest: digest,
                      canonicalInventoryDigest: digest,
                      standardSelection: {
                        scope: "Level 1",
                        levels: ["1"],
                        entryIds: [],
                      },
                      gaps: [],
                      worklist: exact,
                    },
                  }
                : {}),
            },
          }),
    };
  }

  it.each<[string, ItemKind]>([
    ["operation-trace", "endpoint"],
    ["openapi-scan", "endpoint"],
    ["checklist", "item"],
    ["finding-verification", "item"],
    ["something-new", "item"],
  ])("names a %s work item an %s", (kind, expected) => {
    expect(checkItemKind(item(kind))).toBe(expected);
  });

  it("names standard work items by their standard", () => {
    expect(checkItemKind(item("standard-mapping", "owasp-wstg"))).toBe(
      "scenario",
    );
    expect(checkItemKind(item("standard-mapping", "owasp-asvs"))).toBe(
      "requirement",
    );
    expect(checkItemKind(item("standard-mapping", "owasp-web-top10"))).toBe(
      "requirement",
    );
    expect(checkItemKind(item("standard-mapping"))).toBe("requirement");
  });

  it.each<[AuditProfile["inventory"]["implementation"], string[], ItemKind]>([
    ["openapi-operations@1", [], "endpoint"],
    ["openapi-scans@1", [], "endpoint"],
    ["standard-mappings@1", ["owasp-wstg"], "scenario"],
    ["standard-mappings@1", ["owasp-asvs"], "requirement"],
    ["standard-mappings@1", ["owasp-web-top10"], "requirement"],
    ["standard-mappings@1", ["owasp-wstg", "owasp-asvs"], "requirement"],
    ["standard-mappings@1", [], "requirement"],
    ["checklist@1", [], "item"],
    ["finding-candidates@1", [], "item"],
  ])("names %s check types with %j standards: %s", (impl, schemes, kind) => {
    expect(checkItemKind(profile(impl, schemes))).toBe(kind);
  });

  it("names a started check by its pinned standards", () => {
    expect(
      checkItemKind(
        check("owasp-wstg-4-2-source-review", { schemes: ["owasp-wstg"] }),
      ),
    ).toBe("scenario");
    expect(
      checkItemKind(check("custom-review", { schemes: ["owasp-asvs"] })),
    ).toBe("requirement");
    expect(checkItemKind(check("custom-review", { selection: true }))).toBe(
      "requirement",
    );
  });

  it.each<[string, ItemKind]>([
    ["openapi-operation-trace", "endpoint"],
    ["openapi-nuclei-scan", "endpoint"],
    ["owasp-wstg-4-2-fast-active-http", "scenario"],
    ["owasp-asvs-5-0-l1-source-review", "requirement"],
    ["owasp-top10-2025-source-risk", "requirement"],
    ["source-checklist", "item"],
    ["custom-review", "item"],
  ])("names a draft %s check by its check type: %s", (name, kind) => {
    expect(checkItemKind(check(name))).toBe(kind);
    // A baseline without standards falls back to the same rule.
    expect(checkItemKind(check(name, {}))).toBe(kind);
  });
});
