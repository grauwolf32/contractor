import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import type {
  Audit,
  AuditProfile,
  AuditState,
  FindingSeverity,
  FindingState,
} from "../api/audits";
import { STATUS_TONES } from "../test/status-tones";
import {
  capitalize,
  CHECK_STATE_LABELS,
  checkItemKind,
  checkStateLabel,
  FINDING_STATE_LABELS,
  findingStateLabel,
  itemCount,
  readableValue,
  reportStatusLabel,
  RUN_STATE_LABELS,
  runRecoveryReason,
  runStateLabel,
  runStatePhrase,
  SEVERITY_LABELS,
  severityLabel,
  TERMS,
  VERDICT_LABELS,
  verdictLabel,
  type VocabularyLabel,
} from "./vocabulary";

type ItemKind = ReturnType<typeof checkItemKind>;

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
    ["FindingState", FINDING_STATE_LABELS, enumOf("FindingState")],
    [
      "FindingVerdict",
      VERDICT_LABELS,
      [...enumOf("FindingVerdict"), "unreviewed"],
    ],
    ["WorkflowRunState", RUN_STATE_LABELS, enumOf("WorkflowRunState")],
  ])("labels every %s value with a tone", (_schema, table, values) => {
    expect(Object.keys(table).sort()).toEqual([...values].sort());
    for (const value of values) expectLabel(table[value]!);
  });

  // Tables only their lookup reads are typed Record<enum, …> over the
  // generated types, which keeps their keys equal to the contract's enum at
  // compile time. At run time each value must come from the table:
  // labelOf returns the table's own entry, while a value missing from it
  // gets a new fallback object on every call.
  it.each<[string, (value: string) => VocabularyLabel, string[]]>([
    [
      "AuditReportStatus",
      (value) =>
        reportStatusLabel(value as Parameters<typeof reportStatusLabel>[0]),
      enumOf("AuditReportStatus"),
    ],
  ])("labels every %s value from its table", (_schema, lookup, values) => {
    for (const value of values) {
      expect(lookup(value), value).toBe(lookup(value));
      expectLabel(lookup(value));
    }
    expect(lookup("unknown_value")).not.toBe(lookup("unknown_value"));
  });

  it("labels every severity", () => {
    const severities = enumOf("FindingSeverity");
    expect(Object.keys(SEVERITY_LABELS).sort()).toEqual([...severities].sort());
    for (const severity of severities)
      expect(severityLabel(severity as FindingSeverity)).toBe(
        capitalize(severity),
      );
  });

  it("looks labels up through the functions", () => {
    expect(findingStateLabel("needs-evidence")).toBe(
      FINDING_STATE_LABELS["needs-evidence"],
    );
  });

  it("matches the contract tables", () => {
    const states: Record<AuditState, [string, string]> = {
      draft: ["Draft", "idle"],
      active: ["Running", "progress"],
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

    const findings: Record<FindingState, [string, string]> = {
      proposed: ["Needs review", "review"],
      confirmed: ["Confirmed", "success"],
      rejected: ["Not an issue", "neutral"],
      duplicate: ["Duplicate", "neutral"],
      "needs-evidence": ["Needs evidence", "warning"],
    };
    for (const [state, [label, tone]] of Object.entries(findings))
      expect(findingStateLabel(state as FindingState)).toEqual({
        label,
        tone,
      });
  });

  it("names verdicts and severities", () => {
    expect(verdictLabel("true_positive").label).toBe("Confirmed");
    expect(verdictLabel("false_positive").label).toBe("Not an issue");
    expect(verdictLabel("needs_evidence").label).toBe("Needs evidence");
    expect(verdictLabel("duplicate").label).toBe("Duplicate");
    expect(verdictLabel("reopen").label).toBe("Reopened");
    expect(verdictLabel("unreviewed").label).toBe("Not reviewed");
    expect(verdictLabel(undefined).label).toBe("Not reviewed");
    expect(verdictLabel(null).label).toBe("Not reviewed");

    expect(severityLabel("informational")).toBe("Informational");
    expect(severityLabel("critical")).toBe("Critical");
    expect(severityLabel("")).toBe("Not set");
    expect(severityLabel(undefined)).toBe("Not set");
    expect(severityLabel(null)).toBe("Not set");
  });

  it("labels retained reports as ready", () => {
    expect(reportStatusLabel("ready").label).toBe("Ready");
  });

  it("keeps unknown Server values readable", () => {
    expect(checkStateLabel("on_hold" as AuditState)).toEqual({
      label: "On hold",
      tone: "neutral",
    });
    expect(findingStateLabel("toString" as FindingState)).toEqual({
      label: "ToString",
      tone: "neutral",
    });
    expect(severityLabel("severe" as FindingSeverity)).toBe("Severe");
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
    expect(itemCount(kind, count)).toBe(`${count} ${noun}`);
  });

  it("counts items", () => {
    expect(itemCount("endpoint", 1)).toBe("1 endpoint");
    expect(itemCount("requirement", 1204)).toBe("1,204 requirements");
  });
});

describe("readable values", () => {
  it("reads an API value as words, also one this client does not know", () => {
    expect(readableValue("release_pending")).toBe("Release pending");
    expect(readableValue("not-eligible")).toBe("Not eligible");
    expect(readableValue("__")).toBe("Unknown");
  });
});

describe("checkItemKind", () => {
  const digest = `sha256:${"1".repeat(64)}`;

  function profile(name: string): Pick<AuditProfile, "ref"> {
    return { ref: { name, version: "1", digest } };
  }

  function check(
    name: string,
    baseline?: { schemes?: string[] },
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
              activeChecks: "automatic",
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
            },
          }),
    };
  }

  it.each<[string, ItemKind]>([
    ["openapi-operation-trace", "endpoint"],
    ["openapi-sqlmap-scan", "endpoint"],
    ["web-scan", "item"],
    ["custom", "item"],
  ])("names published %s as %s", (name, kind) =>
    expect(checkItemKind(profile(name))).toBe(kind),
  );

  it("names a started check by its pinned standards", () => {
    expect(
      checkItemKind(
        check("owasp-wstg-4-2-source-review", { schemes: ["owasp-wstg"] }),
      ),
    ).toBe("scenario");
    expect(
      checkItemKind(check("custom-review", { schemes: ["owasp-asvs"] })),
    ).toBe("requirement");
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

  // V385-001 A5: Runs, Projects and the Inbox read one Run vocabulary; the
  // words are the API states (v3b-build-contract.md §3, Run unchanged).
  it("words and tones of Run states", () => {
    expect(
      Object.fromEntries(
        Object.entries(RUN_STATE_LABELS).map(([state, { label, tone }]) => [
          state,
          `${label}/${tone}`,
        ]),
      ),
    ).toEqual({
      initializing: "Initializing/progress",
      pending: "Pending/idle",
      waiting: "Waiting/warning",
      running: "Running/progress",
      cancelling: "Cancelling/warning",
      succeeded: "Succeeded/done",
      failed: "Failed/blocked",
      cancelled: "Cancelled/neutral",
    });
    expect(runStatePhrase("waiting")).toBe("is waiting");
    expect(runStatePhrase("cancelled")).toBe("was cancelled");
    const later = "paused" as Parameters<typeof runStateLabel>[0];
    expect(runStateLabel(later)).toEqual({ label: "Paused", tone: "neutral" });
    expect(runStatePhrase(later)).toBe("is paused");
  });

  it("explains every model recovery cause of a waiting Run", () => {
    const codes = enumOf("RunRecovery", "code");
    for (const code of codes)
      expect(
        runRecoveryReason(code as Parameters<typeof runRecoveryReason>[0]),
      ).not.toBe("The model connection is interrupted.");
    expect(
      runRecoveryReason(
        "gateway_lost" as Parameters<typeof runRecoveryReason>[0],
      ),
    ).toBe("The model connection is interrupted.");
  });
});
