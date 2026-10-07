import { useQuery } from "@tanstack/react-query";

import { listAuditPresets } from "../../api/audit-presets";
import type { AuditProfile } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import type { components } from "../../api/generated/public";
import { queryKeys } from "../../api/query-keys";
import { capitalize, checkItemKind, itemNoun } from "../../app/vocabulary";
import { auditPresetLabel } from "../projects/audits/labels";
import { compareWorkflowVersions } from "../workflows/families";

// Check types are AuditProfiles in the API (contract §3: "Check type").

export const auditModeLabels: Record<AuditProfile["mode"], string> = {
  "risk-assessment": "Risk assessment",
  "requirements-verification": "Requirements verification",
  "custom-checklist": "Custom checklist",
  "operation-tracing": "Endpoint tracing",
  "finding-verification": "Possible issue verification",
};

/** What a check of this type works through, in one sentence. */
export function presetScope(profile: AuditProfile): string {
  if (profile.inventory.standardSelection)
    return profile.inventory.standardSelection.scope;
  if (
    profile.ref.name === "owasp-wstg-4-2-fast-source-review" ||
    profile.ref.name === "owasp-wstg-4-2-fast-active-http"
  )
    return `${profile.execution.maxItemsTotal} priority WSTG scenarios. Focused first pass; other scenarios are outside scope.`;
  switch (profile.inventory.implementation) {
    case "standard-mappings@1":
      return `${capitalize(itemNoun(checkItemKind(profile), 2))} defined by the referenced standard.`;
    case "checklist@1":
      return "Items from your uploaded checklist.";
    case "openapi-scans@1":
      return "Scanner tests for the endpoints you select in your OpenAPI document.";
    case "openapi-operations@1":
      return "One endpoint per operation in your OpenAPI document.";
    case "finding-candidates@1":
      return "Items from the finding candidates you supply.";
  }
}

/**
 * Whether the item list is fixed by a standard (and so can be listed here)
 * or built from the inputs of each check.
 */
export function hasFixedItems(profile: AuditProfile): boolean {
  return profile.inventory.implementation === "standard-mappings@1";
}

type CompatibilityReason = components["schemas"]["AuditCompatibilityReason"];

const COMPATIBILITY_REASONS: Readonly<Record<CompatibilityReason, string>> = {
  preparation_unsupported:
    "This server cannot run the preparation step this type needs yet.",
  discovery_unsupported: "This server cannot run the discovery step.",
  assessment_unsupported: "This server cannot run the assessment step.",
  multiple_rounds_unsupported:
    "This type needs more than one round, which this server does not run.",
  batching_unsupported: "This server cannot run items in batches.",
  automatic_active_checks_unsupported:
    "This type runs active tests without asking, which this server does not allow.",
  active_check_approval_unsupported:
    "This server cannot ask you to approve active tests.",
  finding_confirmation_unsupported:
    "Its workflows can propose possible issues, but this type does not let you confirm them.",
  manual_applicability_unsupported:
    "This server cannot ask you whether a requirement applies.",
  report_acceptance_unsupported:
    "This server cannot ask you to accept the report.",
  manual_item_unsupported: "This server cannot run manual items.",
};

/** One readable sentence for a reason the server cannot run a check type. */
export function compatibilityReasonText(reason: string): string {
  return Object.hasOwn(COMPATIBILITY_REASONS, reason)
    ? COMPATIBILITY_REASONS[reason as CompatibilityReason]
    : capitalize(reason.replaceAll("_", " "));
}

/** The Start page with this check type chosen (contract §5). */
export function startCheckPath(name: string): string {
  return `/checks/new?${new URLSearchParams({ type: name }).toString()}`;
}

/** Text a check type is found by: names, version, mode, scope, standards. */
export function checkTypeSearchText(profile: AuditProfile): string {
  return [
    profile.ref.name,
    profile.ref.version,
    auditPresetLabel(profile.ref.name),
    profile.mode,
    auditModeLabels[profile.mode],
    presetScope(profile),
    ...profile.standards.map(
      (standard) => `${standard.scheme}@${standard.version}`,
    ),
  ]
    .join(" ")
    .toLowerCase();
}

export interface CheckTypeFamily {
  name: string;
  /** Newest version first. */
  versions: AuditProfile[];
}

/** Versions grouped by check type name, names in order, newest first. */
export function groupCheckTypes(
  profiles: readonly AuditProfile[],
): CheckTypeFamily[] {
  const families = new Map<string, AuditProfile[]>();
  for (const profile of profiles) {
    const versions = families.get(profile.ref.name) ?? [];
    versions.push(profile);
    families.set(profile.ref.name, versions);
  }
  return [...families]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, versions]) => ({
      name,
      versions: versions.sort((left, right) =>
        compareWorkflowVersions(right.ref.version, left.ref.version),
      ),
    }));
}

export function useAuditPresets() {
  const api = usePublicAPI();
  return useQuery({
    queryKey: queryKeys.catalog.auditPresets,
    queryFn: ({ signal }) => listAuditPresets(api, signal),
  });
}
