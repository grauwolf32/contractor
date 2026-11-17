import { useQuery } from "@tanstack/react-query";

import { listAuditPresets } from "../../api/audit-presets";
import type { AuditProfile } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { compareVersions } from "../../app/format";
import { auditPresetLabel } from "../projects/audits/labels";
import { presentCheckType } from "../checks/start/check-types";

// Check types are AuditProfiles in the API (contract §3: "Check type").

/** Description derived from the published input contract. */
function presetScope(profile: AuditProfile): string {
  return `Up to ${profile.limits.max_tasks} tasks using ${Object.keys(profile.inputs).join(", ") || "selected materials"}.`;
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
    presetScope(profile),
    presentCheckType(profile).description,
    presentCheckType(profile).label,
  ]
    .join(" ")
    .toLowerCase();
}

interface CheckTypeFamily {
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
        compareVersions(right.ref.version, left.ref.version),
      ),
    }));
}

export function useAuditPresets(refetchOnMount = true) {
  const api = usePublicAPI();
  return useQuery({
    refetchOnMount,
    queryKey: queryKeys.catalog.auditPresets,
    queryFn: ({ signal }) => listAuditPresets(api, signal),
  });
}
