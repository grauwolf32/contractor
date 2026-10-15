import { listAuditProfiles, type AuditProfile } from "./audits";
import { collectAuditPages } from "./audit-collections";
import type { PublicAPI } from "./client";
import { stringRule } from "./contract";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";

export type AuditStandard = components["schemas"]["AuditStandardPackage"];

// The OpenAPI AuditStandardRef identity rules.
export const AUDIT_STANDARD_SCHEME_PATTERN = stringRule(
  "AuditStandardRef",
  "properties",
  "scheme",
);
export const AUDIT_STANDARD_VERSION_PATTERN = stringRule(
  "AuditStandardRef",
  "properties",
  "version",
);

// The profile endpoint has no search filter. Read the bounded batch of pages
// before searching or grouping versions so presets on later pages remain
// discoverable; the catalog is far smaller than the page cap.
export async function listAuditPresets(
  api: PublicAPI,
  signal: AbortSignal,
): Promise<AuditProfile[]> {
  const presets = await collectAuditPages((cursor) => {
    signal.throwIfAborted();
    return listAuditProfiles(api, {
      signal,
      ...(cursor === undefined ? {} : { cursor }),
    });
  });
  return presets.items;
}

export async function getAuditStandard(
  api: PublicAPI,
  scheme: string,
  version: string,
  signal?: AbortSignal,
): Promise<AuditStandard> {
  if (
    !AUDIT_STANDARD_SCHEME_PATTERN.test(scheme) ||
    !AUDIT_STANDARD_VERSION_PATTERN.test(version)
  ) {
    throw new TypeError("Audit standard identity is invalid");
  }
  const result = await api.request((client) =>
    client.GET("/v1/audit-standards/{scheme}/versions/{version}", {
      params: { path: { scheme, version } },
      ...(signal === undefined ? {} : { signal }),
    }),
  );
  const standard = requireData(result).standard;
  // Cross-field: the requested standard, listing every entry it may disclose
  // and, with full disclosure, every mapping and evidence contract.
  if (
    standard.reference.scheme !== scheme ||
    standard.reference.version !== version ||
    (standard.license.disclosure !== "metadata" &&
      (standard.entries?.length ?? 0) !== standard.entryCount) ||
    (standard.license.disclosure === "full" &&
      ((standard.mappings?.length ?? 0) !== standard.mappingCount ||
        (standard.evidenceContracts?.length ?? 0) !==
          standard.evidenceContractCount))
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an invalid audit standard response",
    );
  }
  return standard;
}

export function auditPresetPath(name: string, version: string): string {
  return `/catalog/audit-presets/${encodeURIComponent(name)}/${encodeURIComponent(version)}`;
}
