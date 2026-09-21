import { listAuditProfiles, type AuditProfile } from "./audits";
import { collectAuditPages } from "./audit-collections";
import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import type { components } from "./generated/public";
import { CONFIG_ID_PATTERN, CONFIG_VERSION_PATTERN } from "./workflows";

export type AuditStandard = components["schemas"]["AuditStandardPackage"];

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
    !CONFIG_ID_PATTERN.test(scheme) ||
    !CONFIG_VERSION_PATTERN.test(version)
  ) {
    throw new TypeError("Audit standard identity is invalid");
  }
  const result = await api.request((client) =>
    client.GET("/v1/audit-standards/{scheme}/versions/{version}", {
      params: { path: { scheme, version } },
      ...(signal === undefined ? {} : { signal }),
    }),
  );
  if (result.data === undefined)
    throw publicAPIError(result.response.status, result.error);
  const standard = result.data.standard;
  if (
    standard?.reference?.scheme !== scheme ||
    standard.reference.version !== version ||
    !/^sha256:[a-f0-9]{64}$/.test(standard.digest) ||
    typeof standard.title !== "string" ||
    !standard.source ||
    !standard.license ||
    !["full", "identifiers", "metadata"].includes(
      standard.license.disclosure,
    ) ||
    !Number.isSafeInteger(standard.entryCount) ||
    standard.entryCount < 0 ||
    !Number.isSafeInteger(standard.mappingCount) ||
    standard.mappingCount < 0 ||
    !Number.isSafeInteger(standard.evidenceContractCount) ||
    standard.evidenceContractCount < 0 ||
    (standard.entries !== undefined && !Array.isArray(standard.entries)) ||
    (standard.mappings !== undefined && !Array.isArray(standard.mappings)) ||
    (standard.evidenceContracts !== undefined &&
      !Array.isArray(standard.evidenceContracts)) ||
    (standard.license.disclosure !== "metadata" &&
      (standard.entries?.length ?? 0) !== standard.entryCount) ||
    (standard.license.disclosure === "full" &&
      ((standard.mappings?.length ?? 0) !== standard.mappingCount ||
        (standard.evidenceContracts?.length ?? 0) !==
          standard.evidenceContractCount))
  ) {
    throw new PublicAPIError({
      status: result.response.status,
      code: "invalid_api_response",
      message: "Server returned an invalid audit standard response",
    });
  }
  return structuredClone(standard);
}

export function auditPresetPath(name: string, version: string): string {
  return `/catalog/audit-presets/${encodeURIComponent(name)}/${encodeURIComponent(version)}`;
}
