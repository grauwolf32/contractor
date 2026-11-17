import { collectAuditPages } from "./audit-collections";
import { listAuditProfiles, type AuditProfile } from "./audits";
import type { PublicAPI } from "./client";

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
  if (presets.truncated) {
    throw new Error(
      "The check type inventory has more entries than can be read at once, so it is incomplete.",
    );
  }
  return presets.items;
}

export function auditPresetPath(name: string, version: string): string {
  return `/catalog/audit-presets/${encodeURIComponent(name)}/${encodeURIComponent(version)}`;
}
