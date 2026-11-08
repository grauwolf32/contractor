import type { PublishedFindingExport } from "../api/finding-exports";
import { sha256Hex } from "../app/digest";

export const EXPORT_BYTES = '{"version":"2.1.0","runs":[]}\n';

export async function publishedExport(
  selectedCount = 2,
  resultCount = 1,
): Promise<PublishedFindingExport> {
  const bytes = new TextEncoder().encode(EXPORT_BYTES);
  const digest = `sha256:${await sha256Hex(bytes)}`;
  const exact = (name: string) => ({
    ref: { namespace: "audit-fixture", name, revision: "r1" },
    digest,
    mediaType: "application/json",
    sizeBytes: bytes.byteLength,
  });
  return {
    exportId: `export-${"a".repeat(64)}`,
    snapshot: exact("snapshot"),
    artifact: exact("sarif"),
    diagnostics: exact("diagnostics"),
    snapshotAt: "2026-10-30T12:00:00Z",
    selectedCount,
    resultCount,
    warningCount: 3,
    replayed: false,
  };
}
