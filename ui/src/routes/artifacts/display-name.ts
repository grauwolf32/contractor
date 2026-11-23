import type { ExactArtifactRef } from "../../api/artifacts";
import { middleTruncate } from "../../app/format";

/** Display only: retain a recognizable prefix and suffix of a long material name. */
export function materialDisplayName(name: string): string {
  return name.length > 48 ? middleTruncate(name, 28, 8) : name;
}

/** Compact visible reference; callers retain the exact identity in labels and links. */
export function materialDisplayRef(ref: ExactArtifactRef): string {
  const revision =
    ref.name.length > 48 ? middleTruncate(ref.revision, 8, 4) : ref.revision;
  return `${ref.namespace}/${materialDisplayName(ref.name)}@${revision}`;
}
