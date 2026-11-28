import type { ArtifactMetadata } from "../../api/artifacts";
import {
  MATERIAL_KIND_ORDER,
  materialKindOf,
  type MaterialKind,
} from "./kinds";

export type MaterialCategory = "all" | MaterialKind;
export function materialCategory(value: string | null): MaterialCategory {
  return MATERIAL_KIND_ORDER.includes(value as MaterialKind)
    ? (value as MaterialKind)
    : "all";
}
export function matchingMaterials(
  items: readonly ArtifactMetadata[],
  category: MaterialCategory,
  search: string,
): ArtifactMetadata[] {
  const needle = search.trim().toLowerCase();
  return items.filter((item) => {
    if (category !== "all" && materialKindOf(item) !== category) return false;
    const path = `${item.artifact.namespace}/${item.artifact.name}`;
    const revision = item.artifact.revision;
    const fields = `${path}@${revision} ${path} ${revision} ${item.mediaType} ${item.gitSource?.resolvedCommit ?? ""}`;
    return fields.toLowerCase().includes(needle);
  });
}

/** The key that identifies one exact material in selects and choices. */
export function materialKey(
  material: Pick<ArtifactMetadata, "artifact">,
): string {
  return JSON.stringify([
    material.artifact.namespace,
    material.artifact.name,
    material.artifact.revision,
  ]);
}
