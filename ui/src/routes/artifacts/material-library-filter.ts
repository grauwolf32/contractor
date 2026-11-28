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
  return items.filter(
    (item) =>
      (category === "all" || materialKindOf(item) === category) &&
      `${item.artifact.namespace}/${item.artifact.name} ${item.artifact.revision} ${item.mediaType} ${item.gitSource?.resolvedCommit ?? ""}`
        .toLowerCase()
        .includes(needle),
  );
}
