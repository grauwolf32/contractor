import type { ArtifactMetadata } from "../../api/artifacts";
import { formatBytes } from "../../app/format";
import { workflowFormats } from "../workflows/formats";

/**
 * What a project material is, read from its namespace and media type: the
 * Materials vocabulary of the coverage map (Source code, API spec,
 * Architecture, Docs, Diffs, Other) plus Run results published to the
 * project's `outputs` namespace.
 */
export type MaterialKind =
  "sources" | "openapi" | "likec4" | "docs" | "diffs" | "results" | "other";

const KIND_LABELS: Readonly<Record<MaterialKind, string>> = {
  sources: "Source code",
  openapi: "API spec",
  likec4: "Architecture",
  docs: "Docs",
  diffs: "Diffs",
  results: "Result",
  other: "Other",
};

const NAMESPACE_KINDS: Readonly<Record<string, MaterialKind>> = {
  source: "sources",
  sources: "sources",
  openapi: "openapi",
  likec4: "likec4",
  docs: "docs",
  diffs: "diffs",
  outputs: "results",
};

const MEDIA_TYPE_KINDS: Readonly<Record<string, MaterialKind>> = {
  "application/zip": "sources",
  "text/vnd.likec4": "likec4",
  "text/markdown": "docs",
  "application/pdf": "docs",
  "text/x-diff": "diffs",
};

/** The kind of a material: its namespace first, then its media type. */
export function materialKind(
  material: Pick<ArtifactMetadata, "artifact" | "mediaType">,
): MaterialKind {
  const namespace = material.artifact.namespace.toLowerCase();
  if (Object.hasOwn(NAMESPACE_KINDS, namespace))
    return NAMESPACE_KINDS[namespace]!;
  return Object.hasOwn(MEDIA_TYPE_KINDS, material.mediaType)
    ? MEDIA_TYPE_KINDS[material.mediaType]!
    : "other";
}

export function materialKindLabel(kind: MaterialKind): string {
  return KIND_LABELS[kind];
}

/** "ZIP · 68.7 KiB": the readable format and size of one revision. */
export function materialFormat(
  material: Pick<ArtifactMetadata, "mediaType" | "size">,
): string {
  const format = workflowFormats[material.mediaType] ?? material.mediaType;
  return `${format} · ${formatBytes(material.size)}`;
}

/** Order for showing materials: inputs first, results last. */
export const MATERIAL_KIND_ORDER: readonly MaterialKind[] = [
  "sources",
  "openapi",
  "likec4",
  "docs",
  "diffs",
  "other",
  "results",
];
