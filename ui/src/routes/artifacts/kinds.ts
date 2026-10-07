import type { ArtifactMetadata } from "../../api/artifacts";
import { workflowFormats } from "../workflows/formats";
import { SKILL_ARCHIVE_MEDIA_TYPE } from "./artifact-file";

/**
 * What a material is for, read from where it is stored and its format
 * (coverage map: "Namespaces read as kinds"). The kind only groups and labels
 * materials; it never claims more about the content than its namespace or
 * media type says.
 */
export type MaterialKind =
  "source" | "api" | "architecture" | "docs" | "diffs" | "other";

/** Display order of the kind groups. */
export const MATERIAL_KIND_ORDER: readonly MaterialKind[] = [
  "source",
  "api",
  "architecture",
  "docs",
  "diffs",
  "other",
];

export const MATERIAL_KIND_LABELS: Readonly<Record<MaterialKind, string>> = {
  source: "Source code",
  api: "API spec",
  architecture: "Architecture",
  docs: "Docs",
  diffs: "Diffs",
  other: "Other",
};

// The namespaces the "Add material" kinds suggest.
const NAMESPACE_KINDS: Readonly<Record<string, MaterialKind>> = {
  sources: "source",
  openapi: "api",
  likec4: "architecture",
  docs: "docs",
  diffs: "diffs",
};

// Formats that name their purpose. Generic formats (ZIP, YAML, JSON, text)
// say nothing about what a file is for, so they fall through to Other.
const MEDIA_TYPE_KINDS: Readonly<Record<string, MaterialKind>> = {
  "application/vnd.oai.openapi": "api",
  "application/vnd.oai.openapi+json": "api",
  "application/vnd.oai.openapi+yaml": "api",
  "text/vnd.likec4": "architecture",
  "text/markdown": "docs",
  "text/x-markdown": "docs",
  "application/pdf": "docs",
  "text/x-diff": "diffs",
};

type KindSource = Pick<ArtifactMetadata, "artifact" | "mediaType"> & {
  gitSource?: ArtifactMetadata["gitSource"];
};

/**
 * The kind of a material: a Git import is source code; otherwise its
 * namespace decides, then a format that names its purpose; anything else is
 * Other.
 */
export function materialKindOf(item: KindSource): MaterialKind {
  if (item.gitSource !== undefined) return "source";
  const namespace = item.artifact.namespace;
  if (Object.hasOwn(NAMESPACE_KINDS, namespace))
    return NAMESPACE_KINDS[namespace]!;
  return Object.hasOwn(MEDIA_TYPE_KINDS, item.mediaType)
    ? MEDIA_TYPE_KINDS[item.mediaType]!
    : "other";
}

/** "Source code", "API spec", …; a Skill package in the skills namespace. */
export function materialKindLabel(item: KindSource): string {
  if (
    item.artifact.namespace === "skills" &&
    item.mediaType === SKILL_ARCHIVE_MEDIA_TYPE
  )
    return "Skill package";
  return MATERIAL_KIND_LABELS[materialKindOf(item)];
}

/** Materials in kind order, each group in the order the Server listed them. */
export function groupMaterialsByKind<T extends KindSource>(
  items: readonly T[],
): { kind: MaterialKind; label: string; items: T[] }[] {
  const groups = new Map<MaterialKind, T[]>();
  for (const item of items) {
    const kind = materialKindOf(item);
    const group = groups.get(kind);
    if (group === undefined) groups.set(kind, [item]);
    else group.push(item);
  }
  return MATERIAL_KIND_ORDER.flatMap((kind) => {
    const group = groups.get(kind);
    return group === undefined
      ? []
      : [{ kind, label: MATERIAL_KIND_LABELS[kind], items: group }];
  });
}

/** Short format name ("ZIP", "Markdown"), else the media type itself. */
export function formatLabel(mediaType: string): string {
  return Object.hasOwn(workflowFormats, mediaType)
    ? workflowFormats[mediaType]!
    : mediaType;
}

/** One way to add a material, offered by the "Add material" sheet. */
export interface MaterialUploadKind {
  id: MaterialKind;
  label: string;
  description: string;
  /** Suggested, editable namespace and media type of the new material. */
  namespace: string;
  mediaType: string;
}

/**
 * Upload kinds of the "Add material" sheet. The kind only suggests a
 * namespace and media type; both stay editable in the upload form.
 */
export const MATERIAL_UPLOAD_KINDS: readonly MaterialUploadKind[] = [
  {
    id: "source",
    label: "Source code ZIP",
    description: "A ZIP archive of the code to check",
    namespace: "sources",
    mediaType: "application/zip",
  },
  {
    id: "api",
    label: "OpenAPI",
    description: "API spec in YAML or JSON",
    namespace: "openapi",
    mediaType: "application/yaml",
  },
  {
    id: "architecture",
    label: "LikeC4",
    description: "Architecture model and views",
    namespace: "likec4",
    mediaType: "text/vnd.likec4",
  },
  {
    id: "docs",
    label: "Docs",
    description: "Markdown, text or PDF documentation",
    namespace: "docs",
    mediaType: "text/markdown",
  },
  {
    id: "diffs",
    label: "Diffs",
    description: "Unified patches for review",
    namespace: "diffs",
    mediaType: "text/x-diff",
  },
  {
    id: "other",
    label: "Other",
    description: "Any file, with a namespace and media type you choose",
    namespace: "artifacts",
    mediaType: "application/octet-stream",
  },
];
