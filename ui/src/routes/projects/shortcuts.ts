export type ProjectArtifactShortcut =
  "sources" | "openapi" | "likec4" | "docs" | "diffs" | "other";

export interface ShortcutDefinition {
  id: ProjectArtifactShortcut;
  label: string;
  description: string;
  namespace: string;
  mediaType: string;
}

export const PROJECT_ARTIFACT_SHORTCUTS: readonly ShortcutDefinition[] = [
  {
    id: "sources",
    label: "Sources",
    description: "Source archives or individual source files",
    namespace: "sources",
    mediaType: "application/zip",
  },
  {
    id: "openapi",
    label: "OpenAPI",
    description: "OpenAPI documents in YAML or JSON",
    namespace: "openapi",
    mediaType: "application/yaml",
  },
  {
    id: "likec4",
    label: "LikeC4",
    description: "Architecture models and views",
    namespace: "likec4",
    mediaType: "text/vnd.likec4",
  },
  {
    id: "docs",
    label: "Docs",
    description: "Markdown, text, or PDF documentation",
    namespace: "docs",
    mediaType: "text/markdown",
  },
  {
    id: "diffs",
    label: "Diffs",
    description: "Unified text patches and review input",
    namespace: "diffs",
    mediaType: "text/x-diff",
  },
  {
    id: "other",
    label: "Other",
    description: "Any supported Artifact identity and media type",
    namespace: "artifacts",
    mediaType: "application/octet-stream",
  },
] as const;
