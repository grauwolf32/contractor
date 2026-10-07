import { describe, expect, it } from "vitest";

import type { ArtifactMetadata } from "../../api/artifacts";
import { SKILL_ARCHIVE_MEDIA_TYPE } from "./artifact-file";
import {
  formatLabel,
  groupMaterialsByKind,
  materialKindLabel,
  materialKindOf,
  MATERIAL_KIND_ORDER,
  MATERIAL_UPLOAD_KINDS,
} from "./kinds";

function item(
  namespace: string,
  mediaType: string,
  extra: Partial<ArtifactMetadata> = {},
): ArtifactMetadata {
  return {
    artifact: { namespace, name: `${namespace}-item`, revision: "r1" },
    mediaType,
    size: 10,
    current: true,
    frozen: false,
    createdAt: "2026-10-01T10:00:00Z",
    ...extra,
  };
}

const gitSource = {
  repositoryUrl: "https://example.test/repo.git",
  requestedRef: null,
  resolvedCommit: "a".repeat(40),
  importedAt: "2026-10-01T10:00:00Z",
};

describe("material kinds", () => {
  it("reads the kind from the namespace first", () => {
    expect(materialKindOf(item("sources", "text/plain"))).toBe("source");
    expect(materialKindOf(item("openapi", "application/json"))).toBe("api");
    expect(materialKindOf(item("likec4", "text/plain"))).toBe("architecture");
    expect(materialKindOf(item("docs", "application/zip"))).toBe("docs");
    expect(materialKindOf(item("diffs", "text/plain"))).toBe("diffs");
  });

  it("falls back to formats that name their purpose, never to generic ones", () => {
    expect(
      materialKindOf(item("outputs", "application/vnd.oai.openapi+yaml")),
    ).toBe("api");
    expect(materialKindOf(item("outputs", "text/vnd.likec4"))).toBe(
      "architecture",
    );
    expect(materialKindOf(item("outputs", "text/markdown"))).toBe("docs");
    expect(materialKindOf(item("outputs", "text/x-diff"))).toBe("diffs");
    // A ZIP, YAML or JSON file says nothing about what it is for.
    expect(materialKindOf(item("outputs", "application/zip"))).toBe("other");
    expect(materialKindOf(item("artifacts", "application/yaml"))).toBe("other");
    expect(materialKindOf(item("artifacts", "application/json"))).toBe("other");
  });

  it("treats every Git import as source code", () => {
    expect(
      materialKindOf(item("imports", "application/zip", { gitSource })),
    ).toBe("source");
  });

  it("does not trust inherited object keys as namespaces", () => {
    expect(materialKindOf(item("constructor", "text/plain"))).toBe("other");
    expect(materialKindOf(item("toString", "text/plain"))).toBe("other");
  });

  it("labels kinds and Skill packages", () => {
    expect(materialKindLabel(item("openapi", "application/yaml"))).toBe(
      "API spec",
    );
    expect(materialKindLabel(item("skills", SKILL_ARCHIVE_MEDIA_TYPE))).toBe(
      "Skill package",
    );
    expect(materialKindLabel(item("skills", "text/plain"))).toBe("Other");
  });

  it("groups in kind order and keeps the Server order inside a group", () => {
    const groups = groupMaterialsByKind([
      item("artifacts", "application/octet-stream"),
      item("docs", "text/markdown", {
        artifact: { namespace: "docs", name: "b", revision: "r" },
      }),
      item("sources", "application/zip"),
      item("docs", "text/markdown", {
        artifact: { namespace: "docs", name: "a", revision: "r" },
      }),
    ]);
    expect(groups.map((group) => group.label)).toEqual([
      "Source code",
      "Docs",
      "Other",
    ]);
    expect(groups[1]?.items.map((entry) => entry.artifact.name)).toEqual([
      "b",
      "a",
    ]);
    expect(groupMaterialsByKind([])).toEqual([]);
  });

  it("names formats and keeps unknown media types as written", () => {
    expect(formatLabel("application/zip")).toBe("ZIP");
    expect(formatLabel("text/markdown")).toBe("Markdown");
    expect(formatLabel("application/x-custom")).toBe("application/x-custom");
  });

  it("offers one upload kind per material kind with a valid suggestion", () => {
    expect(MATERIAL_UPLOAD_KINDS.map((kind) => kind.id)).toEqual(
      MATERIAL_KIND_ORDER,
    );
    expect(MATERIAL_UPLOAD_KINDS.map((kind) => kind.label)).toEqual([
      "Source code ZIP",
      "OpenAPI",
      "LikeC4",
      "Docs",
      "Diffs",
      "Other",
    ]);
    for (const kind of MATERIAL_UPLOAD_KINDS) {
      // Each suggestion lands in the group of the kind that suggested it.
      expect(materialKindOf(item(kind.namespace, kind.mediaType))).toBe(
        kind.id,
      );
    }
  });
});
