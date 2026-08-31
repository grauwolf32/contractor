export const queryKeys = {
  session: ["auth", "session"] as const,
  workflows: {
    all: ["workflows"] as const,
    list: (cursor?: string) => ["workflows", "list", cursor ?? null] as const,
    detail: (name: string, version: string) =>
      ["workflows", "detail", name, version] as const,
  },
  artifacts: {
    all: ["artifacts"] as const,
    picker: ["artifacts", "picker"] as const,
    list: (namespace: string | undefined, cursor: string | undefined) =>
      ["artifacts", "list", namespace ?? null, cursor ?? null] as const,
    metadata: (namespace: string, name: string, revision?: string) =>
      ["artifacts", "metadata", namespace, name, revision ?? null] as const,
    versions: (namespace: string, name: string, cursor?: string) =>
      ["artifacts", "versions", namespace, name, cursor ?? null] as const,
    lineage: (
      namespace: string,
      name: string,
      revision: string,
      cursor?: string,
    ) =>
      [
        "artifacts",
        "lineage",
        namespace,
        name,
        revision,
        cursor ?? null,
      ] as const,
  },
  configurations: {
    all: ["configurations"] as const,
    picker: (kind: string) => ["configurations", "picker", kind] as const,
  },
  credentials: {
    all: ["credentials"] as const,
    picker: ["credentials", "picker"] as const,
  },
  runs: {
    all: ["runs"] as const,
    list: (state: string | undefined, cursor: string | undefined) =>
      ["runs", "list", state ?? null, cursor ?? null] as const,
    detail: (runId: string) => ["runs", "detail", runId] as const,
    artifacts: (
      runId: string,
      namespace: string | undefined,
      cursor: string | undefined,
    ) =>
      [
        "runs",
        "artifacts",
        runId,
        "list",
        namespace ?? null,
        cursor ?? null,
      ] as const,
    artifactMetadata: (
      runId: string,
      namespace: string,
      name: string,
      revision?: string,
    ) =>
      [
        "runs",
        "artifacts",
        runId,
        "metadata",
        namespace,
        name,
        revision ?? null,
      ] as const,
    artifactVersions: (
      runId: string,
      namespace: string,
      name: string,
      cursor?: string,
    ) =>
      [
        "runs",
        "artifacts",
        runId,
        "versions",
        namespace,
        name,
        cursor ?? null,
      ] as const,
    artifactLineage: (
      runId: string,
      namespace: string,
      name: string,
      revision: string,
      cursor?: string,
    ) =>
      [
        "runs",
        "artifacts",
        runId,
        "lineage",
        namespace,
        name,
        revision,
        cursor ?? null,
      ] as const,
  },
  operations: ["operations"] as const,
};
