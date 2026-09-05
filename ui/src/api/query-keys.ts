export const queryKeys = {
  session: ["auth", "session"] as const,
  workflows: {
    all: ["workflows"] as const,
    picker: ["workflows", "picker"] as const,
    list: (cursor?: string) => ["workflows", "list", cursor ?? null] as const,
    detail: (name: string, version: string) =>
      ["workflows", "detail", name, version] as const,
  },
  projects: {
    all: ["projects"] as const,
    list: (kind: string, cursor?: string) =>
      ["projects", "list", kind, cursor ?? null] as const,
    detail: (projectId: string) => ["projects", "detail", projectId] as const,
    artifacts: {
      all: (projectId: string) =>
        ["projects", "detail", projectId, "artifacts"] as const,
      picker: (projectId: string) =>
        ["projects", "detail", projectId, "artifacts", "picker"] as const,
      list: (
        projectId: string,
        namespace: string | undefined,
        cursor: string | undefined,
      ) =>
        [
          "projects",
          "detail",
          projectId,
          "artifacts",
          "list",
          namespace ?? null,
          cursor ?? null,
        ] as const,
      metadata: (
        projectId: string,
        namespace: string,
        name: string,
        revision?: string,
      ) =>
        [
          "projects",
          "detail",
          projectId,
          "artifacts",
          "metadata",
          namespace,
          name,
          revision ?? null,
        ] as const,
      versions: (
        projectId: string,
        namespace: string,
        name: string,
        cursor?: string,
      ) =>
        [
          "projects",
          "detail",
          projectId,
          "artifacts",
          "versions",
          namespace,
          name,
          cursor ?? null,
        ] as const,
      lineage: (
        projectId: string,
        namespace: string,
        name: string,
        revision: string,
        cursor?: string,
      ) =>
        [
          "projects",
          "detail",
          projectId,
          "artifacts",
          "lineage",
          namespace,
          name,
          revision,
          cursor ?? null,
        ] as const,
    },
    runs: (projectId: string, cursor?: string) =>
      ["projects", "detail", projectId, "runs", cursor ?? null] as const,
  },
  queue: {
    all: ["queue"] as const,
    list: (
      state: string | undefined,
      membership: string | undefined,
      cursor: string | undefined,
    ) =>
      [
        "queue",
        "list",
        state ?? null,
        membership ?? null,
        cursor ?? null,
      ] as const,
  },
  artifacts: {
    all: ["artifacts"] as const,
    picker: ["artifacts", "picker"] as const,
    list: (
      namespace: string | undefined,
      cursor: string | undefined,
      excludeNamespace?: string,
    ) =>
      [
        "artifacts",
        "list",
        namespace ?? null,
        excludeNamespace ?? null,
        cursor ?? null,
      ] as const,
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
    infinitePicker: (kind: string) =>
      ["configurations", "infinite-picker", kind] as const,
    list: (kind: string, cursor?: string) =>
      ["configurations", "list", kind, cursor ?? null] as const,
    detail: (kind: string, name: string, version: string) =>
      ["configurations", "detail", kind, name, version] as const,
  },
  credentials: {
    all: ["credentials"] as const,
    picker: ["credentials", "picker"] as const,
    list: (cursor?: string) => ["credentials", "list", cursor ?? null] as const,
    detail: (credentialId: string) =>
      ["credentials", "detail", credentialId] as const,
  },
  runs: {
    all: ["runs"] as const,
    list: (
      state: string | undefined,
      cursor: string | undefined,
      labelSelectors: readonly string[] = [],
    ) =>
      [
        "runs",
        "list",
        state ?? null,
        [...labelSelectors],
        cursor ?? null,
      ] as const,
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
  operations: {
    all: ["operations"] as const,
    snapshot: ["operations", "snapshot"] as const,
    runtimeConfigs: {
      all: ["operations", "runtime-configs"] as const,
      list: (cursor?: string) =>
        ["operations", "runtime-configs", "list", cursor ?? null] as const,
      detail: (name: string, version: string) =>
        ["operations", "runtime-configs", "detail", name, version] as const,
    },
    runtimeLabels: {
      all: ["operations", "runtime-labels"] as const,
      picker: ["operations", "runtime-labels", "picker"] as const,
      list: (cursor?: string) =>
        ["operations", "runtime-labels", "list", cursor ?? null] as const,
      detail: (label: string) =>
        ["operations", "runtime-labels", "detail", label] as const,
    },
    runtimeCredentials: {
      all: ["operations", "runtime-credentials"] as const,
      list: (cursor?: string) =>
        ["operations", "runtime-credentials", "list", cursor ?? null] as const,
    },
    runtimeAgentPrincipals: {
      all: ["operations", "runtime-agent-principals"] as const,
      list: (cursor?: string) =>
        [
          "operations",
          "runtime-agent-principals",
          "list",
          cursor ?? null,
        ] as const,
      detail: (runtimeAgentId: string) =>
        [
          "operations",
          "runtime-agent-principals",
          "detail",
          runtimeAgentId,
        ] as const,
    },
  },
};
