export const queryKeys = {
  session: ["auth", "session"] as const,
  workflows: ["workflows"] as const,
  artifacts: {
    all: ["artifacts"] as const,
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
  runs: ["runs"] as const,
  operations: ["operations"] as const,
};
