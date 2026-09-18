import { useQuery } from "@tanstack/react-query";
import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import type { ArtifactMetadata } from "../../api/artifacts";
import { queryKeys } from "../../api/query-keys";

export function useProjectArtifactInventory(projectId: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: [...queryKeys.projects.artifacts.picker(projectId), "complete"],
    queryFn: async ({ signal }) => {
      const items: ArtifactMetadata[] = [];
      const seen = new Set<string>();
      let cursor: string | undefined;
      for (;;) {
        signal.throwIfAborted();
        const page = await listProjectArtifacts(api, {
          projectId,
          ...(cursor === undefined ? {} : { cursor }),
        });
        signal.throwIfAborted();
        items.push(...page.items);
        if (!page.page.hasMore) return items;
        cursor = page.page.nextCursor;
        if (!cursor || seen.has(cursor))
          throw new Error(
            "Project materials could not be fully loaded. Retry before choosing inputs.",
          );
        seen.add(cursor);
      }
    },
  });
}
