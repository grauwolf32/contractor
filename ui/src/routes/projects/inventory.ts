import { useQuery } from "@tanstack/react-query";
import { usePublicAPI } from "../../api/context";
import { loadCompletePages } from "../../api/pages";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";

export function useProjectArtifactInventory(projectId: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: [...queryKeys.projects.artifacts.picker(projectId), "complete"],
    // Workflow matching needs every material; a partial list is refused.
    queryFn: ({ signal }) =>
      loadCompletePages(
        "The Project's material list",
        (cursor) =>
          listProjectArtifacts(api, {
            projectId,
            ...(cursor === undefined ? {} : { cursor }),
          }),
        { signal },
      ),
  });
}
