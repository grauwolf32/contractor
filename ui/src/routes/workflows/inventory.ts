import { useQuery } from "@tanstack/react-query";
import { usePublicAPI } from "../../api/context";
import { loadCompletePages } from "../../api/pages";
import { queryKeys } from "../../api/query-keys";
import { listWorkflows } from "../../api/workflows";

export function useWorkflowInventory() {
  const api = usePublicAPI();
  return useQuery({
    queryKey: queryKeys.workflows.inventory,
    queryFn: ({ signal }) =>
      loadCompletePages(
        "The Workflow inventory",
        (cursor) =>
          listWorkflows(api, {
            ...(cursor === undefined ? {} : { cursor }),
            signal,
          }),
        { signal },
      ),
  });
}
