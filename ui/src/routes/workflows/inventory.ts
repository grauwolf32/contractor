import { useQuery } from "@tanstack/react-query";
import { usePublicAPI } from "../../api/context";
import { listWorkflows, type WorkflowSummary } from "../../api/workflows";

export function useWorkflowInventory() {
  const api = usePublicAPI();
  return useQuery({
    queryKey: ["workflows", "inventory"],
    queryFn: async ({ signal }) => {
      const items: WorkflowSummary[] = [];
      const cursors = new Set<string>();
      let cursor: string | undefined;
      do {
        const page = await listWorkflows(api, {
          ...(cursor === undefined ? {} : { cursor }),
          signal,
        });
        items.push(...page.items);
        if (!page.page.hasMore) return items;
        cursor = page.page.nextCursor;
        if (!cursor || cursors.has(cursor))
          throw new Error(
            "Workflow inventory could not be completed. Refresh to retry.",
          );
        cursors.add(cursor);
      } while (!signal.aborted);
      signal.throwIfAborted();
      return items;
    },
  });
}
