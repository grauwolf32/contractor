import { useQuery } from "@tanstack/react-query";
import { getProjectFindingSource, type ProjectFinding } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
export function useFindingSource(finding: ProjectFinding | undefined) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: [
      "project-findings",
      finding?.projectId,
      "sources",
      finding?.firstReceiptId,
    ],
    queryFn: () =>
      getProjectFindingSource(api, finding!.projectId, finding!.firstReceiptId),
    enabled: finding !== undefined,
    retry: false,
  });
}
