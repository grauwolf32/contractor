import { useQuery } from "@tanstack/react-query";
import { usePublicAPI } from "../../api/context";
import {
  evalInventory,
  EVAL_POLL_MS,
  getEvalCapabilities,
  getEvalExperiment,
  listEvalCases,
  listEvalDatasets,
  type EvalCapabilities,
} from "../../api/evals";
import { listProjects } from "../../api/projects";
import { useSession } from "../../auth/session";

export function useEvalOwner() {
  const api = usePublicAPI();
  const { session } = useSession();
  return `${api.apiBaseUrl}/${session?.principal.userId ?? ""}`;
}

export function useEvalProjects() {
  const api = usePublicAPI();
  return useQuery({
    queryKey: ["evals", "projects"],
    queryFn: ({ signal }) =>
      evalInventory(
        (cursor) =>
          listProjects(api, {
            kind: "evaluation",
            ...(cursor ? { cursor } : {}),
          }),
        signal,
      ),
  });
}

export function useEvalCapabilities() {
  const api = usePublicAPI();
  return useQuery({
    queryKey: ["evals", "capabilities"],
    queryFn: async ({ signal }) => {
      let metadata: EvalCapabilities | undefined;
      const bindings = await evalInventory(async (cursor) => {
        const page = await getEvalCapabilities(api, cursor, signal);
        metadata ??= page;
        return {
          items: page.bindings ?? [],
          page: page.page ?? { hasMore: false, nextCursor: null },
        };
      }, signal);
      return { ...metadata!, bindings };
    },
  });
}

export function useEvalDatasets(projectId: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: ["evals", "datasets", projectId],
    enabled: !!projectId,
    queryFn: ({ signal }) =>
      evalInventory(
        (cursor) => listEvalDatasets(api, projectId, cursor, signal),
        signal,
      ),
  });
}

export function useEvalCases(projectId: string, id: string, revision: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: ["evals", "cases", projectId, id, revision],
    enabled: !!projectId && !!id && !!revision,
    queryFn: ({ signal }) =>
      evalInventory(
        (cursor) => listEvalCases(api, projectId, id, revision, cursor, signal),
        signal,
      ),
  });
}

export function useEvalExperiment(id: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: ["evals", "experiment", id],
    enabled: !!id,
    queryFn: ({ signal }) => getEvalExperiment(api, id, signal),
    refetchInterval: (query) =>
      query.state.data?.state === "draft" ? false : EVAL_POLL_MS,
  });
}
