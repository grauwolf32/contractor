import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import type { ArtifactMetadata } from "../../api/artifacts";
import {
  AUDIT_PAGE_SIZE,
  listProjectFindings,
  listProjectAudits,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { loadAllPages } from "../../api/pages";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { listProjectRuns } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { pollInterval } from "../../app/polling";
import { MATERIAL_KIND_ORDER, materialKindOf } from "../artifacts/kinds";

/** How often the overview refreshes checks and Runs while it is open. */
export const OVERVIEW_POLL_MS = 10_000;
export const SUCCESSFUL_RUNS_POLL_MS = 15_000;

/** Overview reads poll while shown and back off while reads fail. */
const overviewPoll = pollInterval({ interval: OVERVIEW_POLL_MS });

/**
 * The project's newest checks: the first page, as the cross-project lists
 * read it (same key and request), so the overview shares the read the rail's
 * Inbox badge already polls. Decisions and check lifecycle actions (start,
 * pause, continue, stop, delete) refresh it through invalidateCrossProject,
 * so an overview that opens after one reads it again.
 */
export function useProjectChecks(projectId: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: queryKeys.crossProject.audits(projectId, null),
    queryFn: () =>
      listProjectAudits(api, {
        projectId,
        limit: AUDIT_PAGE_SIZE,
      }),
    staleTime: OVERVIEW_POLL_MS,
    refetchInterval: overviewPoll,
    refetchOnWindowFocus: false,
  });
}

export type PossibleIssuesToReview = ReturnType<
  typeof usePossibleIssuesToReview
>;

/** Canonical Project Findings, counted once across all execution contexts. */
export function usePossibleIssuesToReview(projectId: string) {
  const api = usePublicAPI();
  const query = useQuery({
    queryKey: ["project-findings", projectId, "overview"],
    queryFn: () =>
      loadAllPages((cursor) =>
        listProjectFindings(
          api,
          projectId,
          cursor === undefined ? {} : { cursor },
        ),
      ),
    retry: false,
    refetchInterval: OVERVIEW_POLL_MS,
  });
  const items = (query.data?.items ?? []).filter(
    (finding) =>
      finding.analystVerdict === null || finding.state === "proposed",
  );
  return {
    items,
    total: items.length,
    settled: !query.isPending,
    partial:
      query.data?.truncated === true ||
      query.isError ||
      (query.data?.error !== null && query.data?.error !== undefined),
    refetch: async () => {
      await query.refetch();
    },
  };
}

/** Up to five recent Runs of any state. */
export function useRecentRuns(projectId: string) {
  const api = usePublicAPI();
  const options = { limit: 5 };
  return useQuery({
    queryKey: queryKeys.projects.runView(projectId, options),
    queryFn: () => listProjectRuns(api, { projectId, ...options }),
    refetchInterval: overviewPoll,
  });
}

/** The three most recent successful Runs, for their results. */
export function useSuccessfulRuns(projectId: string) {
  const api = usePublicAPI();
  const options = { limit: 3, state: "succeeded" as const };
  return useQuery({
    queryKey: queryKeys.projects.runView(projectId, options),
    queryFn: () => listProjectRuns(api, { projectId, ...options }),
    refetchInterval: pollInterval({ interval: SUCCESSFUL_RUNS_POLL_MS }),
  });
}

export interface MaterialsSample {
  /** Current materials read, inputs first (source code, API spec, …). */
  items: ArtifactMetadata[];
  /** The project has more materials than were read. */
  more: boolean;
  isPending: boolean;
  /** Every read failed: nothing is known about the materials. */
  error: Error | null;
  /** Some read failed while others answered. */
  partial: boolean;
  isFetching: boolean;
  refetch: () => void;
}

/**
 * A bounded sample of the project's materials (S06: the overview never loads
 * the complete inventory): five category-ordered current materials, and up to three source
 * archives and three API specs, which the check types need most.
 */
export function useMaterialsSample(projectId: string): MaterialsSample {
  const api = usePublicAPI();
  const base = queryKeys.projects.artifacts.all(projectId);
  const recent = useQuery({
    queryKey: [...base, "overview-material-order"],
    queryFn: () =>
      listProjectArtifacts(api, { projectId, limit: 5, order: "material" }),
  });
  const sources = useQuery({
    queryKey: [...base, "overview-sources"],
    queryFn: () =>
      listProjectArtifacts(api, { projectId, namespace: "sources", limit: 3 }),
  });
  const specs = useQuery({
    queryKey: [...base, "overview-openapi"],
    queryFn: () =>
      listProjectArtifacts(api, { projectId, namespace: "openapi", limit: 3 }),
  });
  const reads = [recent, sources, specs];
  const items = useMemo(() => {
    const unique = new Map<string, ArtifactMetadata>();
    for (const page of [sources.data, specs.data, recent.data]) {
      for (const item of page?.items ?? []) {
        if (!item.current) continue;
        unique.set(`${item.artifact.namespace}/${item.artifact.name}`, item);
      }
    }
    return [...unique.values()].sort(
      (left, right) =>
        MATERIAL_KIND_ORDER.indexOf(materialKindOf(left)) -
          MATERIAL_KIND_ORDER.indexOf(materialKindOf(right)) ||
        `${left.artifact.namespace}/${left.artifact.name}`.localeCompare(
          `${right.artifact.namespace}/${right.artifact.name}`,
        ),
    );
  }, [recent.data, sources.data, specs.data]);
  const failed = reads.filter((read) => read.error !== null);
  return {
    items,
    more: reads.some((read) => read.data?.page.hasMore === true),
    isPending: reads.some((read) => read.isPending),
    error: failed.length === reads.length ? (failed[0]?.error ?? null) : null,
    partial: failed.length > 0 && failed.length < reads.length,
    isFetching: reads.some((read) => read.isFetching),
    refetch: () => {
      for (const read of reads) void read.refetch();
    },
  };
}
