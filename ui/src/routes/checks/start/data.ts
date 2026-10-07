/**
 * Reads behind the Start page: the project, the published check types with
 * the details of their preferred versions, and the project's current
 * materials.
 */
import {
  type UseQueryResult,
  useInfiniteQuery,
  useQueries,
  useQuery,
} from "@tanstack/react-query";
import { useEffect, useMemo } from "react";

import type { ArtifactMetadata } from "../../../api/artifacts";
import {
  getAuditProfile,
  listAuditProfiles,
  type AuditProfile,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { getProject } from "../../../api/projects";
import { listProjectArtifacts } from "../../../api/project-artifacts";
import { queryKeys } from "../../../api/query-keys";
import { nextPageCursor } from "../../../app/pagination";
import {
  groupFamilies,
  groupRows,
  type CheckTypeFamily,
  type CheckTypeRow,
} from "./check-types";

const INITIAL_CURSOR = null as string | null;

/**
 * The Project Artifact API filters by namespace only, not by media type, so
 * the page reads the inventory itself: at most this many pages on its own,
 * the rest on request ("Load more materials"), so a large project does not
 * download its whole inventory every time.
 */
export const MATERIAL_AUTOLOAD_PAGES = 4;

// A check type version does not change once published.
const PROFILE_DETAIL_STALE_MS = 5 * 60_000;

export function useProject(projectId: string) {
  const api = usePublicAPI();
  return useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
  });
}

/** "name@version", the key of a check type version. */
export function versionKey(profile: Pick<AuditProfile, "ref">): string {
  return `${profile.ref.name}@${profile.ref.version}`;
}

interface DetailReads {
  profiles: AuditProfile[];
  settled: boolean;
}

// Stable, so the combined result keeps its identity while nothing changed.
function combineDetails(results: UseQueryResult<AuditProfile>[]): DetailReads {
  return {
    profiles: results.flatMap((result) =>
      result.data === undefined ? [] : [result.data],
    ),
    settled: results.every((result) => result.status !== "pending"),
  };
}

export interface CheckTypeCatalog {
  /** The paged list of published check types (every version). */
  query: ReturnType<typeof useProfilePages>;
  families: CheckTypeFamily[];
  rows: CheckTypeRow[];
  /**
   * Details (with workflows) of each family's preferred version that have
   * loaded, by versionKey. A failed detail read is left out.
   */
  details: ReadonlyMap<string, AuditProfile>;
  /** Every detail read has an answer (data or error). */
  detailsSettled: boolean;
}

function useProfilePages() {
  const api = usePublicAPI();
  return useInfiniteQuery({
    queryKey: [...queryKeys.auditProfiles.all, "pages"],
    initialPageParam: INITIAL_CURSOR,
    queryFn: ({ pageParam, signal }) =>
      listAuditProfiles(api, {
        signal,
        ...(pageParam === null ? {} : { cursor: pageParam }),
      }),
    getNextPageParam: (page) => nextPageCursor(page.page),
  });
}

export function useCheckTypeCatalog(): CheckTypeCatalog {
  const api = usePublicAPI();
  const query = useProfilePages();
  const families = useMemo(
    () => groupFamilies(query.data?.pages.flatMap((page) => page.items) ?? []),
    [query.data],
  );
  const rows = useMemo(() => groupRows(families), [families]);
  const reads = useQueries({
    queries: families.map(({ preferred }) => ({
      queryKey: queryKeys.auditProfiles.detail(
        preferred.ref.name,
        preferred.ref.version,
      ),
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        getAuditProfile(api, preferred.ref.name, preferred.ref.version, signal),
      staleTime: PROFILE_DETAIL_STALE_MS,
    })),
    combine: combineDetails,
  });
  const details = useMemo(
    () =>
      new Map(reads.profiles.map((profile) => [versionKey(profile), profile])),
    [reads.profiles],
  );
  return { query, families, rows, details, detailsSettled: reads.settled };
}

/** The exact profile version the form submits, with its workflows. */
export function useProfileDetail(profile: AuditProfile | undefined) {
  const api = usePublicAPI();
  const name = profile?.ref.name ?? "";
  const version = profile?.ref.version ?? "";
  return useQuery({
    queryKey: queryKeys.auditProfiles.detail(name, version),
    queryFn: ({ signal }) => getAuditProfile(api, name, version, signal),
    enabled: profile !== undefined,
    staleTime: PROFILE_DETAIL_STALE_MS,
  });
}

export interface ProjectMaterials {
  /** Current materials read so far; undefined until the first page arrives. */
  items: ArtifactMetadata[] | undefined;
  /**
   * Every page is read, so a single format match is the only one. A failed
   * refresh keeps the pages read before, so it keeps this too.
   */
  complete: boolean;
  /** Pages are still being read on their own. */
  loading: boolean;
  /** More pages wait for "Load more materials". */
  hasMore: boolean;
  loadingMore: boolean;
  error: Error | null;
  /** The error came from reading a further page; the pages before it stay. */
  moreFailed: boolean;
  loadMore: () => void;
  /** Reads the failed page again, or every page after a failed refresh. */
  retry: () => void;
  retrying: boolean;
}

export function useProjectMaterials(projectId: string): ProjectMaterials {
  const api = usePublicAPI();
  const query = useInfiniteQuery({
    queryKey: queryKeys.projects.artifacts.picker(projectId),
    initialPageParam: INITIAL_CURSOR,
    queryFn: ({ pageParam }) =>
      listProjectArtifacts(api, {
        projectId,
        ...(pageParam === null ? {} : { cursor: pageParam }),
      }),
    getNextPageParam: (page) => nextPageCursor(page.page),
  });
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetching,
    isError,
    isFetchNextPageError,
    refetch,
  } = query;
  const loadedPages = data?.pages.length ?? 0;
  const autoloading =
    hasNextPage && !isError && loadedPages < MATERIAL_AUTOLOAD_PAGES;
  // The page count is a dependency too: a fast page can finish before its
  // fetching state is ever rendered, which alone would not re-run the effect.
  useEffect(() => {
    if (autoloading && !isFetching) void fetchNextPage();
  }, [autoloading, fetchNextPage, isFetching, loadedPages]);
  const items = useMemo(
    () => data?.pages.flatMap((page) => page.items),
    [data],
  );
  return {
    items,
    // From the pages read, not the query status: a failed refresh (on mount,
    // reconnect or invalidation) keeps them, and with them the attachments.
    complete: data !== undefined && !hasNextPage && !isFetchNextPageError,
    loading: query.isPending || autoloading,
    hasMore: hasNextPage && !autoloading && !isError,
    loadingMore: query.isFetchingNextPage,
    error: query.error,
    moreFailed: isFetchNextPageError,
    loadMore: () => void fetchNextPage(),
    retry: () => void (isFetchNextPageError ? fetchNextPage() : refetch()),
    retrying: isFetching,
  };
}
