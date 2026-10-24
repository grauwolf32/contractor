import { useInfiniteQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { invalidAPIResponse } from "../../api/error";
import { listProjects } from "../../api/projects";
import { nextPageCursor, type PageContinuation } from "../../app/pagination";

// The pickers retain at most 20 pages; a selected exact revision is kept
// separately so it survives page eviction or a mutable binding update.
export function pickerCursor(
  page: { page: PageContinuation },
  _pages: unknown[],
  _last: string | null,
  params: (string | null)[],
) {
  const next = nextPageCursor(page.page);
  if (page.page.hasMore && (next === undefined || params.includes(next)))
    throw invalidAPIResponse(
      200,
      "Server returned a repeated or missing picker cursor",
    );
  return next;
}
export function useScanProjects() {
  const api = usePublicAPI();
  return useInfiniteQuery({
    queryKey: ["web-scans", "projects"],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) =>
      listProjects(api, {
        kind: "project",
        ...(pageParam === null ? {} : { cursor: pageParam }),
      }),
    getNextPageParam: pickerCursor,
    getPreviousPageParam: () => undefined,
    maxPages: 20,
  });
}
export function useScanSettings(initialTarget = "") {
  const [parameters, setParameters] = useState<Record<string, string>>({
    target: initialTarget,
  });
  const [custom, setCustom] = useState(false);
  const [pack, setPack] = useState<ArtifactMetadata>();
  return {
    parameters,
    custom,
    pack,
    setCustom,
    setPack,
    change: (name: string, value: string) =>
      setParameters((current) => {
        const next = { ...current };
        if (value === "" && name !== "target") delete next[name];
        else next[name] = value;
        return next;
      }),
    request: () => ({
      parameters: Object.fromEntries(
        Object.entries(parameters).filter(([name]) => name !== "target"),
      ),
      ...(custom && pack !== undefined ? { rulepack: pack.artifact } : {}),
    }),
  };
}
