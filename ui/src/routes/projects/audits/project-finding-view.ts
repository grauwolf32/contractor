import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";

import { useSearchParams } from "react-router";
import { collectAuditPages } from "../../../api/audit-collections";
import {
  findingNeedsReview,
  findingSeverity,
  listProjectFindings,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";

import { FINDING_STATE_LABELS } from "../../../app/vocabulary";
import { matchesIssueFilters, readIssueFilters } from "../../issues/links";
import { FINDING_STATES, isVerdictFilter } from "./finding-options";

export function useProjectFindingView(projectId: string, facts: boolean) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const queryKey = ["project-findings", projectId, "list"];
  const findings = useInfiniteQuery({
    queryKey,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) =>
      collectAuditPages(
        (cursor) =>
          listProjectFindings(
            api,
            projectId,
            cursor === undefined ? {} : { cursor },
          ),
        pageParam === undefined ? {} : { cursor: pageParam },
      ),
    getNextPageParam: (last) => (last.truncated ? last.nextCursor : undefined),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: false,
  });
  const rows = findings.data?.pages.flatMap((page) => page.items) ?? [];
  const filters = readIssueFilters(
    new URLSearchParams({
      ...Object.fromEntries(params),
      state: params.get("state") ?? "all",
      ...(facts ? { severity: "informational" } : {}),
    }),
  );
  const search = (params.get("q") ?? "").trim().toLocaleLowerCase();
  const rawVerdict = params.get("verdict") ?? "";
  const verdict = isVerdictFilter(rawVerdict) ? rawVerdict : undefined;
  const matching = rows.filter(
    (row) =>
      (search === "" ||
        [row.title, row.subjectKey, row.findingId]
          .join(" ")
          .toLocaleLowerCase()
          .includes(search)) &&
      (filters.severity === undefined ||
        findingSeverity(row) === filters.severity) &&
      (verdict === undefined ||
        (verdict === "needs_review"
          ? findingNeedsReview(row)
          : row.analystVerdict === verdict)),
  );
  const visible = matching
    .filter((row) => matchesIssueFilters(row, filters))
    .sort(
      (a, b) =>
        Number(a.analystVerdict !== null) - Number(b.analystVerdict !== null) ||
        b.createdAt.localeCompare(a.createdAt) ||
        a.findingId.localeCompare(b.findingId),
    );
  const chips = [
    {
      value: "unassessed" as const,
      label: "Not assessed",
      count: matching.filter((row) => row.state === null).length,
    },
    ...FINDING_STATES.map((value) => ({
      value,
      label: FINDING_STATE_LABELS[value].label,
      count: matching.filter((row) => row.state === value).length,
    })),
    { value: "all" as const, label: "All", count: matching.length },
  ];
  function change(name: string, value: string) {
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        if (value === "" || value === "all") next.delete(name);
        else next.set(name, value);
        return next;
      },
      { replace: true },
    );
  }
  const refresh = () => {
    void client.resetQueries({ queryKey, exact: true });
  };
  return {
    params,
    findings,
    filters,
    verdict,
    visible,
    chips,
    change,
    refresh,
    total: rows.length,
    clear: () => setParams({}, { replace: true }),
  };
}
