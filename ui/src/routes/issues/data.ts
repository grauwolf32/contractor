import { useMemo } from "react";
import {
  useAllPossibleIssues,
  useAllChecks,
  type CrossProjectIssue,
  type CrossProjectError,
  type CrossProjectCheck,
} from "../../api/cross-project";
import { WORKING_AUDIT_STATES } from "../../api/states";
import { FINDING_STATES } from "../projects/audits/finding-options";
import {
  matchesIssueFilters,
  type IssueFilters,
  type StateFilter,
} from "./links";
export interface IssueList {
  issues: CrossProjectIssue[];
  counts: Readonly<Record<StateFilter, number | string | undefined>>;
  pending: boolean;
  error: Error | null;
  errors: CrossProjectError[];
  truncated: boolean;
  checks: CrossProjectCheck[];
  running: CrossProjectCheck[];
  refetch: () => Promise<void>;
}
/** One bounded canonical list; counts and sorting describe only loaded rows. */
export function useIssueList(filters: IssueFilters): IssueList {
  const read = useAllPossibleIssues(
    filters.severity === undefined ? {} : { severities: [filters.severity] },
  );
  const checks = useAllChecks({ states: WORKING_AUDIT_STATES });
  return useMemo(() => {
    const scoped = read.issues.filter(
      (issue) =>
        filters.project === undefined ||
        issue.project.projectId === filters.project,
    );
    const counts = {} as Record<StateFilter, number | string | undefined>;
    for (const state of [
      ...FINDING_STATES,
      "unassessed",
      "needs-review",
      "all",
    ] as StateFilter[]) {
      const n = scoped.filter((issue) =>
        matchesIssueFilters(issue.finding, { ...filters, state }),
      ).length;
      counts[state] =
        read.isPending && n === 0
          ? undefined
          : read.partial || read.truncated
            ? `${n}+`
            : n;
    }
    return {
      issues: scoped.filter((issue) =>
        matchesIssueFilters(issue.finding, filters),
      ),
      counts,
      pending: read.isPending,
      error: read.error,
      errors: read.errors,
      truncated: read.truncated,
      checks: checks.checks,
      running: checks.checks.filter(
        (check) =>
          filters.project === undefined ||
          check.project.projectId === filters.project,
      ),
      refetch: async () => {
        await Promise.all([read.refetch(), checks.refetch()]);
      },
    };
  }, [read, checks, filters]);
}
