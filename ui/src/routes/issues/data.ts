/**
 * The Issues list: possible issues of every listed project, one bounded
 * cross-project read per finding state (src/api/cross-project.ts), so each
 * state chip has its own count and its own first page per check.
 */
import { useMemo } from "react";

import type {
  AuditFindingSeverity,
  AuditFindingState,
  AuditState,
} from "../../api/audits";
import {
  useAllChecks,
  useAllPossibleIssues,
  type AllPossibleIssues,
  type CrossProjectCheck,
  type CrossProjectError,
  type CrossProjectIssue,
} from "../../api/cross-project";
import { FINDING_STATES } from "../projects/audits/finding-options";
import type { IssueFilters, StateFilter } from "./links";

/** Check states that can hold possible issues (every state but draft and deleting). */
const ISSUE_CHECK_STATES: readonly AuditState[] = [
  "active",
  "waiting_review",
  "paused",
  "finalizing",
  "cancelling",
  "completed",
  "cancelled",
  "failed",
];

/** Checks that can still propose possible issues. */
const RUNNING_CHECK_STATES: ReadonlySet<AuditState> = new Set([
  "active",
  "finalizing",
]);

/** A count for a filter chip: exact, a lower bound ("12+") or unknown. */
export type IssueCount = number | string | undefined;

export interface IssueList {
  /** Possible issues matching every filter, newest first. */
  issues: CrossProjectIssue[];
  /** Per state chip and "all". */
  counts: Readonly<Record<StateFilter, IssueCount>>;
  /** Some read has not settled yet. */
  pending: boolean;
  /** The project index failed: nothing can be listed. */
  error: Error | null;
  /** Failed reads (deduplicated); everything else is listed. */
  errors: CrossProjectError[];
  /** Checks of the shown states with more possible issues than listed. */
  truncatedChecks: CrossProjectCheck[];
  /** The project index or a project's check page had more than it read. */
  checksTruncated: boolean;
  /** Listed checks in scope, for names of failed reads. */
  checks: CrossProjectCheck[];
  /** Checks in scope that are still running, so more may arrive. */
  running: CrossProjectCheck[];
  /** Refetches the polled heads and every failed read. */
  refetch: () => Promise<void>;
}

function issueKey(issue: CrossProjectIssue): string {
  return `${issue.finding.auditId}/${issue.finding.findingId}`;
}

function time(value: string): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

function byText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

/** Newest first, like the cross-project lists; IDs break ties. */
export function newerIssue(
  left: CrossProjectIssue,
  right: CrossProjectIssue,
): number {
  return (
    time(right.finding.createdAt) - time(left.finding.createdAt) ||
    time(right.finding.updatedAt) - time(left.finding.updatedAt) ||
    byText(left.finding.auditId, right.finding.auditId) ||
    byText(left.finding.findingId, right.finding.findingId)
  );
}

function sumCounts(counts: readonly IssueCount[]): IssueCount {
  let total = 0;
  let lowerBound = false;
  for (const count of counts) {
    if (count === undefined) return undefined;
    if (typeof count === "number") total += count;
    else {
      total += Number.parseInt(count, 10) || 0;
      lowerBound = true;
    }
  }
  return lowerBound ? `${total}+` : total;
}

function errorKey(error: CrossProjectError): string {
  return `${error.scope}:${error.projectId ?? ""}:${error.auditId ?? ""}`;
}

/**
 * The possible issues the filters select, with chip counts. Without a
 * project filter a count is the Server's total over the listed checks (it
 * includes possible issues beyond each check's first page); with one it
 * counts the listed rows of that project, "+" when one of its checks has
 * more than listed.
 */
export function useIssueList(filters: IssueFilters): IssueList {
  const severity: { severities?: AuditFindingSeverity[] } =
    filters.severity === undefined ? {} : { severities: [filters.severity] };
  const proposed = useAllPossibleIssues({ states: ["proposed"], ...severity });
  const confirmed = useAllPossibleIssues({
    states: ["confirmed"],
    ...severity,
  });
  const rejected = useAllPossibleIssues({ states: ["rejected"], ...severity });
  const duplicate = useAllPossibleIssues({
    states: ["duplicate"],
    ...severity,
  });
  const needsEvidence = useAllPossibleIssues({
    states: ["needs-evidence"],
    ...severity,
  });
  const checks = useAllChecks({ states: ISSUE_CHECK_STATES });
  const { state, project } = filters;

  return useMemo(() => {
    const byState: Readonly<Record<AuditFindingState, AllPossibleIssues>> = {
      proposed,
      confirmed,
      rejected,
      duplicate,
      "needs-evidence": needsEvidence,
    };
    const results = FINDING_STATES.map((value) => byState[value]);
    const inScope = (check: CrossProjectCheck) =>
      project === undefined || check.project.projectId === project;
    const scopedChecks = checks.checks.filter(inScope);
    const checkById = new Map(
      scopedChecks.map((check) => [check.audit.auditId, check]),
    );

    const scoped = new Map<AuditFindingState, CrossProjectIssue[]>();
    const counts: Partial<Record<StateFilter, IssueCount>> = {};
    for (const value of FINDING_STATES) {
      const result = byState[value];
      const issues = result.issues.filter(inScope);
      scoped.set(value, issues);
      const truncatedHere = result.truncatedAuditIds.some((auditId) =>
        issues.some((issue) => issue.audit.auditId === auditId),
      );
      counts[value] =
        project === undefined
          ? result.total
          : result.isPending && issues.length === 0
            ? undefined
            : truncatedHere
              ? `${issues.length}+`
              : issues.length;
    }
    counts.all = sumCounts(FINDING_STATES.map((value) => counts[value]));

    let issues: CrossProjectIssue[];
    if (state === "all") {
      // During a refetch the previous revision of a check stays listed, so
      // one possible issue can sit in two state lists; the newest wins.
      const newest = new Map<string, CrossProjectIssue>();
      for (const issue of FINDING_STATES.flatMap(
        (value) => scoped.get(value) ?? [],
      )) {
        const key = issueKey(issue);
        const known = newest.get(key);
        if (
          known === undefined ||
          issue.finding.revision > known.finding.revision
        )
          newest.set(key, issue);
      }
      issues = [...newest.values()].sort(newerIssue);
    } else {
      issues = scoped.get(state) ?? [];
    }

    const shownStates = state === "all" ? FINDING_STATES : [state];
    const truncatedIds = new Set(
      shownStates.flatMap((value) => byState[value].truncatedAuditIds),
    );
    const truncatedChecks = [...truncatedIds].flatMap((auditId) => {
      const check = checkById.get(auditId);
      return check === undefined ? [] : [check];
    });

    const errors = new Map<string, CrossProjectError>();
    for (const error of [
      ...checks.errors,
      ...results.flatMap((result) => result.errors),
    ])
      errors.set(errorKey(error), error);

    return {
      issues,
      counts: counts as Record<StateFilter, IssueCount>,
      pending: checks.isPending || results.some((result) => result.isPending),
      error: checks.error,
      errors: [...errors.values()].filter(
        (error) =>
          error.scope === "index" ||
          project === undefined ||
          error.projectId === project,
      ),
      truncatedChecks,
      checksTruncated: checks.truncated,
      checks: scopedChecks,
      running: scopedChecks.filter((check) =>
        RUNNING_CHECK_STATES.has(check.audit.state),
      ),
      refetch: checks.refetch,
    };
  }, [
    checks,
    confirmed,
    duplicate,
    needsEvidence,
    project,
    proposed,
    rejected,
    state,
  ]);
}
