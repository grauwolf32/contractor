import { useMemo } from "react";

import type { Audit, AuditState } from "../../api/audits";
import { CROSS_PROJECT_LIMITS, useAllChecks } from "../../api/cross-project";
import { usePossibleIssuesToReview } from "./overview-data";

/** What the projects list says about one project's checks. */
export interface ProjectActivity {
  /** How many of the project's listed checks are in each state. */
  states: Partial<Record<AuditState, number>>;
  /**
   * Possible issues that need review in the project's listed checks (its 50
   * newest), counted by the Server per check: the count the project's
   * overview shows. `settled`: every read of the project's checks has
   * answered. `more`: the count is a lower bound, because the project has
   * more checks than were read or a check could not be read.
   */
  possibleIssues: { count: number; more: boolean; settled: boolean };
  /** The project's most recently updated check. */
  latest: Audit | undefined;
}

const NO_ACTIVITY = new Map<string, ProjectActivity>();

/**
 * Per-project activity for the projects list, from the bounded cross-project
 * reads (api/cross-project.ts): the first page of checks of every project in
 * the first page of projects (the reads the rail's Inbox badge polls), and
 * the possible issues of each of those checks that can hold any. The
 * possible-issue reads are the ones the project's overview and the Issues
 * page use: keyed by the check's revision, each is read once per change of
 * its check.
 */
export function useProjectActivity(): Map<string, ProjectActivity> {
  const checks = useAllChecks();
  const audits = useMemo(
    () => checks.checks.map(({ audit }) => audit),
    [checks.checks],
  );
  const issues = usePossibleIssuesToReview(audits);
  return useMemo(() => {
    if (checks.checks.length === 0) return NO_ACTIVITY;
    const activity = new Map<string, ProjectActivity>();
    const listed = new Map<string, number>();
    const entry = (projectId: string): ProjectActivity => {
      let current = activity.get(projectId);
      if (current === undefined) {
        current = {
          states: {},
          possibleIssues: { count: 0, more: false, settled: true },
          latest: undefined,
        };
        activity.set(projectId, current);
      }
      return current;
    };
    const pending = new Set(issues.pendingAuditIds);
    const failed = new Set(issues.failedAuditIds);
    // Checks arrive newest first by last update.
    for (const { project, audit } of checks.checks) {
      const current = entry(project.projectId);
      current.latest ??= audit;
      current.states[audit.state] = (current.states[audit.state] ?? 0) + 1;
      listed.set(project.projectId, (listed.get(project.projectId) ?? 0) + 1);
      if (pending.has(audit.auditId)) current.possibleIssues.settled = false;
      if (failed.has(audit.auditId)) current.possibleIssues.more = true;
    }
    for (const { audit, total } of issues.checks) {
      const current = activity.get(audit.projectId);
      if (current !== undefined) current.possibleIssues.count += total;
    }
    // Only a project with a full page of checks can have more of them.
    if (checks.truncated) {
      for (const [projectId, count] of listed) {
        if (count >= CROSS_PROJECT_LIMITS.auditsPerProject)
          entry(projectId).possibleIssues.more = true;
      }
    }
    return activity;
  }, [checks.checks, checks.truncated, issues]);
}
