import { useMemo } from "react";

import type { Audit, AuditState } from "../../api/audits";
import {
  INBOX_CHECK_STATES,
  useAllChecks,
  useAllPossibleIssues,
} from "../../api/cross-project";

/** States in which a check is doing work on its own. */
export const RUNNING_CHECK_STATES: readonly AuditState[] = [
  "active",
  "finalizing",
  "cancelling",
];

/** What the projects list says about one project's checks. */
export interface ProjectActivity {
  /** Checks that are running, finishing or stopping. */
  running: number;
  /** Checks waiting for a decision. */
  waiting: number;
  /** Checks paused by the owner or a time limit. */
  paused: number;
  /**
   * Possible issues that need review in running and waiting checks (the
   * Inbox's scope). `more` says some check has more than were read.
   */
  possibleIssues: { count: number; more: boolean };
  /** The project's most recently updated check. */
  latest: Audit | undefined;
}

const NO_ACTIVITY = new Map<string, ProjectActivity>();

/**
 * Per-project activity for the projects list, from the bounded cross-project
 * reads (api/cross-project.ts): every check of the first page of projects,
 * and the possible issues of running and waiting checks. These are the reads
 * the rail's Inbox badge already polls, so the list adds no requests of its
 * own; possible issues of finished checks are counted on the project's own
 * overview instead.
 */
export function useProjectActivity(): Map<string, ProjectActivity> {
  const checks = useAllChecks();
  const issues = useAllPossibleIssues({
    checkStates: INBOX_CHECK_STATES,
    states: ["proposed"],
  });
  return useMemo(() => {
    if (checks.checks.length === 0) return NO_ACTIVITY;
    const truncated = new Set(issues.truncatedAuditIds);
    const activity = new Map<string, ProjectActivity>();
    const entry = (projectId: string): ProjectActivity => {
      let current = activity.get(projectId);
      if (current === undefined) {
        current = {
          running: 0,
          waiting: 0,
          paused: 0,
          possibleIssues: { count: 0, more: false },
          latest: undefined,
        };
        activity.set(projectId, current);
      }
      return current;
    };
    // Checks arrive newest first by last update.
    for (const { project, audit } of checks.checks) {
      const current = entry(project.projectId);
      current.latest ??= audit;
      if (RUNNING_CHECK_STATES.includes(audit.state)) current.running += 1;
      else if (audit.state === "waiting_review") current.waiting += 1;
      else if (audit.state === "paused") current.paused += 1;
      if (truncated.has(audit.auditId)) current.possibleIssues.more = true;
    }
    for (const { project } of issues.issues) {
      entry(project.projectId).possibleIssues.count += 1;
    }
    return activity;
  }, [checks.checks, issues.issues, issues.truncatedAuditIds]);
}
