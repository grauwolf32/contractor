import { useMemo } from "react";

import type { Audit, AuditState } from "../../api/audits";
import { useAllChecks, useAllPossibleIssues } from "../../api/cross-project";

/** What the projects list says about one project's checks. */
export interface ProjectActivity {
  /** How many of the project's listed checks are in each state. */
  states: Partial<Record<AuditState, number>>;
  /**
   * Possible issues that need review across every check of the project. `settled`: every read of the project's checks has
   * answered. `more`: the count is a lower bound, because the project has
   * more checks than were read or a check could not be read.
   */
  possibleIssues: { count: number; more: boolean; settled: boolean };
  /** The project's most recently updated check. */
  latest: Audit | undefined;
}

const NO_ACTIVITY = new Map<string, ProjectActivity>();

/** Activity from complete owner-wide lists, shared with Checks and Issues. */
export function useProjectActivity(): Map<string, ProjectActivity> {
  const checks = useAllChecks();
  const issues = useAllPossibleIssues({ states: ["proposed"] });
  return useMemo(() => {
    if (checks.checks.length === 0) return NO_ACTIVITY;
    const activity = new Map<string, ProjectActivity>();
    for (const { project, audit } of checks.checks) {
      let current = activity.get(project.projectId);
      if (current === undefined) {
        current = {
          states: {},
          possibleIssues: {
            count: 0,
            more: checks.partial || issues.partial,
            settled: !checks.isPending && !issues.isPending,
          },
          latest: audit,
        };
        activity.set(project.projectId, current);
      }
      current.states[audit.state] = (current.states[audit.state] ?? 0) + 1;
    }
    for (const issue of issues.issues) {
      const current = activity.get(issue.project.projectId);
      if (current !== undefined) current.possibleIssues.count += 1;
    }
    return activity;
  }, [checks, issues]);
}
