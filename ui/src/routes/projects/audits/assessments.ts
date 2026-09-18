import type { AuditCoverageRow } from "../../../api/audits";

export type Status = AuditCoverageRow["coverage"]["status"];
export type Group =
  "all" | "issues" | "uncertain" | "not-tested" | "complete" | "out-of-scope";

export const ASSESSMENTS: Record<
  Status,
  { label: string; description: string; group: Group }
> = {
  violated: {
    label: "Issue found",
    description: "The check found evidence that the requirement is violated.",
    group: "issues",
  },
  satisfied: {
    label: "Requirement met",
    description: "The available evidence satisfies this check.",
    group: "complete",
  },
  inconclusive: {
    label: "Inconclusive",
    description: "There is not enough evidence to reach a conclusion.",
    group: "uncertain",
  },
  blocked: {
    label: "Blocked",
    description:
      "The check could not be completed. Review the limitations below.",
    group: "uncertain",
  },
  "not-tested": {
    label: "Not checked yet",
    description: "No assessment has been recorded for this check.",
    group: "not-tested",
  },
  "not-applicable": {
    label: "Not applicable",
    description: "Reviewed and marked as not applicable to this audit.",
    group: "out-of-scope",
  },
  excluded: {
    label: "Excluded",
    description: "This check was excluded from the assessment.",
    group: "out-of-scope",
  },
  "traced-complete": {
    label: "Fully traced",
    description:
      "All requested parts of this operation were traced. This is not a security verdict.",
    group: "complete",
  },
  "traced-partial": {
    label: "Partially traced",
    description: "Some parts of this operation could not be traced.",
    group: "uncertain",
  },
  unmapped: {
    label: "Not mapped",
    description: "The operation could not be mapped to its implementation.",
    group: "uncertain",
  },
};

export const GROUPS: { id: Group; label: string }[] = [
  { id: "all", label: "All checks" },
  { id: "issues", label: "Issues found" },
  { id: "uncertain", label: "Need follow-up" },
  { id: "not-tested", label: "Not checked yet" },
  { id: "complete", label: "Met / fully traced" },
  { id: "out-of-scope", label: "Not applicable / excluded" },
];
