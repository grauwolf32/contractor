import type { AuditCoverageRow } from "../../../api/audits";
import { capitalize, itemNoun, type ItemKind } from "../../../app/vocabulary";

export type Status = AuditCoverageRow["coverage"]["status"];

/**
 * Result filter of a check's item list, kept in the URL as `result`. The
 * values are the URL contract: progress links and the report use them.
 */
export type Group =
  "all" | "issues" | "uncertain" | "not-tested" | "complete" | "out-of-scope";

export const GROUP_IDS: readonly Group[] = [
  "all",
  "issues",
  "uncertain",
  "not-tested",
  "complete",
  "out-of-scope",
];

/** The result filter each coverage status belongs to. */
export const STATUS_GROUPS: Readonly<Record<Status, Exclude<Group, "all">>> = {
  violated: "issues",
  satisfied: "complete",
  "traced-complete": "complete",
  inconclusive: "uncertain",
  blocked: "uncertain",
  "traced-partial": "uncertain",
  unmapped: "uncertain",
  "not-tested": "not-tested",
  "not-applicable": "out-of-scope",
  excluded: "out-of-scope",
};

/**
 * What a coverage status means. Coverage is not a verdict on security: a
 * traced endpoint is only traced, and excluded items never count as met.
 */
export const STATUS_DESCRIPTIONS: Readonly<Record<Status, string>> = {
  violated: "The evidence shows that the requirement is not met.",
  satisfied: "The available evidence meets the requirement.",
  inconclusive: "There is not enough evidence to reach a conclusion.",
  blocked: "The work could not be completed. Read the limitations below.",
  "not-tested": "No result has been recorded yet.",
  "not-applicable": "Reviewed and marked as not applicable to this check.",
  excluded: "Excluded from this check. It does not count as met.",
  "traced-complete":
    "All requested parts of this operation were traced. This is not a security verdict.",
  "traced-partial": "Some parts of this operation could not be traced.",
  unmapped: "The operation could not be mapped to its implementation.",
};

/** The filter group of a status, also for a status this client does not know. */
export function statusGroup(status: Status): Exclude<Group, "all"> {
  return Object.hasOwn(STATUS_GROUPS, status)
    ? STATUS_GROUPS[status]
    : "uncertain";
}

export function statusDescription(status: Status): string | undefined {
  return Object.hasOwn(STATUS_DESCRIPTIONS, status)
    ? STATUS_DESCRIPTIONS[status]
    : undefined;
}

/** Filter chip label; the "complete" group is named after the item kind. */
export function groupLabel(group: Group, kind: ItemKind): string {
  switch (group) {
    case "all":
      return "All";
    case "issues":
      return "Issues found";
    case "uncertain":
      return "Need follow-up";
    case "not-tested":
      return "Not checked yet";
    case "complete":
      return kind === "endpoint"
        ? "Fully traced"
        : kind === "item"
          ? "Met / fully traced"
          : "Met";
    case "out-of-scope":
      return "Not applicable / excluded";
  }
}

/** The `result` URL parameter as a group; anything else is "all". */
export function parseGroup(value: string | null): Group {
  return GROUP_IDS.find((group) => group === value) ?? "all";
}

/** "Endpoints", "Requirements", … for headings and tabs. */
export function itemHeading(kind: ItemKind): string {
  return capitalize(itemNoun(kind, 2));
}
