/**
 * The check page's view of its work items: coverage rows joined with the
 * item collection (kind, execution state, attempts) and the check's possible
 * issues. Pure functions, so the list, the item view and the progress header
 * read the same model.
 */
import type {
  AuditCoverageRow,
  AuditFinding,
  AuditItem,
  AuditWorkspace,
} from "../../../api/audits";
import type { StatusTone } from "../../../app/status-tone";
import {
  checkItemKind,
  coverageStatusLabel,
  type ItemKind,
} from "../../../app/vocabulary";
import { httpOperation } from "../../decisions/text";
import { statusGroup, type Group } from "./assessments";
import {
  isOpaqueSubject,
  itemOperation,
  type ItemOperation,
} from "./check-title";

/** How an item reads in lists, chips and the progress line. */
export interface ItemStatus {
  label: string;
  tone: StatusTone;
  /** The result filter the item belongs to. */
  group: Exclude<Group, "all">;
}

/** One endpoint, requirement, scenario or item of the current round. */
export interface CheckEntry {
  row: AuditCoverageRow;
  /** From the item collection; undefined while it loads or beyond its pages. */
  item: AuditItem | undefined;
  kind: ItemKind;
  /** The traced operation of an endpoint. */
  operation: ItemOperation | undefined;
  /** Requirement or scenario key; for an endpoint its subject key. */
  key: string;
  /** First line of the task (a requirement's statement), or "". */
  summary: string;
  status: ItemStatus;
  /** Possible issues and issues proposed for this item. */
  findings: AuditFinding[];
  /** 1-based position in the round. */
  position: number;
}

/** A coverage status, or the work going on while it has none yet. */
export function itemStatus(
  row: Pick<AuditCoverageRow, "coverage">,
  item: Pick<AuditItem, "state"> | undefined,
): ItemStatus {
  const status = row.coverage.status;
  if (status === "not-tested" && item !== undefined) {
    if (item.state === "submitted" || item.state === "collecting")
      return { label: "Checking now", tone: "progress", group: "not-tested" };
    if (item.state === "awaiting_review")
      return { label: "Waiting for you", tone: "review", group: "not-tested" };
  }
  const { label, tone } = coverageStatusLabel(status);
  return { label, tone, group: statusGroup(status) };
}

/** First non-empty line of Markdown text, without list or heading marks. */
export function firstLine(text: string, limit = 200): string {
  const line =
    text
      .split("\n")
      .map((candidate) => candidate.replace(/^[#>*\-\s]+/u, "").trim())
      .find((candidate) => candidate !== "") ?? "";
  return line.length > limit ? `${line.slice(0, limit)}…` : line;
}

/** The path of an HTTP operation subject ("GET https://h/p" → "/p"). */
function subjectOperation(key: string): ItemOperation | undefined {
  const operation = httpOperation(key);
  if (operation === undefined) return undefined;
  if (operation.path.startsWith("/")) return operation;
  try {
    return { method: operation.method, path: new URL(operation.path).pathname };
  } catch {
    return undefined;
  }
}

/**
 * Whether a possible issue belongs to an item: its current assessment names
 * the item, or its subject is the item's subject or traced operation.
 */
export function findingOnItem(
  finding: AuditFinding,
  row: Pick<AuditCoverageRow, "itemId" | "itemKey" | "subjectKey">,
  operation: ItemOperation | undefined,
): boolean {
  if (finding.currentAssessment?.itemId === row.itemId) return true;
  const subject = finding.firstProposal.document.subject;
  if (subject === null) return false;
  const key = subject.key.trim();
  if (key === row.subjectKey || key === row.itemKey) return true;
  if (operation === undefined) return false;
  const proposed = subjectOperation(key);
  return (
    proposed !== undefined &&
    proposed.method === operation.method &&
    proposed.path === operation.path
  );
}

/**
 * Joins coverage rows, items and possible issues. `fallbackKind` names items
 * whose collection entry is not loaded (from the check type or baseline).
 */
export function buildEntries(
  rows: readonly AuditCoverageRow[],
  items: readonly AuditItem[],
  findings: readonly AuditFinding[],
  fallbackKind: ItemKind,
): CheckEntry[] {
  const itemsById = new Map(items.map((item) => [item.itemId, item]));
  return rows.map((row, index) => {
    const item = itemsById.get(row.itemId);
    const operation = itemOperation(row);
    const kind =
      item === undefined
        ? operation === undefined
          ? fallbackKind
          : "endpoint"
        : checkItemKind(item);
    const summary =
      row.details === undefined ? "" : firstLine(row.details.objective);
    return {
      row,
      item,
      kind: operation !== undefined && kind === "item" ? "endpoint" : kind,
      operation,
      key: row.subjectKey,
      summary:
        operation !== undefined &&
        summary === `${operation.method} ${operation.path}`
          ? ""
          : summary,
      status: itemStatus(row, item),
      findings: findings.filter((finding) =>
        findingOnItem(finding, row, operation),
      ),
      position: index + 1,
    };
  });
}

/** Plain-text name: "GET /orders/{id}", "A01:2025 Trace access control". */
export function entryName(entry: CheckEntry): string {
  if (entry.operation !== undefined)
    return `${entry.operation.method} ${entry.operation.path}`;
  if (isOpaqueSubject(entry.key)) return `Item ${entry.row.ordinal + 1}`;
  return entry.summary === "" ? entry.key : `${entry.key} ${entry.summary}`;
}

/** Text an item search matches: task, result, evidence and attempts. */
export function entrySearchText(entry: CheckEntry): string {
  const { row, item } = entry;
  return [
    entryName(entry),
    row.subjectKey,
    row.itemKey,
    row.details?.objective,
    row.details?.resultSummary,
    row.coverage.rationale,
    entry.status.label,
    ...row.coverage.gaps,
    ...row.coverage.requested,
    ...(row.details?.evidence.map((evidence) => evidence.summary) ?? []),
    ...(item === undefined
      ? []
      : [
          item.state,
          item.workflowRole,
          item.finalDisposition,
          ...item.attempts.flatMap((attempt) => [
            attempt.state,
            attempt.terminalOutcome,
            attempt.collectionDisposition,
            attempt.runId,
          ]),
        ]),
    ...entry.findings.map((finding) => finding.firstProposal.document.title),
  ]
    .filter((value): value is string => typeof value === "string")
    .join(" ")
    .toLocaleLowerCase();
}

/** Entries in a result group whose text contains the search. */
export function filterEntries(
  entries: readonly CheckEntry[],
  group: Group,
  search: string,
): CheckEntry[] {
  const needle = search.trim().toLocaleLowerCase();
  return entries.filter(
    (entry) =>
      (group === "all" || entry.status.group === group) &&
      (needle === "" || entrySearchText(entry).includes(needle)),
  );
}

/** Entries of one kind, in round order, under their list heading. */
export interface EntryGroup {
  kind: ItemKind;
  entries: CheckEntry[];
}

/** Groups entries by kind, in order of each kind's first entry. */
export function groupByKind(entries: readonly CheckEntry[]): EntryGroup[] {
  const groups: EntryGroup[] = [];
  for (const entry of entries) {
    const group = groups.find((candidate) => candidate.kind === entry.kind);
    if (group === undefined)
      groups.push({ kind: entry.kind, entries: [entry] });
    else group.entries.push(entry);
  }
  return groups;
}

/**
 * The longest shared directory of endpoint paths ("/workshop/api/"), or ""
 * when there is none worth naming. Every path keeps at least its last part.
 */
export function commonPathPrefix(paths: readonly string[]): string {
  if (paths.length < 2) return "";
  const split = paths.map((path) =>
    path.split("/").filter((segment) => segment !== ""),
  );
  const shortest = Math.min(...split.map((segments) => segments.length - 1));
  const shared: string[] = [];
  for (let index = 0; index < shortest; index += 1) {
    const segment = split[0]![index]!;
    if (split.every((segments) => segments[index] === segment))
      shared.push(segment);
    else break;
  }
  return shared.length === 0 ? "" : `/${shared.join("/")}/`;
}

/** A path without the list's shared prefix. */
export function relativePath(path: string, prefix: string): string {
  return prefix !== "" && path.startsWith(prefix)
    ? path.slice(prefix.length)
    : path;
}

/** "mechanic" for "mechanic/mechanic_report": the first part of a nested path. */
export function endpointArea(path: string, prefix: string): string | undefined {
  const segments = relativePath(path, prefix)
    .split("/")
    .filter((segment) => segment !== "");
  return segments.length > 1 && !segments[0]!.startsWith("{")
    ? segments[0]
    : undefined;
}

function plural(count: number, singular: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? singular : many}`;
}

/** "1 issue · 2 possible issues" for the issues proposed on one item. */
export function issueSummary(findings: readonly AuditFinding[]): string[] {
  const confirmed = findings.filter(
    (finding) => finding.state === "confirmed",
  ).length;
  const possible = findings.filter(
    (finding) =>
      finding.state === "proposed" || finding.state === "needs-evidence",
  ).length;
  return [
    confirmed === 0 ? "" : plural(confirmed, "issue", "issues"),
    possible === 0 ? "" : plural(possible, "possible issue", "possible issues"),
  ].filter((part) => part !== "");
}

/** Concluded coverage: what the Server counts as completed work. */
const CONCLUDED = new Set([
  "satisfied",
  "violated",
  "traced-complete",
  "not-applicable",
  "excluded",
]);

export function isConcluded(entry: CheckEntry): boolean {
  return CONCLUDED.has(entry.row.coverage.status);
}

/** One legend entry: a status, its tone, how many items have it. */
export interface LegendEntry {
  label: string;
  tone: StatusTone;
  /** The result filter its link opens; "all" when no one filter holds it. */
  group: Group;
  count: number;
}

const TONE_ORDER: readonly StatusTone[] = [
  "done",
  "success",
  "partial",
  "warning",
  "review",
  "progress",
  "info",
  "blocked",
  "neutral",
  "idle",
];

/** Item statuses with their counts, finished work first, waiting work last. */
export function legend(entries: readonly CheckEntry[]): LegendEntry[] {
  const byLabel = new Map<string, LegendEntry>();
  for (const { status } of entries) {
    const known = byLabel.get(status.label);
    if (known === undefined) byLabel.set(status.label, { ...status, count: 1 });
    else known.count += 1;
  }
  return [...byLabel.values()].sort(
    (left, right) =>
      TONE_ORDER.indexOf(left.tone) - TONE_ORDER.indexOf(right.tone),
  );
}

/** "1 partially traced, 2 not checked yet" for an accessible summary. */
export function legendSentence(entries: readonly LegendEntry[]): string {
  return entries
    .map((entry) => `${entry.count} ${entry.label.toLocaleLowerCase()}`)
    .join(", ");
}

/** Work counts of a check, from its workspace snapshot. */
export interface WorkCounts {
  total: number;
  done: number;
  issues: number;
  followUp: number;
  unchecked: number;
}

export function workCounts(workspace: AuditWorkspace): WorkCounts {
  return {
    total: workspace.totalChecks,
    done: Math.max(0, workspace.completedChecks - workspace.issues),
    issues: workspace.issues,
    followUp: workspace.gaps,
    unchecked: workspace.unchecked,
  };
}

/**
 * Legend of a check whose items are not listed: concluded work (met, traced,
 * excluded), issues found, work that needs follow-up and unchecked work.
 */
export function countLegend(counts: WorkCounts): LegendEntry[] {
  return [
    // Done also holds not applicable and excluded work: no one filter.
    { label: "Done", tone: "done", group: "all", count: counts.done },
    {
      label: "Issues found",
      tone: "blocked",
      group: "issues",
      count: counts.issues,
    },
    {
      label: "Need follow-up",
      tone: "partial",
      group: "uncertain",
      count: counts.followUp,
    },
    {
      label: "Not checked yet",
      tone: "idle",
      group: "not-tested",
      count: counts.unchecked,
    },
  ];
}

/** One segment per counted item, in legend order. */
export function countSegments(
  entries: readonly LegendEntry[],
): { tone: StatusTone; label: string }[] {
  return entries.flatMap((entry) =>
    Array.from({ length: Math.max(0, entry.count) }, () => ({
      tone: entry.tone,
      label: entry.label,
    })),
  );
}

/** "3 of 10 done", from the Server's counts when it has them. */
export function doneSummary(done: number, total: number): string {
  return `${done.toLocaleString("en-US")} of ${total.toLocaleString("en-US")} done`;
}
