import type { AuditState } from "../../../api/audits";
import { AUDIT_ID_PATTERN } from "../../../api/audits";
import { PROJECT_ID_PATTERN } from "../../../api/projects";
import { CHECK_STATE_LABELS } from "../../../app/vocabulary";

/**
 * The state filter of the Checks list, `?state=` in the URL. Running holds
 * every check that has started and not ended yet and is not waiting for
 * the user (running, paused, finishing, stopping); deleting checks are
 * listed under All only.
 */
export type CheckFilter =
  "running" | "waiting" | "drafts" | "finished" | "stopped" | "all";

export interface CheckFilterOption {
  value: CheckFilter;
  label: string;
  /** Check states in this filter; empty for All. */
  states: readonly AuditState[];
}

export const CHECK_FILTERS: readonly CheckFilterOption[] = [
  {
    value: "running",
    label: CHECK_STATE_LABELS.active.label,
    states: ["active", "paused", "finalizing", "cancelling"],
  },
  {
    value: "waiting",
    label: CHECK_STATE_LABELS.waiting_review.label,
    states: ["waiting_review"],
  },
  { value: "drafts", label: "Drafts", states: ["draft"] },
  {
    value: "finished",
    label: CHECK_STATE_LABELS.completed.label,
    states: ["completed"],
  },
  {
    value: "stopped",
    label: `${CHECK_STATE_LABELS.cancelled.label} / ${CHECK_STATE_LABELS.failed.label.toLowerCase()}`,
    states: ["cancelled", "failed"],
  },
  { value: "all", label: "All", states: [] },
];

/**
 * `?state=` as a filter. Filter names are the canonical values; a check
 * state (`active`, `waiting_review`, `draft`, …) selects the filter that
 * holds it, so links built from states land on the right list. Anything
 * else is All.
 */
export function parseCheckFilter(value: string | null): CheckFilter {
  if (value === null) return "all";
  const named = CHECK_FILTERS.find((option) => option.value === value);
  if (named !== undefined) return named.value;
  return (
    CHECK_FILTERS.find((option) =>
      option.states.some((state) => state === value),
    )?.value ?? "all"
  );
}

/** Whether a check in `state` is listed under `filter`. */
export function inFilter(filter: CheckFilter, state: AuditState): boolean {
  if (filter === "all") return true;
  return (
    CHECK_FILTERS.find((option) => option.value === filter)?.states.includes(
      state,
    ) ?? false
  );
}

/** `?project=`, when it is a valid project ID. */
export function parseProject(value: string | null): string | undefined {
  return value !== null && PROJECT_ID_PATTERN.test(value) ? value : undefined;
}

/** `?check=`, when it is a valid check ID. */
export function parseCheck(value: string | null): string | undefined {
  return value !== null && AUDIT_ID_PATTERN.test(value) ? value : undefined;
}

/** The Checks list URL with these filters and selection. */
export function checksHref({
  filter,
  projectId,
  checkId,
}: {
  filter: CheckFilter;
  projectId: string | undefined;
  checkId: string | undefined;
}): string {
  const params = new URLSearchParams();
  if (filter !== "all") params.set("state", filter);
  if (projectId !== undefined) params.set("project", projectId);
  if (checkId !== undefined) params.set("check", checkId);
  const search = params.toString();
  return search === "" ? "/checks" : `/checks?${search}`;
}
