/** Local checkbox selection, bounded to the four event kinds. */
export type TimelineKind = "checks" | "issues" | "runs" | "materials";
export type TimelineFilter = "all" | readonly TimelineKind[];

export const TIMELINE_FILTERS: readonly {
  value: TimelineKind;
  label: string;
}[] = [
  { value: "checks", label: "Checks" },
  { value: "issues", label: "Possible issues" },
  { value: "runs", label: "Runs" },
  { value: "materials", label: "Materials" },
];

/** Single and combined selections use the same bounded URL grammar. */
export function parseTimelineFilter(value: string | null): TimelineFilter {
  if (value === "none") return [];
  if (!value || value === "all") return "all";
  const parts = value.split(",");
  if (
    parts.length > TIMELINE_FILTERS.length ||
    parts.some((part) => !TIMELINE_FILTERS.some(({ value }) => value === part))
  )
    return "all";
  const selected = TIMELINE_FILTERS.filter(({ value }) =>
    parts.includes(value),
  ).map(({ value }) => value);
  return selected.length === TIMELINE_FILTERS.length ? "all" : selected;
}

export function serializeTimelineFilter(filter: TimelineFilter): string {
  return filter === "all"
    ? "all"
    : filter.length === 0
      ? "none"
      : TIMELINE_FILTERS.filter(({ value }) => filter.includes(value))
          .map(({ value }) => value)
          .join(",");
}
