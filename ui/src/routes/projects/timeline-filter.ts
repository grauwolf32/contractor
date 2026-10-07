/** Which events the project timeline shows (`?timeline=` on the Overview). */
export type TimelineFilter = "all" | "checks" | "issues" | "runs" | "materials";

export const TIMELINE_FILTERS: readonly {
  value: TimelineFilter;
  label: string;
}[] = [
  { value: "all", label: "All" },
  { value: "checks", label: "Checks" },
  { value: "issues", label: "Possible issues" },
  { value: "runs", label: "Runs" },
  { value: "materials", label: "Materials" },
];

/** The filter a URL value names; anything else is "all". */
export function parseTimelineFilter(value: string | null): TimelineFilter {
  return (
    TIMELINE_FILTERS.find((option) => option.value === value)?.value ?? "all"
  );
}
