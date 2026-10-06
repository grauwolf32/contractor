import type { ReactNode } from "react";

import type { StatusTone } from "../app/status-tone";

// 24×24 stroke shapes from the V3B mockups.
const SHAPES: Readonly<Record<StatusTone, ReactNode>> = {
  done: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M8.3 12.3l2.5 2.5 5-5" />
    </>
  ),
  partial: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 3.5a8.5 8.5 0 0 0 0 17z" fill="currentColor" />
    </>
  ),
  progress: (
    <>
      <circle cx="12" cy="12" r="8.5" opacity="0.3" />
      <path d="M12 3.5a8.5 8.5 0 0 1 8.5 8.5" />
    </>
  ),
  blocked: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M6 6l12 12" />
    </>
  ),
  idle: <circle cx="12" cy="12" r="8.5" strokeDasharray="2.6 2.8" />,
  review: (
    <>
      <path d="M12 3.2l8.8 8.8-8.8 8.8L3.2 12z" />
      <path d="M12 8.2v4.6M12 15.8v.01" />
    </>
  ),
  warning: (
    <>
      <path d="M12 3.8 21 19.5H3z" />
      <path d="M12 9.8v4.2M12 16.9v.01" />
    </>
  ),
  success: <path d="M5 12.5l4.2 4.2L19 7" />,
  info: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 11.2v5M12 7.9v.01" />
    </>
  ),
  neutral: <circle cx="12" cy="12" r="3.5" fill="currentColor" stroke="none" />,
};

export interface StatusGlyphProps {
  tone: StatusTone;
  /** Pixel size of the square icon. Default 16. */
  size?: number | undefined;
  /**
   * Accessible name. Without it the glyph is decorative (aria-hidden) and the
   * status word must be visible next to it.
   */
  label?: string | undefined;
}

/** Stroke icon for a status tone, coloured by tone. */
export function StatusGlyph({ tone, size = 16, label }: StatusGlyphProps) {
  const accessibility =
    label === undefined || label === ""
      ? ({ "aria-hidden": true } as const)
      : ({ role: "img", "aria-label": label } as const);
  return (
    <svg
      className="ui-glyph"
      data-tone={tone}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      focusable="false"
      {...accessibility}
    >
      {SHAPES[tone]}
    </svg>
  );
}

export interface StatusChipProps {
  tone: StatusTone;
  /** The status word. Always visible: a chip never relies on colour alone. */
  children: ReactNode;
  /** Show the tone glyph before the word. Default true. */
  glyph?: boolean | undefined;
  /** Default "md". */
  size?: "sm" | "md" | undefined;
}

/** Pill with a status glyph and word on the tone's tinted background. */
export function StatusChip({
  tone,
  children,
  glyph = true,
  size = "md",
}: StatusChipProps) {
  return (
    <span className="ui-status-chip" data-tone={tone} data-size={size}>
      {glyph ? (
        <StatusGlyph tone={tone} size={size === "sm" ? 12 : 13} />
      ) : null}
      {children}
    </span>
  );
}

export interface ProgressSegment {
  tone: StatusTone;
  /** Shown as the segment's hover title. */
  label: string;
}

export interface ProgressSegmentsProps {
  /** One segment per item, in list order. */
  segments: readonly ProgressSegment[];
  /** Accessible summary of the whole line, e.g. "0 of 5 done: 1 blocked, …". */
  label: string;
  /** "md" = 8px bars (default), "sm" = 4px bars for list rows. */
  size?: "sm" | "md" | undefined;
}

/** Above this many items, per-item bars sit 2 px apart instead of 3 px. */
const COMPACT_ABOVE = 40;
/**
 * Above this many items, per-item bars get too thin to see in a list row, so
 * consecutive items of one tone merge into one bar.
 */
const RUNS_ABOVE = 60;
/** Labels named in a merged bar's hover title before "and N more". */
const TITLE_LABELS = 3;

interface SegmentRun {
  tone: StatusTone;
  labels: string[];
}

function toRuns(segments: readonly ProgressSegment[]): SegmentRun[] {
  const runs: SegmentRun[] = [];
  for (const segment of segments) {
    const last = runs[runs.length - 1];
    if (last !== undefined && last.tone === segment.tone) {
      last.labels.push(segment.label);
    } else {
      runs.push({ tone: segment.tone, labels: [segment.label] });
    }
  }
  return runs;
}

/** "Met (12)", "Met (3), Fully traced (2)": labels in order, with counts. */
function runTitle(labels: readonly string[]): string {
  const counts = new Map<string, number>();
  for (const label of labels) counts.set(label, (counts.get(label) ?? 0) + 1);
  const parts = Array.from(counts, ([label, count]) =>
    count === 1 ? label : `${label} (${count})`,
  );
  const named = parts.slice(0, TITLE_LABELS).join(", ");
  return parts.length > TITLE_LABELS
    ? `${named} and ${parts.length - TITLE_LABELS} more`
    : named;
}

/**
 * Segment line coloured by tone. Up to 60 items it shows one equal-width bar
 * per item; longer lines merge consecutive items of one tone into a bar as
 * wide as their share, so order and proportions survive in a narrow row.
 */
export function ProgressSegments({
  segments,
  label,
  size = "md",
}: ProgressSegmentsProps) {
  const merged = segments.length > RUNS_ABOVE;
  const density = merged
    ? "runs"
    : segments.length > COMPACT_ABOVE
      ? "compact"
      : undefined;
  return (
    <div
      className="ui-segments"
      role="img"
      aria-label={label}
      data-size={size}
      data-density={density}
    >
      {merged
        ? toRuns(segments).map((run, position) => (
            <span
              // Runs are positional, in list order.
              key={position}
              className="ui-segment"
              data-tone={run.tone}
              title={runTitle(run.labels)}
              style={{ flexGrow: run.labels.length }}
            />
          ))
        : segments.map((segment, position) => (
            <span
              // Segments are positional: the n-th bar is the n-th item.
              key={position}
              className="ui-segment"
              data-tone={segment.tone}
              title={segment.label}
            />
          ))}
    </div>
  );
}
