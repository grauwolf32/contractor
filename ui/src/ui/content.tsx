import type { ReactNode } from "react";

import type { StatusTone } from "../app/status-tone";
import { clockTime } from "./format";
import { StatusGlyph } from "./status";

export interface TechnicalDetailsProps {
  /** Summary text. Default "Technical details". */
  summary?: string | undefined;
  /** One quiet line under the summary, e.g. who it is for. */
  description?: ReactNode;
  children: ReactNode;
  /** Start expanded. Default false. */
  defaultOpen?: boolean | undefined;
  /** Added next to "ui-tech" on the `<details>`, e.g. a hook for specs. */
  className?: string | undefined;
  /**
   * Called with the new state whenever it opens or closes, e.g. to render
   * costly content only while it is open.
   */
  onToggle?: ((open: boolean) => void) | undefined;
}

/**
 * Native disclosure styled as a quiet link: internals (revisions, digests,
 * rounds, slots, tokens, allocations) live behind it.
 */
export function TechnicalDetails({
  summary = "Technical details",
  description,
  children,
  defaultOpen = false,
  className,
  onToggle,
}: TechnicalDetailsProps) {
  return (
    <details
      className={className === undefined ? "ui-tech" : `ui-tech ${className}`}
      open={defaultOpen || undefined}
      onToggle={
        onToggle === undefined
          ? undefined
          : (event) => onToggle(event.currentTarget.open)
      }
    >
      <summary>
        <span className="ui-tech-label">
          <svg
            className="ui-tech-chevron"
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            focusable="false"
          >
            <path d="M9.5 6l6 6-6 6" />
          </svg>
          {summary}
        </span>
        {description === undefined ? null : (
          <span className="ui-tech-description">{description}</span>
        )}
      </summary>
      <div className="ui-tech-body">{children}</div>
    </details>
  );
}

export interface ActivityEntry {
  id: string;
  /** A date (shown as local HH:MM) or a short text such as "now". */
  time?: string | Date | undefined;
  /** Glyph tone for a timed entry. Default "neutral". */
  tone?: StatusTone | undefined;
  /** Bold lead-in. */
  title?: ReactNode;
  text?: ReactNode;
}

export interface ActivityLogProps {
  /** Entries in display order (newest first or oldest first is the caller's choice). */
  entries: readonly ActivityEntry[];
  "aria-label": string;
}

/**
 * Timeline: a time column, a glyph and the text. Entries without a time are
 * steps of the entry above and get a small dot instead of a glyph.
 */
export function ActivityLog({
  entries,
  "aria-label": ariaLabel,
}: ActivityLogProps) {
  return (
    <ol role="list" className="ui-activity" aria-label={ariaLabel}>
      {entries.map((entry) => {
        const time = clockTime(entry.time);
        const tone = entry.tone ?? "neutral";
        return (
          <li
            key={entry.id}
            className="ui-activity-entry"
            data-timed={time === undefined ? undefined : ""}
          >
            <span className="ui-activity-time">
              {time === undefined ? null : time.dateTime === undefined ? (
                time.label
              ) : (
                <time dateTime={time.dateTime} title={time.title}>
                  {time.label}
                </time>
              )}
            </span>
            {time === undefined ? (
              <span className="ui-activity-marker">
                <span className="ui-activity-dot" />
              </span>
            ) : (
              <span className="ui-activity-marker" data-tone={tone}>
                <StatusGlyph tone={tone} size={14} />
              </span>
            )}
            <p className="ui-activity-text">
              {entry.title === undefined ? null : (
                <strong>{entry.title}</strong>
              )}
              {entry.title !== undefined && entry.text !== undefined
                ? " "
                : null}
              {entry.text}
            </p>
          </li>
        );
      })}
    </ol>
  );
}

export interface EmptyStateProps {
  title: ReactNode;
  /** Explanation under the title. */
  children?: ReactNode;
  /** A next step, e.g. a link or button. */
  action?: ReactNode;
}

/** What an empty list or an unselected detail pane shows. */
export function EmptyState({ title, children, action }: EmptyStateProps) {
  return (
    <div className="ui-empty">
      <p className="ui-empty-title">{title}</p>
      {children === undefined || children === null ? null : (
        <div className="ui-empty-body">{children}</div>
      )}
      {action === undefined || action === null ? null : (
        <div className="ui-empty-action">{action}</div>
      )}
    </div>
  );
}
