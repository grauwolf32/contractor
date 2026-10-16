/**
 * The parts every Inbox detail shares: the detail bar with its position and
 * previous / next, the heading block, titled blocks and the "Decision
 * recorded" line.
 */
import { useEffect, useId, useRef, type ReactNode } from "react";

import { ContextLink } from "../../app/context-navigation";
import { Icon } from "../../app/icon";
import {
  DetailPane,
  IconButton,
  Kbd,
  StatusChip,
  StatusGlyph,
  type StatusTone,
} from "../../ui";
import type { InboxData } from "./data";
import type { InboxRow, InboxSectionId } from "./model";
import { SECTIONS } from "./present";

/** What every detail view needs from the page. */
export interface DetailContext {
  data: InboxData;
  /** The selected item's previous and next rows in the list (J and K). */
  previous: InboxRow | undefined;
  next: InboxRow | undefined;
  /** The decision after the selected one, for the decision bar's Next. */
  nextDecision: InboxRow | undefined;
  /** Selects a row (or nothing); keyboard moves replace the history entry. */
  onSelect: (
    row: InboxRow | undefined,
    options?: { replace?: boolean },
  ) => void;
  /**
   * A decision was recorded on the item with this key and title; the
   * outcome is its label ("Not an issue").
   */
  onDecided: (key: string, outcome: string, title: string) => void;
  /** The item to move focus to once it shows, after a decision. */
  focusKey: string | undefined;
  /** Focus has moved there: the request is used up. */
  onFocused: () => void;
  /** "Decision recorded: …" while it is announced. */
  recorded: string;
}

/**
 * Moves focus to `target` once when `focus` turns on, then reports the
 * request used up, so a later visit to the same item leaves focus alone.
 */
function useFocusRequest(
  target: { readonly current: HTMLElement | null },
  focus: boolean,
  onFocused: () => void,
) {
  useEffect(() => {
    if (!focus) return;
    target.current?.focus();
    onFocused();
  }, [focus, onFocused, target]);
}

/** A link to an item's own page; going back returns to this selection. */
export function OpenLink({
  to,
  children,
}: {
  to: string;
  children: ReactNode;
}) {
  return (
    <ContextLink returnLabel="Inbox" to={to} className="inbox-open">
      {children}
      <Icon name="open" size={14} />
    </ContextLink>
  );
}

/**
 * The top bar of the detail pane (V3B home mockup): where the item is and
 * its position, the link to its own page, and previous / next.
 */
export function DetailBar({
  where,
  label,
  actions,
  context,
  focus,
}: {
  where: string;
  label: string;
  actions?: ReactNode;
  context: DetailContext;
  /** Move focus here once shown: the item arrived after a decision. */
  focus: boolean;
}) {
  const place = useRef<HTMLParagraphElement>(null);
  useFocusRequest(place, focus, context.onFocused);
  const { previous, next, onSelect } = context;
  return (
    <div className="inbox-detail-bar">
      <p className="inbox-detail-where" ref={place} tabIndex={-1}>
        <strong>{where}</strong> <span>{label}</span>
      </p>
      <div className="inbox-detail-actions">
        {actions}
        <IconButton
          className="inbox-step"
          size="sm"
          label="Previous item"
          aria-keyshortcuts="K"
          icon={<Icon name="chevron-up" size={16} />}
          disabled={previous === undefined}
          onClick={() => onSelect(previous, { replace: true })}
        />
        <IconButton
          className="inbox-step"
          size="sm"
          label="Next item"
          aria-keyshortcuts="J"
          icon={<Icon name="chevron-down" size={16} />}
          disabled={next === undefined}
          onClick={() => onSelect(next, { replace: true })}
        />
      </div>
    </div>
  );
}

/** "Decision recorded: …" after the previous item left the list. */
export function Recorded({ text }: { text: string }) {
  if (text === "") return null;
  // The page-level status region announces it; this copy is for the eye.
  return (
    <p className="inbox-recorded" aria-hidden="true">
      <StatusGlyph tone="success" />
      {text}
    </p>
  );
}

/** Heading block of a detail: state chip, title and a meta line. */
export function Head({
  tone,
  state,
  title,
  meta,
}: {
  tone: StatusTone;
  state: string;
  title: ReactNode;
  meta?: ReactNode;
}) {
  return (
    <header className="inbox-head">
      <StatusChip tone={tone}>{state}</StatusChip>
      <h2 className="inbox-title">{title}</h2>
      {meta === undefined ? null : (
        <div className="inbox-head-meta">{meta}</div>
      )}
    </header>
  );
}

/** A titled part of a detail: "Progress", "Primary cause". */
export function Block({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  const id = useId();
  return (
    <section className="inbox-block" aria-labelledby={id}>
      <h3 id={id} className="inbox-block-title">
        {title}
      </h3>
      {children}
    </section>
  );
}

const OVERVIEW_ORDER: readonly InboxSectionId[] = [
  "decide",
  "unblock",
  "ready",
  "running",
];

const OVERVIEW_TONES: Readonly<Record<InboxSectionId, StatusTone>> = {
  decide: "review",
  unblock: "blocked",
  ready: "done",
  running: "progress",
};

/** What the detail pane shows while nothing is selected. */
export function Overview({
  context,
  counts,
}: {
  context: DetailContext;
  counts: Record<InboxSectionId, string>;
}) {
  const { model } = context.data;
  const heading = useRef<HTMLHeadingElement>(null);
  // "": the last decision left nothing to select.
  useFocusRequest(heading, context.focusKey === "", context.onFocused);
  return (
    <DetailPane>
      <Recorded text={context.recorded} />
      <header className="inbox-head">
        <h2 className="inbox-title" ref={heading} tabIndex={-1}>
          Overview
        </h2>
        <p className="inbox-head-meta">
          What needs you across your projects. Choose an item in the list, or
          press <Kbd>J</Kbd> to start with the first one.
        </p>
      </header>
      <ul className="inbox-overview" role="list">
        {OVERVIEW_ORDER.map((id) => {
          const first = model[id][0];
          const content = (
            <>
              <StatusGlyph tone={OVERVIEW_TONES[id]} />
              <span className="inbox-overview-title">{SECTIONS[id].title}</span>
              <span className="inbox-overview-count">{counts[id]}</span>
            </>
          );
          return (
            <li key={id}>
              {first === undefined ? (
                <div className="inbox-overview-item">{content}</div>
              ) : (
                <button
                  type="button"
                  className="inbox-overview-item"
                  onClick={() => context.onSelect(first)}
                >
                  {content}
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </DetailPane>
  );
}
