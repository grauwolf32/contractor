/**
 * The parts every Inbox detail shares: the detail bar with its position and
 * previous / next, the heading block, titled blocks and the "Decision
 * recorded" line.
 */
import { useId, useLayoutEffect, useRef, type ReactNode } from "react";

import { Link } from "react-router";

import { ContextLink } from "../../app/context-navigation";
import { Icon } from "../../app/icon";
import {
  DetailPane,
  IconButton,
  Button,
  ButtonLink,
  StatusChip,
  StatusGlyph,
  type StatusTone,
} from "../../ui";
import type { InboxData } from "./data";
import type { InboxRow } from "./model";
import { SECTIONS, checkName } from "./present";

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
 *
 * A layout effect, so focus moves in the commit that shows the target. The
 * move after a decision is a navigation, which React commits in a
 * transition and runs passive effects for in a later task: until then
 * focus sat on <body>, as the decision bar that had it was gone.
 */
function useFocusRequest(
  target: { readonly current: HTMLElement | null },
  focus: boolean,
  onFocused: () => void,
) {
  useLayoutEffect(() => {
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

/** A compact continuation when entry did not select an item. */
export function Overview({ context }: { context: DetailContext }) {
  const { data } = context;
  const first = data.model.order[0];
  const heading = useRef<HTMLHeadingElement>(null);
  useFocusRequest(heading, context.focusKey === "", context.onFocused);
  const pending = Object.values(data.pending).some(Boolean);
  let title = "Your Inbox is clear";
  let description =
    "No decisions, stopped work, recent results or running checks are listed.";
  if (pending) {
    title = "Loading your Inbox";
    description = "Reading possible issues, stopped work and results.";
  } else if (data.incomplete) {
    title = "Some Inbox items couldn't be loaded";
    description =
      "Loaded items are still available in the queue. Retry to refresh the missing information.";
  } else if (first !== undefined) {
    title =
      first.type === "issue"
        ? first.issue.finding.title
        : first.type === "report"
          ? `${first.report.project.name} report`
          : first.type === "run"
            ? first.run.workflow
            : checkName(first.check);
    description = "Choose an item from the queue, or continue with this one.";
  }
  return (
    <DetailPane>
      <Recorded text={context.recorded} />
      <div className="inbox-focus-state">
        <StatusGlyph
          tone={
            pending
              ? "progress"
              : data.incomplete
                ? "warning"
                : first === undefined
                  ? "done"
                  : "info"
          }
        />
        {first !== undefined && !pending && !data.incomplete ? (
          <p className="inbox-focus-eyebrow">{SECTIONS[first.section].title}</p>
        ) : null}
        <h2 ref={heading} tabIndex={-1}>
          {title}
        </h2>
        <p role={pending ? "status" : undefined}>{description}</p>
        {pending ? null : data.incomplete ? (
          <Button onClick={() => void data.retry()}>Retry missing items</Button>
        ) : first === undefined ? (
          <div className="inbox-focus-actions">
            <ButtonLink variant="primary" to="/checks/new">
              Start a check
            </ButtonLink>
            <Link to="/projects">Browse projects</Link>
          </div>
        ) : (
          <Button variant="primary" onClick={() => context.onSelect(first)}>
            Continue
          </Button>
        )}
      </div>
    </DetailPane>
  );
}
