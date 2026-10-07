import { useId, type ReactNode } from "react";
import { Link, type To } from "react-router";

export interface ListSectionProps {
  /** Small uppercase heading. Without it the section is a plain list. */
  title?: ReactNode;
  /** Shown in monospace after the heading. */
  count?: number | string | undefined;
  /** Short note at the right of the heading ("only you can confirm"). */
  aside?: ReactNode;
  /** Heading level. Default "h2" (under a ListPane h1 title). */
  titleAs?: "h2" | "h3" | undefined;
  /** Names the list (`<ul>`) itself, e.g. a list without a title. */
  "aria-label"?: string | undefined;
  /**
   * Keys that move through the list, declared on the list (`<ul>`) itself,
   * e.g. "J K ArrowDown ArrowUp Enter". Show them with Kbd elsewhere.
   */
  "aria-keyshortcuts"?: string | undefined;
  /** ListRow elements. */
  children: ReactNode;
}

/** A titled group of list rows: heading, count and aside, then the rows. */
export function ListSection({
  title,
  count,
  aside,
  titleAs: Heading = "h2",
  "aria-label": ariaLabel,
  "aria-keyshortcuts": ariaKeyShortcuts,
  children,
}: ListSectionProps) {
  const headingId = useId();
  const rows = (
    // Explicit role: list-style: none drops list semantics in WebKit.
    <ul
      role="list"
      className="ui-list"
      aria-label={ariaLabel}
      aria-keyshortcuts={ariaKeyShortcuts}
    >
      {children}
    </ul>
  );
  if (title === undefined || title === null) return rows;
  return (
    <section className="ui-list-section" aria-labelledby={headingId}>
      <div className="ui-list-section-heading">
        <Heading id={headingId} className="ui-list-section-title">
          {title}
        </Heading>
        {count === undefined ? null : (
          <span className="ui-list-section-count">{count}</span>
        )}
        {aside === undefined ? null : (
          <span className="ui-list-section-aside">{aside}</span>
        )}
      </div>
      {rows}
    </section>
  );
}

function isPresent(part: ReactNode): boolean {
  return part !== null && part !== undefined && part !== false && part !== "";
}

/**
 * An array meta: the parts with "·" separators. Each separator starts the
 * part it precedes, so on a wrapped line it travels with that part and the
 * meta line clips it (ui.css) instead of leaving a "·" at the end of the
 * line above.
 */
function renderMeta(meta: ReactNode): ReactNode {
  if (!Array.isArray(meta)) return meta;
  return (
    <span className="ui-row-parts">
      {(meta as ReactNode[]).filter(isPresent).map((part, position) => (
        // Meta parts are positional.
        <span key={position} className="ui-row-part">
          {position > 0 ? (
            <span className="ui-row-sep" aria-hidden="true">
              ·
            </span>
          ) : null}
          <span>{part}</span>
        </span>
      ))}
    </span>
  );
}

export interface ListRowProps {
  /** Selecting the row navigates here (selection lives in the URL). */
  to?: To | undefined;
  /** Selects the row: the title becomes a button, or with `to` also runs on click. */
  onSelect?: (() => void) | undefined;
  /** Selected rows get aria-current="true", the tint and the left bar. */
  selected?: boolean | undefined;
  /** Leading icon, usually a StatusGlyph. Decorative: say the status in `meta`. */
  glyph?: ReactNode;
  /** Row title: the link or button text. */
  title: ReactNode;
  /** Line under the title. An array is joined with "·" separators. */
  meta?: ReactNode;
  /** Right-hand slot, e.g. a "Live" indicator. */
  trailing?: ReactNode;
  /** Title lines before clamping: 1, 2 (default) or false for no clamp. */
  clamp?: 1 | 2 | false | undefined;
  /** Below the title and meta: inline actions (separate buttons), progress. */
  children?: ReactNode;
  /** Id of the `<li>`. */
  id?: string | undefined;
  /**
   * Keys that act on the row, declared on its link or button
   * (`aria-keyshortcuts`), e.g. "J K ArrowDown ArrowUp Enter".
   */
  ariaKeyShortcuts?: string | undefined;
}

/**
 * One list row (`<li>`). The title is a router Link (`to`) or a button
 * (`onSelect`) that covers the whole row; inline actions in `children` and
 * `trailing` stay separate controls above it, never inside the link.
 */
export function ListRow({
  to,
  onSelect,
  selected = false,
  glyph,
  title,
  meta,
  trailing,
  clamp = 2,
  children,
  id,
  ariaKeyShortcuts,
}: ListRowProps) {
  const current = selected ? "true" : undefined;
  const text = (
    <span
      className="ui-row-title"
      data-clamp={clamp === false ? "none" : clamp}
    >
      {title}
    </span>
  );
  let control: ReactNode;
  if (to !== undefined) {
    control = (
      <Link
        to={to}
        className="ui-row-control"
        aria-current={current}
        aria-keyshortcuts={ariaKeyShortcuts}
        onClick={onSelect}
      >
        {text}
      </Link>
    );
  } else if (onSelect !== undefined) {
    control = (
      <button
        type="button"
        className="ui-row-control"
        aria-current={current}
        aria-keyshortcuts={ariaKeyShortcuts}
        onClick={onSelect}
      >
        {text}
      </button>
    );
  } else {
    control = <span className="ui-row-control">{text}</span>;
  }
  const interactive = to !== undefined || onSelect !== undefined;
  return (
    <li
      id={id}
      className="ui-row"
      data-selected={selected ? "" : undefined}
      data-interactive={interactive ? "" : undefined}
    >
      {glyph === undefined || glyph === null ? null : (
        <span className="ui-row-glyph">{glyph}</span>
      )}
      <div className="ui-row-main">
        {control}
        {isPresent(meta) ? (
          <div
            className="ui-row-meta"
            data-parts={Array.isArray(meta) ? "" : undefined}
          >
            {renderMeta(meta)}
          </div>
        ) : null}
        {children === undefined || children === null ? null : (
          <div className="ui-row-extra">{children}</div>
        )}
      </div>
      {trailing === undefined || trailing === null ? null : (
        <div className="ui-row-trailing">{trailing}</div>
      )}
    </li>
  );
}

export interface FilterChipOption<T extends string = string> {
  value: T;
  label: string;
  count?: number | string | undefined;
}

export interface FilterChipsProps<T extends string = string> {
  /** Accessible name of the group, e.g. "Filter by status". */
  label: string;
  options: readonly FilterChipOption<T>[];
  /** The pressed option. */
  value: T;
  onChange: (value: T) => void;
}

/**
 * One-of-many filter as a group of aria-pressed pill buttons with counts.
 * Pressing the chip that is already pressed changes nothing.
 */
export function FilterChips<T extends string = string>({
  label,
  options,
  value,
  onChange,
}: FilterChipsProps<T>) {
  return (
    <div role="group" aria-label={label} className="ui-filter-chips">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="ui-filter-chip"
          aria-pressed={option.value === value}
          onClick={() => {
            if (option.value !== value) onChange(option.value);
          }}
        >
          {option.label}
          {/* The space keeps the accessible name "Needs review 1". */}
          {option.count === undefined ? null : (
            <>
              {" "}
              <span className="ui-filter-chip-count">{option.count}</span>
            </>
          )}
        </button>
      ))}
    </div>
  );
}
