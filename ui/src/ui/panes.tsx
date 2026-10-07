import { useEffect, useRef, type ReactNode, type Ref } from "react";
import { Link, type To } from "react-router";

export interface PaneLayoutProps {
  /** The list pane content, usually a ListPane. */
  list: ReactNode;
  /** The detail pane content, usually a DetailPane or an EmptyState. */
  detail: ReactNode;
  /** On one-pane screens (≤ 820 px) show the detail instead of the list. */
  showDetail: boolean;
  /** One-pane screens show this link above the detail, e.g. "Back to checks". */
  backLink?: { to: To; label: string } | undefined;
  /** Accessible name of the list `<section>`. */
  listLabel: string;
  /** Accessible name of the detail `<section>`. */
  detailLabel: string;
}

/** The row ListRow marks as selected. */
const SELECTED_ROW = '[aria-current="true"]';

/**
 * List + detail frame. From 1100 px the list is 360 px wide, from 821 px
 * 300 px; both panes then fill the viewport below the shell top bar and
 * scroll on their own. At 820 px and below one pane shows at a time in
 * normal page flow, and switching panes moves focus to the pane that
 * appears. Going back to the list moves focus to the row the user was on,
 * which scrolls it into view, even when Back clears the selection.
 */
export function PaneLayout({
  list,
  detail,
  showDetail,
  backLink,
  listLabel,
  detailLabel,
}: PaneLayoutProps) {
  const listPane = useRef<HTMLElement>(null);
  const detailPane = useRef<HTMLElement>(null);
  const shown = useRef(showDetail);
  // Where going back returns to: the list stays mounted, so the node survives.
  const place = useRef<HTMLElement | null>(null);
  // The last element focused inside the list, for lists without a selection.
  const listFocus = useRef<HTMLElement | null>(null);

  // Runs after every render: the selected row can change while the detail
  // shows (J / K), not only when the pane switches.
  useEffect(() => {
    const listNode = listPane.current;
    const detailNode = detailPane.current;
    if (listNode === null || detailNode === null) return;
    const switched = shown.current !== showDetail;
    shown.current = showDetail;
    if (showDetail) {
      const row = listNode.querySelector<HTMLElement>(SELECTED_ROW);
      if (row !== null) place.current = row;
      else if (switched) place.current = listFocus.current;
    }
    if (!switched) return;
    const target = showDetail ? detailNode : listNode;
    const hidden = showDetail ? listNode : detailNode;
    // Only the one-pane layout hides a pane; elsewhere focus stays put.
    if (getComputedStyle(hidden).display !== "none") return;
    const active = document.activeElement;
    if (active !== null && active !== document.body && !hidden.contains(active))
      return;
    if (!showDetail) {
      const remembered = place.current;
      place.current = null;
      const row =
        listNode.querySelector<HTMLElement>(SELECTED_ROW) ??
        (remembered !== null &&
        remembered.isConnected &&
        listNode.contains(remembered)
          ? remembered
          : null);
      if (row !== null) {
        // Focusing scrolls the row back into view.
        row.focus();
        if (document.activeElement === row) return;
      }
    }
    target.focus({ preventScroll: true });
    if (typeof target.scrollIntoView === "function") {
      target.scrollIntoView({ block: "start" });
    }
  });

  return (
    <div className="ui-panes" data-show={showDetail ? "detail" : "list"}>
      <section
        ref={listPane}
        className="ui-panes-list"
        aria-label={listLabel}
        tabIndex={-1}
        onFocus={(event) => {
          if (
            event.target !== event.currentTarget &&
            event.target instanceof HTMLElement
          ) {
            listFocus.current = event.target;
          }
        }}
      >
        {list}
      </section>
      <section
        ref={detailPane}
        className="ui-panes-detail"
        aria-label={detailLabel}
        tabIndex={-1}
      >
        {backLink === undefined ? null : (
          <Link className="ui-panes-back" to={backLink.to}>
            <span aria-hidden="true">←</span>
            {backLink.label}
          </Link>
        )}
        {detail}
      </section>
    </div>
  );
}

export interface ListPaneProps {
  /** Pane title, e.g. "Possible issues". */
  title?: ReactNode;
  /** Title level. Default "h1". */
  titleAs?: "h1" | "h2" | undefined;
  /**
   * The title heading. With a ref the heading can take focus (tabIndex -1),
   * e.g. once the selected item left the list. Not used with `header`.
   */
  titleRef?: Ref<HTMLHeadingElement> | undefined;
  /** One line under the title. */
  subtitle?: ReactNode;
  /**
   * Controls at the right of the title. They wrap under it in a narrow pane
   * and shrink with it, so a select with long options needs no width cap.
   */
  actions?: ReactNode;
  /** Replaces the title block, e.g. a DetailHeader with a breadcrumb. */
  header?: ReactNode;
  /** Under the title, e.g. FilterChips or a filter field. */
  toolbar?: ReactNode;
  /** Bottom of the pane, e.g. key hints or Technical details. */
  footer?: ReactNode;
  /** ListSection elements, EmptyState, notices. */
  children: ReactNode;
  /** Renders a `<section>` with this name; omit inside PaneLayout. */
  "aria-label"?: string | undefined;
}

/** The list pane: title, optional toolbar, the lists and a footer. */
export function ListPane({
  title,
  titleAs: Title = "h1",
  titleRef,
  subtitle,
  actions,
  header,
  toolbar,
  footer,
  children,
  "aria-label": ariaLabel,
}: ListPaneProps) {
  const hasTitleBlock =
    title !== undefined || subtitle !== undefined || actions !== undefined;
  const content = (
    <>
      {header !== undefined ? (
        header
      ) : hasTitleBlock ? (
        <header className="ui-list-pane-header">
          <div className="ui-list-pane-heading">
            {title === undefined ? null : (
              <Title
                ref={titleRef}
                className="ui-list-pane-title"
                tabIndex={titleRef === undefined ? undefined : -1}
              >
                {title}
              </Title>
            )}
            {subtitle === undefined ? null : (
              <p className="ui-list-pane-subtitle">{subtitle}</p>
            )}
          </div>
          {actions === undefined ? null : (
            <div className="ui-list-pane-actions">{actions}</div>
          )}
        </header>
      ) : null}
      {toolbar === undefined ? null : (
        <div className="ui-list-pane-toolbar">{toolbar}</div>
      )}
      <div className="ui-list-pane-body">{children}</div>
      {footer === undefined ? null : (
        <footer className="ui-list-pane-footer">{footer}</footer>
      )}
    </>
  );
  return ariaLabel === undefined ? (
    <div className="ui-list-pane">{content}</div>
  ) : (
    <section className="ui-list-pane" aria-label={ariaLabel}>
      {content}
    </section>
  );
}

export interface DetailPaneProps {
  /** Usually a DetailHeader. */
  header?: ReactNode;
  /**
   * Pinned to the bottom of the pane on wide screens, e.g. a DecisionBar.
   * Content with the `ui-footer-record` class (a recorded decision) is
   * capped there and scrolls on its own.
   */
  footer?: ReactNode;
  children: ReactNode;
  /** Renders a `<section>` with this name; omit inside PaneLayout. */
  "aria-label"?: string | undefined;
}

/** The detail pane: header, padded body and a sticky footer. */
export function DetailPane({
  header,
  footer,
  children,
  "aria-label": ariaLabel,
}: DetailPaneProps) {
  const content = (
    <>
      {header}
      <div className="ui-detail-pane-body">{children}</div>
      {footer === undefined || footer === null ? null : (
        <div className="ui-detail-pane-footer">{footer}</div>
      )}
    </>
  );
  return ariaLabel === undefined ? (
    <div className="ui-detail-pane">{content}</div>
  ) : (
    <section className="ui-detail-pane" aria-label={ariaLabel}>
      {content}
    </section>
  );
}

export interface BreadcrumbItem {
  label: ReactNode;
  /** Items with `to` are links; the last item without `to` is the current page. */
  to?: To | undefined;
}

export interface DetailHeaderProps {
  breadcrumb?: readonly BreadcrumbItem[] | undefined;
  title: ReactNode;
  /** Title level. Default "h2". */
  titleAs?: "h1" | "h2" | undefined;
  /**
   * The title heading. With a ref the heading can take focus (tabIndex -1),
   * e.g. when the control that had focus went away.
   */
  titleRef?: Ref<HTMLHeadingElement> | undefined;
  /** Next to the title, usually a StatusChip. */
  status?: ReactNode;
  /** Line under the title (dates, owner, IdChip). */
  meta?: ReactNode;
  /** Controls at the right of the title row. */
  actions?: ReactNode;
}

function Chevron() {
  return (
    <svg
      className="ui-breadcrumb-sep"
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
  );
}

/** Header of a detail (or list) pane: breadcrumb, title, status, meta, actions. */
export function DetailHeader({
  breadcrumb,
  title,
  titleAs: Title = "h2",
  titleRef,
  status,
  meta,
  actions,
}: DetailHeaderProps) {
  return (
    <header className="ui-detail-header">
      {breadcrumb === undefined || breadcrumb.length === 0 ? null : (
        <nav aria-label="Breadcrumb" className="ui-breadcrumb">
          <ol>
            {breadcrumb.map((item, position) => {
              const last = position === breadcrumb.length - 1;
              return (
                // Breadcrumb items are positional.
                <li key={position}>
                  {item.to === undefined ? (
                    <span aria-current={last ? "page" : undefined}>
                      {item.label}
                    </span>
                  ) : (
                    <Link to={item.to}>{item.label}</Link>
                  )}
                  {last ? null : <Chevron />}
                </li>
              );
            })}
          </ol>
        </nav>
      )}
      <div className="ui-detail-header-main">
        <div className="ui-detail-header-title">
          <Title
            ref={titleRef}
            className="ui-detail-header-heading"
            tabIndex={titleRef === undefined ? undefined : -1}
          >
            {title}
          </Title>
          {status}
        </div>
        {actions === undefined ? null : (
          <div className="ui-detail-header-actions">{actions}</div>
        )}
      </div>
      {meta === undefined ? null : (
        <div className="ui-detail-header-meta">{meta}</div>
      )}
    </header>
  );
}
