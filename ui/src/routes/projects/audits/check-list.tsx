import { useId, type ReactNode } from "react";
import { Link } from "react-router";

import {
  auditPollInterval,
  type Audit,
  type AuditCoverageRow,
} from "../../../api/audits";
import { ErrorNotice } from "../../../app/error-notice";
import { StaleDataWarning } from "../../../app/query-view";
import { Icon } from "../../../app/icon";
import {
  checkStateLabel,
  itemNoun,
  type ItemKind,
} from "../../../app/vocabulary";
import {
  DetailHeader,
  EmptyState,
  FilterChips,
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  MethodChip,
  StatusChip,
  StatusGlyph,
  type ListNavigationContainerProps,
} from "../../../ui";
import { GROUP_IDS, groupLabel, itemHeading, type Group } from "./assessments";
import { startedText } from "./check-format";
import type { CheckLinks } from "./check-links";
import {
  commonPathPrefix,
  issueSummary,
  relativePath,
  type CheckEntry,
  type EntryGroup,
} from "./check-model";
import type { AuditCollectionQuery } from "./collections";
import { auditProfileLabel } from "./labels";
import { LoadMoreControl } from "./load-more";
import { PathText } from "./path-text";

/** Keys that move the selection through the list (useListNavigation). */
const LIST_KEYS = "J K ArrowDown ArrowUp Home End";

function ActivityGlyph() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M9 6.5h10.5M9 12h10.5M9 17.5h10.5" />
      <path d="M4.5 6.5h.01M4.5 12h.01M4.5 17.5h.01" />
    </svg>
  );
}

function EntryTitle({
  entry,
  prefix,
}: {
  entry: CheckEntry;
  prefix: string;
}): ReactNode {
  if (entry.operation !== undefined)
    return (
      <>
        <MethodChip method={entry.operation.method} />{" "}
        <span className="checks-row-path">
          <PathText path={relativePath(entry.operation.path, prefix)} />
        </span>
      </>
    );
  return (
    <>
      <span className="checks-item-key">{entry.key}</span>
      {entry.summary === "" ? null : <> {entry.summary}</>}
    </>
  );
}

function EntryRow({
  entry,
  prefix,
  selected,
  links,
}: {
  entry: CheckEntry;
  prefix: string;
  selected: boolean;
  links: CheckLinks;
}) {
  return (
    <ListRow
      id={`check-${entry.row.itemId}`}
      to={links.item(entry.row.itemId)}
      selected={selected}
      glyph={<StatusGlyph tone={entry.status.tone} />}
      title={<EntryTitle entry={entry} prefix={prefix} />}
      meta={[
        <span
          key="status"
          className="checks-status-word"
          data-tone={entry.status.tone}
        >
          {entry.status.label}
        </span>,
        ...issueSummary(entry.findings),
      ]}
    />
  );
}

/**
 * The check's list pane: its identity (breadcrumb, check type, state, start),
 * the result filter and search, "All activity" and every endpoint,
 * requirement or scenario grouped by kind. Filters, search and the revision
 * pin live in the URL; so does the selection.
 */
export function CheckListPane({
  audit,
  projectName,
  kind,
  entries,
  groups,
  shown,
  group,
  search,
  coverage,
  pin,
  activitySelected,
  selectedItemId,
  links,
  navigation,
  onFilter,
  onClearFilters,
  onRefresh,
}: {
  audit: Audit;
  projectName: string | undefined;
  kind: ItemKind;
  entries: readonly CheckEntry[];
  /** The items the filters show, grouped by kind in the order J / K walk. */
  groups: readonly EntryGroup[];
  /** How many items the filters show. */
  shown: number;
  group: Group;
  search: string;
  coverage: AuditCollectionQuery<AuditCoverageRow>;
  /** The revision the list is pinned to (`auditRevision`), if any. */
  pin: string | null;
  activitySelected: boolean;
  selectedItemId: string | undefined;
  links: CheckLinks;
  navigation: ListNavigationContainerProps;
  onFilter: (key: "result" | "q", value: string) => void;
  onClearFilters: () => void;
  onRefresh: () => void;
}) {
  const searchId = useId();
  const state = checkStateLabel(audit.state);
  const plural = itemNoun(kind, 2);
  const filtered = group !== "all" || search.trim() !== "";
  const live = auditPollInterval([audit]) !== false;
  const newerRevision = pin !== null && String(audit.revision) !== pin;
  const options = GROUP_IDS.map((id) => ({
    value: id,
    label: groupLabel(id, kind),
    count:
      id === "all"
        ? entries.length
        : entries.filter((entry) => entry.status.group === id).length,
  }));
  let body: ReactNode;
  if (coverage.isPending)
    body = (
      <p className="checks-pane-note" role="status">
        Loading {plural}…
      </p>
    );
  else if (coverage.error !== null && entries.length === 0)
    body = (
      <div className="checks-pane-note">
        <ErrorNotice
          error={coverage.error}
          onRetry={onRefresh}
          retryPending={coverage.isFetching}
        />
      </div>
    );
  else if (entries.length === 0)
    body = (
      <EmptyState title={`No ${plural} yet`}>
        {audit.state === "draft"
          ? `Start the check to create its ${plural} and follow their results here.`
          : `This round has no ${plural} yet.`}
      </EmptyState>
    );
  else if (shown === 0)
    body = (
      <EmptyState title={`No ${plural} match these filters`}>
        Try another search, or clear the filters to see every{" "}
        {itemNoun(kind, 1)}.
      </EmptyState>
    );
  else body = null;
  return (
    <ListPane
      header={
        <DetailHeader
          breadcrumb={[
            { label: "Checks", to: "/checks" },
            {
              label: projectName ?? "Project",
              to: `/projects/${encodeURIComponent(audit.projectId)}/audits`,
            },
          ]}
          title={auditProfileLabel(audit)}
          titleAs="h1"
          status={
            <StatusChip tone={state.tone} size="sm">
              {state.label}
            </StatusChip>
          }
          meta={<span>{startedText(audit)}</span>}
        />
      }
      toolbar={
        <div className="checks-list-toolbar">
          <FilterChips
            label="Filter by result"
            options={options}
            value={group}
            onChange={(value) => onFilter("result", value)}
          />
          <div className="checks-search">
            <label htmlFor={searchId}>Search {plural}</label>
            <input
              id={searchId}
              type="search"
              placeholder="Result, evidence, run or ID…"
              value={search}
              onChange={(event) => onFilter("q", event.currentTarget.value)}
            />
          </div>
          <div className="checks-list-count">
            <p role="status">
              Showing {shown.toLocaleString("en-US")} of{" "}
              {coverage.truncated ? "≥" : ""}
              {entries.length.toLocaleString("en-US")} {plural}
            </p>
            <button
              type="button"
              className="ui-btn"
              data-variant="ghost"
              data-size="xs"
              data-icon-only=""
              aria-label={`Refresh ${plural}`}
              title={`Refresh ${plural}`}
              aria-busy={coverage.isFetching || undefined}
              disabled={coverage.isFetching}
              onClick={onRefresh}
            >
              <Icon name="refresh" />
            </button>
            {filtered ? (
              <button
                type="button"
                className="checks-text-button"
                onClick={onClearFilters}
              >
                Clear filters
              </button>
            ) : null}
          </div>
        </div>
      }
      footer={
        <>
          {/* The rows declare their keys (aria-keyshortcuts); this is the
              visible hint. */}
          <span className="checks-key-hint" aria-hidden="true">
            <Kbd>J</Kbd> <Kbd>K</Kbd> move
          </span>
          <span className="checks-footer-tech">
            <Link
              to={{ ...toObject(links.overview), hash: "#technical-details" }}
            >
              Technical details
            </Link>
            <span>Setup, limits and inputs, for admins and debugging.</span>
          </span>
        </>
      }
    >
      {newerRevision ? (
        <div
          className="checks-notice checks-pane-note"
          data-tone="warning"
          role="status"
        >
          <strong>A newer revision of this check is available.</strong>
          <p>This list shows revision {pin}.</p>
          <button
            className="ui-btn"
            data-size="xs"
            type="button"
            onClick={onRefresh}
          >
            Show the latest results
          </button>
        </div>
      ) : null}
      {coverage.error !== null && entries.length > 0 ? (
        <div className="checks-pane-note">
          <StaleDataWarning
            error={coverage.error}
            onRetry={onRefresh}
            retryPending={coverage.isFetching}
          />
        </div>
      ) : null}
      <div
        className="checks-list-rows"
        aria-keyshortcuts={LIST_KEYS}
        {...navigation}
      >
        <ListSection>
          <ListRow
            to={links.overview}
            selected={activitySelected}
            glyph={<ActivityGlyph />}
            title="All activity"
            meta="What the whole check is doing, in plain words"
            trailing={
              live ? (
                <span className="checks-live">
                  <span className="checks-live-dot" aria-hidden="true" />
                  Live
                </span>
              ) : undefined
            }
          />
        </ListSection>
        {groups.map((section) => {
          const paths = section.entries.flatMap((entry) =>
            entry.operation === undefined ? [] : [entry.operation.path],
          );
          const prefix =
            section.kind === "endpoint" ? commonPathPrefix(paths) : "";
          return (
            <ListSection
              key={section.kind}
              title={itemHeading(section.kind)}
              count={section.entries.length}
              aside={
                prefix === "" ? undefined : (
                  <>
                    under{" "}
                    <code className="checks-mono">
                      <PathText path={prefix} />
                    </code>
                  </>
                )
              }
            >
              {section.entries.map((entry) => (
                <EntryRow
                  key={entry.row.itemId}
                  entry={entry}
                  prefix={prefix}
                  selected={entry.row.itemId === selectedItemId}
                  links={links}
                />
              ))}
            </ListSection>
          );
        })}
      </div>
      {body}
      <div className="checks-pane-note">
        <LoadMoreControl
          shown={entries.length}
          noun={plural}
          truncated={coverage.truncated}
          loading={coverage.isLoadingMore}
          error={coverage.moreError}
          onLoadMore={coverage.loadMore}
        />
      </div>
      {entries.length === 0 ? null : (
        <p className="checks-pane-note checks-quiet">
          These results describe single {plural}. A check can finish with issues
          or with incomplete {plural}; excluded {plural} do not count as met.
        </p>
      )}
    </ListPane>
  );
}

function toObject(to: CheckLinks["overview"]) {
  return typeof to === "string" ? { pathname: to } : to;
}
