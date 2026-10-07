import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  Link,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router";

import {
  useProjectsIndex,
  type CrossProjectCheck,
  type CrossProjectError,
  type CrossProjectIssue,
} from "../../api/cross-project";
import { useDocumentTitle } from "../../app/document-title";
import {
  FINDING_STATE_LABELS,
  severityLabel,
  verdictLabel,
} from "../../app/vocabulary";
import {
  EmptyState,
  FilterChips,
  Kbd,
  ListPane,
  ListSection,
  PaneLayout,
  StatusGlyph,
  useListNavigation,
  type FilterChipOption,
} from "../../ui";
import type { AuditFindingDecisionResult } from "../decisions";
import {
  FINDING_STATES,
  SEVERITY_OPTIONS,
  isFindingSeverity,
} from "../projects/audits/finding-options";
import { auditProfileLabel } from "../projects/audits/labels";
import { useIssueList, type IssueList } from "./data";
import { IssueDetail } from "./issue-detail";
import { IssueRow } from "./issue-row";
import {
  issuePath,
  issueSearch,
  matchesIssueFilters,
  readIssueFilters,
  type IssueFilters,
  type StateFilter,
} from "./links";

import "./issues.css";

const ANNOUNCEMENT_MS = 8_000;

/**
 * Empty-list titles that claim the state is clear: only for a complete list
 * (every read succeeded, nothing truncated) without a severity filter.
 */
const EMPTY_TITLES: Readonly<Record<StateFilter, string>> = {
  proposed: "That is everything that needs review.",
  confirmed: "No confirmed issues.",
  rejected: "Nothing is marked as not an issue.",
  duplicate: "No duplicates.",
  "needs-evidence": "Nothing waits for evidence.",
  all: "No possible issues yet.",
};

function keyOf(issue: Pick<CrossProjectIssue, "finding">): string {
  return `${issue.finding.auditId}/${issue.finding.findingId}`;
}

function checkName(check: CrossProjectCheck): string {
  return `${auditProfileLabel(check.audit)} on ${check.project.name}`;
}

function outcomeOf(result: AuditFindingDecisionResult): string {
  const verdict = verdictLabel(result.decision.verdict).label;
  return result.decision.severity === undefined
    ? verdict
    : `${verdict} · ${severityLabel(result.decision.severity)}`;
}

function failureText(
  error: CrossProjectError,
  list: IssueList,
  projectName: (projectId: string) => string,
): string {
  if (error.scope === "list")
    return `The possible issues list could not be refreshed: ${error.error.message}`;
  if (error.scope === "index")
    return `The project list could not be refreshed: ${error.error.message}`;
  if (error.scope === "project")
    return `Checks of ${projectName(error.projectId)} could not be loaded: ${error.error.message}`;
  const check = list.checks.find(
    (candidate) => candidate.audit.auditId === error.auditId,
  );
  return `Possible issues of ${check === undefined ? `check ${error.auditId}` : checkName(check)} could not be loaded: ${error.error.message}`;
}

function runningNote(list: IssueList): string | undefined {
  const [first, ...rest] = list.running;
  if (first === undefined) return undefined;
  return rest.length === 0
    ? `${checkName(first)} is still running, so more may arrive.`
    : `${list.running.length} checks are still running, so more may arrive.`;
}

/**
 * What an empty list says. It claims a state is clear only when every read
 * succeeded, nothing was cut off and no severity filter narrows it: possible
 * issues that need review have no analyst rating to match yet.
 */
function emptyText(
  list: IssueList,
  filters: IssueFilters,
  complete: boolean,
  running: string | undefined,
): { title: string; note?: string | undefined } {
  if (list.errors.length > 0)
    return {
      title: "No possible issues listed",
      note: "There may be more in what could not be loaded.",
    };
  if (filters.severity !== undefined)
    return {
      title: "No possible issues match these filters",
      note:
        filters.state === "proposed"
          ? "Severity is the analyst's rating, given when an issue is confirmed, so possible issues that need review are usually not rated yet."
          : undefined,
    };
  if (!complete) return { title: "No possible issues listed" };
  return {
    title: EMPTY_TITLES[filters.state],
    note:
      filters.state === "all"
        ? "Possible issues arrive here as checks find them."
        : running,
  };
}

interface FocusRequest {
  key: string;
  id: number;
}

/**
 * Issues: possible issues of every project, newest first, with the
 * selected one in the detail pane (`/issues/:auditId/:findingId`). Filters
 * live in the query string; J / K move, C / R / E decide.
 */
export function IssuesRoute() {
  const { auditId, findingId } = useParams();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();
  const projectSelect = useId();
  const severitySelect = useId();
  const filters = readIssueFilters(params);
  const search = issueSearch(filters);
  const list = useIssueList(filters);
  const index = useProjectsIndex();
  const selectedKey =
    auditId === undefined || findingId === undefined
      ? undefined
      : `${auditId}/${findingId}`;
  const issues = list.issues;
  const selectedIndex =
    selectedKey === undefined
      ? -1
      : issues.findIndex((issue) => keyOf(issue) === selectedKey);
  const selected = selectedIndex < 0 ? undefined : issues[selectedIndex];
  const previous = selectedIndex > 0 ? issues[selectedIndex - 1] : undefined;
  const next = selectedIndex < 0 ? issues[0] : issues[selectedIndex + 1];
  const project =
    filters.project === undefined
      ? undefined
      : index.projects.find(
          (candidate) => candidate.projectId === filters.project,
        );
  const [focusRequest, setFocusRequest] = useState<FocusRequest | null>(null);
  const [announcement, setAnnouncement] = useState<{
    text: string;
    id: number;
  } | null>(null);
  // The neighbours of the selected possible issue while it was listed: a
  // decision can take it out of the list before onDecided runs.
  const neighbours = useRef<{
    key: string;
    next: CrossProjectIssue | undefined;
    previous: CrossProjectIssue | undefined;
  } | null>(null);
  const listTitle = useRef<HTMLHeadingElement>(null);
  // Set when the last possible issue of the list was decided: the decision
  // and the row it came from go away, so focus moves to the list's title
  // once the list shows (the navigation renders in a later transition).
  const focusListTitle = useRef(false);

  useDocumentTitle(
    selectedKey === undefined
      ? "Possible issues"
      : (selected?.finding.firstProposal.document.title ?? "Possible issue"),
  );

  useLayoutEffect(() => {
    if (selectedKey !== undefined && selectedIndex >= 0)
      neighbours.current = {
        key: selectedKey,
        next: issues[selectedIndex + 1],
        previous: issues[selectedIndex - 1],
      };
  });

  useEffect(() => {
    if (announcement === null) return undefined;
    const timer = window.setTimeout(
      () => setAnnouncement(null),
      ANNOUNCEMENT_MS,
    );
    return () => window.clearTimeout(timer);
  }, [announcement]);

  // Runs after PaneLayout's own effect (children first), which on one-pane
  // screens focuses the row of the decided possible issue: that row leaves
  // the list once its check is read again.
  useEffect(() => {
    if (selectedKey !== undefined || !focusListTitle.current) return;
    focusListTitle.current = false;
    listTitle.current?.focus();
  }, [selectedKey]);

  const open = useCallback(
    (
      issue: CrossProjectIssue,
      options: { replace?: boolean; focus?: boolean } = {},
    ) => {
      if (options.focus === true)
        setFocusRequest((current) => ({
          key: keyOf(issue),
          id: (current?.id ?? 0) + 1,
        }));
      void navigate(
        {
          pathname: issuePath(issue.finding.auditId, issue.finding.findingId),
          search,
        },
        { replace: options.replace ?? false },
      );
    },
    [navigate, search],
  );

  function setFilters(change: Partial<IssueFilters>) {
    void navigate(
      {
        pathname: location.pathname,
        search: issueSearch({ ...filters, ...change }),
      },
      { replace: true },
    );
  }

  const { containerProps } = useListNavigation({
    count: issues.length,
    index: selectedIndex,
    onMove: (position) => {
      const issue = issues[position];
      if (issue !== undefined) open(issue, { replace: true });
    },
    onOpen: (position) => {
      const issue = issues[position];
      if (issue !== undefined)
        setFocusRequest((current) => ({
          key: keyOf(issue),
          id: (current?.id ?? 0) + 1,
        }));
    },
  });

  function announce(text: string) {
    setAnnouncement((current) => ({ text, id: (current?.id ?? 0) + 1 }));
  }

  function onDecided(result: AuditFindingDecisionResult) {
    const decided = keyOf(result);
    const around =
      neighbours.current?.key === decided ? neighbours.current : undefined;
    // Still in the list, or opened from a link without being in it: the
    // decision announces itself and the page stays on the possible issue.
    if (around === undefined || matchesIssueFilters(result.finding, filters))
      return;
    const target = around.next ?? around.previous;
    const outcome = outcomeOf(result);
    if (target === undefined) {
      announce(
        `Decision recorded: ${outcome}. That was the last possible issue in this list.`,
      );
      focusListTitle.current = true;
      void navigate({ pathname: "/issues", search }, { replace: true });
      return;
    }
    announce(
      `Decision recorded: ${outcome}. Showing the next possible issue: ${target.finding.firstProposal.document.title}.`,
    );
    open(target, { replace: true, focus: true });
  }

  function dropReview() {
    const nextParams = new URLSearchParams(params);
    nextParams.delete("review");
    const text = nextParams.toString();
    void navigate(
      { pathname: location.pathname, search: text === "" ? "" : `?${text}` },
      { replace: true },
    );
  }

  const projectName = (projectId: string) =>
    index.projects.find((candidate) => candidate.projectId === projectId)
      ?.name ?? projectId;

  const chips: FilterChipOption<StateFilter>[] = [
    ...FINDING_STATES.map((state) => ({
      value: state,
      label: FINDING_STATE_LABELS[state].label,
      count: list.counts[state],
    })),
    { value: "all", label: "All", count: list.counts.all },
  ];
  const projectOptions = [...index.projects].sort((left, right) =>
    left.name.localeCompare(right.name),
  );
  const unknownProject = filters.project !== undefined && project === undefined;
  const complete =
    !list.pending &&
    list.error === null &&
    list.errors.length === 0 &&
    list.truncatedChecks.length === 0 &&
    !list.checksTruncated;
  const running = runningNote(list);
  const empty = emptyText(list, filters, complete, running);
  const liveText = announcement?.text ?? "";

  const listPane = (
    <ListPane
      title="Possible issues"
      titleRef={listTitle}
      subtitle={
        <span className="issues-list-subtitle">
          {filters.project === undefined
            ? "Across all projects, newest first"
            : `In ${project?.name ?? filters.project}, newest first`}
        </span>
      }
      actions={
        <span className="issues-select-field">
          <label htmlFor={projectSelect} className="ui-visually-hidden">
            Project
          </label>
          <select
            id={projectSelect}
            className="issues-select"
            value={filters.project ?? ""}
            onChange={(event) =>
              setFilters({
                project:
                  event.target.value === "" ? undefined : event.target.value,
              })
            }
          >
            <option value="">All projects</option>
            {unknownProject ? (
              <option value={filters.project}>{filters.project}</option>
            ) : null}
            {projectOptions.map((candidate) => (
              <option key={candidate.projectId} value={candidate.projectId}>
                {candidate.name}
              </option>
            ))}
          </select>
        </span>
      }
      toolbar={
        <div className="issues-toolbar">
          <FilterChips
            label="Filter by state"
            options={chips}
            value={filters.state}
            onChange={(state) => setFilters({ state })}
          />
          <span className="issues-select-field">
            <label htmlFor={severitySelect} className="issues-field-label">
              Severity
            </label>
            <select
              id={severitySelect}
              className="issues-select"
              aria-describedby={`${severitySelect}-hint`}
              value={filters.severity ?? ""}
              onChange={(event) =>
                setFilters({
                  severity: isFindingSeverity(event.target.value)
                    ? event.target.value
                    : undefined,
                })
              }
            >
              {SEVERITY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <span id={`${severitySelect}-hint`} className="issues-quiet">
              The analyst&apos;s rating; AI suggestions are not filtered.
            </span>
          </span>
        </div>
      }
      footer={
        <>
          <span className="issues-keys">
            <Kbd>J</Kbd> <Kbd>K</Kbd> move
          </span>
          <span className="issues-keys">
            <Kbd>C</Kbd> <Kbd>R</Kbd> <Kbd>E</Kbd> decide
          </span>
        </>
      }
    >
      {/* Shown, not announced: the page-level status region speaks it. */}
      <p className="issues-announcement" aria-hidden="true">
        {selectedKey === undefined ? liveText : ""}
      </p>
      {list.error !== null ? (
        <div className="issues-notice" data-tone="error" role="alert">
          <p>
            <strong>Possible issues could not be loaded.</strong>{" "}
            {list.error.message}
          </p>
          <div className="issues-notice-actions">
            <button
              type="button"
              className="ui-btn"
              data-size="xs"
              onClick={() => void list.refetch()}
            >
              Retry
            </button>
          </div>
        </div>
      ) : null}
      {list.errors.length === 0 ? null : (
        <div className="issues-notice" data-tone="warning" role="alert">
          <p>
            <strong>Some possible issues could not be loaded.</strong> The rest
            are listed.
          </p>
          <ul>
            {list.errors.map((error) => (
              <li
                key={`${error.scope}:${error.projectId ?? ""}:${error.auditId ?? ""}`}
              >
                {failureText(error, list, projectName)}
              </li>
            ))}
          </ul>
          <div className="issues-notice-actions">
            <button
              type="button"
              className="ui-btn"
              data-size="xs"
              onClick={() => void list.refetch()}
            >
              Retry
            </button>
          </div>
        </div>
      )}
      {list.pending && issues.length === 0 && list.error === null ? (
        <p className="issues-list-status" role="status">
          Loading possible issues…
        </p>
      ) : null}
      <div className="issues-list-nav" {...containerProps}>
        {issues.length === 0 ? (
          list.pending || list.error !== null ? null : (
            <EmptyState
              title={empty.title}
              action={
                filters.project !== undefined ||
                filters.severity !== undefined ? (
                  <Link
                    to={{
                      pathname: "/issues",
                      search: issueSearch({ state: filters.state }),
                    }}
                  >
                    Show every project and severity
                  </Link>
                ) : undefined
              }
            >
              {empty.note === undefined ? null : <p>{empty.note}</p>}
            </EmptyState>
          )
        ) : (
          <ListSection>
            {issues.map((issue) => (
              <IssueRow
                key={keyOf(issue)}
                id={`issue-${issue.finding.auditId}-${issue.finding.findingId}`}
                finding={issue.finding}
                to={{
                  pathname: issuePath(
                    issue.finding.auditId,
                    issue.finding.findingId,
                  ),
                  search,
                }}
                selected={keyOf(issue) === selectedKey}
                context={issue.project.name}
              />
            ))}
          </ListSection>
        )}
      </div>
      {issues.length > 0 &&
      complete &&
      filters.state === "proposed" &&
      filters.severity === undefined ? (
        <p className="issues-end-note">
          <StatusGlyph tone="done" size={15} />
          <span>
            That is everything that needs review.
            {running === undefined ? null : ` ${running}`}
          </span>
        </p>
      ) : null}
    </ListPane>
  );

  const detail =
    auditId === undefined || findingId === undefined ? (
      <div className="issues-detail-empty">
        <EmptyState title="Choose a possible issue">
          <p>
            Pick one from the list, or press <Kbd>J</Kbd> to start with the
            newest.
          </p>
        </EmptyState>
      </div>
    ) : (
      <>
        <p className="issues-announcement" aria-hidden="true">
          {liveText}
        </p>
        <IssueDetail
          key={selectedKey}
          auditId={auditId}
          findingId={findingId}
          listed={selected}
          projects={index.projects}
          position={
            selectedIndex < 0
              ? undefined
              : { index: selectedIndex, count: issues.length }
          }
          listPending={list.pending}
          onPrevious={
            previous === undefined
              ? undefined
              : () => open(previous, { replace: true })
          }
          onNext={
            next === undefined ? undefined : () => open(next, { replace: true })
          }
          reviewId={params.get("review")}
          onDropReview={dropReview}
          onDecided={onDecided}
          focusRequest={
            focusRequest !== null && focusRequest.key === selectedKey
              ? focusRequest.id
              : undefined
          }
          backTo={{ pathname: "/issues", search }}
        />
      </>
    );

  return (
    <>
      <PaneLayout
        listLabel="Possible issues"
        detailLabel="Review"
        showDetail={selectedKey !== undefined}
        backLink={{
          to: { pathname: "/issues", search },
          label: "Back to possible issues",
        }}
        list={listPane}
        detail={detail}
      />
      {/*
       * Decisions that move on are announced here: outside the panes, so
       * it is never inside a pane that one-pane screens hide (display:
       * none) and was rendered before the message arrives.
       */}
      <p className="ui-visually-hidden" role="status">
        {liveText}
      </p>
    </>
  );
}
