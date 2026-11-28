import { useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useId, useState } from "react";
import { useNavigate } from "react-router";

import type { Audit, AuditState } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import {
  invalidateCrossProject,
  useProjectsIndex,
} from "../../api/cross-project";
import { listProjects, type Project } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { ErrorNotice } from "../../app/error-notice";
import { Icon } from "../../app/icon";
import { useCursorStack } from "../../app/pagination";
import { StaleDataWarning } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import type { StatusTone } from "../../app/status-tone";
import { checkStateLabel } from "../../app/vocabulary";
import {
  Button,
  IconButton,
  EmptyState,
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  StatusGlyph,
  useListNavigation,
} from "../../ui";
import { type ProjectActivity, useProjectActivity } from "./project-activity";
import { projectPath } from "./project-sections";
import { plural } from "../../app/format";
import { pollWhileAnyDeleting } from "./use-project-deletion";
import { LIVE_AUDIT_STATES } from "../../api/states";

interface RowStatus {
  tone: StatusTone;
  parts: ReactNode[];
}

function Status({ tone, children }: { tone: StatusTone; children: ReactNode }) {
  return (
    <strong className="projects-row-status" data-tone={tone}>
      {children}
    </strong>
  );
}

/**
 * Check states a row reports while a check is in them, in this order, with
 * the state's own label: "Check running", "Check stopping", …
 */
const IN_FLIGHT_STATES: readonly AuditState[] = LIVE_AUDIT_STATES;

/** "Check running", "2 checks waiting for you". */
function checksIn(count: number, stateLabel: string): string {
  const phrase = stateLabel.toLowerCase();
  return count === 1 ? `Check ${phrase}` : `${count} checks ${phrase}`;
}

/** "Check finished 6 minutes ago" for a check that is not running. */
function lastCheck(audit: Audit): { tone: StatusTone; text: ReactNode } {
  const when = <RecordedTime value={audit.finishedAt ?? audit.updatedAt} />;
  switch (audit.state) {
    case "completed":
      return { tone: "done", text: <>Check finished {when}</> };
    case "failed":
      return { tone: "blocked", text: <>Check failed {when}</> };
    case "cancelled":
      return { tone: "neutral", text: <>Check stopped {when}</> };
    case "draft":
      return { tone: "idle", text: <>Draft check saved {when}</> };
    default:
      return { tone: "idle", text: <>Check updated {when}</> };
  }
}

/** The one-line status of a project row: what needs the user, else news. */
function rowStatus(
  project: Project,
  activity: ProjectActivity | undefined,
): RowStatus {
  if (project.lifecycle === "deleting") {
    return {
      tone: "neutral",
      parts: [
        <Status key="state" tone="neutral">
          Deleting
        </Status>,
        project.deletion === undefined ? null : (
          <span key="when">
            requested <RecordedTime value={project.deletion.requestedAt} />
          </span>
        ),
      ],
    };
  }
  const parts: ReactNode[] = [];
  let tone: StatusTone = "idle";
  if (activity !== undefined) {
    for (const state of IN_FLIGHT_STATES) {
      const count = activity.states[state] ?? 0;
      if (count === 0) continue;
      const label = checkStateLabel(state);
      if (tone === "idle") tone = label.tone;
      parts.push(
        <Status key={state} tone={label.tone}>
          {checksIn(count, label.label)}
        </Status>,
      );
    }
    const { count, more, settled } = activity.possibleIssues;
    if (settled && count > 0) {
      if (tone === "idle") tone = "review";
      parts.push(
        <Status key="issues" tone="review">
          {more
            ? `${count}+ possible issues to review`
            : plural(count, "possible issue", "possible issues") + " to review"}
        </Status>,
      );
    }
    const paused = activity.states.paused ?? 0;
    if (paused > 0 && parts.length === 0) {
      const label = checkStateLabel("paused");
      tone = label.tone;
      parts.push(
        <Status key="paused" tone={label.tone}>
          {checksIn(paused, label.label)}
        </Status>,
      );
    }
    if (parts.length === 0 && activity.latest !== undefined) {
      const last = lastCheck(activity.latest);
      tone = last.tone;
      parts.push(<span key="last">{last.text}</span>);
    }
  }
  if (parts.length === 0) {
    parts.push(
      <span key="updated">
        Updated <RecordedTime value={project.updatedAt} />
      </span>,
    );
  }
  return { tone, parts };
}

function matches(project: Project, filter: string): boolean {
  const needle = filter.trim().toLowerCase();
  return (
    needle === "" ||
    project.name.toLowerCase().includes(needle) ||
    project.description.toLowerCase().includes(needle) ||
    project.projectId.toLowerCase().includes(needle)
  );
}

/**
 * The Projects list pane: one row per project with a one-line status,
 * paging, deletion in progress (polled every second) and "New project".
 */
export function ProjectListPane({
  selectedId,
  keyboard,
  onNewProject,
}: {
  selectedId: string | undefined;
  /** J / K move through the projects (off where a section has its own list). */
  keyboard: boolean;
  onNewProject: () => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const filterId = useId();
  const [filter, setFilter] = useState("");
  const searching = filter.trim() !== "";
  const [searchPage, setSearchPage] = useState(0);
  const index = useProjectsIndex({ enabled: searching, includeDeleting: true });
  const pageQuery = useQuery({
    queryKey: queryKeys.projects.list("project", cursor),
    queryFn: () =>
      listProjects(api, {
        kind: "project",
        ...(cursor === undefined ? {} : { cursor }),
      }),
    refetchInterval: pollWhileAnyDeleting,
  });
  const activity = useProjectActivity();
  const query = searching
    ? {
        data:
          (index.isPending || index.error !== null) &&
          index.projects.length === 0
            ? undefined
            : { items: index.projects },
        error: index.error,
        isFetching: index.isFetching,
        refetch: index.refetch,
      }
    : pageQuery;
  const items = query.data?.items ?? [];
  const matchesFilter = items.filter((project) => matches(project, filter));
  const searchPageCount = Math.max(1, Math.ceil(matchesFilter.length / 50));
  const currentSearchPage = Math.min(searchPage, searchPageCount - 1);
  const visible = searching
    ? matchesFilter.slice(currentSearchPage * 50, (currentSearchPage + 1) * 50)
    : matchesFilter;
  const { containerProps } = useListNavigation({
    count: visible.length,
    index: visible.findIndex((project) => project.projectId === selectedId),
    onMove: (position) => {
      const project = visible[position];
      if (project !== undefined) void navigate(projectPath(project.projectId));
    },
    enabled: keyboard,
  });
  const controls = pages.controls(pageQuery.data?.page);
  const paged =
    !searching && (controls.canGoBack || controls.nextCursor !== undefined);
  // Beside a selected project the list is secondary: its loading and
  // failures are shown quietly, without live regions that would compete
  // with the project's own announcements.
  const secondary = selectedId !== undefined;
  const subtitle =
    query.data === undefined
      ? undefined
      : filter.trim() !== ""
        ? `${matchesFilter.length} of ${plural(items.length, "project", "projects")}${index.partial || index.truncated ? " loaded" : ""}`
        : items.length === 0
          ? undefined
          : `${plural(items.length, "project", "projects")}${paged ? " on this page" : ""}${items.length > 1 ? ", newest first" : ""}`;

  return (
    <ListPane
      title="Projects"
      subtitle={subtitle}
      actions={
        <>
          <IconButton
            className="refresh-button"
            variant="ghost"
            size="sm"
            label="Refresh projects"
            icon={<Icon name="refresh" size={16} />}
            pending={query.isFetching}
            data-fetching={query.isFetching || undefined}
            onClick={() => {
              void query.refetch();
              void invalidateCrossProject(queryClient);
            }}
          />
          <Button
            size="sm"
            aria-label="New project"
            onClick={onNewProject}
            icon={<Icon name="plus" />}
          >
            <span>New</span>
          </Button>
        </>
      }
      toolbar={
        items.length === 0 && filter === "" ? undefined : (
          <div className="projects-filter">
            <label className="ui-visually-hidden" htmlFor={filterId}>
              Filter projects
            </label>
            <Icon name="search" />
            <input
              id={filterId}
              type="search"
              placeholder="Filter projects"
              autoComplete="off"
              value={filter}
              onChange={(event) => {
                setFilter(event.target.value);
                setSearchPage(0);
              }}
            />
          </div>
        )
      }
      footer={
        keyboard && visible.length > 1 ? (
          <span className="projects-keys">
            <Kbd>J</Kbd> <Kbd>K</Kbd> move between projects
          </span>
        ) : undefined
      }
    >
      <div className="projects-list-body">
        {query.data === undefined ? (
          query.error === null ? (
            <p className="loading-copy" role={secondary ? undefined : "status"}>
              Loading projects…
            </p>
          ) : secondary ? (
            <div className="projects-list-error">
              <p>Projects could not be loaded: {query.error.message}</p>
              <Button
                size="xs"
                disabled={query.isFetching}
                onClick={() => void query.refetch()}
              >
                Try again
              </Button>
            </div>
          ) : (
            <ErrorNotice
              error={query.error}
              context="Could not load Projects"
              onRetry={() => void query.refetch()}
              retryPending={query.isFetching}
            />
          )
        ) : (
          <>
            {query.error === null ? null : secondary ? (
              <p className="projects-list-error">
                Could not refresh the list; it shows the projects loaded last.
              </p>
            ) : (
              <StaleDataWarning
                error={query.error}
                onRetry={() => void query.refetch()}
                retryPending={query.isFetching}
              />
            )}
            {searching && (index.partial || index.truncated) ? (
              <div
                className="projects-list-error"
                role={secondary ? undefined : "status"}
              >
                <p>
                  Search covers {items.length} loaded projects.{" "}
                  {index.partial
                    ? "Some projects could not be loaded."
                    : "More projects are available than can be searched at once."}
                </p>
                <Button size="xs" onClick={() => void index.refetch()}>
                  Retry search
                </Button>
              </div>
            ) : null}
            {items.length === 0 && !searching ? (
              <EmptyState
                title="No projects yet"
                action={
                  <Button variant="primary" size="sm" onClick={onNewProject}>
                    New project
                  </Button>
                }
              >
                A project keeps your materials, the checks you run on them and
                their results together.
              </EmptyState>
            ) : visible.length === 0 ? (
              <EmptyState
                title={
                  index.partial || index.truncated
                    ? "No match in loaded projects"
                    : "No project matches the filter"
                }
              >
                {index.partial || index.truncated
                  ? "Try another name or retry loading projects."
                  : "Try another name."}
              </EmptyState>
            ) : (
              <div
                {...containerProps}
                aria-keyshortcuts={keyboard ? "J K" : undefined}
              >
                <ListSection>
                  {visible.map((project) => {
                    const status = rowStatus(
                      project,
                      activity.get(project.projectId),
                    );
                    return (
                      <ListRow
                        key={project.projectId}
                        to={projectPath(project.projectId)}
                        selected={project.projectId === selectedId}
                        glyph={<StatusGlyph tone={status.tone} />}
                        title={project.name}
                        clamp={1}
                        meta={status.parts}
                      />
                    );
                  })}
                </ListSection>
              </div>
            )}
          </>
        )}
      </div>
      <div className="projects-list-pages">
        {searching ? (
          searchPageCount > 1 ? (
            <nav className="pagination" aria-label="Project search pages">
              <Button
                disabled={currentSearchPage === 0}
                onClick={() => setSearchPage(currentSearchPage - 1)}
              >
                Previous
              </Button>
              <span className="ui-caption">
                Page {currentSearchPage + 1} of {searchPageCount}
              </span>
              <Button
                disabled={currentSearchPage + 1 >= searchPageCount}
                onClick={() => setSearchPage(currentSearchPage + 1)}
              >
                Next
              </Button>
            </nav>
          ) : null
        ) : (
          <CursorControls label="Projects pages" {...controls} />
        )}
      </div>
    </ListPane>
  );
}
