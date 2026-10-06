import { useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useId, useState } from "react";
import { useNavigate } from "react-router";

import type { Audit } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { invalidateCrossProject } from "../../api/cross-project";
import { listProjects, type Project } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { ErrorNotice } from "../../app/error-notice";
import { Icon } from "../../app/icon";
import { useCursorStack } from "../../app/pagination";
import { StaleDataWarning } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import type { StatusTone } from "../../app/status-tone";
import {
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

function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
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
    if (activity.running > 0) {
      tone = "progress";
      parts.push(
        <Status key="running" tone="progress">
          {activity.running === 1
            ? "Check running"
            : `${activity.running} checks running`}
        </Status>,
      );
    }
    if (activity.waiting > 0) {
      if (tone === "idle") tone = "review";
      parts.push(
        <Status key="waiting" tone="review">
          {activity.waiting === 1
            ? "Check waiting for you"
            : `${activity.waiting} checks waiting for you`}
        </Status>,
      );
    }
    const { count, more } = activity.possibleIssues;
    if (count > 0) {
      if (tone === "idle") tone = "review";
      parts.push(
        <Status key="issues" tone="review">
          {more
            ? `${count}+ possible issues to review`
            : plural(count, "possible issue", "possible issues") + " to review"}
        </Status>,
      );
    }
    if (activity.paused > 0 && parts.length === 0) {
      tone = "warning";
      parts.push(
        <Status key="paused" tone="warning">
          {activity.paused === 1
            ? "Check paused"
            : `${activity.paused} checks paused`}
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
  const query = useQuery({
    queryKey: queryKeys.projects.list("project", cursor),
    queryFn: () =>
      listProjects(api, {
        kind: "project",
        ...(cursor === undefined ? {} : { cursor }),
      }),
    refetchInterval: (current) =>
      current.state.data?.items.some(
        (project) => project.lifecycle === "deleting",
      )
        ? 1_000
        : false,
  });
  const activity = useProjectActivity();
  const items = query.data?.items ?? [];
  const visible = items.filter((project) => matches(project, filter));
  const { containerProps } = useListNavigation({
    count: visible.length,
    index: visible.findIndex((project) => project.projectId === selectedId),
    onMove: (position) => {
      const project = visible[position];
      if (project !== undefined) void navigate(projectPath(project.projectId));
    },
    enabled: keyboard,
  });
  const controls = pages.controls(query.data?.page);
  const paged = controls.canGoBack || controls.nextCursor !== undefined;
  // Beside a selected project the list is secondary: its loading and
  // failures are shown quietly, without live regions that would compete
  // with the project's own announcements.
  const secondary = selectedId !== undefined;
  const subtitle =
    query.data === undefined
      ? undefined
      : filter.trim() !== ""
        ? `${visible.length} of ${plural(items.length, "project", "projects")}`
        : items.length === 0
          ? undefined
          : `${plural(items.length, "project", "projects")}${paged ? " on this page" : ""}${items.length > 1 ? ", newest first" : ""}`;

  return (
    <ListPane
      title="Projects"
      subtitle={subtitle}
      actions={
        <>
          <button
            type="button"
            className="ui-btn refresh-button projects-icon-button"
            data-variant="ghost"
            data-size="sm"
            aria-label="Refresh projects"
            title="Refresh projects"
            aria-busy={query.isFetching || undefined}
            data-fetching={query.isFetching || undefined}
            disabled={query.isFetching}
            onClick={() => {
              void query.refetch();
              void invalidateCrossProject(queryClient);
            }}
          >
            <Icon name="refresh" />
          </button>
          <button
            type="button"
            className="ui-btn"
            data-size="sm"
            aria-label="New project"
            onClick={onNewProject}
          >
            <Icon name="plus" />
            <span>New</span>
          </button>
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
              onChange={(event) => setFilter(event.target.value)}
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
              <button
                type="button"
                className="ui-btn"
                data-size="xs"
                disabled={query.isFetching}
                onClick={() => void query.refetch()}
              >
                Try again
              </button>
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
            {items.length === 0 ? (
              <EmptyState
                title="No projects yet"
                action={
                  <button
                    type="button"
                    className="ui-btn"
                    data-variant="primary"
                    data-size="sm"
                    onClick={onNewProject}
                  >
                    New project
                  </button>
                }
              >
                A project keeps your materials, the checks you run on them and
                their results together.
              </EmptyState>
            ) : visible.length === 0 ? (
              <EmptyState title="No project matches the filter">
                {paged
                  ? "The filter searches the projects on this page."
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
        <CursorControls label="Projects pages" {...controls} />
      </div>
    </ListPane>
  );
}
