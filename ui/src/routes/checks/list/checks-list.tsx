import { useId, useMemo, type ReactNode } from "react";
import { Link } from "react-router";

import type { AuditWorkspace } from "../../../api/audits";
import type {
  AllChecks,
  CrossProjectCheck,
  ProjectsIndex,
} from "../../../api/cross-project";
import { ErrorNotice } from "../../../app/error-notice";
import { RecordedTime } from "../../../app/recorded-time";
import { checkStateLabel } from "../../../app/vocabulary";
import {
  EmptyState,
  FilterChips,
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  ProgressSegments,
  StatusChip,
  StatusGlyph,
  type ListNavigationContainerProps,
} from "../../../ui";
import {
  countLegend,
  countSegments,
  doneSummary,
  legendSentence,
  workCounts,
} from "../../projects/audits/check-model";
import { auditProfileLabel } from "../../projects/audits/labels";
import { CHECK_FILTERS, type CheckFilter } from "./filters";

/** Keys that move through the list and open a check (useListNavigation). */
const LIST_KEYS = "J K ArrowDown ArrowUp Home End Enter";

/** A check's progress line in a list row, from its workspace counts. */
function RowProgress({ workspace }: { workspace: AuditWorkspace }) {
  const counts = workCounts(workspace);
  if (counts.total === 0) return null;
  const legend = countLegend(counts).filter((entry) => entry.count > 0);
  const done = doneSummary(workspace.completedChecks, counts.total);
  return (
    <div className="checks-row-progress">
      <ProgressSegments
        size="sm"
        segments={countSegments(legend)}
        label={`${done}: ${legendSentence(legend)}`}
      />
      <span>{done}</span>
    </div>
  );
}

function ProjectFilter({
  projects,
  value,
  onChange,
}: {
  projects: ProjectsIndex["projects"];
  value: string | undefined;
  onChange: (projectId: string | undefined) => void;
}) {
  const id = useId();
  const sorted = useMemo(
    () =>
      [...projects].sort((left, right) => left.name.localeCompare(right.name)),
    [projects],
  );
  const known =
    value === undefined || sorted.some((p) => p.projectId === value);
  return (
    <div className="checks-project-filter">
      <label htmlFor={id}>Project</label>
      <select
        id={id}
        value={value ?? ""}
        onChange={(event) =>
          onChange(
            event.currentTarget.value === ""
              ? undefined
              : event.currentTarget.value,
          )
        }
      >
        <option value="">All projects</option>
        {known ? null : <option value={value}>{value}</option>}
        {sorted.map((project) => (
          <option key={project.projectId} value={project.projectId}>
            {project.name}
          </option>
        ))}
      </select>
    </div>
  );
}

function projectNames(
  checks: AllChecks,
  projects: ProjectsIndex["projects"],
): string {
  const names = new Map(projects.map((p) => [p.projectId, p.name]));
  return [
    ...new Set(
      checks.errors.flatMap((error) =>
        error.projectId === undefined
          ? []
          : [names.get(error.projectId) ?? error.projectId],
      ),
    ),
  ].join(", ");
}

/**
 * The Checks list pane: every check of the first page of projects, newest
 * first, filtered by state and project (both in the URL), with a progress
 * line where the Server's counts are read.
 */
export function ChecksListPane({
  checks,
  projects,
  rows,
  counts,
  filter,
  projectId,
  selectedId,
  workspaces,
  rowHref,
  startHref,
  navigation,
  onFilter,
  onProject,
}: {
  checks: AllChecks;
  projects: ProjectsIndex["projects"];
  /** The checks the filters show, newest first. */
  rows: readonly CrossProjectCheck[];
  /** How many checks each filter holds (within the project filter). */
  counts: Readonly<Record<CheckFilter, number>>;
  filter: CheckFilter;
  projectId: string | undefined;
  selectedId: string | undefined;
  workspaces: ReadonlyMap<string, AuditWorkspace>;
  rowHref: (auditId: string) => string;
  startHref: string;
  navigation: ListNavigationContainerProps;
  onFilter: (filter: CheckFilter) => void;
  onProject: (projectId: string | undefined) => void;
}) {
  const running = counts.running;
  const waiting = counts.waiting;
  const parts = [
    running === 0 ? "" : `${running.toLocaleString("en-US")} running`,
    waiting === 0 ? "" : `${waiting.toLocaleString("en-US")} waiting for you`,
  ].filter((part) => part !== "");
  const subtitle =
    checks.isPending && checks.checks.length === 0
      ? undefined
      : parts.length === 0
        ? "Nothing is running or waiting for you."
        : `${parts.join(", ")}.`;
  let state: ReactNode = null;
  if (checks.error !== null)
    state = (
      <div className="checks-pane-note">
        <ErrorNotice
          error={checks.error}
          context="Checks could not be loaded"
          onRetry={() => void checks.refetch()}
        />
      </div>
    );
  else if (checks.isPending && checks.checks.length === 0)
    state = (
      <p className="checks-pane-note checks-quiet" role="status">
        Loading checks…
      </p>
    );
  else if (rows.length === 0)
    state = (
      <EmptyState
        title={checks.checks.length === 0 ? "No checks yet" : "No checks match"}
        action={
          <Link className="ui-btn" data-size="sm" to={startHref}>
            Start a check
          </Link>
        }
      >
        {checks.checks.length === 0
          ? "A check reads a project's materials and reports what it finds, item by item."
          : "Choose another state or project to see more checks."}
      </EmptyState>
    );
  const failed = projectNames(checks, projects);
  return (
    <ListPane
      title="Checks"
      subtitle={subtitle}
      actions={
        <Link
          className="ui-btn"
          data-variant="primary"
          data-size="sm"
          to={startHref}
        >
          Start a check
        </Link>
      }
      toolbar={
        <div className="checks-list-toolbar">
          <FilterChips
            label="Filter checks by state"
            options={CHECK_FILTERS.map((option) => ({
              value: option.value,
              label: option.label,
              count: counts[option.value],
            }))}
            value={filter}
            onChange={onFilter}
          />
          <ProjectFilter
            projects={projects}
            value={projectId}
            onChange={onProject}
          />
        </div>
      }
      footer={
        <span className="checks-key-hint" aria-hidden="true">
          <Kbd>J</Kbd> <Kbd>K</Kbd> move <Kbd>Enter</Kbd> open
        </span>
      }
    >
      {checks.partial && checks.error === null ? (
        <div
          className="checks-notice checks-pane-note"
          data-tone="warning"
          role="status"
        >
          <p>
            {failed === ""
              ? "Some checks could not be refreshed; the list may be out of date."
              : `Checks of ${failed} could not be loaded; they are missing here.`}
          </p>
          <button
            className="ui-btn"
            data-size="xs"
            type="button"
            onClick={() => void checks.refetch()}
          >
            Try again
          </button>
        </div>
      ) : null}
      <div className="checks-list-rows" {...navigation}>
        <ListSection>
          {rows.map(({ project, audit }) => {
            const label = checkStateLabel(audit.state);
            const workspace = workspaces.get(audit.auditId);
            return (
              <ListRow
                key={audit.auditId}
                id={`check-${audit.auditId}`}
                to={rowHref(audit.auditId)}
                selected={audit.auditId === selectedId}
                // The footer shows these keys; the rows declare them.
                ariaKeyShortcuts={LIST_KEYS}
                glyph={<StatusGlyph tone={label.tone} />}
                title={
                  <>
                    {auditProfileLabel(audit)}{" "}
                    {/* Checks of one type are told apart by their project. */}
                    <span className="ui-visually-hidden">· {project.name}</span>
                  </>
                }
                meta={[
                  project.name,
                  <StatusChip key="state" tone={label.tone} size="sm">
                    {label.label}
                  </StatusChip>,
                  <span key="updated">
                    Updated <RecordedTime value={audit.updatedAt} />
                  </span>,
                ]}
              >
                {workspace === undefined ? null : (
                  <RowProgress workspace={workspace} />
                )}
              </ListRow>
            );
          })}
        </ListSection>
      </div>
      {state}
    </ListPane>
  );
}
