import type { ReactNode } from "react";
import { Link } from "react-router";

import { ContextLink } from "../../../app/context-navigation";
import { ErrorNotice } from "../../../app/error-notice";
import {
  EmptyState,
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  StatusGlyph,
  modKeyLabel,
  type ListNavigationContainerProps,
  type StatusTone,
} from "../../../ui";
import { inSentence, missingLabel } from "./check-types";
import { projectPaths } from "./paths";
import type { ReadinessGroup, RowView, StartCheck } from "./use-start-check";

const GROUP_TITLES: Readonly<Record<ReadinessGroup, string>> = {
  ready: "Ready with your materials",
  missing: "Needs more materials",
  unavailable: "Can't run on this server",
  unknown: "All check types",
};

const GROUP_TONES: Readonly<Record<ReadinessGroup, StatusTone>> = {
  ready: "done",
  missing: "idle",
  unavailable: "blocked",
  unknown: "neutral",
};

/** "source code and an API spec" style list: "Source code, API spec". */
function wordList(words: readonly string[]): string {
  const lower = words.map(inSentence);
  if (lower.length <= 1) return lower[0] ?? "";
  return `${lower.slice(0, -1).join(", ")} and ${lower.at(-1)}`;
}

/** The second meta part: what the check type uses or still needs. */
function rowNeeds(row: RowView): ReactNode {
  const { readiness } = row;
  if (readiness?.state === "missing")
    return (
      <>
        <strong className="start-row-missing">Missing:</strong>{" "}
        {readiness.missing.map(missingLabel).join(", ")}
      </>
    );
  if (readiness?.state === "unavailable") return "Not supported by this server";
  return row.uses.length === 0 ? undefined : `Uses ${wordList(row.uses)}`;
}

function RowLinks({ row, projectId }: { row: RowView; projectId: string }) {
  if (row.readiness?.state !== "missing") return null;
  const paths = projectPaths(projectId);
  const needsInput = row.readiness.missing.some(
    (need) => need.kind !== "live-target",
  );
  const needsTarget = row.readiness.missing.some(
    (need) => need.kind === "live-target",
  );
  return (
    <span className="start-row-links">
      {needsInput ? <Link to={paths.addMaterial}>Add materials</Link> : null}
      {needsTarget ? (
        <Link to={paths.settings}>Set the live target</Link>
      ) : null}
    </span>
  );
}

function TypeRow({ row, projectId }: { row: RowView; projectId: string }) {
  return (
    <ListRow
      to={row.href}
      selected={row.selected}
      glyph={<StatusGlyph tone={GROUP_TONES[row.group]} />}
      title={row.presentation.label}
      meta={
        <span className="start-row-meta">
          <span>{row.presentation.description}</span>
          <span>{rowNeeds(row)}</span>
        </span>
      }
      trailing={
        row.suggested ? (
          <span className="start-badge">Suggested</span>
        ) : undefined
      }
    >
      {row.readiness?.state === "missing" ? (
        <RowLinks row={row} projectId={projectId} />
      ) : undefined}
    </ListRow>
  );
}

/** The list pane: check types grouped by readiness, J / K to move. */
export function TypeList({
  model,
  containerProps,
}: {
  model: StartCheck;
  containerProps: ListNavigationContainerProps;
}) {
  const { catalog, materials, project, projectId, rows } = model;
  const { query } = catalog;
  const paths = projectPaths(projectId);
  const projectName = project?.name;
  const groups = (
    ["ready", "missing", "unavailable", "unknown"] as const
  ).flatMap((group) => {
    const members = rows.filter((row) => row.group === group);
    return members.length === 0 ? [] : [{ group, members }];
  });
  const count = rows.length;
  // Changing the project keeps what the user typed and the chosen type.
  const changeProject = model.href({
    project: undefined,
    objective: model.objective.trim() || undefined,
  });

  let body;
  if (query.error !== null && query.data === undefined)
    body = (
      <div className="start-pane-block">
        <ErrorNotice
          error={query.error}
          context="Check types could not be loaded."
          onRetry={() => void query.refetch()}
          retryPending={query.isFetching}
        />
      </div>
    );
  else if (!model.listReady)
    body = (
      <p className="start-pane-block start-quiet" role="status">
        Checking which check types your materials support…
      </p>
    );
  else if (count === 0)
    body = (
      <EmptyState title="No check types are published">
        An operator publishes check types on the server. None are listed yet.
      </EmptyState>
    );
  else
    body = (
      <div {...containerProps} className="start-type-groups">
        {groups.map(({ group, members }) => (
          <ListSection
            key={group}
            title={GROUP_TITLES[group]}
            count={members.length}
            aside={group === "ready" ? "by file format" : undefined}
          >
            {members.map((row) => (
              <TypeRow key={row.key} row={row} projectId={projectId} />
            ))}
          </ListSection>
        ))}
      </div>
    );

  return (
    <ListPane
      title={
        projectName === undefined
          ? "Start a check"
          : `Start a check on ${projectName}`
      }
      subtitle={
        <>
          {model.listReady && count > 0
            ? `${count} check ${count === 1 ? "type" : "types"} · `
            : null}
          <Link to={changeProject}>Change project</Link>
        </>
      }
      footer={
        <>
          <span>
            <Kbd>J</Kbd> <Kbd>K</Kbd> move
          </span>
          <span>
            <Kbd>{modKeyLabel()}</Kbd> <Kbd>Enter</Kbd> starts the check
          </span>
        </>
      }
    >
      {model.unknownType === undefined ? null : (
        <p className="start-pane-block start-notice" role="status">
          {model.unknownType.partial ? (
            <>
              No check type named <code>{model.unknownType.name}</code> is among
              those loaded so far. Load more check types to look further.
            </>
          ) : (
            <>
              This server lists no check type named{" "}
              <code>{model.unknownType.name}</code>. Choose one below.
            </>
          )}
        </p>
      )}
      {project === undefined && model.listReady ? (
        <p className="start-pane-block start-notice" role="status">
          The project could not be loaded, so its live target is unknown.
        </p>
      ) : null}
      {model.listReady &&
      materials.items === undefined &&
      materials.error !== null ? (
        <div className="start-pane-block">
          <ErrorNotice
            error={materials.error}
            context="The project's materials could not be loaded, so readiness is unknown."
          />
          <button
            type="button"
            className="ui-btn"
            data-size="sm"
            disabled={materials.retrying}
            onClick={materials.retry}
          >
            Retry loading materials
          </button>
        </div>
      ) : null}
      {model.listReady && materials.hasMore ? (
        <p className="start-pane-block start-quiet">
          Readiness uses the first {materials.items?.length ?? 0} materials of
          this project. Load more from a check type&apos;s Materials.
        </p>
      ) : null}
      {/* A failed refresh or "Load more" keeps the check types read before. */}
      {query.error !== null && query.data !== undefined ? (
        <div className="start-pane-block">
          <ErrorNotice
            error={query.error}
            context={
              query.isFetchNextPageError
                ? "More check types could not be loaded."
                : "Check types could not be refreshed. The list shows the ones read before."
            }
            onRetry={() =>
              void (query.isFetchNextPageError
                ? query.fetchNextPage()
                : query.refetch())
            }
            retryPending={query.isFetching}
          />
        </div>
      ) : null}
      {body}
      {query.hasNextPage ? (
        <div className="start-pane-block">
          <button
            type="button"
            className="ui-btn"
            data-size="sm"
            disabled={query.isFetchingNextPage}
            onClick={() => void query.fetchNextPage()}
          >
            {query.isFetchingNextPage ? "Loading…" : "Load more check types"}
          </button>
        </div>
      ) : null}
      {model.listReady && rows.some((row) => row.group === "missing") ? (
        <p className="start-pane-block start-list-note">
          Add materials or a live target to unlock more check types.{" "}
          <Link to={paths.addMaterial}>Add materials</Link> ·{" "}
          <Link to={paths.settings}>Project settings</Link>
        </p>
      ) : null}
      {model.listReady && count > 0 ? (
        <p className="start-pane-block start-quiet">
          <ContextLink to="/catalog/audit-presets" returnLabel="Start a check">
            See all check types
          </ContextLink>{" "}
          in the Library, with what each one covers.
        </p>
      ) : null}
    </ListPane>
  );
}
