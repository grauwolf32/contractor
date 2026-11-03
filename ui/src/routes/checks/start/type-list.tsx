import type { ReactNode } from "react";
import { Link } from "react-router";

import { ContextLink } from "../../../app/context-navigation";
import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import {
  Button,
  EmptyState,
  Field,
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  modKeyLabel,
  type ListNavigationContainerProps,
} from "../../../ui";
import {
  checkTypeIcon,
  inSentence,
  joinWords,
  missingLabel,
} from "./check-types";
import { projectPaths } from "./paths";
import type { ReadinessGroup, RowView, StartCheck } from "./use-start-check";

const GROUP_TITLES: Readonly<Record<ReadinessGroup, string>> = {
  ready: "Ready with your materials",
  missing: "Needs more materials",
  unavailable: "Can't run on this server",
  unknown: "All check types",
};

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
  return row.uses.length === 0
    ? undefined
    : `Uses ${joinWords(row.uses.map(inSentence))}`;
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
      glyph={<Icon name={checkTypeIcon()} size={16} />}
      title={row.presentation.label}
      meta={
        <span className="start-row-meta">
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
  rows,
  search,
  onSearch,
}: {
  model: StartCheck;
  containerProps: ListNavigationContainerProps;
  rows: readonly RowView[];
  search: string;
  onSearch: (value: string) => void;
}) {
  const { catalog, materials, project, projectId } = model;
  const { query } = catalog;
  const paths = projectPaths(projectId);
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
      <p className="start-pane-block ui-caption" role="status">
        Checking which check types your materials support…
      </p>
    );
  else if (count === 0)
    body = (
      <EmptyState
        title={
          model.rows.length === 0
            ? "No check types are published"
            : "No matching check types"
        }
      >
        {model.rows.length === 0
          ? "An operator publishes check types on the server. None are listed yet."
          : "Try another name or description."}
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
      title="Choose a check type"
      titleAs="h2"
      toolbar={
        <Field label="Find a check type">
          {(control) => (
            <input
              {...control}
              type="search"
              value={search}
              onChange={(event) => onSearch(event.target.value)}
              placeholder="Search check types"
            />
          )}
        </Field>
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
          <Button
            size="sm"
            pending={materials.retrying}
            onClick={materials.retry}
          >
            Retry loading materials
          </Button>
        </div>
      ) : null}
      {model.listReady && materials.hasMore ? (
        <p className="start-pane-block ui-caption">
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
          <Button
            size="sm"
            pending={query.isFetchingNextPage}
            pendingLabel="Loading…"
            onClick={() => void query.fetchNextPage()}
          >
            Load more check types
          </Button>
        </div>
      ) : null}
      {model.listReady && rows.some((row) => row.group === "missing") ? (
        <p className="start-pane-block ui-caption start-list-note">
          Add materials or a live target to unlock more check types.{" "}
          <Link to={paths.addMaterial}>Add materials</Link> ·{" "}
          <Link to={paths.settings}>Project settings</Link>
        </p>
      ) : null}
      {model.listReady && count > 0 ? (
        <p className="start-pane-block ui-caption">
          <ContextLink to="/catalog/audit-presets" returnLabel="Start a check">
            See all check types
          </ContextLink>{" "}
          in the Library, with what each one covers.
        </p>
      ) : null}
    </ListPane>
  );
}
