import { Link, useSearchParams } from "react-router";

import {
  CROSS_PROJECT_LIMITS,
  useProjectsIndex,
} from "../../../api/cross-project";
import { ErrorNotice } from "../../../app/error-notice";
import {
  DetailPane,
  EmptyState,
  ListPane,
  ListRow,
  ListSection,
  PaneLayout,
  StatusGlyph,
} from "../../../ui";

/** First line of a project description, for the row's meta line. */
function firstLine(text: string): string | undefined {
  return text.split("\n", 1)[0]?.trim() || undefined;
}

/**
 * Step one when /checks/new has no project: choose the project to check.
 * The other query parameters (objective, type) carry over.
 */
export function ProjectPicker({ notice }: { notice?: string | undefined }) {
  const [params] = useSearchParams();
  const index = useProjectsIndex();

  function startIn(projectId: string): string {
    const next = new URLSearchParams(params);
    next.set("project", projectId);
    return `/checks/new?${next.toString()}`;
  }

  let body;
  if (index.error !== null)
    body = (
      <div className="start-pane-block">
        <ErrorNotice
          error={index.error}
          context="Projects could not be loaded."
          onRetry={() => void index.refetch()}
        />
      </div>
    );
  else if (index.isPending && index.projects.length === 0)
    body = (
      <p className="start-pane-block start-quiet" role="status">
        Loading projects…
      </p>
    );
  else if (index.projects.length === 0)
    body = (
      <EmptyState
        title="No projects yet"
        action={<Link to="/projects?new=1">New project</Link>}
      >
        A check runs on a project&apos;s materials. Create a project and add its
        source code or API spec first.
      </EmptyState>
    );
  else
    body = (
      <ListSection title="Projects" count={index.projects.length}>
        {index.projects.map((project) => (
          <ListRow
            key={project.projectId}
            to={startIn(project.projectId)}
            glyph={<StatusGlyph tone="neutral" />}
            title={project.name}
            meta={[
              firstLine(project.description),
              project.httpTarget === undefined ? undefined : "Live target set",
            ]}
          />
        ))}
      </ListSection>
    );

  return (
    <PaneLayout
      listLabel="Projects"
      detailLabel="Set up the check"
      showDetail={false}
      list={
        <ListPane title="Start a check" subtitle="Choose the project to check.">
          {notice === undefined ? null : (
            <p className="start-pane-block start-notice" role="alert">
              {notice}
            </p>
          )}
          {body}
          {index.truncated ? (
            <p className="start-pane-block start-quiet">
              Showing the first {CROSS_PROJECT_LIMITS.projects} projects. To
              check another one, open it from{" "}
              <Link to="/projects">Projects</Link> and start the check there.
            </p>
          ) : null}
        </ListPane>
      }
      detail={
        <DetailPane>
          <EmptyState
            title="Choose a project"
            action={<Link to="/projects?new=1">New project</Link>}
          >
            A check runs on one project&apos;s materials: its source code, API
            specs and live target. Pick the project on the left, then the check
            type.
          </EmptyState>
        </DetailPane>
      }
    />
  );
}
