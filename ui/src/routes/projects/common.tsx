import { useQuery } from "@tanstack/react-query";
import { type ReactNode } from "react";

import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { useCursorStack } from "../../app/pagination";
import { ProjectSectionActions } from "./navigation";
import { QueryView } from "../../app/query-view";
import { ArtifactBindingsTable } from "../artifacts/bindings";
import { artifactDetailPath } from "../artifacts/paths";

export function ProjectRegion({
  eyebrow,
  title,
  action,
  children,
  id,
  compact = false,
}: {
  eyebrow: string;
  title: string;
  action?: ReactNode;
  children: ReactNode;
  id?: string;
  /** Workspace tab: the tab names the section, actions go to the tab bar. */
  compact?: boolean;
}) {
  return (
    <section className="panel project-region" id={id}>
      {compact ? (
        action === undefined ? null : (
          <ProjectSectionActions>{action}</ProjectSectionActions>
        )
      ) : (
        <div className="section-heading">
          <div>
            <p className="eyebrow">{eyebrow}</p>
            <h3>{title}</h3>
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

/**
 * Read-only bindings table for legacy evaluation workspaces: the recorded
 * inputs of a workspace without upload shortcuts or Git import.
 */
export function ProjectArtifactBindings({
  projectId,
  detailRoot,
}: {
  projectId: string;
  detailRoot: "/projects" | "/evals";
}) {
  const api = usePublicAPI();
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.projects.artifacts.list(projectId, undefined, cursor),
    queryFn: () =>
      listProjectArtifacts(api, {
        projectId,
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });
  return (
    <ProjectRegion
      eyebrow="Current bindings"
      title="Artifacts"
      id="project-artifacts"
    >
      <QueryView
        query={query}
        loading={
          <p className="loading-copy" role="status">
            Loading Project Artifacts…
          </p>
        }
        onRetry={() => void query.refetch()}
        isEmpty={(queryData) => queryData.items.length === 0}
        empty={
          <div className="compact-empty">
            <strong>No Artifact bindings in this workspace.</strong>
          </div>
        }
      >
        {(queryData) => (
          <ArtifactBindingsTable
            items={queryData.items}
            returnLabel="Project Artifacts"
            returnHash="#project-artifacts"
            detailPath={(item) =>
              artifactDetailPath(
                { kind: "project", id: projectId },
                {
                  namespace: item.artifact.namespace,
                  name: item.artifact.name,
                },
                detailRoot,
              )
            }
          />
        )}
      </QueryView>
      <CursorControls
        label="Project Artifact pages"
        {...pages.controls(query.data?.page)}
      />
    </ProjectRegion>
  );
}
