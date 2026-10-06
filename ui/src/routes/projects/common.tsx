import { useQuery } from "@tanstack/react-query";
import { type ReactNode, useId, useRef } from "react";

import { type ArtifactWriteResponse } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { Dialog } from "../../app/dialog";
import { ArtifactWriteForm } from "../artifacts/common";
import { CursorControls } from "../../app/cursor-controls";
import { useCursorStack } from "../../app/pagination";
import { GitRepositoryIcon } from "../artifacts/git-repository-icon";
import { MaterialKindIcon } from "./material-icon";
import {
  PROJECT_ARTIFACT_SHORTCUTS,
  type ShortcutDefinition,
} from "./shortcuts";
import { ProjectSectionActions } from "./navigation";
import { QueryView } from "../../app/query-view";
import { ArtifactBindingsTable } from "../artifacts/bindings";
import { artifactDetailPath } from "../artifacts/paths";

export function ProjectArtifactShortcutGrid({
  onSelect,
  onImportGit,
}: {
  onSelect: (shortcut: ShortcutDefinition) => void;
  onImportGit: () => void;
}) {
  return (
    <div className="project-shortcut-grid" aria-label="Artifact shortcuts">
      <button
        className="project-shortcut"
        type="button"
        aria-label="Import Git repository"
        onClick={onImportGit}
      >
        <span className="project-shortcut-icon">
          <GitRepositoryIcon />
        </span>
        <span>
          <strong>Git</strong>
          <small>Import a repository as a source archive</small>
        </span>
      </button>
      {PROJECT_ARTIFACT_SHORTCUTS.map((shortcut) => (
        <button
          className="project-shortcut"
          key={shortcut.id}
          type="button"
          aria-label={shortcut.label}
          onClick={() => onSelect(shortcut)}
        >
          <span className="project-shortcut-icon">
            <MaterialKindIcon kind={shortcut.id} />
          </span>
          <span>
            <strong>{shortcut.label}</strong>
            <small>{shortcut.description}</small>
          </span>
        </button>
      ))}
    </div>
  );
}

export function ProjectArtifactDialog({
  projectId,
  shortcut,
  onClose,
  onWritten,
}: {
  projectId: string;
  shortcut: ShortcutDefinition;
  onClose: () => void;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  const heading = useId();
  const closeButton = useRef<HTMLButtonElement>(null);

  return (
    <Dialog
      className="project-dialog panel"
      labelledBy={heading}
      initialFocusRef={closeButton}
      onRequestClose={onClose}
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Artifact shortcut</p>
          <h2 id={heading}>{shortcut.label}</h2>
        </div>
        <button
          ref={closeButton}
          className="project-dialog-close"
          type="button"
          aria-label="Close upload dialog"
          onClick={onClose}
        >
          ×
        </button>
      </div>
      <p className="muted-copy">
        The category only suggests editable Artifact metadata.
      </p>
      <ArtifactWriteForm
        scope={{ kind: "project", id: projectId }}
        suggested={shortcut}
        headingId={heading}
        onCancel={onClose}
        onWritten={onWritten}
      />
    </Dialog>
  );
}

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
