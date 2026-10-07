import { useId, type ReactNode, type Ref } from "react";
import { Link } from "react-router";

import type { ArtifactWriteResponse } from "../../api/artifacts";
import type { GitImportResult, GitSource } from "../../api/git-artifacts";
import { ContextLink } from "../../app/context-navigation";
import { formatBytes } from "../../app/format";
import { IdChip, StatusGlyph } from "../../ui";
import { MaterialIcon } from "./icons";
import { formatLabel, materialKindLabel } from "./kinds";
import { GitCommitChip, MaterialName } from "./material-row";
import { artifactDetailPath } from "./paths";
import "./materials.css";

/** Where a Git import came from: repository, branch or tag, commit. */
export function GitSourceLine({ source }: { source: GitSource }) {
  return (
    <p className="materials-notice-line">
      <span>
        From{" "}
        <span className="materials-mono materials-wrap">
          {source.repositoryUrl}
        </span>
      </span>
      <span aria-hidden="true">·</span>
      <span>{source.requestedRef ?? "default branch"}</span>
      <span aria-hidden="true">·</span>
      <GitCommitChip commit={source.resolvedCommit} />
    </p>
  );
}

/**
 * Success notice after a write: what was stored, where, its exact revision
 * (US-01) and the next steps. It takes focus (tabIndex -1) when the dialog
 * that made the write closes, so the outcome is read out.
 */
function StoredNotice({
  title,
  result,
  links,
  onDismiss,
  ref,
}: {
  title: string;
  result: ArtifactWriteResponse | GitImportResult;
  links: ReactNode;
  onDismiss: () => void;
  ref?: Ref<HTMLDivElement> | undefined;
}) {
  const titleId = useId();
  const gitSource = "gitSource" in result ? result.gitSource : undefined;
  return (
    <div
      ref={ref}
      className="materials-notice"
      role="status"
      aria-labelledby={titleId}
      tabIndex={-1}
    >
      <StatusGlyph tone="success" size={18} />
      <div className="materials-notice-body">
        <p id={titleId} className="materials-notice-title">
          {title}
        </p>
        <p className="materials-notice-line">
          <strong className="materials-notice-name">
            <MaterialName
              namespace={result.artifact.namespace}
              name={result.artifact.name}
            />
          </strong>
          <span>
            {materialKindLabel({ ...result, gitSource })} ·{" "}
            {formatLabel(result.mediaType)} · {formatBytes(result.size)}
          </span>
          <span className="materials-notice-revision">
            revision{" "}
            <IdChip value={result.artifact.revision} label="revision" />
          </span>
        </p>
        {gitSource === undefined ? null : <GitSourceLine source={gitSource} />}
        <p className="materials-notice-links">{links}</p>
      </div>
      <button
        type="button"
        className="materials-icon-button"
        aria-label="Dismiss"
        title="Dismiss"
        onClick={onDismiss}
      >
        <MaterialIcon name="close" />
      </button>
    </div>
  );
}

/** A material added to a project, with the ways to use it next. */
export function MaterialAddedNotice({
  projectId,
  result,
  onDismiss,
  ref,
}: {
  projectId: string;
  result: ArtifactWriteResponse | GitImportResult;
  onDismiss: () => void;
  ref?: Ref<HTMLDivElement> | undefined;
}) {
  const project = encodeURIComponent(projectId);
  return (
    <StoredNotice
      ref={ref}
      title={
        "gitSource" in result
          ? "Git repository imported into this project"
          : "Material added to this project"
      }
      result={result}
      onDismiss={onDismiss}
      links={
        <>
          <ContextLink
            returnLabel="Materials"
            to={artifactDetailPath(
              { kind: "project", id: projectId },
              result.artifact,
            )}
          >
            Open material
          </ContextLink>
          <Link to={`/checks/new?project=${project}`}>Start a check</Link>
          <Link to={`/projects/${project}/workflows`}>Run a workflow</Link>
        </>
      }
    />
  );
}

/** A file uploaded to the personal library. */
export function FileStoredNotice({
  result,
  onDismiss,
  ref,
}: {
  result: ArtifactWriteResponse;
  onDismiss: () => void;
  ref?: Ref<HTMLDivElement> | undefined;
}) {
  return (
    <StoredNotice
      ref={ref}
      title="File uploaded to your library"
      result={result}
      onDismiss={onDismiss}
      links={
        <ContextLink
          returnLabel="Files"
          to={artifactDetailPath({ kind: "user" }, result.artifact)}
        >
          Open file
        </ContextLink>
      }
    />
  );
}
