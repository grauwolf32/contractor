import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router";
import {
  ARTIFACT_NAME_PATTERN,
  getArtifactMetadata,
  type ArtifactMetadata,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import {
  importGitArtifact,
  type GitImportResult,
  type GitSource,
} from "../../api/git-artifacts";
import { getProjectArtifactMetadata } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { Dialog } from "../../app/dialog";
import { ErrorNotice } from "./common";
import "./git-artifacts.css";

export function GitSourceDetails({
  source,
}: {
  source: GitSource | undefined;
}) {
  return source === undefined ? null : (
    <div className="git-source-details">
      <strong>Git source</strong>
      <code>{source.repositoryUrl}</code>
      <span>{source.requestedRef ?? "Default branch"}</span>
      <code>Commit {source.resolvedCommit}</code>
    </div>
  );
}
export function GitImportDialog({
  projectId,
  suggestedName = "source",
  onClose,
  onImported,
}: {
  projectId?: string;
  suggestedName?: string;
  onClose: () => void;
  onImported: (result: GitImportResult) => void;
}) {
  const api = usePublicAPI();
  const cache = useQueryClient();
  const heading = useId();
  const [repositoryUrl, setURL] = useState("");
  const [ref, setRef] = useState("");
  const [namespace, setNamespace] = useState("sources");
  const [name, setName] = useState(suggestedName);
  const [baseline, setBaseline] = useState<ArtifactMetadata>();
  const [replace, setReplace] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>();
  const operation = useRef<AbortController | null>(null);
  const initialFocus = useRef<HTMLInputElement>(null);
  useEffect(() => () => operation.current?.abort(), []);
  function close() {
    operation.current?.abort();
    onClose();
  }
  function changeTarget(update: () => void) {
    update();
    setBaseline(undefined);
    setReplace(false);
  }
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (pending) return;
    if (
      !ARTIFACT_NAME_PATTERN.test(namespace) ||
      !ARTIFACT_NAME_PATTERN.test(name)
    ) {
      setError(
        new Error(
          "Namespace and name must start with an ASCII letter or digit, use only letters, digits, dots, underscores or hyphens, and contain at most 128 characters. Spaces are not allowed.",
        ),
      );
      return;
    }
    if (!repositoryUrl || repositoryUrl.trim() !== repositoryUrl) {
      setError(new Error("Enter a repository URL without surrounding spaces."));
      return;
    }
    const controller = new AbortController();
    operation.current = controller;
    setPending(true);
    setError(undefined);
    try {
      if (baseline === undefined) {
        let current: ArtifactMetadata | undefined;
        try {
          current =
            projectId === undefined
              ? await getArtifactMetadata(api, { namespace, name })
              : await getProjectArtifactMetadata(api, {
                  projectId,
                  namespace,
                  name,
                });
        } catch (error) {
          if (!(error instanceof PublicAPIError && error.status === 404))
            throw error;
        }
        if (controller.signal.aborted) return;
        if (current !== undefined) {
          setBaseline(current);
          setReplace(false);
          return;
        }
      } else if (!replace || baseline.frozen) return;
      const result = await importGitArtifact(
        api,
        {
          ...(projectId === undefined ? {} : { projectId }),
          namespace,
          name,
          repositoryUrl,
          ...(ref === "" ? {} : { ref }),
          ...(baseline === undefined
            ? {}
            : { expectedRevision: baseline.artifact.revision }),
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      void cache.invalidateQueries({
        queryKey:
          projectId === undefined
            ? queryKeys.artifacts.all
            : queryKeys.projects.artifacts.all(projectId),
      });
      onImported(result);
    } catch (error) {
      if (!controller.signal.aborted) {
        setError(error);
        if (error instanceof PublicAPIError && error.status === 409) {
          setBaseline(undefined);
          setReplace(false);
        }
      }
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }
  return (
    <Dialog
      className="project-dialog panel git-import-dialog"
      labelledBy={heading}
      initialFocusRef={initialFocus}
      onRequestClose={close}
    >
      <div className="project-dialog-heading">
        <h2 id={heading}>Import Git repository</h2>
        <button
          className="project-dialog-close"
          type="button"
          aria-label="Close Git import"
          onClick={close}
        >
          ×
        </button>
      </div>
      <p>
        Import a tracked source snapshot as a ZIP. Private SSH uses your{" "}
        <Link to="/operations/settings#repository-access" onClick={close}>
          Git key in Operations Settings
        </Link>
        .
      </p>
      <form onSubmit={(event) => void submit(event)}>
        <label>
          Repository URL
          <input
            ref={initialFocus}
            value={repositoryUrl}
            onChange={(event) => setURL(event.target.value)}
            placeholder="https://host/team/repository.git"
            disabled={pending}
            required
          />
        </label>
        <label>
          Branch or tag (optional)
          <input
            value={ref}
            onChange={(event) => setRef(event.target.value)}
            placeholder="Default branch"
            disabled={pending}
            maxLength={1024}
          />
        </label>
        <label>
          Artifact namespace
          <input
            value={namespace}
            onChange={(event) =>
              changeTarget(() => setNamespace(event.target.value))
            }
            disabled={pending}
            required
          />
        </label>
        <label>
          Artifact name
          <input
            value={name}
            onChange={(event) =>
              changeTarget(() => setName(event.target.value))
            }
            disabled={pending}
            required
          />
        </label>
        {baseline === undefined ? null : (
          <div className="notice notice-warning">
            <p>
              This binding exists at <code>{baseline.artifact.revision}</code>.
            </p>
            {baseline.frozen ? (
              <p>This binding is frozen. Choose another name.</p>
            ) : (
              <label>
                <input
                  type="checkbox"
                  checked={replace}
                  onChange={(event) => setReplace(event.target.checked)}
                  disabled={pending}
                />
                Replace this exact revision
              </label>
            )}
          </div>
        )}
        {error === undefined ? null : <ErrorNotice error={error} />}
        <p className="muted-copy">
          Import can take up to two minutes. If you cancel or lose the response,
          inspect the Artifact library before retrying: a complete import may
          already be stored.
        </p>
        <div className="inline-actions">
          <button
            type="submit"
            disabled={
              pending ||
              (baseline !== undefined && (!replace || baseline.frozen))
            }
          >
            {pending ? "Importing…" : "Import snapshot"}
          </button>
          <button type="button" className="secondary-button" onClick={close}>
            {pending ? "Cancel import" : "Cancel"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
