import { useQueryClient } from "@tanstack/react-query";
import { useId, useRef, useState, type FormEvent } from "react";
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
import { writeConflict } from "../../app/conflict";
import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";
import { RecordedTime } from "../../app/recorded-time";
import { Button } from "../../ui";
import { useUploadOperation } from "./upload-operation";
import "./git-artifacts.css";
import "./materials.css";

/**
 * Plain-text Git provenance (repository, ref and the full commit) for forms
 * that show it next to a selected input, such as Run setup.
 */
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

/**
 * Imports one Git snapshot as a source ZIP into the project (with
 * `projectId`) or the personal library. An existing binding is shown with
 * its exact revision and replaced only after explicit consent; a locked one
 * is refused. Closing the dialog cancels a running import.
 */
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
  const description = useId();
  const [repositoryUrl, setURL] = useState("");
  const [ref, setRef] = useState("");
  const [namespace, setNamespace] = useState("sources");
  const [name, setName] = useState(suggestedName);
  const [baseline, setBaseline] = useState<ArtifactMetadata>();
  const [replace, setReplace] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>();
  const operation = useUploadOperation(onClose);
  const close = operation.close;
  const initialFocus = useRef<HTMLInputElement>(null);
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
    const signal = operation.start();
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
        if (signal.aborted) return;
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
        signal,
      );
      if (signal.aborted) return;
      void cache.invalidateQueries({
        queryKey:
          projectId === undefined
            ? queryKeys.artifacts.all
            : queryKeys.projects.artifacts.all(projectId),
      });
      onImported(result);
    } catch (error) {
      if (!signal.aborted) {
        setError(error);
        if (writeConflict(error) !== undefined) {
          setBaseline(undefined);
          setReplace(false);
        }
      }
    } finally {
      if (!signal.aborted) setPending(false);
    }
  }
  return (
    <Dialog
      className="ui-dialog ui-panel materials-sheet"
      labelledBy={heading}
      describedBy={description}
      initialFocusRef={initialFocus}
      onRequestClose={close}
    >
      <DialogHeader
        id={heading}
        title="Import Git repository"
        close={{ label: "Close Git import", onClose: close }}
      />
      <p id={description} className="materials-sheet-intro">
        Imports one branch or tag as a source ZIP and records the exact commit.
        Private repositories over SSH use your{" "}
        <Link to="/operations/settings#repository-access" onClick={close}>
          Git key in Settings
        </Link>
        .
      </p>
      <form className="materials-form" onSubmit={(event) => void submit(event)}>
        <label>
          Repository URL
          <input
            ref={initialFocus}
            value={repositoryUrl}
            onChange={(event) => setURL(event.target.value)}
            placeholder="https://host/team/repository.git"
            autoComplete="off"
            spellCheck={false}
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
            autoComplete="off"
            spellCheck={false}
            disabled={pending}
            maxLength={1024}
          />
        </label>
        <div className="materials-form-pair">
          <label>
            Namespace
            <input
              value={namespace}
              onChange={(event) =>
                changeTarget(() => setNamespace(event.target.value))
              }
              autoComplete="off"
              spellCheck={false}
              disabled={pending}
              required
            />
          </label>
          <label>
            Name
            <input
              value={name}
              onChange={(event) =>
                changeTarget(() => setName(event.target.value))
              }
              autoComplete="off"
              spellCheck={false}
              disabled={pending}
              required
            />
          </label>
        </div>
        {baseline === undefined ? null : (
          <div className="materials-callout" data-tone="warning">
            <p>
              <strong>
                {baseline.artifact.namespace}/{baseline.artifact.name} already
                exists.
              </strong>{" "}
              Its current version is from{" "}
              <RecordedTime value={baseline.createdAt} />, revision{" "}
              <code>{baseline.artifact.revision}</code>.
            </p>
            {baseline.frozen ? (
              <p>It is locked and cannot be replaced. Choose another name.</p>
            ) : (
              <label className="materials-check">
                <input
                  type="checkbox"
                  checked={replace}
                  onChange={(event) => setReplace(event.target.checked)}
                  disabled={pending}
                />
                Replace this revision
              </label>
            )}
          </div>
        )}
        {error === undefined ? null : <ErrorNotice error={error} />}
        <p className="ui-caption">
          An import can take up to two minutes. If you cancel it or lose the
          response, look for it before trying again: a complete import may
          already be stored.
        </p>
        <div className="ui-toolbar" data-justify="end">
          <Button onClick={close}>
            {pending ? "Cancel import" : "Cancel"}
          </Button>
          <Button
            type="submit"
            variant="primary"
            pending={pending}
            pendingLabel="Importing…"
            disabled={baseline !== undefined && (!replace || baseline.frozen)}
          >
            Import snapshot
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
