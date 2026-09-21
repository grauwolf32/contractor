import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type FormEvent,
  type ReactNode,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";

import {
  ARTIFACT_NAME_PATTERN,
  MAXIMUM_ARTIFACT_BYTES,
  MEDIA_TYPE_PATTERN,
  type ArtifactWriteResponse,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import {
  listProjectArtifacts,
  writeProjectArtifact,
  type ProjectArtifactWriteRequest,
} from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { ContextLink } from "../../app/context-navigation";
import { Dialog } from "../../app/dialog";
import {
  artifactFileStem,
  inferredArtifactMediaType,
} from "../artifacts/artifact-file";
import { ArtifactMediaTypeField } from "../artifacts/media-type-field";
import {
  ArtifactFileDrop,
  CursorControls,
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../artifacts/common";
import { GitRepositoryIcon } from "../artifacts/git-repository-icon";
import {
  PROJECT_ARTIFACT_SHORTCUTS,
  type ProjectArtifactShortcut,
  type ShortcutDefinition,
} from "./shortcuts";
import { ProjectSectionActions } from "./navigation";

function ArtifactShortcutIcon({
  shortcut,
}: {
  shortcut: ProjectArtifactShortcut;
}) {
  const common = {
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  switch (shortcut) {
    case "sources":
      return (
        <svg {...common}>
          <path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14" />
        </svg>
      );
    case "openapi":
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="2.5" />
          <path d="M12 3v6.5M12 14.5V21M3 12h6.5M14.5 12H21M5.6 5.6l4.6 4.6M13.8 13.8l4.6 4.6M18.4 5.6l-4.6 4.6M10.2 13.8l-4.6 4.6" />
        </svg>
      );
    case "likec4":
      return (
        <svg {...common}>
          <rect x="3" y="4" width="7" height="5" rx="1" />
          <rect x="14" y="15" width="7" height="5" rx="1" />
          <path d="M10 6.5h5a2 2 0 0 1 2 2V15M7 9v5a3 3 0 0 0 3 3h4" />
        </svg>
      );
    case "docs":
      return (
        <svg {...common}>
          <path d="M6 3h8l4 4v14H6zM14 3v5h4M9 12h6M9 16h6" />
        </svg>
      );
    case "diffs":
      return (
        <svg {...common}>
          <path d="M4 7h8M8 3v8M4 17h8M16 5h4M18 3v4M16 17h4" />
        </svg>
      );
    case "other":
      return (
        <svg {...common}>
          <path d="M4 7.5h6l2 2h8v10H4zM4 7.5v-3h6l2 3" />
        </svg>
      );
  }
}

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
            <ArtifactShortcutIcon shortcut={shortcut.id} />
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

export function ProjectArtifactWriteForm({
  projectId,
  suggested,
  fixedIdentity,
  fixedNamespace,
  fixedMediaType,
  acceptedMediaTypes,
  expectedRevision,
  submitLabel,
  signal,
  headingId,
  onPendingChange,
  onCancel,
  onWritten,
}: {
  projectId: string;
  suggested?: ShortcutDefinition;
  fixedIdentity?: { namespace: string; name: string };
  fixedNamespace?: string;
  fixedMediaType?: string;
  acceptedMediaTypes?: readonly string[];
  expectedRevision?: string;
  submitLabel?: string;
  signal?: AbortSignal;
  /** Labels the form by an outer (dialog) heading instead of its own. */
  headingId?: string;
  onPendingChange?: (pending: boolean) => void;
  /** Renders a Cancel button next to the submit button. */
  onCancel?: () => void;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const ownHeading = useId();
  const formHeading = headingId ?? ownHeading;
  const [namespace, setNamespace] = useState(
    fixedIdentity?.namespace ??
      fixedNamespace ??
      suggested?.namespace ??
      "artifacts",
  );
  const [name, setName] = useState(fixedIdentity?.name ?? "");
  const [mediaType, setMediaType] = useState(
    fixedMediaType ?? suggested?.mediaType ?? "application/octet-stream",
  );
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: (request: ProjectArtifactWriteRequest) =>
      writeProjectArtifact(api, request),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.artifacts.all(projectId),
      });
      onWritten(result);
    },
    onError: async () => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.artifacts.all(projectId),
      });
    },
  });
  useEffect(() => {
    onPendingChange?.(mutation.isPending);
  }, [mutation.isPending, onPendingChange]);

  function selectFile(selected: File | undefined): void {
    const next = selected ?? null;
    setFile(next);
    mutation.reset();
    setValidationError(null);
    if (next === null) {
      return;
    }
    if (fixedIdentity === undefined && name === "") {
      setName(artifactFileStem(next.name));
    }
    if (fixedMediaType === undefined) {
      setMediaType(inferredArtifactMediaType(next, mediaType));
    }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    mutation.reset();
    const effectiveNamespace =
      fixedIdentity?.namespace ?? fixedNamespace ?? namespace.trim();
    const effectiveName = fixedIdentity?.name ?? name.trim();
    const effectiveMediaType = fixedMediaType ?? mediaType;
    if (
      !ARTIFACT_NAME_PATTERN.test(effectiveNamespace) ||
      !ARTIFACT_NAME_PATTERN.test(effectiveName)
    ) {
      setValidationError(
        "Namespace and name must use 1–128 letters, digits, dot, dash, or underscore.",
      );
      return;
    }
    if (!MEDIA_TYPE_PATTERN.test(effectiveMediaType)) {
      setValidationError(
        "Media type must be a lowercase type/subtype without parameters.",
      );
      return;
    }
    if (
      acceptedMediaTypes !== undefined &&
      !acceptedMediaTypes.includes("*/*") &&
      !acceptedMediaTypes.includes(effectiveMediaType)
    ) {
      setValidationError(
        `Media type must be one accepted by this input: ${acceptedMediaTypes.join(", ")}.`,
      );
      return;
    }
    if (file === null) {
      setValidationError("Choose or drop one local file.");
      return;
    }
    if (file.size > MAXIMUM_ARTIFACT_BYTES) {
      setValidationError("Artifact exceeds the 64 MiB upload limit.");
      return;
    }
    mutation.mutate({
      projectId,
      namespace: effectiveNamespace,
      name: effectiveName,
      mediaType: effectiveMediaType,
      payload: file,
      ...(expectedRevision === undefined ? {} : { expectedRevision }),
      ...(signal === undefined ? {} : { signal }),
    });
  }

  const update = expectedRevision !== undefined;
  return (
    <form
      className="project-artifact-form"
      onSubmit={submit}
      aria-labelledby={formHeading}
    >
      {headingId === undefined ? (
        <div className="section-heading">
          <div>
            <p className="eyebrow">
              {update ? "New version" : "Project artifact"}
            </p>
            <h3 id={formHeading}>
              {update
                ? "Upload a new version"
                : `Add ${suggested?.label ?? "Artifact"}`}
            </h3>
          </div>
          {update ? (
            <code>If-Match: &quot;{expectedRevision}&quot;</code>
          ) : null}
        </div>
      ) : null}

      <ArtifactFileDrop file={file} onSelect={selectFile} />

      {acceptedMediaTypes === undefined ? null : (
        <small>Accepted by this input: {acceptedMediaTypes.join(", ")}</small>
      )}

      <div className="form-grid project-artifact-fields">
        <label>
          Namespace
          <input
            name="namespace"
            required
            maxLength={128}
            disabled={
              fixedIdentity !== undefined || fixedNamespace !== undefined
            }
            value={fixedIdentity?.namespace ?? fixedNamespace ?? namespace}
            onChange={(event) => setNamespace(event.target.value)}
          />
        </label>
        <label>
          Name
          <input
            name="name"
            required
            maxLength={128}
            disabled={fixedIdentity !== undefined}
            value={fixedIdentity?.name ?? name}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
        <ArtifactMediaTypeField
          disabled={fixedMediaType !== undefined}
          value={fixedMediaType ?? mediaType}
          onChange={setMediaType}
        />
      </div>
      {validationError === null ? null : (
        <p className="form-error" role="alert">
          {validationError}
        </p>
      )}
      {mutation.error === null ? null : (
        <ErrorNotice error={mutation.error} reconcileWrite />
      )}
      <div
        className={
          onCancel === undefined ? undefined : "project-dialog-actions"
        }
      >
        {onCancel === undefined ? null : (
          <button
            type="button"
            className="secondary-button"
            disabled={mutation.isPending}
            onClick={onCancel}
          >
            Cancel
          </button>
        )}
        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending
            ? "Uploading…"
            : (submitLabel ??
              (update ? "Upload new version" : "Create binding"))}
        </button>
      </div>
    </form>
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
      <ProjectArtifactWriteForm
        projectId={projectId}
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
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
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
      {query.isPending ? (
        <p className="loading-copy" role="status">
          Loading Project Artifacts…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="compact-empty">
          <strong>No Artifact bindings in this workspace.</strong>
        </div>
      ) : (
        <div className="table-scroll">
          <table className="responsive-table">
            <thead>
              <tr>
                <th>Binding</th>
                <th>Current revision</th>
                <th>Media type</th>
                <th>Size</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((item) => (
                <tr key={`${item.artifact.namespace}/${item.artifact.name}`}>
                  <td data-label="Binding">
                    <ContextLink
                      returnLabel="Project Artifacts"
                      returnHash="#project-artifacts"
                      to={`${detailRoot}/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(item.artifact.namespace)}/${encodeURIComponent(item.artifact.name)}`}
                    >
                      {item.artifact.namespace}/{item.artifact.name}
                    </ContextLink>
                  </td>
                  <td data-label="Current revision">
                    <code>{item.artifact.revision}</code>
                  </td>
                  <td data-label="Media type">{item.mediaType}</td>
                  <td data-label="Size">{formatBytes(item.size)}</td>
                  <td data-label="Created">
                    {formatTimestamp(item.createdAt)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <CursorControls
        label="Project Artifact pages"
        canGoBack={cursors.length > 1}
        {...(query.data?.page.hasMore === true &&
        query.data.page.nextCursor !== undefined
          ? { nextCursor: query.data.page.nextCursor }
          : {})}
        onBack={() =>
          setCursors((current) =>
            current.slice(0, Math.max(1, current.length - 1)),
          )
        }
        onNext={(next) => setCursors((current) => [...current, next])}
      />
    </ProjectRegion>
  );
}
