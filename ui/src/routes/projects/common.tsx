import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  type DragEvent,
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
  writeProjectArtifact,
  type ProjectArtifactWriteRequest,
} from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice, formatBytes } from "../artifacts/common";
import {
  PROJECT_ARTIFACT_SHORTCUTS,
  type ProjectArtifactShortcut,
  type ShortcutDefinition,
} from "./shortcuts";

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
}: {
  onSelect: (shortcut: ShortcutDefinition) => void;
}) {
  return (
    <div className="project-shortcut-grid" aria-label="Artifact shortcuts">
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

function fileStem(filename: string): string {
  const basename = filename.replace(/^.*[\\/]/, "");
  const extension = basename.lastIndexOf(".");
  const stem = extension > 0 ? basename.slice(0, extension) : basename;
  const normalized = stem
    .normalize("NFKD")
    .replace(/[^A-Za-z0-9_.-]+/g, "-")
    .replace(/^[^A-Za-z0-9]+/, "")
    .slice(0, 128);
  return normalized === "" ? "artifact" : normalized;
}

function inferredMediaType(file: File, fallback: string): string {
  if (MEDIA_TYPE_PATTERN.test(file.type)) {
    return file.type;
  }
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".yaml") || lower.endsWith(".yml")) {
    return "application/yaml";
  }
  if (lower.endsWith(".json")) {
    return "application/json";
  }
  if (lower.endsWith(".md")) {
    return "text/markdown";
  }
  if (lower.endsWith(".diff") || lower.endsWith(".patch")) {
    return "text/x-diff";
  }
  if (lower.endsWith(".c4") || lower.endsWith(".likec4")) {
    return "text/vnd.likec4";
  }
  if (lower.endsWith(".zip")) {
    return "application/zip";
  }
  return fallback;
}

export function ProjectArtifactWriteForm({
  projectId,
  suggested,
  fixedIdentity,
  expectedRevision,
  submitLabel,
  onWritten,
}: {
  projectId: string;
  suggested?: ShortcutDefinition;
  fixedIdentity?: { namespace: string; name: string };
  expectedRevision?: string;
  submitLabel?: string;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const formHeading = useId();
  const fileInput = useRef<HTMLInputElement>(null);
  const [namespace, setNamespace] = useState(
    fixedIdentity?.namespace ?? suggested?.namespace ?? "artifacts",
  );
  const [name, setName] = useState(fixedIdentity?.name ?? "");
  const [mediaType, setMediaType] = useState(
    suggested?.mediaType ?? "application/octet-stream",
  );
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
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

  function selectFile(selected: File | undefined): void {
    const next = selected ?? null;
    setFile(next);
    mutation.reset();
    setValidationError(null);
    if (next === null) {
      return;
    }
    if (fixedIdentity === undefined && name === "") {
      setName(fileStem(next.name));
    }
    setMediaType(inferredMediaType(next, mediaType));
  }

  function receiveDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files[0]);
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    mutation.reset();
    const effectiveNamespace = fixedIdentity?.namespace ?? namespace.trim();
    const effectiveName = fixedIdentity?.name ?? name.trim();
    if (
      !ARTIFACT_NAME_PATTERN.test(effectiveNamespace) ||
      !ARTIFACT_NAME_PATTERN.test(effectiveName)
    ) {
      setValidationError(
        "Namespace and name must use 1–128 letters, digits, dot, dash, or underscore.",
      );
      return;
    }
    if (!MEDIA_TYPE_PATTERN.test(mediaType)) {
      setValidationError(
        "Media type must be a lowercase type/subtype without parameters.",
      );
      return;
    }
    if (file === null) {
      setValidationError("Choose or drop one local file.");
      return;
    }
    if (file.size > MAXIMUM_ARTIFACT_BYTES) {
      setValidationError("Artifact exceeds the 16 MiB upload limit.");
      return;
    }
    mutation.mutate({
      projectId,
      namespace: effectiveNamespace,
      name: effectiveName,
      mediaType,
      payload: file,
      ...(expectedRevision === undefined ? {} : { expectedRevision }),
    });
  }

  const update = expectedRevision !== undefined;
  return (
    <form className="project-artifact-form" onSubmit={submit}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">
            {update ? "Exact CAS update" : "ProjectScope"}
          </p>
          <h3 id={formHeading}>
            {update
              ? "Upload a new version"
              : `Add ${suggested?.label ?? "Artifact"}`}
          </h3>
        </div>
        {update ? <code>If-Match: &quot;{expectedRevision}&quot;</code> : null}
      </div>

      <div
        className={`project-file-drop ${dragging ? "is-dragging" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (
            !event.currentTarget.contains(event.relatedTarget as Node | null)
          ) {
            setDragging(false);
          }
        }}
        onDrop={receiveDrop}
      >
        <input
          ref={fileInput}
          className="visually-hidden"
          name="file"
          type="file"
          aria-labelledby={`${formHeading}-file-copy`}
          onChange={(event) => selectFile(event.target.files?.[0])}
        />
        <span className="project-file-drop-mark" aria-hidden="true">
          ↑
        </span>
        <strong id={`${formHeading}-file-copy`}>
          {file === null ? "Drop a file here" : file.name}
        </strong>
        <small>
          {file === null
            ? "or choose one local file, up to 16 MiB"
            : `${formatBytes(file.size)} · ready to upload`}
        </small>
        <button
          className="secondary-button"
          type="button"
          onClick={() => fileInput.current?.click()}
        >
          {file === null ? "Choose file" : "Choose another file"}
        </button>
      </div>

      <div className="form-grid project-artifact-fields">
        <label>
          Namespace
          <input
            name="namespace"
            required
            maxLength={128}
            disabled={fixedIdentity !== undefined}
            value={fixedIdentity?.namespace ?? namespace}
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
        <label>
          Media type
          <input
            name="mediaType"
            required
            value={mediaType}
            onChange={(event) => setMediaType(event.target.value)}
          />
        </label>
      </div>
      {validationError === null ? null : (
        <p className="form-error" role="alert">
          {validationError}
        </p>
      )}
      {mutation.error === null ? null : (
        <ErrorNotice error={mutation.error} reconcileWrite />
      )}
      <button type="submit" disabled={mutation.isPending}>
        {mutation.isPending
          ? "Uploading…"
          : (submitLabel ??
            (update ? "Upload exact update" : "Create binding"))}
      </button>
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
  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <div className="project-dialog-backdrop" role="presentation">
      <section
        className="project-dialog panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby={heading}
      >
        <div className="project-dialog-heading">
          <div>
            <p className="eyebrow">Artifact shortcut</p>
            <h2 id={heading}>{shortcut.label}</h2>
          </div>
          <button
            className="project-dialog-close"
            type="button"
            aria-label="Close upload dialog"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <p className="muted-copy">
          The category only suggests editable Artifact metadata. The Server
          stores the same arbitrary ProjectScope binding as every other upload.
        </p>
        <ProjectArtifactWriteForm
          projectId={projectId}
          suggested={shortcut}
          onWritten={onWritten}
        />
      </section>
    </div>
  );
}

export function ProjectRegion({
  eyebrow,
  title,
  action,
  children,
  id,
}: {
  eyebrow: string;
  title: string;
  action?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section className="panel project-region" id={id}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
