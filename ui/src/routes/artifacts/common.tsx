import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  type DragEvent,
  type FormEvent,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { Link } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  MAXIMUM_ARTIFACT_BYTES,
  MEDIA_TYPE_PATTERN,
  type ArtifactWriteRequest,
  type ArtifactWriteResponse,
  writeArtifact,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { queryKeys } from "../../api/query-keys";
import { artifactFileStem, inferredArtifactMediaType } from "./artifact-file";
import { ArtifactMediaTypeField } from "./media-type-field";

export function formatBytes(size: number): string {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KiB`;
  }
  if (size >= 1024 ** 4) return `${(size / 1024 ** 4).toFixed(1)} TiB`;
  if (size >= 1024 ** 3) return `${(size / 1024 ** 3).toFixed(1)} GiB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}

export function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString();
}

export function ArtifactFileDrop({
  file,
  inputRevision,
  maximumBytes = MAXIMUM_ARTIFACT_BYTES,
  onSelect,
}: {
  file: File | null;
  inputRevision?: number;
  maximumBytes?: number;
  onSelect: (file: File | undefined) => void;
}) {
  const copyId = useId();
  const fileInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function receiveDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setDragging(false);
    onSelect(event.dataTransfer.files[0]);
  }

  return (
    <div
      className={
        dragging ? "artifact-file-drop is-dragging" : "artifact-file-drop"
      }
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setDragging(false);
        }
      }}
      onDrop={receiveDrop}
    >
      <input
        key={inputRevision}
        ref={fileInput}
        className="visually-hidden"
        name="file"
        type="file"
        aria-labelledby={copyId}
        onChange={(event) => onSelect(event.target.files?.[0])}
      />
      <span className="artifact-file-drop-mark" aria-hidden="true">
        ↑
      </span>
      <strong id={copyId}>
        {file === null ? "Drop a file here" : file.name}
      </strong>
      <small>
        {file === null
          ? `or choose one local file, up to ${maximumBytes / (1024 * 1024)} MiB`
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
  );
}

export function ErrorNotice({
  error,
  reconcileWrite = false,
  context,
  onRetry,
  retryLabel = "Try again",
  retryPending = false,
}: {
  error: unknown;
  reconcileWrite?: boolean;
  context?: string;
  onRetry?: () => void;
  retryLabel?: string;
  retryPending?: boolean;
}) {
  const message = error instanceof Error ? error.message : "Request failed";
  const requestId =
    error instanceof PublicAPIError ? error.requestId : undefined;
  return (
    <div className="notice notice-error" role="alert">
      <strong>{context ?? message}</strong>
      {context === undefined ? null : <p>{message}</p>}
      {reconcileWrite &&
      error instanceof PublicAPIError &&
      error.code === "conflict" ? (
        <p>
          The record or binding changed. Refresh its current revision before
          choosing an explicit new update; this change was not retried.
        </p>
      ) : reconcileWrite &&
        error instanceof PublicAPIError &&
        error.status === 0 ? (
        <p>
          The Server response was not received, so the change may have been
          applied. Refresh the current state before deciding whether to submit
          it again.
        </p>
      ) : null}
      {onRetry === undefined || reconcileWrite ? null : (
        <button
          className="secondary-button"
          type="button"
          disabled={retryPending}
          onClick={onRetry}
        >
          {retryPending ? "Loading…" : retryLabel}
        </button>
      )}
      {error instanceof PublicAPIError ? (
        <details className="error-details">
          <summary>Request details</summary>
          <small>
            Code {error.code} · Status {error.status}
          </small>
          {requestId === undefined ? null : <small>Request {requestId}</small>}
        </details>
      ) : null}
    </div>
  );
}

export function CursorControls({
  label,
  canGoBack,
  nextCursor,
  onBack,
  onNext,
  onFirst,
}: {
  label: string;
  canGoBack: boolean;
  nextCursor?: string;
  onBack: () => void;
  onNext: (cursor: string) => void;
  onFirst?: () => void;
}) {
  if (!canGoBack && nextCursor === undefined && onFirst === undefined) {
    return null;
  }
  return (
    <nav className="pagination" aria-label={label}>
      {onFirst === undefined ? null : (
        <button className="secondary-button" type="button" onClick={onFirst}>
          First page
        </button>
      )}
      <button
        className="secondary-button"
        type="button"
        disabled={!canGoBack}
        onClick={onBack}
      >
        Previous
      </button>
      <button
        className="secondary-button"
        type="button"
        disabled={nextCursor === undefined}
        onClick={() => {
          if (nextCursor !== undefined) {
            onNext(nextCursor);
          }
        }}
      >
        Next
      </button>
    </nav>
  );
}

export function ArtifactWriteForm({
  fixedIdentity,
  fixedNamespace,
  fixedMediaType,
  excludedNamespace,
  expectedRevision,
  acceptedMediaTypes,
  signal,
  submitLabel,
  onPendingChange,
  onWritten,
}: {
  fixedIdentity?: { namespace: string; name: string };
  fixedNamespace?: string;
  fixedMediaType?: string;
  excludedNamespace?: {
    namespace: string;
    destination: string;
    label: string;
  };
  expectedRevision?: string;
  acceptedMediaTypes?: readonly string[];
  signal?: AbortSignal;
  submitLabel?: string;
  onPendingChange?: (pending: boolean) => void;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const formId = useId();
  const [namespace, setNamespace] = useState(
    fixedIdentity?.namespace ?? fixedNamespace ?? "projects",
  );
  const [name, setName] = useState(fixedIdentity?.name ?? "");
  const [mediaType, setMediaType] = useState(
    fixedMediaType ?? "application/octet-stream",
  );
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [excludedNamespaceError, setExcludedNamespaceError] = useState(false);
  const [inputRevision, setInputRevision] = useState(0);
  const mutation = useMutation({
    mutationFn: (request: ArtifactWriteRequest) => writeArtifact(api, request),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.artifacts.all,
      });
      onWritten(result);
      setFile(null);
      setInputRevision((value) => value + 1);
      if (fixedIdentity === undefined) {
        setName("");
      }
    },
    onError: async () => {
      // A conflict or lost response is ambiguous by design. Reconcile every
      // active Artifact view with Server state, but never retry the unsafe PUT.
      await queryClient.invalidateQueries({
        queryKey: queryKeys.artifacts.all,
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
    setExcludedNamespaceError(false);
    if (next === null) {
      return;
    }
    if (fixedIdentity === undefined && name.trim() === "") {
      setName(artifactFileStem(next.name));
    }
    if (fixedMediaType === undefined) {
      setMediaType(inferredArtifactMediaType(next, mediaType));
    }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    setExcludedNamespaceError(false);
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
    if (effectiveNamespace === excludedNamespace?.namespace) {
      setExcludedNamespaceError(true);
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
      setValidationError("Choose one local file to upload.");
      return;
    }
    if (file.size > MAXIMUM_ARTIFACT_BYTES) {
      setValidationError("Artifact exceeds the 64 MiB upload limit.");
      return;
    }
    mutation.mutate({
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
    <form className="artifact-form" onSubmit={submit} aria-labelledby={formId}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">{update ? "New version" : "New Artifact"}</p>
          <h3 id={formId}>
            {update ? "Upload a new version" : "Upload Artifact"}
          </h3>
        </div>
        {update ? <code>If-Match: &quot;{expectedRevision}&quot;</code> : null}
      </div>
      <ArtifactFileDrop
        file={file}
        inputRevision={inputRevision}
        onSelect={selectFile}
      />
      {acceptedMediaTypes === undefined ? null : (
        <small>Accepted by this input: {acceptedMediaTypes.join(", ")}</small>
      )}
      <div className="form-grid artifact-fields">
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
      {!excludedNamespaceError || excludedNamespace === undefined ? null : (
        <p className="form-error" role="alert">
          The {excludedNamespace.namespace} namespace is managed in the{" "}
          <Link to={excludedNamespace.destination}>
            {excludedNamespace.label}
          </Link>{" "}
          tab.
        </p>
      )}
      {mutation.error === null ? null : (
        <ErrorNotice error={mutation.error} reconcileWrite />
      )}
      <button type="submit" disabled={mutation.isPending}>
        {mutation.isPending
          ? "Uploading…"
          : (submitLabel ?? (update ? "Upload new version" : "Create binding"))}
      </button>
    </form>
  );
}
