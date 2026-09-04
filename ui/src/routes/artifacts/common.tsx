import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useId, useState } from "react";

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

export function formatBytes(size: number): string {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KiB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}

export function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString();
}

export function ErrorNotice({
  error,
  reconcileWrite = false,
}: {
  error: unknown;
  reconcileWrite?: boolean;
}) {
  const message =
    error instanceof Error ? error.message : "Artifact request failed";
  const requestId =
    error instanceof PublicAPIError ? error.requestId : undefined;
  return (
    <div className="notice notice-error" role="alert">
      <strong>{message}</strong>
      {requestId === undefined ? null : <small>Request {requestId}</small>}
      {reconcileWrite &&
      error instanceof PublicAPIError &&
      error.code === "conflict" ? (
        <p>
          The binding changed. Refresh its current revision before choosing an
          explicit new update; this upload was not retried.
        </p>
      ) : reconcileWrite &&
        error instanceof PublicAPIError &&
        error.status === 0 ? (
        <p>
          The Server response was not received, so the upload may have
          succeeded. Refresh the authoritative binding before deciding whether
          to submit another PUT.
        </p>
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
}: {
  label: string;
  canGoBack: boolean;
  nextCursor?: string;
  onBack: () => void;
  onNext: (cursor: string) => void;
}) {
  if (!canGoBack && nextCursor === undefined) {
    return null;
  }
  return (
    <nav className="pagination" aria-label={label}>
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
  expectedRevision,
  onWritten,
}: {
  fixedIdentity?: { namespace: string; name: string };
  fixedNamespace?: string;
  fixedMediaType?: string;
  expectedRevision?: string;
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

  function selectFile(selected: File | undefined): void {
    setFile(selected ?? null);
    if (
      fixedMediaType === undefined &&
      selected?.type !== undefined &&
      selected.type !== "" &&
      MEDIA_TYPE_PATTERN.test(selected.type)
    ) {
      setMediaType(selected.type);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    mutation.reset();
    const effectiveNamespace =
      fixedIdentity?.namespace ?? fixedNamespace ?? namespace;
    const effectiveName = fixedIdentity?.name ?? name;
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
    if (file === null) {
      setValidationError("Choose one local file to upload.");
      return;
    }
    if (file.size > MAXIMUM_ARTIFACT_BYTES) {
      setValidationError("Artifact exceeds the 16 MiB upload limit.");
      return;
    }
    mutation.mutate({
      namespace: effectiveNamespace,
      name: effectiveName,
      mediaType: effectiveMediaType,
      payload: file,
      ...(expectedRevision === undefined ? {} : { expectedRevision }),
    });
  }

  const update = expectedRevision !== undefined;
  return (
    <form className="artifact-form" onSubmit={submit} aria-labelledby={formId}>
      <div className="section-heading">
        <div>
          <p className="eyebrow">
            {update ? "Exact CAS write" : "New binding"}
          </p>
          <h3 id={formId}>
            {update ? "Upload a new version" : "Upload Artifact"}
          </h3>
        </div>
        {update ? <code>If-Match: &quot;{expectedRevision}&quot;</code> : null}
      </div>
      <div className="form-grid">
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
        <label>
          Media type
          <input
            name="mediaType"
            required
            disabled={fixedMediaType !== undefined}
            value={fixedMediaType ?? mediaType}
            onChange={(event) => setMediaType(event.target.value)}
          />
        </label>
        <label>
          Local file (maximum 16 MiB)
          <input
            key={inputRevision}
            name="file"
            type="file"
            onChange={(event) => selectFile(event.target.files?.[0])}
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
          : update
            ? "Upload exact update"
            : "Create binding"}
      </button>
    </form>
  );
}
