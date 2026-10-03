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
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice } from "../../app/error-notice";
import { formatBytes } from "../../app/format";
import { artifactFileStem, inferredArtifactMediaType } from "./artifact-file";
import { ArtifactMediaTypeField } from "./media-type-field";

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

export function ArtifactWriteForm({
  fixedIdentity,
  fixedNamespace,
  fixedMediaType,
  initialMediaType,
  excludedNamespace,
  expectedRevision,
  acceptedMediaTypes,
  maximumBytes = MAXIMUM_ARTIFACT_BYTES,
  startOperation,
  submitLabel,
  headingId,
  onPendingChange,
  onCancel,
  onWritten,
}: {
  fixedIdentity?: { namespace: string; name: string };
  fixedNamespace?: string;
  fixedMediaType?: string;
  initialMediaType?: string;
  excludedNamespace?: {
    namespace: string;
    destination: string;
    label: string;
  };
  expectedRevision?: string;
  acceptedMediaTypes?: readonly string[];
  maximumBytes?: number;
  /** Starts an upload and returns the signal that aborts it. */
  startOperation?: () => AbortSignal;
  submitLabel?: string;
  /** Labels the form by an outer (dialog) heading instead of its own. */
  headingId?: string;
  onPendingChange?: (pending: boolean) => void;
  /** Renders a Cancel button next to the submit button. */
  onCancel?: () => void;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const ownHeadingId = useId();
  const formId = headingId ?? ownHeadingId;
  const [namespace, setNamespace] = useState(
    fixedIdentity?.namespace ?? fixedNamespace ?? "projects",
  );
  const [name, setName] = useState(fixedIdentity?.name ?? "");
  const [mediaType, setMediaType] = useState(
    fixedMediaType ?? initialMediaType ?? "application/octet-stream",
  );
  const preserveMediaType = useRef(initialMediaType !== undefined);
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
      setMediaType(
        inferredArtifactMediaType(next, mediaType, preserveMediaType.current),
      );
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
    if (file.size > maximumBytes) {
      setValidationError(
        `Artifact exceeds the ${maximumBytes / (1024 * 1024)} MiB upload limit.`,
      );
      return;
    }
    mutation.mutate({
      namespace: effectiveNamespace,
      name: effectiveName,
      mediaType: effectiveMediaType,
      payload: file,
      ...(expectedRevision === undefined ? {} : { expectedRevision }),
      ...(startOperation === undefined ? {} : { signal: startOperation() }),
    });
  }

  const update = expectedRevision !== undefined;
  return (
    <form className="artifact-form" onSubmit={submit} aria-labelledby={formId}>
      {headingId === undefined ? (
        <div className="section-heading">
          <div>
            <p className="eyebrow">{update ? "New version" : "New Artifact"}</p>
            <h3 id={formId}>
              {update ? "Upload a new version" : "Upload Artifact"}
            </h3>
          </div>
          {update ? (
            <code>If-Match: &quot;{expectedRevision}&quot;</code>
          ) : null}
        </div>
      ) : null}
      <ArtifactFileDrop
        file={file}
        inputRevision={inputRevision}
        maximumBytes={maximumBytes}
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
          onChange={(value) => {
            preserveMediaType.current = true;
            setMediaType(value);
          }}
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
