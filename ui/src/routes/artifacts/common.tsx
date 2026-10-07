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
import { Link } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  MAXIMUM_ARTIFACT_BYTES,
  MEDIA_TYPE_PATTERN,
  type ArtifactWriteRequest,
  type ArtifactWriteResponse,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { artifactScopeKeys } from "../../api/query-keys";
import {
  writeScopeArtifact,
  type WritableArtifactScope,
} from "../../api/scoped-artifacts";
import { ErrorNotice } from "../../app/error-notice";
import { formatBytes } from "../../app/format";
import { artifactFileStem, inferredArtifactMediaType } from "./artifact-file";
import { MaterialIcon } from "./icons";
import { ArtifactMediaTypeField } from "./media-type-field";
import "./materials.css";

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
        <MaterialIcon name="upload" size={20} />
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
        className="ui-btn"
        data-size="sm"
        type="button"
        onClick={() => fileInput.current?.click()}
      >
        {file === null ? "Choose file" : "Choose another file"}
      </button>
    </div>
  );
}

/** One uploaded revision for a User or Project Artifact binding. */
export function ArtifactWriteForm({
  scope = { kind: "user" },
  suggested,
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
  onWriteError,
  writeErrorHint,
  onWritten,
}: {
  scope?: WritableArtifactScope;
  /** Suggested namespace, media type and heading label for a new binding. */
  suggested?: { label: string; namespace: string; mediaType: string };
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
  /**
   * Renders a Cancel button next to the submit button. With
   * `startOperation` it stays enabled while an upload runs ("Cancel
   * upload"), and the caller aborts that upload's signal.
   */
  onCancel?: () => void;
  /**
   * Receives a failed write in place of the form's own notice, so a view
   * whose reconcile remounts or replaces the form keeps it visible.
   */
  onWriteError?: (error: unknown) => void;
  /** Extra guidance under the form's own notice of a failed write. */
  writeErrorHint?: (error: unknown) => ReactNode;
  onWritten: (result: ArtifactWriteResponse) => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const ownHeadingId = useId();
  const formId = headingId ?? ownHeadingId;
  const project = scope.kind === "project";
  const keys = artifactScopeKeys(scope);
  const [namespace, setNamespace] = useState(
    fixedIdentity?.namespace ??
      fixedNamespace ??
      suggested?.namespace ??
      (project ? "artifacts" : "projects"),
  );
  const [name, setName] = useState(fixedIdentity?.name ?? "");
  const [mediaType, setMediaType] = useState(
    fixedMediaType ??
      initialMediaType ??
      suggested?.mediaType ??
      "application/octet-stream",
  );
  const preserveMediaType = useRef(
    initialMediaType !== undefined || suggested?.mediaType !== undefined,
  );
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [excludedNamespaceError, setExcludedNamespaceError] = useState(false);
  const [inputRevision, setInputRevision] = useState(0);
  const mutation = useMutation({
    mutationFn: (request: ArtifactWriteRequest) =>
      writeScopeArtifact(api, scope, request),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: keys.all });
      onWritten(result);
      setFile(null);
      setInputRevision((value) => value + 1);
      if (fixedIdentity === undefined) {
        setName("");
      }
    },
    onError: async (error) => {
      onWriteError?.(error);
      // A conflict or lost response is ambiguous by design. Reconcile every
      // active Artifact view with Server state, but never retry the unsafe PUT.
      await queryClient.invalidateQueries({ queryKey: keys.all });
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
        `The file exceeds the ${maximumBytes / (1024 * 1024)} MiB upload limit.`,
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
  // Only a caller that owns the upload's signal can cancel it mid-flight.
  const cancelsUpload = mutation.isPending && startOperation !== undefined;
  return (
    <form
      className={`${project ? "project-artifact-form" : "artifact-form"} materials-write-form`}
      onSubmit={submit}
      aria-labelledby={formId}
    >
      {headingId === undefined ? (
        <div className="materials-form-head">
          <p className="materials-kicker">
            {update ? "New version" : project ? "Material" : "File"}
          </p>
          <h3 id={formId}>
            {update
              ? "Upload a new version"
              : suggested !== undefined
                ? `Add ${suggested.label}`
                : project
                  ? "Upload material"
                  : "Upload file"}
          </h3>
        </div>
      ) : null}
      <ArtifactFileDrop
        file={file}
        inputRevision={inputRevision}
        maximumBytes={maximumBytes}
        onSelect={selectFile}
      />
      {acceptedMediaTypes === undefined ? null : (
        <small className="materials-quiet">
          Accepted by this input: {acceptedMediaTypes.join(", ")}
        </small>
      )}
      <div
        className={
          project
            ? "form-grid project-artifact-fields materials-fields"
            : "form-grid artifact-fields materials-fields"
        }
      >
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
      {mutation.error === null || onWriteError !== undefined ? null : (
        <>
          <ErrorNotice error={mutation.error} reconcileWrite />
          {writeErrorHint?.(mutation.error)}
        </>
      )}
      <div className="materials-form-actions">
        {onCancel === undefined ? null : (
          <button
            type="button"
            className="ui-btn"
            disabled={mutation.isPending && !cancelsUpload}
            onClick={onCancel}
          >
            {cancelsUpload ? "Cancel upload" : "Cancel"}
          </button>
        )}
        <button
          type="submit"
          className="ui-btn"
          data-variant="primary"
          disabled={mutation.isPending}
        >
          {mutation.isPending
            ? "Uploading…"
            : (submitLabel ??
              (update ? "Upload new version" : "Create binding"))}
        </button>
      </div>
    </form>
  );
}
