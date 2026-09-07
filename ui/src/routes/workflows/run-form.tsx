import {
  GitImportDialog,
  GitSourceDetails,
} from "../artifacts/git-import-dialog";
import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import {
  type FormEvent,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useNavigate } from "react-router";

import { listArtifacts, type ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { listProjectArtifacts } from "../../api/project-artifacts";
import {
  listRuntimeLabels,
  type RuntimeLabelBinding,
} from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import {
  RUN_METADATA_LABEL_LIMIT,
  type RunMetadataLabelDraft,
} from "../../api/run-metadata-labels";
import {
  createProjectRun,
  createRun,
  listConfigurations,
  listCredentials,
  type ConfigurationResource,
  type CredentialResource,
  type CreateRunRequest,
  type WorkflowResource,
} from "../../api/workflows";
import { Dialog } from "../../app/dialog";
import {
  initialRunDraftState,
  type RunDraftEntry,
  type RunDraftIdentity,
  type RunDraftState,
  type RunDraftSummary,
  type RunDraftMemoryStore,
} from "../../run-drafts/memory";
import { useRunDraftStore } from "../../run-drafts/context";
import {
  artifactAccepts,
  artifactOptionKey,
  NO_CREDENTIAL_OVERRIDE,
  validateRunDraft,
  type ConsumerOverrideDraft,
  type ExecutionOverrideDraft,
} from "../../run-drafts/validation";
import { ErrorNotice, formatBytes, formatTimestamp } from "../artifacts/common";
import { GitRepositoryIcon } from "../artifacts/git-repository-icon";
import { RunInputUploadDialog } from "./run-input-upload-dialog";

import "./run-drafts.css";

const INITIAL_CURSOR = null;
const EVAL_METADATA_PRESET = [
  { key: "purpose", value: "eval" },
  { key: "eval.name", value: "" },
  { key: "eval.id", value: "" },
  { key: "eval.leg", value: "" },
] as const;

function nextCursor(page: { page: { hasMore: boolean; nextCursor?: string } }) {
  return page.page.hasMore ? page.page.nextCursor : undefined;
}

function configurationSelector(resource: ConfigurationResource): string {
  return `${resource.ref.name}@${resource.ref.version}`;
}

function artifactLabel(metadata: ArtifactMetadata): string {
  return `${artifactOptionKey(metadata.artifact)} · ${metadata.mediaType} · ${formatBytes(metadata.size)}`;
}

function artifactDetailPath(
  metadata: ArtifactMetadata,
  projectId?: string,
): string {
  const exactPath = `${encodeURIComponent(metadata.artifact.namespace)}/${encodeURIComponent(metadata.artifact.name)}?revision=${encodeURIComponent(metadata.artifact.revision)}`;
  return projectId === undefined
    ? `/artifacts/${exactPath}`
    : `/projects/${encodeURIComponent(projectId)}/artifacts/${exactPath}`;
}

function RuntimeLabelPreview({ binding }: { binding: RuntimeLabelBinding }) {
  return (
    <small className="runtime-label-preview">
      <code>
        {binding.config.name}@{binding.config.version}
      </code>
      <span>binding revision {binding.revision}</span>
      <code title={binding.config.digest}>
        {binding.config.digest.slice(0, 18)}…
      </code>
    </small>
  );
}

function DraftDisclosureSummary({
  title,
  description,
  status,
  active = false,
}: {
  title: string;
  description: string;
  status: string;
  active?: boolean;
}) {
  return (
    <>
      <span className="run-draft-disclosure-title">
        <strong>{title}</strong>
        <small>{description}</small>
      </span>
      <span
        className={`run-draft-disclosure-status ${active ? "is-active" : ""}`}
      >
        {status}
      </span>
    </>
  );
}

function RunMetadataLabelEditor({
  labels,
  labelInput,
  errors,
  onAdd,
  onAddEvalPreset,
  onChange,
  onLabelInputChange,
  onRemove,
}: {
  labels: readonly RunMetadataLabelDraft[];
  labelInput: string;
  errors: Readonly<Record<string, string>>;
  onAdd: (key?: string, value?: string) => void;
  onAddEvalPreset: () => void;
  onChange: (id: string, field: "key" | "value", value: string) => void;
  onLabelInputChange: (value: string) => void;
  onRemove: (id: string) => void;
}) {
  const labelInputRef = useRef<HTMLInputElement>(null);
  const addButtonRef = useRef<HTMLButtonElement>(null);
  const atLimit = labels.length >= RUN_METADATA_LABEL_LIMIT;
  const existingKeys = new Set(labels.map((label) => label.key));
  const missingEvalLabels = EVAL_METADATA_PRESET.filter(
    (label) => !existingKeys.has(label.key),
  ).length;

  function addInput(): void {
    if (labelInput.trim() === "" || atLimit) {
      return;
    }
    const separator = labelInput.indexOf(":");
    onAdd(
      (separator < 0 ? labelInput : labelInput.slice(0, separator)).trim(),
      separator < 0 ? "" : labelInput.slice(separator + 1),
    );
    onLabelInputChange("");
  }

  return (
    <fieldset
      className="run-draft-section run-metadata-label-editor"
      aria-label="Run metadata labels"
    >
      <p className="muted-copy">
        Add labels to find and group Runs. Edit a badge directly or remove it
        with × before starting the Run.
      </p>
      <div className="notice notice-warning run-label-secret-warning">
        <strong>Do not put secrets in labels.</strong>
        <p>
          Keys and values are visible through the Run API, list filters and
          telemetry.
        </p>
      </div>
      <div className="run-label-composer">
        <div className="run-field">
          <label htmlFor="run-label-input">Add a label</label>
          <input
            ref={labelInputRef}
            id="run-label-input"
            placeholder="team:platform or debug"
            autoComplete="off"
            spellCheck={false}
            value={labelInput}
            disabled={atLimit}
            aria-describedby="run-label-input-help"
            onChange={(event) => onLabelInputChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                event.preventDefault();
                addInput();
              }
            }}
            onBlur={(event) => {
              if (event.relatedTarget !== addButtonRef.current) {
                addInput();
              }
            }}
          />
          <small id="run-label-input-help">
            {atLimit
              ? `Limit reached: ${RUN_METADATA_LABEL_LIMIT} labels.`
              : "Type key:value or just key and press Enter. Labels are fixed once the Run starts."}
          </small>
        </div>
        <button
          ref={addButtonRef}
          className="secondary-button"
          type="button"
          disabled={atLimit}
          onClick={() => {
            if (labelInput.trim() === "") {
              onAdd();
            } else {
              addInput();
              labelInputRef.current?.focus();
            }
          }}
        >
          Add metadata label
        </button>
      </div>
      <div className="run-metadata-label-actions">
        <button
          className="secondary-button"
          type="button"
          disabled={
            missingEvalLabels === 0 ||
            labels.length + missingEvalLabels > RUN_METADATA_LABEL_LIMIT
          }
          onClick={onAddEvalPreset}
        >
          Add eval metadata preset
        </button>
        <small>
          {labels.length}/{RUN_METADATA_LABEL_LIMIT} labels
        </small>
      </div>
      {errors.metadataLabels === undefined ? null : (
        <p className="field-error" role="alert">
          {errors.metadataLabels}
        </p>
      )}
      {labels.length === 0 ? (
        <p className="compact-empty">No Run metadata labels.</p>
      ) : (
        <div className="run-label-badges">
          {labels.map((label, index) => {
            const keyError = errors[`metadataLabel:${label.id}:key`];
            const valueError = errors[`metadataLabel:${label.id}:value`];
            const keyErrorID = `metadata-label-${label.id}-key-error`;
            const valueErrorID = `metadata-label-${label.id}-value-error`;
            return (
              <div className="run-label-badge-field" key={label.id}>
                <div
                  className={`run-label-badge${label.value === "" ? " is-key-only" : ""}${keyError !== undefined || valueError !== undefined ? " has-error" : ""}`}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.nativeEvent.isComposing &&
                      event.target instanceof HTMLInputElement
                    ) {
                      event.preventDefault();
                      labelInputRef.current?.focus();
                    }
                  }}
                >
                  <label
                    className="visually-hidden"
                    htmlFor={`metadata-label-${label.id}-key`}
                  >
                    Run metadata label key {index + 1}
                  </label>
                  <input
                    id={`metadata-label-${label.id}-key`}
                    name={`metadata-label-key-${index + 1}`}
                    type="text"
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="key"
                    className="run-label-badge-key"
                    style={{
                      width: `calc(${Math.max(3, Math.min(24, label.key.length))}ch + 0.2rem)`,
                    }}
                    value={label.key}
                    title={label.key}
                    aria-invalid={keyError === undefined ? undefined : true}
                    aria-describedby={
                      keyError === undefined ? undefined : keyErrorID
                    }
                    onChange={(event) =>
                      onChange(label.id, "key", event.target.value)
                    }
                  />
                  <span
                    className="run-label-badge-separator"
                    aria-hidden="true"
                  >
                    :
                  </span>
                  <label
                    className="visually-hidden"
                    htmlFor={`metadata-label-${label.id}-value`}
                  >
                    Run metadata label value {index + 1}
                  </label>
                  <input
                    id={`metadata-label-${label.id}-value`}
                    name={`metadata-label-value-${index + 1}`}
                    type="text"
                    autoComplete="off"
                    spellCheck={false}
                    placeholder="value"
                    className="run-label-badge-value"
                    style={{
                      width: `calc(${Math.max(5, Math.min(28, label.value.length))}ch + 0.2rem)`,
                    }}
                    value={label.value}
                    title={label.value || "Optional value"}
                    aria-invalid={valueError === undefined ? undefined : true}
                    aria-describedby={
                      valueError === undefined ? undefined : valueErrorID
                    }
                    onChange={(event) =>
                      onChange(label.id, "value", event.target.value)
                    }
                  />
                  <button
                    className="run-label-badge-remove"
                    type="button"
                    aria-label={`Remove Run metadata label ${index + 1}`}
                    onClick={() => onRemove(label.id)}
                  >
                    <span aria-hidden="true">×</span>
                  </button>
                </div>
                {keyError === undefined ? null : (
                  <p className="field-error" id={keyErrorID} role="alert">
                    {keyError}
                  </p>
                )}
                {valueError === undefined ? null : (
                  <p className="field-error" id={valueErrorID} role="alert">
                    {valueError}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </fieldset>
  );
}

function updateRecord(
  source: Record<string, string>,
  name: string,
  value: string,
): Record<string, string> {
  if (value === "") {
    const result = { ...source };
    delete result[name];
    return result;
  }
  return { ...source, [name]: value };
}

function ConsumerOverrides({
  role,
  value,
  modelPolicies,
  gateways,
  credentials,
  disabled = false,
  onChange,
}: {
  role: "Planner" | "Workers";
  value: ConsumerOverrideDraft;
  modelPolicies: ConfigurationResource[];
  gateways: ConfigurationResource[];
  credentials: CredentialResource[];
  disabled?: boolean;
  onChange: (field: keyof ConsumerOverrideDraft, value: string) => void;
}) {
  const selectedModelKnown =
    value.modelPolicy === "" ||
    modelPolicies.some(
      (resource) => configurationSelector(resource) === value.modelPolicy,
    );
  const selectedGatewayKnown =
    value.llmGateway === "" ||
    gateways.some(
      (resource) => configurationSelector(resource) === value.llmGateway,
    );
  const selectedCredentialKnown =
    value.credential === "" ||
    value.credential === NO_CREDENTIAL_OVERRIDE ||
    credentials.some(
      (credential) => credential.credentialId === value.credential,
    );
  return (
    <fieldset className="override-consumer" disabled={disabled}>
      <legend>{role}</legend>
      {disabled ? (
        <p className="muted-copy">
          This Workflow has no model-backed Planner selection to override.
        </p>
      ) : null}
      <label>
        Model policy
        <select
          value={value.modelPolicy}
          onChange={(event) => onChange("modelPolicy", event.target.value)}
        >
          <option value="">Use Workflow default</option>
          {selectedModelKnown ? null : (
            <option value={value.modelPolicy}>
              {value.modelPolicy} · retained selection not in loaded catalog
            </option>
          )}
          {modelPolicies.map((resource) => (
            <option
              key={`${resource.ref.digest}-${configurationSelector(resource)}`}
              value={configurationSelector(resource)}
            >
              {configurationSelector(resource)} · {resource.source}
            </option>
          ))}
        </select>
      </label>
      <label>
        LLM Gateway
        <select
          value={value.llmGateway}
          onChange={(event) => onChange("llmGateway", event.target.value)}
        >
          <option value="">Use Workflow default</option>
          {selectedGatewayKnown ? null : (
            <option value={value.llmGateway}>
              {value.llmGateway} · retained selection not in loaded catalog
            </option>
          )}
          {gateways.map((resource) => (
            <option
              key={`${resource.ref.digest}-${configurationSelector(resource)}`}
              value={configurationSelector(resource)}
            >
              {configurationSelector(resource)} · {resource.source}
            </option>
          ))}
        </select>
      </label>
      <label>
        Credential
        <select
          value={value.credential}
          onChange={(event) => onChange("credential", event.target.value)}
        >
          <option value="">Use Workflow default</option>
          <option value={NO_CREDENTIAL_OVERRIDE}>
            Explicitly use no credential
          </option>
          {selectedCredentialKnown ? null : (
            <option value={value.credential}>
              {value.credential} · retained credential unavailable or not loaded
            </option>
          )}
          {credentials.map((credential) => (
            <option
              key={credential.credentialId}
              value={credential.credentialId}
            >
              {credential.credentialId}
              {credential.label === undefined
                ? ""
                : ` · ${credential.label}`} · {credential.llmGateway.gatewayId}@
              {credential.llmGateway.version}
            </option>
          ))}
        </select>
      </label>
    </fieldset>
  );
}

function RepeatRunReview({
  repeat,
  error,
  onReviewed,
}: {
  repeat: NonNullable<RunDraftState["repeat"]>;
  error?: string;
  onReviewed: (reviewed: boolean) => void;
}) {
  return (
    <section
      className={`run-repeat-review${repeat.reviewed ? " is-reviewed" : " needs-review"}`}
      aria-labelledby="run-repeat-review-title"
    >
      <div className="run-repeat-review-heading">
        <div>
          <p className="eyebrow">Configure another Run</p>
          <h4 id="run-repeat-review-title">Review retained request values</h4>
        </div>
        <Link to={`/runs/${encodeURIComponent(repeat.sourceRunId)}`}>
          Source Run
        </Link>
      </div>
      <p>
        This draft creates a new Run after you submit it. It does not retry an
        attempt, mutate the source Run or promise identical placement.
      </p>
      {repeat.notices.length === 0 ? (
        <p className="compact-empty">
          Exact source revisions and caller-controlled settings were retained.
        </p>
      ) : (
        <ul className="run-repeat-notices">
          {repeat.notices.map((notice, index) => (
            <li
              className={`run-repeat-notice is-${notice.severity}`}
              key={`${notice.code}:${notice.field ?? "none"}:${index}`}
            >
              <span>{notice.severity}</span>
              <div>
                <code>{notice.code}</code>
                <p>{notice.message}</p>
                {notice.field === undefined ? null : (
                  <small>Retained field: {notice.field}</small>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
      <label className="checkbox-label run-repeat-confirmation">
        <input
          type="checkbox"
          checked={repeat.reviewed}
          onChange={(event) => onReviewed(event.target.checked)}
        />
        I reviewed the retained inputs, labels and execution settings for this
        new Run.
      </label>
      {error === undefined ? null : (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

interface WorkflowRunFormProps {
  workflow: WorkflowResource;
  projectId?: string;
  initialArtifactSelections?: Readonly<Record<string, string>>;
  initialArtifacts?: readonly ArtifactMetadata[];
}

function draftScopeLabel(draft: RunDraftSummary): string {
  return draft.projectId === undefined
    ? "standalone UserScope"
    : `Project ${draft.projectId}`;
}

function RunDraftCapacity({
  drafts,
  onDiscard,
}: {
  drafts: RunDraftSummary[];
  onDiscard: (key: string) => void;
}) {
  return (
    <section className="notice notice-warning run-draft-capacity" role="alert">
      <div>
        <strong>The in-memory Run draft limit is reached.</strong>
        <p>
          Choose one retained draft to discard. Contractor will not evict
          unsaved input automatically.
        </p>
      </div>
      <ul className="run-draft-capacity-list">
        {drafts.map((draft) => (
          <li key={draft.key}>
            <span>
              <code>
                {draft.workflowName}@{draft.workflowVersion}
              </code>
              <small>
                {draftScopeLabel(draft)}
                {draft.ambiguousSubmission
                  ? " · submission outcome unknown"
                  : ""}
              </small>
            </span>
            <button
              className="danger-button"
              type="button"
              onClick={() => onDiscard(draft.key)}
            >
              Discard this draft
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function WorkflowRunDraftBoundary(props: WorkflowRunFormProps) {
  const store = useRunDraftStore();
  const identity: RunDraftIdentity = {
    workflowName: props.workflow.ref.name,
    workflowVersion: props.workflow.ref.version,
    ...(props.projectId === undefined ? {} : { projectId: props.projectId }),
  };
  const initialState = initialRunDraftState(
    props.initialArtifactSelections,
    props.initialArtifacts,
  );
  const [acquisition, setAcquisition] = useState(() =>
    store.acquire(identity, initialState),
  );

  function discardAtCapacity(key: string): void {
    store.discard(key);
    setAcquisition(store.acquire(identity, initialState));
  }

  if (acquisition.kind === "capacity") {
    return (
      <RunDraftCapacity
        drafts={acquisition.drafts}
        onDiscard={discardAtCapacity}
      />
    );
  }
  const entry = acquisition.entry;

  function discardCurrent(): void {
    store.discard(entry);
    setAcquisition(store.acquire(identity, initialState));
  }

  return (
    <WorkflowRunFormBody
      {...props}
      key={`${entry.key}:${entry.generation}`}
      draftEntry={entry}
      draftStore={store}
      onDiscardDraft={discardCurrent}
    />
  );
}

export function WorkflowRunForm(props: WorkflowRunFormProps) {
  const key = `${props.projectId ?? "standalone"}\u0000${props.workflow.ref.name}\u0000${props.workflow.ref.version}`;
  return <WorkflowRunDraftBoundary {...props} key={key} />;
}

function DiscardRunDraftDialog({
  onClose,
  onDiscard,
}: {
  onClose: () => void;
  onDiscard: () => void;
}) {
  const heading = useId();
  const safeAction = useRef<HTMLButtonElement>(null);
  return (
    <Dialog
      className="project-dialog panel run-draft-discard-dialog"
      labelledBy={heading}
      initialFocusRef={safeAction}
      onRequestClose={onClose}
      role="alertdialog"
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Unsaved Run setup</p>
          <h2 id={heading}>Discard this Run draft?</h2>
        </div>
        <button
          className="project-dialog-close"
          type="button"
          aria-label="Close discard confirmation"
          onClick={onClose}
        >
          ×
        </button>
      </div>
      <p>
        Parameters, exact Artifact selections, labels, overrides and any
        ambiguous submission identity in this tab will be removed.
      </p>
      <div className="run-draft-actions">
        <button
          ref={safeAction}
          className="secondary-button"
          type="button"
          onClick={onClose}
        >
          Keep editing
        </button>
        <button className="danger-button" type="button" onClick={onDiscard}>
          Discard Run draft
        </button>
      </div>
    </Dialog>
  );
}

function WorkflowRunFormBody({
  workflow,
  projectId,
  draftEntry,
  draftStore,
  onDiscardDraft,
}: WorkflowRunFormProps & {
  draftEntry: RunDraftEntry;
  draftStore: RunDraftMemoryStore;
  onDiscardDraft: () => void;
}) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [gitSlot, setGitSlot] = useState<string | null>(null);
  const [uploadSlot, setUploadSlot] = useState<string | null>(null);
  const [discardRequested, setDiscardRequested] = useState(false);
  const [draft, setDraft] = useState<RunDraftState>(() =>
    structuredClone(draftEntry.state),
  );
  const [ambiguousSubmission, setAmbiguousSubmission] = useState(
    draftEntry.ambiguousSubmission,
  );
  const metadataLabelSequence = useRef(
    Math.max(
      0,
      ...draft.metadataLabels.map(
        (label) => Number.parseInt(label.id, 10) || 0,
      ),
    ),
  );
  const [validationErrors, setValidationErrors] = useState<
    Record<string, string>
  >({});
  const [runtimeOptionsOpen, setRuntimeOptionsOpen] = useState(
    () =>
      draft.runtimeLabels.length > 0 ||
      draft.repeat?.notices.some((notice) =>
        notice.field?.startsWith("runtimeLabels"),
      ) === true,
  );
  const [metadataOptionsOpen, setMetadataOptionsOpen] = useState(
    () => draft.metadataLabels.length > 0,
  );
  const [executionOptionsOpen, setExecutionOptionsOpen] = useState(
    () =>
      draft.overrides.planner.modelPolicy !== "" ||
      draft.overrides.planner.llmGateway !== "" ||
      draft.overrides.planner.credential !== "" ||
      draft.overrides.workers.modelPolicy !== "" ||
      draft.overrides.workers.llmGateway !== "" ||
      draft.overrides.workers.credential !== "" ||
      Object.keys(draft.overrides.stages ?? {}).length > 0 ||
      draft.repeat?.notices.some(
        (notice) => notice.field === "executionConfig",
      ) === true,
  );
  const {
    parameters,
    runtimeLabels: selectedRuntimeLabels,
    metadataLabels,
    metadataLabelInput,
    artifactSelections,
    artifactSuggestions,
    artifactReviews,
    knownArtifacts,
    overrides,
  } = draft;
  const keyring = draftEntry.keyring;

  useEffect(() => {
    draftStore.retain(draftEntry);
    return () => draftStore.release(draftEntry);
  }, [draftEntry, draftStore]);

  function updateDraft(
    update: (current: RunDraftState) => RunDraftState,
  ): void {
    if (!draftStore.isCurrent(draftEntry)) return;
    const next = update(structuredClone(draftEntry.state));
    if (draftStore.replaceState(draftEntry, next)) {
      setDraft(next);
    }
  }

  function setParameters(
    update: (
      current: Record<string, string | undefined>,
    ) => Record<string, string | undefined>,
  ): void {
    updateDraft((current) => ({
      ...current,
      parameters: update(current.parameters),
    }));
  }

  function setSelectedRuntimeLabels(
    update: (current: string[]) => string[],
  ): void {
    updateDraft((current) => ({
      ...current,
      runtimeLabels: update(current.runtimeLabels),
    }));
  }

  function setMetadataLabels(
    update: (current: RunMetadataLabelDraft[]) => RunMetadataLabelDraft[],
  ): void {
    updateDraft((current) => ({
      ...current,
      metadataLabels: update(current.metadataLabels),
    }));
  }

  function setMetadataLabelInput(value: string): void {
    updateDraft((current) => ({
      ...current,
      metadataLabelInput: value,
    }));
  }

  function setOverrides(
    update: (current: ExecutionOverrideDraft) => ExecutionOverrideDraft,
  ): void {
    updateDraft((current) => ({
      ...current,
      overrides: update(current.overrides),
    }));
  }

  function setRepeatReviewed(reviewed: boolean): void {
    updateDraft((current) => ({
      ...current,
      ...(current.repeat === undefined
        ? {}
        : { repeat: { ...current.repeat, reviewed } }),
    }));
    if (reviewed) clearError("repeatReview");
  }

  function selectArtifact(
    slot: string,
    selected: string,
    metadata?: ArtifactMetadata,
  ): void {
    updateDraft((current) => {
      const known =
        metadata === undefined
          ? current.knownArtifacts
          : [
              ...current.knownArtifacts.filter(
                (candidate) =>
                  artifactOptionKey(candidate.artifact) !== selected,
              ),
              metadata,
            ];
      return {
        ...current,
        artifactSelections: updateRecord(
          current.artifactSelections,
          slot,
          selected,
        ),
        artifactReviews: updateRecord(current.artifactReviews, slot, selected),
        knownArtifacts: known,
      };
    });
    clearError(`artifact:${slot}`);
    clearError(`artifactReview:${slot}`);
  }

  function confirmArtifact(slot: string): void {
    const selected = artifactSelections[slot];
    if (selected === undefined || selected === "") return;
    updateDraft((current) => ({
      ...current,
      artifactReviews: { ...current.artifactReviews, [slot]: selected },
    }));
    clearError(`artifactReview:${slot}`);
  }

  const artifactInventory = useInfiniteQuery({
    queryKey:
      projectId === undefined
        ? queryKeys.artifacts.picker
        : queryKeys.projects.artifacts.picker(projectId),
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      projectId === undefined
        ? listArtifacts(api, pageParam === null ? {} : { cursor: pageParam })
        : listProjectArtifacts(api, {
            projectId,
            ...(pageParam === null ? {} : { cursor: pageParam }),
          }),
    getNextPageParam: nextCursor,
  });
  const modelPolicyInventory = useInfiniteQuery({
    queryKey: queryKeys.configurations.infinitePicker("model-policies"),
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listConfigurations(
        api,
        "model-policies",
        pageParam === null ? {} : { cursor: pageParam },
      ),
    getNextPageParam: nextCursor,
    enabled: executionOptionsOpen,
  });
  const gatewayInventory = useInfiniteQuery({
    queryKey: queryKeys.configurations.infinitePicker("llm-gateways"),
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listConfigurations(
        api,
        "llm-gateways",
        pageParam === null ? {} : { cursor: pageParam },
      ),
    getNextPageParam: nextCursor,
    enabled: executionOptionsOpen,
  });
  const credentialInventory = useInfiniteQuery({
    queryKey: queryKeys.credentials.picker,
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listCredentials(api, pageParam === null ? {} : { cursor: pageParam }),
    getNextPageParam: nextCursor,
    enabled: executionOptionsOpen,
  });
  const runtimeLabelInventory = useInfiniteQuery({
    queryKey: queryKeys.operations.runtimeLabels.picker,
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listRuntimeLabels(api, pageParam === null ? {} : { cursor: pageParam }),
    getNextPageParam: nextCursor,
    enabled: runtimeOptionsOpen,
  });

  const artifacts = useMemo(
    () =>
      Array.from(
        new Map(
          [
            ...(artifactInventory.data?.pages.flatMap((page) => page.items) ??
              []),
            ...knownArtifacts,
          ].map((item) => [artifactOptionKey(item.artifact), item]),
        ).values(),
      ),
    [artifactInventory.data, knownArtifacts],
  );
  const artifactMap = useMemo(
    () =>
      new Map(
        artifacts.map((metadata) => [
          artifactOptionKey(metadata.artifact),
          metadata,
        ]),
      ),
    [artifacts],
  );
  const modelPolicies = useMemo(
    () => modelPolicyInventory.data?.pages.flatMap((page) => page.items) ?? [],
    [modelPolicyInventory.data],
  );
  const gateways = useMemo(
    () => gatewayInventory.data?.pages.flatMap((page) => page.items) ?? [],
    [gatewayInventory.data],
  );
  const credentials = useMemo(
    () => credentialInventory.data?.pages.flatMap((page) => page.items) ?? [],
    [credentialInventory.data],
  );
  const runtimeLabels = useMemo(
    () =>
      (runtimeLabelInventory.data?.pages.flatMap((page) => page.items) ?? [])
        .filter((binding) => binding.label !== "default")
        .sort((left, right) => left.label.localeCompare(right.label)),
    [runtimeLabelInventory.data],
  );
  const retainedRuntimeLabels = useMemo(() => {
    const visible = new Set(runtimeLabels.map((binding) => binding.label));
    return selectedRuntimeLabels.filter((label) => !visible.has(label));
  }, [runtimeLabels, selectedRuntimeLabels]);
  const defaultRuntimeConfig = useMemo(
    () =>
      runtimeLabelInventory.data?.pages
        .flatMap((page) => page.items)
        .find((binding) => binding.label === "default"),
    [runtimeLabelInventory.data],
  );
  const currentValidation = validateRunDraft(
    workflow,
    {
      runtimeLabels: selectedRuntimeLabels,
      metadataLabels,
      parameters,
      artifacts: artifactSelections,
      artifactReviews,
      overrides,
    },
    artifactMap,
  );
  const mutation = useMutation({
    mutationFn: ({
      request,
      idempotencyKey,
    }: {
      request: CreateRunRequest;
      idempotencyKey: string;
    }) =>
      projectId === undefined
        ? createRun(api, request, idempotencyKey)
        : createProjectRun(api, projectId, request, idempotencyKey),
    onSuccess: async (result) => {
      draftStore.discard(draftEntry);
      await queryClient.invalidateQueries({ queryKey: queryKeys.runs.all });
      if (projectId !== undefined) {
        await Promise.all([
          queryClient.invalidateQueries({
            queryKey: queryKeys.projects.runs(projectId),
          }),
          queryClient.invalidateQueries({
            queryKey: queryKeys.projects.artifacts.all(projectId),
          }),
        ]);
      }
      await navigate(`/runs/${encodeURIComponent(result.runId)}`);
    },
    onError: (error) => {
      const ambiguous = error instanceof PublicAPIError && error.status === 0;
      if (draftStore.setAmbiguousSubmission(draftEntry, ambiguous)) {
        setAmbiguousSubmission(ambiguous);
      }
    },
  });

  function clearError(key: string): void {
    setValidationErrors((current) => {
      if (!(key in current)) {
        return current;
      }
      const result = { ...current };
      delete result[key];
      return result;
    });
  }

  function clearMetadataLabelErrors(): void {
    setValidationErrors((current) =>
      Object.fromEntries(
        Object.entries(current).filter(
          ([key]) =>
            key !== "metadataLabels" && !key.startsWith("metadataLabel:"),
        ),
      ),
    );
  }

  function newMetadataLabel(key = "", value = ""): RunMetadataLabelDraft {
    metadataLabelSequence.current += 1;
    return {
      id: String(metadataLabelSequence.current),
      key,
      value,
    };
  }

  function addMetadataLabel(key = "", value = ""): void {
    if (metadataLabels.length >= RUN_METADATA_LABEL_LIMIT) {
      setValidationErrors((current) => ({
        ...current,
        metadataLabels: `A Run can have at most ${RUN_METADATA_LABEL_LIMIT} metadata labels.`,
      }));
      return;
    }
    const label = newMetadataLabel(key, value);
    setMetadataLabels((current) => [...current, label]);
    clearMetadataLabelErrors();
  }

  function addEvalMetadataPreset(): void {
    const existingKeys = new Set(metadataLabels.map((label) => label.key));
    const missing = EVAL_METADATA_PRESET.filter(
      (label) => !existingKeys.has(label.key),
    );
    if (metadataLabels.length + missing.length > RUN_METADATA_LABEL_LIMIT) {
      setValidationErrors((current) => ({
        ...current,
        metadataLabels: `The eval preset would exceed ${RUN_METADATA_LABEL_LIMIT} metadata labels.`,
      }));
      return;
    }
    const additions = missing.map((label) =>
      newMetadataLabel(label.key, label.value),
    );
    setMetadataLabels((current) => [...current, ...additions]);
    clearMetadataLabelErrors();
  }

  function updateMetadataLabel(
    id: string,
    field: "key" | "value",
    value: string,
  ): void {
    setMetadataLabels((current) =>
      current.map((label) =>
        label.id === id ? { ...label, [field]: value } : label,
      ),
    );
    clearMetadataLabelErrors();
  }

  function removeMetadataLabel(id: string): void {
    setMetadataLabels((current) => current.filter((label) => label.id !== id));
    clearMetadataLabelErrors();
  }

  function updateOverride(
    role: "planner" | "workers",
    field: keyof ConsumerOverrideDraft,
    value: string,
  ): void {
    setOverrides((current) => ({
      ...current,
      [role]: { ...current[role], [field]: value },
    }));
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    mutation.reset();
    if (draft.repeat !== undefined && !draft.repeat.reviewed) {
      setValidationErrors((current) => ({
        ...current,
        repeatReview:
          "Review and confirm the retained values before creating another Run.",
      }));
      return;
    }
    const validation = validateRunDraft(
      workflow,
      {
        runtimeLabels: selectedRuntimeLabels,
        metadataLabels,
        parameters,
        artifacts: artifactSelections,
        artifactReviews,
        overrides,
      },
      artifactMap,
    );
    setValidationErrors(validation.errors);
    if (validation.request === undefined) {
      const errorKeys = Object.keys(validation.errors);
      if (errorKeys.includes("runtimeLabels")) {
        setRuntimeOptionsOpen(true);
      }
      if (
        errorKeys.some(
          (key) => key === "metadataLabels" || key.startsWith("metadataLabel:"),
        )
      ) {
        setMetadataOptionsOpen(true);
      }
      return;
    }
    const idempotencyKey = keyring.keyFor(
      validation.request,
      projectId === undefined ? "standalone" : `project:${projectId}`,
    );
    draftStore.markSubmitted(draftEntry);
    draftStore.setAmbiguousSubmission(draftEntry, false);
    setAmbiguousSubmission(false);
    mutation.mutate({
      request: validation.request,
      idempotencyKey,
    });
  }

  const responseLost =
    ambiguousSubmission ||
    (mutation.error instanceof PublicAPIError && mutation.error.status === 0);
  const exactRetry =
    currentValidation.request !== undefined &&
    keyring.matches(
      currentValidation.request,
      projectId === undefined ? "standalone" : `project:${projectId}`,
    );
  const plannerSupported = Object.values(workflow.stages).some(
    (stage) => stage.executionConfig.planner !== undefined,
  );
  const inventoryErrors = [
    modelPolicyInventory.error,
    gatewayInventory.error,
    credentialInventory.error,
  ].filter((error) => error !== null);
  const requiredParameterNames = Object.entries(workflow.parameters)
    .filter(([, slot]) => slot.required)
    .map(([name]) => name);
  const requiredArtifactNames = Object.entries(workflow.inputs)
    .filter(([, slot]) => slot.required)
    .map(([name]) => name);
  const requiredFieldCount =
    requiredParameterNames.length + requiredArtifactNames.length;
  const completedRequiredFieldCount =
    requiredParameterNames.filter((name) => parameters[name] !== undefined)
      .length +
    requiredArtifactNames.filter(
      (name) => (artifactSelections[name] ?? "") !== "",
    ).length;
  const remainingRequiredFieldCount = Math.max(
    0,
    requiredFieldCount - completedRequiredFieldCount,
  );
  const reviewRequiredNames = Object.keys(workflow.inputs).filter((name) => {
    const selected = artifactSelections[name] ?? "";
    return selected !== "" && artifactReviews[name] !== selected;
  });
  const repeatReviewPending =
    draft.repeat !== undefined && !draft.repeat.reviewed;
  const draftReady =
    currentValidation.request !== undefined &&
    !artifactInventory.isPending &&
    !repeatReviewPending;
  const consumerOverrideCount = (selection: ConsumerOverrideDraft): number =>
    Object.values(selection).filter((value) => value !== "").length;
  const stageOverrideCount = Object.keys(overrides.stages ?? {}).length;
  const overrideCount =
    consumerOverrideCount(overrides.planner) +
    consumerOverrideCount(overrides.workers) +
    stageOverrideCount;
  const readinessValue = draftReady
    ? "Ready"
    : remainingRequiredFieldCount === 0 && reviewRequiredNames.length > 0
      ? `Review ${reviewRequiredNames.length}`
      : requiredFieldCount === 0
        ? "Defaults"
        : `${completedRequiredFieldCount}/${requiredFieldCount}`;
  const readinessCopy = draftReady
    ? "Fields complete · exact inputs reviewed"
    : repeatReviewPending
      ? "Review retained values for this new Run"
      : artifactInventory.isPending && requiredArtifactNames.length > 0
        ? "Loading Artifact choices…"
        : remainingRequiredFieldCount > 0
          ? `${remainingRequiredFieldCount} required ${remainingRequiredFieldCount === 1 ? "field" : "fields"} remaining`
          : reviewRequiredNames.length > 0
            ? `Fields complete · ${reviewRequiredNames.length} input ${reviewRequiredNames.length === 1 ? "review" : "reviews"} needed`
            : "Review highlighted settings";

  return (
    <form
      className="run-draft"
      id="workflow-run-form"
      onSubmit={submit}
      noValidate
    >
      {gitSlot === null ? null : (
        <GitImportDialog
          {...(projectId === undefined ? {} : { projectId })}
          suggestedName={gitSlot}
          onClose={() => setGitSlot(null)}
          onImported={(result) => {
            if (!draftStore.isCurrent(draftEntry)) return;
            const metadata: ArtifactMetadata = {
              artifact: result.artifact,
              mediaType: result.mediaType,
              size: result.size,
              gitSource: result.gitSource,
              current: true,
              frozen: false,
              createdAt: result.gitSource.importedAt,
            };
            selectArtifact(
              gitSlot,
              artifactOptionKey(result.artifact),
              metadata,
            );
            setGitSlot(null);
          }}
        />
      )}
      {uploadSlot === null ? null : (
        <RunInputUploadDialog
          {...(projectId === undefined ? {} : { projectId })}
          slotName={uploadSlot}
          mediaTypes={workflow.inputs[uploadSlot]?.mediaTypes ?? []}
          onClose={() => setUploadSlot(null)}
          onUploaded={(result) => {
            if (!draftStore.isCurrent(draftEntry)) return;
            const metadata: ArtifactMetadata = {
              artifact: result.artifact,
              mediaType: result.mediaType,
              size: result.size,
              current: true,
              frozen: false,
              createdAt: new Date().toISOString(),
            };
            selectArtifact(
              uploadSlot,
              artifactOptionKey(result.artifact),
              metadata,
            );
            setUploadSlot(null);
          }}
        />
      )}
      {discardRequested ? (
        <DiscardRunDraftDialog
          onClose={() => setDiscardRequested(false)}
          onDiscard={onDiscardDraft}
        />
      ) : null}
      <div className="section-heading run-draft-heading">
        <div>
          <p className="eyebrow">Run setup</p>
          <h3>
            {projectId === undefined
              ? "Start Workflow Run"
              : "Start Project Workflow Run"}
          </h3>
          <p className="run-draft-intro">
            Complete the declared inputs. Optional settings keep the published
            Workflow defaults until you change them.
          </p>
          <code className="run-draft-workflow">
            {workflow.ref.name}@{workflow.ref.version}
          </code>
        </div>
        <span
          className={`run-draft-readiness ${draftReady ? "is-ready" : ""}`}
          aria-live="polite"
        >
          <strong>{readinessValue}</strong>
          <small>{readinessCopy}</small>
        </span>
      </div>

      {draft.repeat === undefined ? null : (
        <RepeatRunReview
          repeat={draft.repeat}
          {...(validationErrors.repeatReview === undefined
            ? {}
            : { error: validationErrors.repeatReview })}
          onReviewed={setRepeatReviewed}
        />
      )}

      <fieldset className="run-draft-section">
        <legend>String parameters</legend>
        {Object.entries(workflow.parameters).length === 0 ? (
          <p className="compact-empty">This Workflow declares no parameters.</p>
        ) : (
          <div className="run-field-grid">
            {Object.entries(workflow.parameters)
              .sort(([left], [right]) => left.localeCompare(right))
              .map(([name, slot]) => {
                const error = validationErrors[`parameter:${name}`];
                const included = parameters[name] !== undefined;
                return (
                  <div className="run-field" key={name}>
                    {!slot.required ? (
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={included}
                          onChange={(event) => {
                            setParameters((current) => ({
                              ...current,
                              [name]: event.target.checked ? "" : undefined,
                            }));
                            clearError(`parameter:${name}`);
                          }}
                        />
                        Include optional <code>{name}</code>
                      </label>
                    ) : null}
                    <label>
                      <span>
                        {name}{" "}
                        {slot.required ? <strong>required</strong> : null}
                      </span>
                      <input
                        name={`parameter-${name}`}
                        type="text"
                        disabled={!slot.required && !included}
                        value={parameters[name] ?? ""}
                        aria-invalid={error === undefined ? undefined : true}
                        aria-describedby={
                          error === undefined
                            ? undefined
                            : `parameter-${name}-error`
                        }
                        onChange={(event) => {
                          setParameters((current) => ({
                            ...current,
                            [name]: event.target.value,
                          }));
                          clearError(`parameter:${name}`);
                        }}
                      />
                    </label>
                    {error === undefined ? null : (
                      <p
                        className="field-error"
                        id={`parameter-${name}-error`}
                        role="alert"
                      >
                        {error}
                      </p>
                    )}
                  </div>
                );
              })}
          </div>
        )}
      </fieldset>

      <fieldset className="run-draft-section">
        <legend>
          Exact {projectId === undefined ? "UserScope" : "ProjectScope"}{" "}
          Artifact inputs
        </legend>
        {artifactInventory.error === null ? null : (
          <ErrorNotice error={artifactInventory.error} />
        )}
        {Object.entries(workflow.inputs).length === 0 ? (
          <p className="compact-empty">
            This Workflow declares no Artifact inputs.
          </p>
        ) : (
          <div className="run-field-grid">
            {Object.entries(workflow.inputs)
              .sort(([left], [right]) => left.localeCompare(right))
              .map(([name, slot]) => {
                const artifactError = validationErrors[`artifact:${name}`];
                const reviewError = validationErrors[`artifactReview:${name}`];
                const compatible = artifacts.filter((metadata) =>
                  artifactAccepts(slot.mediaTypes, metadata),
                );
                const selected = artifactSelections[name] ?? "";
                const selectedMetadata = artifactMap.get(selected);
                const reviewed =
                  selected !== "" && artifactReviews[name] === selected;
                const suggested =
                  selected !== "" && artifactSuggestions[name] === selected;
                const needsReview = selected !== "" && !reviewed;
                const describedBy = [
                  artifactError === undefined
                    ? undefined
                    : `artifact-${name}-error`,
                  reviewError === undefined
                    ? undefined
                    : `artifact-${name}-review-error`,
                ]
                  .filter((value) => value !== undefined)
                  .join(" ");
                return (
                  <div className="run-field" key={name}>
                    <label>
                      <span>
                        {name}{" "}
                        {slot.required ? <strong>required</strong> : null}
                      </span>
                      <select
                        name={`artifact-${name}`}
                        value={selected}
                        disabled={artifactInventory.isPending}
                        aria-invalid={
                          artifactError === undefined &&
                          reviewError === undefined
                            ? undefined
                            : true
                        }
                        aria-describedby={
                          describedBy === "" ? undefined : describedBy
                        }
                        onChange={(event) => {
                          const exactRef = event.target.value;
                          selectArtifact(
                            name,
                            exactRef,
                            artifactMap.get(exactRef),
                          );
                        }}
                      >
                        <option value="">
                          {slot.required
                            ? "Select an exact revision"
                            : "No Artifact supplied"}
                        </option>
                        {compatible.map((metadata) => (
                          <option
                            key={artifactOptionKey(metadata.artifact)}
                            value={artifactOptionKey(metadata.artifact)}
                          >
                            {artifactLabel(metadata)}
                          </option>
                        ))}
                      </select>
                    </label>
                    <small>
                      Slot <code>{name}</code> accepts{" "}
                      {slot.mediaTypes.join(", ")}. MIME compatibility does not
                      verify Artifact contents.
                    </small>
                    {selectedMetadata === undefined ? null : (
                      <section
                        className={`run-input-review${needsReview ? " needs-review" : " is-reviewed"}`}
                        aria-label={`Exact input review for ${name}`}
                      >
                        <div className="run-input-review-heading">
                          <span>
                            <small>
                              {needsReview && suggested
                                ? "Format-compatible suggestion"
                                : reviewed
                                  ? "Reviewed selection"
                                  : "Exact selection"}
                            </small>
                            <code>{selected}</code>
                          </span>
                          <strong>
                            {reviewed ? "Confirmed" : "Review needed"}
                          </strong>
                        </div>
                        <dl className="run-input-review-facts">
                          <div>
                            <dt>Scope</dt>
                            <dd>
                              {projectId === undefined
                                ? "UserScope"
                                : `ProjectScope · ${projectId}`}
                            </dd>
                          </div>
                          <div>
                            <dt>Stored type</dt>
                            <dd>{selectedMetadata.mediaType}</dd>
                          </div>
                          <div>
                            <dt>Size</dt>
                            <dd>{formatBytes(selectedMetadata.size)}</dd>
                          </div>
                          <div>
                            <dt>Provenance</dt>
                            <dd>
                              {selectedMetadata.gitSource === undefined
                                ? `Artifact write · ${formatTimestamp(selectedMetadata.createdAt)}`
                                : `Git import · ${formatTimestamp(selectedMetadata.gitSource.importedAt)}`}
                            </dd>
                          </div>
                        </dl>
                        <p>
                          {needsReview && suggested
                            ? `Suggested only because ${selectedMetadata.mediaType} matches this slot. Confirm that this exact revision is the intended ${name} input.`
                            : reviewed
                              ? `Confirmed for ${name}. This records your exact selection, not a semantic validation of its contents.`
                              : `Review this exact revision before using it as ${name}.`}
                        </p>
                        <div className="run-input-review-actions">
                          {needsReview ? (
                            <button
                              className="secondary-button"
                              type="button"
                              onClick={() => confirmArtifact(name)}
                            >
                              Confirm exact input for {name}
                            </button>
                          ) : null}
                          <Link
                            to={artifactDetailPath(selectedMetadata, projectId)}
                          >
                            Preview exact Artifact details
                          </Link>
                        </div>
                      </section>
                    )}
                    <div className="run-draft-actions">
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => setUploadSlot(name)}
                      >
                        Upload local file for {name}
                      </button>
                      {slot.mediaTypes.some((type) =>
                        ["application/zip", "*/*"].includes(type),
                      ) ? (
                        <button
                          className="secondary-button git-import-icon-button"
                          type="button"
                          aria-label={`Import Git for ${name}`}
                          title={`Import Git repository for ${name}`}
                          onClick={() => setGitSlot(name)}
                        >
                          <GitRepositoryIcon />
                        </button>
                      ) : null}
                    </div>
                    <GitSourceDetails
                      source={
                        artifactMap.get(artifactSelections[name] ?? "")
                          ?.gitSource
                      }
                    />
                    {artifactError === undefined ? null : (
                      <p
                        className="field-error"
                        id={`artifact-${name}-error`}
                        role="alert"
                      >
                        {artifactError}
                      </p>
                    )}
                    {reviewError === undefined ? null : (
                      <p
                        className="field-error"
                        id={`artifact-${name}-review-error`}
                        role="alert"
                      >
                        {reviewError}
                      </p>
                    )}
                    {!artifactInventory.isPending && compatible.length === 0 ? (
                      <p className="field-guidance">
                        No matching artifacts found in the loaded list.{" "}
                        <Link
                          to={
                            projectId === undefined
                              ? "/artifacts"
                              : `/projects/${encodeURIComponent(projectId)}#project-artifacts`
                          }
                        >
                          Upload one
                        </Link>
                        .
                      </p>
                    ) : null}
                  </div>
                );
              })}
          </div>
        )}
        {artifactInventory.hasNextPage ? (
          <button
            className="secondary-button"
            type="button"
            disabled={artifactInventory.isFetchingNextPage}
            onClick={() => void artifactInventory.fetchNextPage()}
          >
            {artifactInventory.isFetchingNextPage
              ? "Loading Artifacts…"
              : "Load more Artifacts"}
          </button>
        ) : null}
      </fieldset>

      <div className="run-draft-advanced-heading">
        <div>
          <p className="eyebrow">Optional setup</p>
          <h4>Advanced settings</h4>
        </div>
        <p>
          Keep these collapsed to use the published Workflow and Runtime
          defaults.
        </p>
      </div>

      <details
        className="run-draft-disclosure"
        open={runtimeOptionsOpen}
        onToggle={(event) => setRuntimeOptionsOpen(event.currentTarget.open)}
      >
        <summary>
          <DraftDisclosureSummary
            title="Runtime placement"
            description="Published default is always applied"
            status={
              runtimeLabelInventory.error !== null
                ? "Unavailable"
                : runtimeOptionsOpen && runtimeLabelInventory.isPending
                  ? "Loading…"
                  : selectedRuntimeLabels.length === 0
                    ? "Default only"
                    : `${selectedRuntimeLabels.length} selected`
            }
            active={selectedRuntimeLabels.length > 0}
          />
        </summary>
        <fieldset
          className="run-draft-section runtime-label-picker"
          aria-label="Runtime infrastructure labels"
        >
          <p className="muted-copy">
            Default is always pinned and is not selectable. Explicit labels are
            a sorted immutable set for this Run; they configure infrastructure,
            not Workflow behavior or model budgets.
          </p>
          {runtimeLabelInventory.error === null ? null : (
            <ErrorNotice error={runtimeLabelInventory.error} />
          )}
          <div className="runtime-default-preview">
            <strong>Default · always applied</strong>
            {defaultRuntimeConfig === undefined ? (
              <span className="muted-copy">Loading exact binding…</span>
            ) : (
              <RuntimeLabelPreview binding={defaultRuntimeConfig} />
            )}
          </div>
          {runtimeLabels.length === 0 &&
          retainedRuntimeLabels.length === 0 &&
          !runtimeLabelInventory.isPending ? (
            <p className="compact-empty">
              No explicit Runtime labels are bound.
            </p>
          ) : (
            <div className="runtime-label-options">
              {retainedRuntimeLabels.map((label) => (
                <label
                  className="runtime-label-option retained-selection"
                  key={`retained:${label}`}
                >
                  <input
                    type="checkbox"
                    checked
                    onChange={() => {
                      setSelectedRuntimeLabels((current) =>
                        current.filter((candidate) => candidate !== label),
                      );
                      clearError("runtimeLabels");
                    }}
                  />
                  <span>
                    <strong>{label}</strong>
                    <small>
                      Retained label · current binding unavailable or not loaded
                    </small>
                  </span>
                </label>
              ))}
              {runtimeLabels.map((binding) => (
                <label className="runtime-label-option" key={binding.label}>
                  <input
                    type="checkbox"
                    checked={selectedRuntimeLabels.includes(binding.label)}
                    onChange={(event) => {
                      setSelectedRuntimeLabels((current) =>
                        (event.target.checked
                          ? [...current, binding.label]
                          : current.filter((label) => label !== binding.label)
                        ).sort(),
                      );
                      clearError("runtimeLabels");
                    }}
                  />
                  <span>
                    <strong>{binding.label}</strong>
                    <RuntimeLabelPreview binding={binding} />
                  </span>
                </label>
              ))}
            </div>
          )}
          {validationErrors.runtimeLabels === undefined ? null : (
            <p className="field-error" role="alert">
              {validationErrors.runtimeLabels}
            </p>
          )}
          {runtimeLabelInventory.hasNextPage ? (
            <button
              className="secondary-button"
              type="button"
              disabled={runtimeLabelInventory.isFetchingNextPage}
              onClick={() => void runtimeLabelInventory.fetchNextPage()}
            >
              {runtimeLabelInventory.isFetchingNextPage
                ? "Loading Runtime labels…"
                : "Load more Runtime labels"}
            </button>
          ) : null}
        </fieldset>
      </details>

      <details
        className="run-draft-disclosure"
        open={metadataOptionsOpen}
        onToggle={(event) => setMetadataOptionsOpen(event.currentTarget.open)}
      >
        <summary>
          <DraftDisclosureSummary
            title="Run metadata"
            description="Searchable labels and eval identifiers"
            status={
              metadataLabels.length === 0
                ? "No labels"
                : `${metadataLabels.length} ${metadataLabels.length === 1 ? "label" : "labels"}`
            }
            active={metadataLabels.length > 0}
          />
        </summary>
        <RunMetadataLabelEditor
          labels={metadataLabels}
          labelInput={metadataLabelInput}
          errors={validationErrors}
          onAdd={addMetadataLabel}
          onAddEvalPreset={addEvalMetadataPreset}
          onChange={updateMetadataLabel}
          onLabelInputChange={setMetadataLabelInput}
          onRemove={removeMetadataLabel}
        />
      </details>

      <details
        className="run-draft-disclosure override-panel"
        open={executionOptionsOpen}
        onToggle={(event) => setExecutionOptionsOpen(event.currentTarget.open)}
      >
        <summary>
          <DraftDisclosureSummary
            title="Execution overrides"
            description="Models, Gateway and credentials"
            status={
              executionOptionsOpen && inventoryErrors.length > 0
                ? "Unavailable"
                : overrideCount === 0
                  ? "Workflow defaults"
                  : `${overrideCount} overridden`
            }
            active={overrideCount > 0}
          />
        </summary>
        <div className="run-draft-disclosure-body">
          <p className="muted-copy">
            Empty fields retain the Workflow's resolved defaults. Values below
            are exact published refs or active credential IDs; this form accepts
            no free-form model, Gateway URL, budget, provider, or token.
          </p>
          {inventoryErrors.map((error, index) => (
            <ErrorNotice key={index} error={error} />
          ))}
          {!modelPolicyInventory.isPending && modelPolicies.length === 0 ? (
            <p className="inventory-empty">
              No ModelPolicy versions published.
            </p>
          ) : null}
          {!gatewayInventory.isPending && gateways.length === 0 ? (
            <p className="inventory-empty">
              No LLMGatewayConfig versions published.
            </p>
          ) : null}
          {!credentialInventory.isPending && credentials.length === 0 ? (
            <p className="inventory-empty">No active credentials available.</p>
          ) : null}
          <div className="override-grid">
            <ConsumerOverrides
              role="Planner"
              value={overrides.planner}
              modelPolicies={modelPolicies}
              gateways={gateways}
              credentials={credentials}
              disabled={!plannerSupported}
              onChange={(field, value) =>
                updateOverride("planner", field, value)
              }
            />
            <ConsumerOverrides
              role="Workers"
              value={overrides.workers}
              modelPolicies={modelPolicies}
              gateways={gateways}
              credentials={credentials}
              onChange={(field, value) =>
                updateOverride("workers", field, value)
              }
            />
          </div>
          {stageOverrideCount === 0 ? null : (
            <section
              className="retained-stage-overrides"
              aria-labelledby="retained-stage-overrides-title"
            >
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Retained exact patch</p>
                  <h4 id="retained-stage-overrides-title">
                    Stage-specific overrides
                  </h4>
                </div>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() =>
                    setOverrides((current) => {
                      const next = { ...current };
                      delete next.stages;
                      return next;
                    })
                  }
                >
                  Remove all Stage overrides
                </button>
              </div>
              <p className="muted-copy">
                The current form cannot edit per-Stage selections. It preserves
                the original request exactly or removes this patch as one
                explicit action.
              </p>
              <pre>{JSON.stringify(overrides.stages, null, 2)}</pre>
            </section>
          )}
          <div className="load-more-row">
            {modelPolicyInventory.hasNextPage ? (
              <button
                className="secondary-button"
                type="button"
                disabled={modelPolicyInventory.isFetchingNextPage}
                onClick={() => void modelPolicyInventory.fetchNextPage()}
              >
                Load more ModelPolicies
              </button>
            ) : null}
            {gatewayInventory.hasNextPage ? (
              <button
                className="secondary-button"
                type="button"
                disabled={gatewayInventory.isFetchingNextPage}
                onClick={() => void gatewayInventory.fetchNextPage()}
              >
                Load more Gateways
              </button>
            ) : null}
            {credentialInventory.hasNextPage ? (
              <button
                className="secondary-button"
                type="button"
                disabled={credentialInventory.isFetchingNextPage}
                onClick={() => void credentialInventory.fetchNextPage()}
              >
                Load more credentials
              </button>
            ) : null}
          </div>
        </div>
      </details>

      {mutation.error === null ? null : <ErrorNotice error={mutation.error} />}
      {responseLost ? (
        <div className="notice notice-warning" role="alert">
          <strong>The Run may already exist.</strong>
          {exactRetry ? (
            <p>
              The draft is unchanged. “Retry exact request” reuses the same
              idempotency key and cannot create a second Run for this request.
            </p>
          ) : (
            <p>
              This draft changed after the lost response. Submitting it uses a
              new key and may create another Run; restore the prior values to
              retry the ambiguous request first.
            </p>
          )}
        </div>
      ) : null}
      <div className="run-submit-row">
        <span className="run-submit-status">
          <strong>{draftReady ? "Ready to start" : readinessCopy}</strong>
          <small>The Run opens after the Server accepts the request.</small>
        </span>
        <div className="run-draft-actions">
          {draftEntry.meaningful ? (
            <button
              className="secondary-button"
              type="button"
              disabled={mutation.isPending}
              onClick={() => setDiscardRequested(true)}
            >
              Discard saved draft
            </button>
          ) : null}
          <button
            type="submit"
            disabled={
              mutation.isPending ||
              repeatReviewPending ||
              (Object.keys(workflow.inputs).length > 0 &&
                artifactInventory.isPending)
            }
          >
            {mutation.isPending
              ? projectId === undefined
                ? "Submitting…"
                : "Starting Project Run…"
              : responseLost && exactRetry
                ? "Retry exact request"
                : responseLost
                  ? "Start changed draft with a new key"
                  : projectId === undefined
                    ? "Start Workflow Run"
                    : "Start Project Workflow Run"}
          </button>
        </div>
      </div>
    </form>
  );
}
