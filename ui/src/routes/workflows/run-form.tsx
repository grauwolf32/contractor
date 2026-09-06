import {
  GitImportDialog,
  GitSourceDetails,
} from "../artifacts/git-import-dialog";
import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import { type FormEvent, useMemo, useRef, useState } from "react";
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
import { RunDraftKeyring } from "../../run-drafts/idempotency";
import {
  artifactAccepts,
  artifactOptionKey,
  emptyExecutionOverrides,
  NO_CREDENTIAL_OVERRIDE,
  validateRunDraft,
  type ConsumerOverrideDraft,
  type ExecutionOverrideDraft,
} from "../../run-drafts/validation";
import { ErrorNotice, formatBytes } from "../artifacts/common";

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
  errors,
  onAdd,
  onAddEvalPreset,
  onChange,
  onRemove,
}: {
  labels: readonly RunMetadataLabelDraft[];
  errors: Readonly<Record<string, string>>;
  onAdd: () => void;
  onAddEvalPreset: () => void;
  onChange: (id: string, field: "key" | "value", value: string) => void;
  onRemove: (id: string) => void;
}) {
  const existingKeys = new Set(labels.map((label) => label.key));
  const missingEvalLabels = EVAL_METADATA_PRESET.filter(
    (label) => !existingKeys.has(label.key),
  ).length;
  return (
    <fieldset
      className="run-draft-section run-metadata-label-editor"
      aria-label="Run metadata labels"
    >
      <p className="muted-copy">
        Immutable searchable metadata for this Run and its root traces. Labels
        do not select Runtime infrastructure and are never shown to Planner or
        Workers.
      </p>
      <div className="notice notice-warning run-label-secret-warning">
        <strong>Do not put secrets in labels.</strong>
        <p>
          Keys and values are visible through the Run API, list filters and
          telemetry.
        </p>
      </div>
      <div className="run-metadata-label-actions">
        <button
          className="secondary-button"
          type="button"
          disabled={labels.length >= RUN_METADATA_LABEL_LIMIT}
          onClick={onAdd}
        >
          Add metadata label
        </button>
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
        <div className="run-metadata-label-rows">
          {labels.map((label, index) => {
            const keyError = errors[`metadataLabel:${label.id}:key`];
            const valueError = errors[`metadataLabel:${label.id}:value`];
            const keyErrorID = `metadata-label-${label.id}-key-error`;
            const valueErrorID = `metadata-label-${label.id}-value-error`;
            return (
              <div className="run-metadata-label-row" key={label.id}>
                <div className="run-field">
                  <label htmlFor={`metadata-label-${label.id}-key`}>
                    Run metadata label key {index + 1}
                  </label>
                  <input
                    id={`metadata-label-${label.id}-key`}
                    name={`metadata-label-key-${index + 1}`}
                    type="text"
                    autoComplete="off"
                    value={label.key}
                    aria-invalid={keyError === undefined ? undefined : true}
                    aria-describedby={
                      keyError === undefined ? undefined : keyErrorID
                    }
                    onChange={(event) =>
                      onChange(label.id, "key", event.target.value)
                    }
                  />
                  {keyError === undefined ? null : (
                    <p className="field-error" id={keyErrorID} role="alert">
                      {keyError}
                    </p>
                  )}
                </div>
                <div className="run-field">
                  <label htmlFor={`metadata-label-${label.id}-value`}>
                    Run metadata label value {index + 1}
                  </label>
                  <input
                    id={`metadata-label-${label.id}-value`}
                    name={`metadata-label-value-${index + 1}`}
                    type="text"
                    autoComplete="off"
                    value={label.value}
                    aria-invalid={valueError === undefined ? undefined : true}
                    aria-describedby={
                      valueError === undefined ? undefined : valueErrorID
                    }
                    onChange={(event) =>
                      onChange(label.id, "value", event.target.value)
                    }
                  />
                  {valueError === undefined ? null : (
                    <p className="field-error" id={valueErrorID} role="alert">
                      {valueError}
                    </p>
                  )}
                </div>
                <button
                  className="danger-button run-metadata-label-remove"
                  type="button"
                  aria-label={`Remove Run metadata label ${index + 1}`}
                  onClick={() => onRemove(label.id)}
                >
                  Remove
                </button>
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

export function WorkflowRunForm({
  workflow,
  projectId,
  initialArtifactSelections = {},
}: {
  workflow: WorkflowResource;
  projectId?: string;
  initialArtifactSelections?: Readonly<Record<string, string>>;
}) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [gitSlot, setGitSlot] = useState<string | null>(null);
  const [importedArtifacts, setImportedArtifacts] = useState<
    ArtifactMetadata[]
  >([]);
  const [keyring] = useState(() => new RunDraftKeyring());
  const [parameters, setParameters] = useState<
    Record<string, string | undefined>
  >({});
  const [selectedRuntimeLabels, setSelectedRuntimeLabels] = useState<string[]>(
    [],
  );
  const metadataLabelSequence = useRef(0);
  const [metadataLabels, setMetadataLabels] = useState<RunMetadataLabelDraft[]>(
    [],
  );
  const [artifactSelections, setArtifactSelections] = useState<
    Record<string, string>
  >({ ...initialArtifactSelections });
  const [overrides, setOverrides] = useState<ExecutionOverrideDraft>(
    emptyExecutionOverrides,
  );
  const [validationErrors, setValidationErrors] = useState<
    Record<string, string>
  >({});
  const [runtimeOptionsOpen, setRuntimeOptionsOpen] = useState(false);
  const [metadataOptionsOpen, setMetadataOptionsOpen] = useState(false);
  const [executionOptionsOpen, setExecutionOptionsOpen] = useState(false);

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
            ...importedArtifacts,
          ].map((item) => [artifactOptionKey(item.artifact), item]),
        ).values(),
      ),
    [artifactInventory.data, importedArtifacts],
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

  function addMetadataLabel(): void {
    if (metadataLabels.length >= RUN_METADATA_LABEL_LIMIT) {
      setValidationErrors((current) => ({
        ...current,
        metadataLabels: `A Run can have at most ${RUN_METADATA_LABEL_LIMIT} metadata labels.`,
      }));
      return;
    }
    const label = newMetadataLabel();
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
    role: keyof ExecutionOverrideDraft,
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
    const validation = validateRunDraft(
      workflow,
      {
        runtimeLabels: selectedRuntimeLabels,
        metadataLabels,
        parameters,
        artifacts: artifactSelections,
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
    mutation.mutate({
      request: validation.request,
      idempotencyKey: keyring.keyFor(
        validation.request,
        projectId === undefined ? "standalone" : `project:${projectId}`,
      ),
    });
  }

  const responseLost =
    mutation.error instanceof PublicAPIError && mutation.error.status === 0;
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
  const draftReady =
    currentValidation.request !== undefined && !artifactInventory.isPending;
  const overrideCount = Object.values(overrides).reduce(
    (count, selection) =>
      count + Object.values(selection).filter((value) => value !== "").length,
    0,
  );
  const readinessValue = draftReady
    ? "Ready"
    : requiredFieldCount === 0
      ? "Defaults"
      : `${completedRequiredFieldCount}/${requiredFieldCount}`;
  const readinessCopy = draftReady
    ? "All required fields are complete"
    : artifactInventory.isPending && requiredArtifactNames.length > 0
      ? "Loading Artifact choices…"
      : remainingRequiredFieldCount > 0
        ? `${remainingRequiredFieldCount} required ${remainingRequiredFieldCount === 1 ? "field" : "fields"} remaining`
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
            setImportedArtifacts((current) => [
              ...current,
              {
                artifact: result.artifact,
                mediaType: result.mediaType,
                size: result.size,
                gitSource: result.gitSource,
                current: true,
                frozen: false,
                createdAt: result.gitSource.importedAt,
              },
            ]);
            setArtifactSelections((current) =>
              updateRecord(
                current,
                gitSlot,
                artifactOptionKey(result.artifact),
              ),
            );
            clearError(`artifact:${gitSlot}`);
            setGitSlot(null);
          }}
        />
      )}
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
                const error = validationErrors[`artifact:${name}`];
                const compatible = artifacts.filter((metadata) =>
                  artifactAccepts(slot.mediaTypes, metadata),
                );
                const incompatible = artifacts.filter(
                  (metadata) => !artifactAccepts(slot.mediaTypes, metadata),
                );
                return (
                  <div className="run-field" key={name}>
                    <label>
                      <span>
                        {name}{" "}
                        {slot.required ? <strong>required</strong> : null}
                      </span>
                      <select
                        name={`artifact-${name}`}
                        value={artifactSelections[name] ?? ""}
                        disabled={artifactInventory.isPending}
                        aria-invalid={error === undefined ? undefined : true}
                        aria-describedby={
                          error === undefined
                            ? undefined
                            : `artifact-${name}-error`
                        }
                        onChange={(event) => {
                          setArtifactSelections((current) =>
                            updateRecord(current, name, event.target.value),
                          );
                          clearError(`artifact:${name}`);
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
                        {incompatible.map((metadata) => (
                          <option
                            key={artifactOptionKey(metadata.artifact)}
                            value={artifactOptionKey(metadata.artifact)}
                            disabled
                          >
                            Incompatible · {artifactLabel(metadata)}
                          </option>
                        ))}
                      </select>
                    </label>
                    <small>Accepts {slot.mediaTypes.join(", ")}</small>
                    {slot.mediaTypes.some((type) =>
                      ["application/zip", "*/*"].includes(type),
                    ) ? (
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => setGitSlot(name)}
                      >
                        Import Git for {name}
                      </button>
                    ) : null}
                    <GitSourceDetails
                      source={
                        artifactMap.get(artifactSelections[name] ?? "")
                          ?.gitSource
                      }
                    />
                    {error === undefined ? null : (
                      <p
                        className="field-error"
                        id={`artifact-${name}-error`}
                        role="alert"
                      >
                        {error}
                      </p>
                    )}
                    {!artifactInventory.isPending && compatible.length === 0 ? (
                      <p className="field-guidance">
                        No loaded revision is compatible.{" "}
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
          {runtimeLabels.length === 0 && !runtimeLabelInventory.isPending ? (
            <p className="compact-empty">
              No explicit Runtime labels are bound.
            </p>
          ) : (
            <div className="runtime-label-options">
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
          errors={validationErrors}
          onAdd={addMetadataLabel}
          onAddEvalPreset={addEvalMetadataPreset}
          onChange={updateMetadataLabel}
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
        <button
          type="submit"
          disabled={
            mutation.isPending ||
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
    </form>
  );
}
