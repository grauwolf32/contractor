import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import { type FormEvent, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";

import { listArtifacts, type ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { queryKeys } from "../../api/query-keys";
import {
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

function nextCursor(page: { page: { hasMore: boolean; nextCursor?: string } }) {
  return page.page.hasMore ? page.page.nextCursor : undefined;
}

function configurationSelector(resource: ConfigurationResource): string {
  return `${resource.ref.name}@${resource.ref.version}`;
}

function artifactLabel(metadata: ArtifactMetadata): string {
  return `${artifactOptionKey(metadata.artifact)} · ${metadata.mediaType} · ${formatBytes(metadata.size)}`;
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

export function WorkflowRunForm({ workflow }: { workflow: WorkflowResource }) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [keyring] = useState(() => new RunDraftKeyring());
  const [parameters, setParameters] = useState<
    Record<string, string | undefined>
  >({});
  const [artifactSelections, setArtifactSelections] = useState<
    Record<string, string>
  >({});
  const [overrides, setOverrides] = useState<ExecutionOverrideDraft>(
    emptyExecutionOverrides,
  );
  const [validationErrors, setValidationErrors] = useState<
    Record<string, string>
  >({});

  const artifactInventory = useInfiniteQuery({
    queryKey: queryKeys.artifacts.picker,
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listArtifacts(api, pageParam === null ? {} : { cursor: pageParam }),
    getNextPageParam: nextCursor,
  });
  const modelPolicyInventory = useInfiniteQuery({
    queryKey: queryKeys.configurations.picker("model-policies"),
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listConfigurations(
        api,
        "model-policies",
        pageParam === null ? {} : { cursor: pageParam },
      ),
    getNextPageParam: nextCursor,
  });
  const gatewayInventory = useInfiniteQuery({
    queryKey: queryKeys.configurations.picker("llm-gateways"),
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listConfigurations(
        api,
        "llm-gateways",
        pageParam === null ? {} : { cursor: pageParam },
      ),
    getNextPageParam: nextCursor,
  });
  const credentialInventory = useInfiniteQuery({
    queryKey: queryKeys.credentials.picker,
    initialPageParam: INITIAL_CURSOR as string | null,
    queryFn: ({ pageParam }) =>
      listCredentials(api, pageParam === null ? {} : { cursor: pageParam }),
    getNextPageParam: nextCursor,
  });

  const artifacts = useMemo(
    () => artifactInventory.data?.pages.flatMap((page) => page.items) ?? [],
    [artifactInventory.data],
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
  const currentValidation = validateRunDraft(
    workflow,
    { parameters, artifacts: artifactSelections, overrides },
    artifactMap,
  );
  const mutation = useMutation({
    mutationFn: ({
      request,
      idempotencyKey,
    }: {
      request: CreateRunRequest;
      idempotencyKey: string;
    }) => createRun(api, request, idempotencyKey),
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.runs.all });
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
      { parameters, artifacts: artifactSelections, overrides },
      artifactMap,
    );
    setValidationErrors(validation.errors);
    if (validation.request === undefined) {
      return;
    }
    mutation.mutate({
      request: validation.request,
      idempotencyKey: keyring.keyFor(validation.request),
    });
  }

  const responseLost =
    mutation.error instanceof PublicAPIError && mutation.error.status === 0;
  const exactRetry =
    currentValidation.request !== undefined &&
    keyring.matches(currentValidation.request);
  const plannerSupported = Object.values(workflow.stages).some(
    (stage) => stage.executionConfig.planner !== undefined,
  );
  const inventoryErrors = [
    modelPolicyInventory.error,
    gatewayInventory.error,
    credentialInventory.error,
  ].filter((error) => error !== null);

  return (
    <form className="run-draft" onSubmit={submit} noValidate>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Immutable request draft</p>
          <h3>Create Workflow Run</h3>
        </div>
        <code>
          {workflow.ref.name}@{workflow.ref.version}
        </code>
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
        <legend>Exact UserScope Artifact inputs</legend>
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
                        <Link to="/artifacts">Upload one</Link>.
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

      <details className="run-draft-section override-panel">
        <summary>Optional published execution overrides</summary>
        <p className="muted-copy">
          Empty fields retain the Workflow's resolved defaults. Values below are
          exact published refs or active credential IDs; this form accepts no
          free-form model, Gateway URL, budget, provider, or token.
        </p>
        {inventoryErrors.map((error, index) => (
          <ErrorNotice key={index} error={error} />
        ))}
        {!modelPolicyInventory.isPending && modelPolicies.length === 0 ? (
          <p className="inventory-empty">No ModelPolicy versions published.</p>
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
            onChange={(field, value) => updateOverride("planner", field, value)}
          />
          <ConsumerOverrides
            role="Workers"
            value={overrides.workers}
            modelPolicies={modelPolicies}
            gateways={gateways}
            credentials={credentials}
            onChange={(field, value) => updateOverride("workers", field, value)}
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
        <button
          type="submit"
          disabled={
            mutation.isPending ||
            (Object.keys(workflow.inputs).length > 0 &&
              artifactInventory.isPending)
          }
        >
          {mutation.isPending
            ? "Submitting…"
            : responseLost && exactRetry
              ? "Retry exact request"
              : responseLost
                ? "Start changed draft with a new key"
                : "Start Workflow Run"}
        </button>
        <small>
          No Run is inserted optimistically; navigation waits for Server 202.
        </small>
      </div>
    </form>
  );
}
