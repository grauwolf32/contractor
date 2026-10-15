import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import { type FormEvent, useMemo, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  BUDGET_DURATION_PATTERN,
  createCredential,
  listConfigurations,
  MANAGED_CONFIG_NAME_PATTERN,
  type ConfigurationResource,
  type CreateCredentialRequest,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice } from "../../../app/error-notice";
import { LoadMoreButton } from "../../../app/load-more";
import { nextPageCursor } from "../../../app/pagination";
import { hasCredentialManager } from "../llm-configurations/model";
import { integerBounds, LABEL, validateCredentialRequest } from "./validation";
import { compactDigest } from "../../../app/format";
import { PublicationFeedback, ScopeChip } from "../common";
import { useGuardedNavigate } from "../../../app/page-location";
import { Button, Field } from "../../../ui";

function configKey(resource: ConfigurationResource): string {
  return `${resource.ref.name}@${resource.ref.version}:${resource.ref.digest}`;
}

function optionalNumber(value: string): number | undefined {
  return value === "" ? undefined : Number(value);
}

// V385: credential IDs are ConfigId in the contract, but src/api/operations.ts
// (requireConfigIdentity) accepts only ConfigurationName names, so the form
// checks that rule until the API layer is fixed.
const CREDENTIAL_ID = MANAGED_CONFIG_NAME_PATTERN;

export function CredentialCreateForm() {
  const api = usePublicAPI();
  const navigate = useGuardedNavigate();
  const queryClient = useQueryClient();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<CreateCredentialRequest>("create-credential"),
  );
  const gateways = useInfiniteQuery({
    queryKey: queryKeys.configurations.infinitePicker("llm-gateways"),
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      listConfigurations(
        api,
        "llm-gateways",
        pageParam === null ? { signal } : { cursor: pageParam, signal },
      ),
    getNextPageParam: (page) => nextPageCursor(page.page),
  });
  const policies = useInfiniteQuery({
    queryKey: queryKeys.configurations.infinitePicker("model-policies"),
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      listConfigurations(
        api,
        "model-policies",
        pageParam === null ? { signal } : { cursor: pageParam, signal },
      ),
    getNextPageParam: (page) => nextPageCursor(page.page),
  });
  const [credentialId, setCredentialId] = useState("");
  const [label, setLabel] = useState("");
  const [gatewayKey, setGatewayKey] = useState("");
  const [policyKeys, setPolicyKeys] = useState<string[]>([]);
  const [maxBudget, setMaxBudget] = useState("");
  const [budgetDuration, setBudgetDuration] = useState("");
  const [tpmLimit, setTPMLimit] = useState("");
  const [rpmLimit, setRPMLimit] = useState("");
  const [parallelLimit, setParallelLimit] = useState("");
  const [errors, setErrors] = useState<string[]>([]);

  const managedGateways = useMemo(
    () =>
      (gateways.data?.pages.flatMap((page) => page.items) ?? []).filter(
        hasCredentialManager,
      ),
    [gateways.data],
  );
  const modelPolicies =
    policies.data?.pages.flatMap((page) => page.items) ?? [];
  const mutation = useMutation({
    mutationFn: (request: CreateCredentialRequest) =>
      createCredential(api, request, keyring.keyFor(request)),
    onSuccess: async (credential) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.credentials.all }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.snapshot,
        }),
      ]);
      await navigate(
        `/operations/credentials/${encodeURIComponent(credential.credentialId)}`,
      );
    },
    onError: async () => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.credentials.all,
      });
    },
  });

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    mutation.reset();
    const selectedGateway = managedGateways.find(
      (resource) => configKey(resource) === gatewayKey,
    );
    const selectedPolicies = policyKeys
      .map((key) =>
        modelPolicies.find((resource) => configKey(resource) === key),
      )
      .filter(
        (resource): resource is ConfigurationResource => resource !== undefined,
      );
    const nextErrors: string[] = [];
    if (!CREDENTIAL_ID.test(credentialId)) {
      nextErrors.push(
        `Credential ID must start with a lowercase letter and contain at most ${CREDENTIAL_ID.maxLength} lowercase letters, digits, dashes or underscores.`,
      );
    }
    if (selectedGateway === undefined) {
      nextErrors.push("Select a Gateway with LiteLLM credential management.");
    }
    if (selectedPolicies.length !== policyKeys.length) {
      nextErrors.push("One selected ModelPolicy is no longer available.");
    }
    if (
      budgetDuration !== "" &&
      !BUDGET_DURATION_PATTERN.test(budgetDuration)
    ) {
      nextErrors.push("Budget reset must look like 1h, 7d, or 1mo.");
    }
    if (selectedGateway === undefined) {
      setErrors(nextErrors);
      return;
    }
    const normalizedLabel = label.trim();
    const request: CreateCredentialRequest = {
      credentialId,
      llmGateway: {
        gatewayId: selectedGateway.ref.name,
        version: selectedGateway.ref.version,
        digest: selectedGateway.ref.digest,
      },
      ...(normalizedLabel === "" ? {} : { label: normalizedLabel }),
      gatewayPolicy: {
        modelPolicies: selectedPolicies.map((resource) => ({
          policyId: resource.ref.name,
          version: resource.ref.version,
          digest: resource.ref.digest,
        })),
        ...(optionalNumber(maxBudget) === undefined
          ? {}
          : { maxBudget: optionalNumber(maxBudget)! }),
        ...(budgetDuration === "" ? {} : { budgetDuration }),
        ...(optionalNumber(tpmLimit) === undefined
          ? {}
          : { tpmLimit: optionalNumber(tpmLimit)! }),
        ...(optionalNumber(rpmLimit) === undefined
          ? {}
          : { rpmLimit: optionalNumber(rpmLimit)! }),
        ...(optionalNumber(parallelLimit) === undefined
          ? {}
          : { maxParallelRequests: optionalNumber(parallelLimit)! }),
      },
    };
    nextErrors.push(...validateCredentialRequest(request));
    setErrors([...new Set(nextErrors)]);
    if (nextErrors.length === 0) {
      mutation.mutate(request);
    }
  }

  function togglePolicy(key: string, checked: boolean): void {
    setPolicyKeys((current) =>
      checked
        ? [...current, key]
        : current.filter((candidate) => candidate !== key),
    );
    setErrors([]);
  }

  const inventoryError = gateways.error ?? policies.error;
  return (
    <form
      className="ops-draft credential-create-form"
      onSubmit={submit}
      noValidate
    >
      <div className="ops-draft-head">
        <h3 className="ops-section-title">Create credential</h3>
        <ScopeChip>active when created</ScopeChip>
      </div>
      <p className="ui-caption">
        Server asks the selected LiteLLM manager to generate a key, encrypts it,
        and never returns it to this browser. Replacing access means a new ID,
        not an in-place update or rotation.
      </p>
      {inventoryError === null ? null : <ErrorNotice error={inventoryError} />}
      <div className="form-grid">
        <Field label="Credential ID">
          {(control) => (
            <input
              {...control}
              name="credentialId"
              required
              maxLength={CREDENTIAL_ID.maxLength}
              value={credentialId}
              onChange={(event) => {
                setCredentialId(event.target.value);
                setErrors([]);
              }}
            />
          )}
        </Field>
        <Field label="Managed LLM Gateway">
          {(control) => (
            <select
              {...control}
              required
              value={gatewayKey}
              onChange={(event) => {
                setGatewayKey(event.target.value);
                setErrors([]);
              }}
            >
              <option value="">Select Gateway</option>
              {managedGateways.map((resource) => (
                <option key={configKey(resource)} value={configKey(resource)}>
                  {resource.ref.name}@{resource.ref.version} · {resource.source}
                </option>
              ))}
            </select>
          )}
        </Field>
      </div>
      <LoadMoreButton query={gateways} label="Load more Gateways" />
      <fieldset className="ops-checklist">
        <legend>Allowed ModelPolicies</legend>
        {modelPolicies.length === 0 ? (
          <p className="ui-caption">
            No ModelPolicy is available in loaded pages.
          </p>
        ) : (
          modelPolicies.map((resource) => {
            const key = configKey(resource);
            return (
              <label className="checkbox-label" key={key}>
                <input
                  type="checkbox"
                  checked={policyKeys.includes(key)}
                  onChange={(event) => togglePolicy(key, event.target.checked)}
                />
                <span>
                  {resource.ref.name}@{resource.ref.version} ·{" "}
                  <code>{compactDigest(resource.ref.digest)}</code>
                </span>
              </label>
            );
          })
        )}
      </fieldset>
      <LoadMoreButton query={policies} label="Load more ModelPolicies" />
      <details className="optional-settings">
        <summary>Optional settings</summary>
        <div>
          <div className="form-grid">
            <Field label="Safe label" optional>
              {(control) => (
                <input
                  {...control}
                  name="label"
                  maxLength={LABEL.maxLength}
                  value={label}
                  onChange={(event) => {
                    setLabel(event.target.value);
                    setErrors([]);
                  }}
                />
              )}
            </Field>
            <Field label="Maximum spend · Gateway-enforced (LiteLLM)">
              {(control) => (
                <input
                  {...control}
                  type="number"
                  min={0}
                  step="any"
                  value={maxBudget}
                  onChange={(event) => {
                    setMaxBudget(event.target.value);
                    setErrors([]);
                  }}
                />
              )}
            </Field>
            <Field label="Budget reset · Gateway-enforced (LiteLLM)">
              {(control) => (
                <input
                  {...control}
                  placeholder="1d"
                  maxLength={BUDGET_DURATION_PATTERN.maxLength}
                  value={budgetDuration}
                  onChange={(event) => {
                    setBudgetDuration(event.target.value);
                    setErrors([]);
                  }}
                />
              )}
            </Field>
            {(
              [
                ["tpmLimit", "TPM limit", tpmLimit, setTPMLimit],
                ["rpmLimit", "RPM limit", rpmLimit, setRPMLimit],
                [
                  "maxParallelRequests",
                  "Parallel requests",
                  parallelLimit,
                  setParallelLimit,
                ],
              ] as const
            ).map(([name, title, value, setValue]) => {
              const { minimum, maximum } = integerBounds(name);
              return (
                <Field
                  key={name}
                  label={`${title} · Gateway-enforced (LiteLLM)`}
                >
                  {(control) => (
                    <input
                      {...control}
                      type="number"
                      min={minimum}
                      max={maximum}
                      step={1}
                      value={value}
                      onChange={(event) => {
                        setValue(event.target.value);
                        setErrors([]);
                      }}
                    />
                  )}
                </Field>
              );
            })}
          </div>
        </div>
      </details>
      <PublicationFeedback
        title="Credential request is not valid"
        errors={errors}
        mutationError={mutation.error}
      />
      <div className="ops-form-actions">
        <Button
          variant="primary"
          type="submit"
          disabled={inventoryError !== null}
          pending={mutation.isPending}
          pendingLabel="Creating…"
        >
          Create active credential
        </Button>
      </div>
    </form>
  );
}
