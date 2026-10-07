import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import { type FormEvent, useMemo, useState } from "react";
import { useNavigate } from "react-router";

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
import { validateCredentialRequest } from "./validation";
import { compactDigest } from "../../../app/format";
import { PublicationFeedback, ScopeChip } from "../common";

function configKey(resource: ConfigurationResource): string {
  return `${resource.ref.name}@${resource.ref.version}:${resource.ref.digest}`;
}

function optionalNumber(value: string): number | undefined {
  return value === "" ? undefined : Number(value);
}

export function CredentialCreateForm() {
  const api = usePublicAPI();
  const navigate = useNavigate();
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
    if (!MANAGED_CONFIG_NAME_PATTERN.test(credentialId)) {
      nextErrors.push(
        "Credential ID must contain 1–128 letters, digits, dot, dash, or underscore.",
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
      <p className="ops-note">
        Server asks the selected LiteLLM manager to generate a key, encrypts it,
        and never returns it to this browser. Replacing access means a new ID,
        not an in-place update or rotation.
      </p>
      {inventoryError === null ? null : <ErrorNotice error={inventoryError} />}
      <div className="form-grid">
        <label>
          Credential ID
          <input
            name="credentialId"
            required
            maxLength={128}
            value={credentialId}
            onChange={(event) => {
              setCredentialId(event.target.value);
              setErrors([]);
            }}
          />
        </label>
        <label>
          Managed LLM Gateway
          <select
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
        </label>
      </div>
      <LoadMoreButton query={gateways} label="Load more Gateways" />
      <fieldset className="ops-checklist">
        <legend>Allowed ModelPolicies</legend>
        {modelPolicies.length === 0 ? (
          <p className="ops-note">
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
            <label>
              Safe label (optional)
              <input
                name="label"
                maxLength={256}
                value={label}
                onChange={(event) => {
                  setLabel(event.target.value);
                  setErrors([]);
                }}
              />
            </label>
            <label>
              Maximum spend · Gateway-enforced (LiteLLM)
              <input
                type="number"
                min={0}
                step="any"
                value={maxBudget}
                onChange={(event) => {
                  setMaxBudget(event.target.value);
                  setErrors([]);
                }}
              />
            </label>
            <label>
              Budget reset · Gateway-enforced (LiteLLM)
              <input
                placeholder="1d"
                maxLength={32}
                value={budgetDuration}
                onChange={(event) => {
                  setBudgetDuration(event.target.value);
                  setErrors([]);
                }}
              />
            </label>
            <label>
              TPM limit · Gateway-enforced (LiteLLM)
              <input
                type="number"
                min={1}
                max={2_147_483_647}
                step={1}
                value={tpmLimit}
                onChange={(event) => {
                  setTPMLimit(event.target.value);
                  setErrors([]);
                }}
              />
            </label>
            <label>
              RPM limit · Gateway-enforced (LiteLLM)
              <input
                type="number"
                min={1}
                max={2_147_483_647}
                step={1}
                value={rpmLimit}
                onChange={(event) => {
                  setRPMLimit(event.target.value);
                  setErrors([]);
                }}
              />
            </label>
            <label>
              Parallel requests · Gateway-enforced (LiteLLM)
              <input
                type="number"
                min={1}
                max={2_147_483_647}
                step={1}
                value={parallelLimit}
                onChange={(event) => {
                  setParallelLimit(event.target.value);
                  setErrors([]);
                }}
              />
            </label>
          </div>
        </div>
      </details>
      <PublicationFeedback
        title="Credential request is not valid"
        errors={errors}
        mutationError={mutation.error}
      />
      <div className="ops-form-actions">
        <button
          className="ui-btn"
          data-variant="primary"
          type="submit"
          disabled={mutation.isPending || inventoryError !== null}
        >
          {mutation.isPending ? "Creating…" : "Create active credential"}
        </button>
      </div>
    </form>
  );
}
