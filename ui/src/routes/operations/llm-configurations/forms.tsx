import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

import { CONFIG_VERSION, CONFIGURATION_NAME } from "../../../api/contract";
import { usePublicAPI } from "../../../api/context";
import {
  publishConfiguration,
  type ConfigurationResource,
  type LLMGatewayBody,
  type ModelPolicyBody,
  type PublishConfigurationRequest,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { Button, Field } from "../../../ui";
import {
  GATEWAY_URL_LENGTH,
  llmGatewayBody,
  MANAGEMENT_URL_LENGTH,
  modelPolicyBody,
  policyLimit,
  validateLLMGateway,
  validateModelPolicy,
  type ModelPolicyConsumer,
} from "./model";
import { PublicationFeedback } from "../common";

interface PublicationFormProps {
  source: ConfigurationResource;
  onPublished: (resource: ConfigurationResource) => void;
}

interface IdentityDraft {
  name: string;
  version: string;
}

/**
 * Publishes one configuration version, reusing the idempotency key while the
 * request is unchanged, and refreshes the catalog and Operations snapshot.
 */
function usePublishConfiguration(
  kind: "model-policies" | "llm-gateways",
  onPublished: (resource: ConfigurationResource) => void,
) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<PublishConfigurationRequest>("publish-config"),
  );
  return useMutation({
    mutationFn: (request: PublishConfigurationRequest) =>
      publishConfiguration(api, kind, request, keyring.keyFor(request)),
    onSuccess: async (resource) => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.configurations.all,
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.snapshot,
        }),
      ]);
      onPublished(resource);
    },
    onError: async () => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.configurations.all,
      });
    },
  });
}

function IdentityFields({
  value,
  onChange,
}: {
  value: IdentityDraft;
  onChange: (next: IdentityDraft) => void;
}) {
  return (
    <>
      <Field label="New managed name">
        {(control) => (
          <input
            {...control}
            name="name"
            required
            maxLength={CONFIGURATION_NAME.maxLength}
            value={value.name}
            onChange={(event) =>
              onChange({ ...value, name: event.target.value })
            }
          />
        )}
      </Field>
      <Field label="New version">
        {(control) => (
          <input
            {...control}
            name="version"
            required
            maxLength={CONFIG_VERSION.maxLength}
            placeholder="2"
            value={value.version}
            onChange={(event) =>
              onChange({ ...value, version: event.target.value })
            }
          />
        )}
      </Field>
    </>
  );
}

function identityErrors(identity: IdentityDraft): string[] {
  const errors: string[] = [];
  if (!CONFIGURATION_NAME.test(identity.name)) {
    errors.push(
      `Managed name must start with a lowercase letter and contain at most ${CONFIGURATION_NAME.maxLength} lowercase letters, digits, dashes, or underscores.`,
    );
  }
  if (!CONFIG_VERSION.test(identity.version)) {
    errors.push(
      `Version must start with a letter or digit and contain at most ${CONFIG_VERSION.maxLength} letters, digits, dots, plus signs, dashes, or underscores.`,
    );
  }
  return errors;
}

function cloneIdentityErrors(
  identity: IdentityDraft,
  source: ConfigurationResource,
): string[] {
  const errors = identityErrors(identity);
  if (
    identity.name === source.ref.name &&
    identity.version === source.ref.version
  ) {
    errors.push("Clone must use a new name or version.");
  }
  return errors;
}

interface ModelDraft {
  model: string;
  contextWindowTokens: string;
  maxOutputTokens: string;
  maxModelCalls: string;
  maxToolCalls: string;
  maxWorkerCalls: string;
  maxTotalTokens: string;
  temperature: string;
}

function modelDraft(source: ModelPolicyBody): ModelDraft {
  const text = (value: number | undefined) => value?.toString() ?? "";
  return {
    model: source.model,
    contextWindowTokens: text(source.contextWindowTokens),
    maxOutputTokens: text(source.maxOutputTokens),
    maxModelCalls: text(source.maxModelCalls),
    maxToolCalls: text(source.maxToolCalls),
    maxWorkerCalls: text(source.maxWorkerCalls),
    maxTotalTokens: text(source.maxTotalTokens),
    temperature: text(source.temperature),
  };
}

function optionalNumber(value: string): number | undefined {
  return value === "" ? undefined : Number(value);
}

function modelBody(draft: ModelDraft): ModelPolicyBody {
  const optional = <K extends keyof ModelPolicyBody>(
    key: K,
    value: string,
  ): Partial<ModelPolicyBody> => {
    const parsed = optionalNumber(value);
    return parsed === undefined
      ? {}
      : ({ [key]: parsed } as Partial<ModelPolicyBody>);
  };
  return {
    model: draft.model,
    ...optional("contextWindowTokens", draft.contextWindowTokens),
    ...optional("maxOutputTokens", draft.maxOutputTokens),
    ...optional("maxModelCalls", draft.maxModelCalls),
    ...optional("maxToolCalls", draft.maxToolCalls),
    ...optional("maxWorkerCalls", draft.maxWorkerCalls),
    ...optional("maxTotalTokens", draft.maxTotalTokens),
    ...optional("temperature", draft.temperature),
  };
}

/** A numeric ModelPolicy field; integer fields take the contract's maximum. */
function NumberDraftField({
  label,
  name,
  value,
  integer = true,
  onChange,
}: {
  label: string;
  name: Exclude<keyof ModelDraft, "model">;
  value: string;
  integer?: boolean;
  onChange: (name: keyof ModelDraft, value: string) => void;
}) {
  return (
    <Field label={label}>
      {(control) => (
        <input
          {...control}
          name={name}
          type="number"
          min={integer ? 1 : 0}
          {...(integer ? { max: policyLimit("maximum", name) } : {})}
          step={integer ? 1 : "any"}
          value={value}
          onChange={(event) => onChange(name, event.target.value)}
        />
      )}
    </Field>
  );
}

export function ModelPolicyPublicationForm({
  source,
  onPublished,
}: PublicationFormProps) {
  const [identity, setIdentity] = useState<IdentityDraft>({
    name: source.ref.name,
    version: "",
  });
  const [consumer, setConsumer] = useState<ModelPolicyConsumer>("worker");
  const [draft, setDraft] = useState(() => modelDraft(modelPolicyBody(source)));
  const [errors, setErrors] = useState<string[]>([]);
  const mutation = usePublishConfiguration("model-policies", onPublished);

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    mutation.reset();
    const body = modelBody(draft);
    const nextErrors = [
      ...cloneIdentityErrors(identity, source),
      ...validateModelPolicy(body, consumer),
    ];
    setErrors(nextErrors);
    if (nextErrors.length !== 0) {
      return;
    }
    mutation.mutate({
      name: identity.name,
      version: identity.version,
      modelPolicy: body,
    });
  }

  const change = (name: keyof ModelDraft, value: string) => {
    setDraft((current) => ({ ...current, [name]: value }));
    setErrors([]);
  };
  return (
    <form className="ops-draft" onSubmit={submit} noValidate>
      <div className="ops-draft-head">
        <div>
          <p className="ops-eyebrow">Clone-to-draft</p>
          <h3 className="ops-section-title">Publish a new ModelPolicy</h3>
        </div>
        <code>
          from {source.ref.name}@{source.ref.version}
        </code>
      </div>
      <p className="ui-caption">
        Intended consumer is validation guidance for this draft and is not
        stored in the shared ModelPolicy body.
      </p>
      <div className="ops-form-grid">
        <IdentityFields
          value={identity}
          onChange={(value) => {
            setIdentity(value);
            setErrors([]);
          }}
        />
        <Field label="Intended consumer">
          {(control) => (
            <select
              {...control}
              value={consumer}
              onChange={(event) =>
                setConsumer(event.target.value as ModelPolicyConsumer)
              }
            >
              <option value="worker">Worker</option>
              <option value="planner">Planner</option>
              <option value="both">Planner and Worker</option>
            </select>
          )}
        </Field>
        <Field label="Gateway model alias">
          {(control) => (
            <input
              {...control}
              name="model"
              required
              maxLength={policyLimit("maxLength", "model")}
              value={draft.model}
              onChange={(event) => change("model", event.target.value)}
            />
          )}
        </Field>
        <NumberDraftField
          label="Context window tokens"
          name="contextWindowTokens"
          value={draft.contextWindowTokens}
          onChange={change}
        />
        <NumberDraftField
          label="Maximum output tokens"
          name="maxOutputTokens"
          value={draft.maxOutputTokens}
          onChange={change}
        />
        <NumberDraftField
          label="Maximum model calls"
          name="maxModelCalls"
          value={draft.maxModelCalls}
          onChange={change}
        />
        <NumberDraftField
          label="Maximum tool calls"
          name="maxToolCalls"
          value={draft.maxToolCalls}
          onChange={change}
        />
        <NumberDraftField
          label="Maximum Worker calls"
          name="maxWorkerCalls"
          value={draft.maxWorkerCalls}
          onChange={change}
        />
        <NumberDraftField
          label="Maximum total tokens"
          name="maxTotalTokens"
          value={draft.maxTotalTokens}
          onChange={change}
        />
        <NumberDraftField
          label="Temperature"
          name="temperature"
          value={draft.temperature}
          integer={false}
          onChange={change}
        />
      </div>
      <PublicationFeedback
        title="Draft is not publishable"
        errors={errors}
        mutationError={mutation.error}
      />
      <div className="ops-form-actions">
        <Button
          variant="primary"
          type="submit"
          pending={mutation.isPending}
          pendingLabel="Publishing…"
        >
          Publish version
        </Button>
        <small>The existing version and digest are never edited.</small>
      </div>
    </form>
  );
}

export function LLMGatewayPublicationForm({
  source,
  onPublished,
}: PublicationFormProps) {
  const sourceBody = llmGatewayBody(source);
  const [identity, setIdentity] = useState<IdentityDraft>({
    name: source.ref.name,
    version: "",
  });
  const [url, setURL] = useState(sourceBody.url);
  const [managed, setManaged] = useState(
    sourceBody.credentialManager !== undefined,
  );
  const [managementURL, setManagementURL] = useState(
    sourceBody.credentialManager?.managementUrl ?? "",
  );
  const [errors, setErrors] = useState<string[]>([]);
  const mutation = usePublishConfiguration("llm-gateways", onPublished);

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    mutation.reset();
    const body: LLMGatewayBody = {
      protocol: "openai-compatible@1",
      url,
      ...(managed
        ? {
            credentialManager: {
              implementation: "litellm-virtual-keys@1",
              managementUrl: managementURL,
            },
          }
        : {}),
      // Provider failure signatures are declared in the catalog YAML; a cloned
      // version keeps the source declaration rather than silently dropping it.
      ...(sourceBody.failureSignatures === undefined
        ? {}
        : { failureSignatures: sourceBody.failureSignatures }),
    };
    const nextErrors = [
      ...cloneIdentityErrors(identity, source),
      ...validateLLMGateway(body),
    ];
    setErrors(nextErrors);
    if (nextErrors.length !== 0) {
      return;
    }
    mutation.mutate({
      name: identity.name,
      version: identity.version,
      llmGateway: body,
    });
  }

  return (
    <form className="ops-draft" onSubmit={submit} noValidate>
      <div className="ops-draft-head">
        <div>
          <p className="ops-eyebrow">Clone-to-draft</p>
          <h3 className="ops-section-title">Publish a new LLMGatewayConfig</h3>
        </div>
        <code>
          from {source.ref.name}@{source.ref.version}
        </code>
      </div>
      <div className="ops-form-grid">
        <IdentityFields
          value={identity}
          onChange={(value) => {
            setIdentity(value);
            setErrors([]);
          }}
        />
        <Field label="Protocol">
          {(control) => (
            <input {...control} value="openai-compatible@1" disabled readOnly />
          )}
        </Field>
        <Field label="Inference URL with explicit path">
          {(control) => (
            <input
              {...control}
              name="url"
              required
              maxLength={GATEWAY_URL_LENGTH}
              value={url}
              onChange={(event) => {
                setURL(event.target.value);
                setErrors([]);
              }}
            />
          )}
        </Field>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={managed}
            onChange={(event) => {
              setManaged(event.target.checked);
              setErrors([]);
            }}
          />
          Enable LiteLLM virtual-key management
        </label>
        {managed ? (
          <Field label="LiteLLM management root URL">
            {(control) => (
              <input
                {...control}
                name="managementUrl"
                required
                maxLength={MANAGEMENT_URL_LENGTH}
                value={managementURL}
                onChange={(event) => {
                  setManagementURL(event.target.value);
                  setErrors([]);
                }}
              />
            )}
          </Field>
        ) : null}
      </div>
      <p className="ui-caption">
        Gateway URLs never contain credentials. HTTP management is accepted only
        for an IP-literal loopback origin; production management uses HTTPS.
      </p>
      <PublicationFeedback
        title="Draft is not publishable"
        errors={errors}
        mutationError={mutation.error}
      />
      <div className="ops-form-actions">
        <Button
          variant="primary"
          type="submit"
          pending={mutation.isPending}
          pendingLabel="Publishing…"
        >
          Publish version
        </Button>
        <small>
          The Server validates the complete configuration union before
          publication.
        </small>
      </div>
    </form>
  );
}
