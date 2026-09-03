import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  MANAGED_CONFIG_NAME_PATTERN,
  publishConfiguration,
  type ConfigurationResource,
  type LLMGatewayBody,
  type ModelPolicyBody,
  type PublishConfigurationRequest,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { CONFIG_VERSION_PATTERN } from "../../../api/workflows";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice } from "../../artifacts/common";
import {
  llmGatewayBody,
  modelPolicyBody,
  validateLLMGateway,
  validateModelPolicy,
  type ModelPolicyConsumer,
} from "./model";

interface PublicationFormProps {
  source: ConfigurationResource;
  onPublished: (resource: ConfigurationResource) => void;
}

interface IdentityDraft {
  name: string;
  version: string;
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
      <label>
        New managed name
        <input
          name="name"
          required
          maxLength={128}
          value={value.name}
          onChange={(event) => onChange({ ...value, name: event.target.value })}
        />
      </label>
      <label>
        New immutable version
        <input
          name="version"
          required
          maxLength={64}
          placeholder="2"
          value={value.version}
          onChange={(event) =>
            onChange({ ...value, version: event.target.value })
          }
        />
      </label>
    </>
  );
}

function PublicationFeedback({
  errors,
  mutationError,
}: {
  errors: string[];
  mutationError: unknown;
}) {
  return (
    <>
      {errors.length === 0 ? null : (
        <div className="notice notice-error" role="alert">
          <strong>Draft is not publishable</strong>
          <ul>
            {errors.map((error) => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      )}
      {mutationError === null ? null : <ErrorNotice error={mutationError} />}
    </>
  );
}

function identityErrors(identity: IdentityDraft): string[] {
  const errors: string[] = [];
  if (!MANAGED_CONFIG_NAME_PATTERN.test(identity.name)) {
    errors.push(
      "Managed name must contain 1–128 letters, digits, dot, dash, or underscore.",
    );
  }
  if (!CONFIG_VERSION_PATTERN.test(identity.version)) {
    errors.push(
      "Version must contain 1–64 letters, digits, dot, dash, or underscore.",
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
    errors.push("Clone must use a new name or immutable version.");
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

function NumberDraftField({
  label,
  name,
  value,
  maximum,
  integer = true,
  onChange,
}: {
  label: string;
  name: keyof ModelDraft;
  value: string;
  maximum?: number;
  integer?: boolean;
  onChange: (name: keyof ModelDraft, value: string) => void;
}) {
  return (
    <label>
      {label}
      <input
        name={name}
        type="number"
        min={integer ? 1 : 0}
        {...(maximum === undefined ? {} : { max: maximum })}
        step={integer ? 1 : "any"}
        value={value}
        onChange={(event) => onChange(name, event.target.value)}
      />
    </label>
  );
}

export function ModelPolicyPublicationForm({
  source,
  onPublished,
}: PublicationFormProps) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<PublishConfigurationRequest>("publish-config"),
  );
  const [identity, setIdentity] = useState<IdentityDraft>({
    name: source.ref.name,
    version: "",
  });
  const [consumer, setConsumer] = useState<ModelPolicyConsumer>("worker");
  const [draft, setDraft] = useState(() => modelDraft(modelPolicyBody(source)));
  const [errors, setErrors] = useState<string[]>([]);
  const mutation = useMutation({
    mutationFn: (request: PublishConfigurationRequest) =>
      publishConfiguration(
        api,
        "model-policies",
        request,
        keyring.keyFor(request),
      ),
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
    <form className="configuration-draft" onSubmit={submit} noValidate>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Clone-to-draft</p>
          <h3>Publish a new ModelPolicy</h3>
        </div>
        <code>
          from {source.ref.name}@{source.ref.version}
        </code>
      </div>
      <p className="muted-copy">
        Intended consumer is validation guidance for this draft and is not
        stored in the shared ModelPolicy body.
      </p>
      <div className="form-grid">
        <IdentityFields
          value={identity}
          onChange={(value) => {
            setIdentity(value);
            setErrors([]);
          }}
        />
        <label>
          Intended consumer
          <select
            value={consumer}
            onChange={(event) =>
              setConsumer(event.target.value as ModelPolicyConsumer)
            }
          >
            <option value="worker">Worker</option>
            <option value="planner">Planner</option>
            <option value="both">Planner and Worker</option>
          </select>
        </label>
        <label>
          Gateway model alias
          <input
            name="model"
            required
            maxLength={256}
            value={draft.model}
            onChange={(event) => change("model", event.target.value)}
          />
        </label>
        <NumberDraftField
          label="Context window tokens"
          name="contextWindowTokens"
          value={draft.contextWindowTokens}
          maximum={100_000_000}
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
          maximum={1_000}
          onChange={change}
        />
        <NumberDraftField
          label="Maximum tool calls"
          name="maxToolCalls"
          value={draft.maxToolCalls}
          maximum={10_000}
          onChange={change}
        />
        <NumberDraftField
          label="Maximum Worker calls"
          name="maxWorkerCalls"
          value={draft.maxWorkerCalls}
          maximum={10_000}
          onChange={change}
        />
        <NumberDraftField
          label="Maximum total tokens"
          name="maxTotalTokens"
          value={draft.maxTotalTokens}
          maximum={100_000_000}
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
      <PublicationFeedback errors={errors} mutationError={mutation.error} />
      <div className="run-submit-row">
        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "Publishing…" : "Publish immutable version"}
        </button>
        <small>The existing version and digest are never edited.</small>
      </div>
    </form>
  );
}

export function LLMGatewayPublicationForm({
  source,
  onPublished,
}: PublicationFormProps) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const sourceBody = llmGatewayBody(source);
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<PublishConfigurationRequest>("publish-config"),
  );
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
  const mutation = useMutation({
    mutationFn: (request: PublishConfigurationRequest) =>
      publishConfiguration(
        api,
        "llm-gateways",
        request,
        keyring.keyFor(request),
      ),
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
    <form className="configuration-draft" onSubmit={submit} noValidate>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Clone-to-draft</p>
          <h3>Publish a new LLMGatewayConfig</h3>
        </div>
        <code>
          from {source.ref.name}@{source.ref.version}
        </code>
      </div>
      <div className="form-grid">
        <IdentityFields
          value={identity}
          onChange={(value) => {
            setIdentity(value);
            setErrors([]);
          }}
        />
        <label>
          Protocol
          <input value="openai-compatible@1" disabled readOnly />
        </label>
        <label>
          Inference URL with explicit path
          <input
            name="url"
            required
            maxLength={2048}
            value={url}
            onChange={(event) => {
              setURL(event.target.value);
              setErrors([]);
            }}
          />
        </label>
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
          <label>
            LiteLLM management root URL
            <input
              name="managementUrl"
              required
              maxLength={2048}
              value={managementURL}
              onChange={(event) => {
                setManagementURL(event.target.value);
                setErrors([]);
              }}
            />
          </label>
        ) : null}
      </div>
      <p className="field-guidance">
        Gateway URLs never contain credentials. HTTP management is accepted only
        for an IP-literal loopback origin; production management uses HTTPS.
      </p>
      <PublicationFeedback errors={errors} mutationError={mutation.error} />
      <div className="run-submit-row">
        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "Publishing…" : "Publish immutable version"}
        </button>
        <small>
          The Server validates the complete configuration union before
          publication.
        </small>
      </div>
    </form>
  );
}
