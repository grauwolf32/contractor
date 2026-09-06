import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useRef, useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import {
  createRuntimeCredential,
  deleteRuntimeCredential,
  deleteRuntimeLabel,
  listConfigurations,
  listRuntimeConfigs,
  listRuntimeCredentials,
  listRuntimeLabels,
  publishRuntimeConfig,
  putRuntimeLabel,
  type ConfigurationResource,
  type CreateRuntimeCredentialRequest,
  type RuntimeConfigDocument,
  type RuntimeConfigResource,
  type RuntimeCredentialMetadata,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../../artifacts/common";

const RUNTIME_ID = /^[a-z][a-z0-9_-]{0,62}$/;
const RUNTIME_VERSION = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;

function exactKey(resource: RuntimeConfigResource): string {
  return `${resource.ref.name}@${resource.ref.version}:${resource.ref.digest}`;
}

function gatewayKey(resource: ConfigurationResource): string {
  return `${resource.ref.name}@${resource.ref.version}:${resource.ref.digest}`;
}

function RuntimeRef({ resource }: { resource: RuntimeConfigResource }) {
  return (
    <span className="exact-config-ref">
      <Link
        to={`/runs/configuration/${encodeURIComponent(resource.ref.name)}/${encodeURIComponent(resource.ref.version)}`}
      >
        {resource.ref.name}@{resource.ref.version}
      </Link>
      <code title={resource.ref.digest}>
        {resource.ref.digest.slice(0, 18)}…
      </code>
    </span>
  );
}

interface ConfigDraft {
  name: string;
  version: string;
  gateway: boolean;
  gatewayKey: string;
  llmCredential: string;
  workerTelemetry: boolean;
  workerTelemetryEndpoint: string;
  workerTelemetryCredential: string;
  workerTelemetryTimeout: string;
  workerTelemetryCaptureContent: boolean;
  plannerTelemetry: boolean;
  plannerTelemetryEndpoint: string;
  plannerTelemetryCredential: string;
  plannerTelemetryTimeout: string;
  plannerTelemetryCaptureContent: boolean;
  httpProxy: boolean;
  proxyURL: string;
  proxyCredential: string;
  proxyCA: string;
  proxyTargets: Array<"llm-gateway" | "tool-http" | "tool-subprocess">;
}

const emptyConfigDraft = (): ConfigDraft => ({
  name: "",
  version: "1",
  gateway: false,
  gatewayKey: "",
  llmCredential: "",
  workerTelemetry: false,
  workerTelemetryEndpoint: "",
  workerTelemetryCredential: "",
  workerTelemetryTimeout: "5",
  workerTelemetryCaptureContent: false,
  plannerTelemetry: false,
  plannerTelemetryEndpoint: "",
  plannerTelemetryCredential: "",
  plannerTelemetryTimeout: "5",
  plannerTelemetryCaptureContent: false,
  httpProxy: false,
  proxyURL: "",
  proxyCredential: "",
  proxyCA: "",
  proxyTargets: [],
});

function safeHTTPURL(value: string): boolean {
  try {
    const parsed = new URL(value);
    return (
      (parsed.protocol === "http:" || parsed.protocol === "https:") &&
      parsed.username === "" &&
      parsed.password === "" &&
      parsed.search === "" &&
      parsed.hash === ""
    );
  } catch {
    return false;
  }
}

function telemetry(
  endpoint: string,
  credential: string,
  timeout: string,
  captureContent: boolean,
): NonNullable<
  NonNullable<RuntimeConfigDocument["spec"]["worker"]>["telemetry"]
> {
  return {
    adapter: "otlp-http@1",
    endpoint,
    captureContent,
    ...(credential === "" ? {} : { credential }),
    ...(timeout === "" ? {} : { flushTimeoutSeconds: Number(timeout) }),
  };
}

function buildDocument(
  draft: ConfigDraft,
  gateways: ConfigurationResource[],
): { document?: RuntimeConfigDocument; errors: string[] } {
  const errors: string[] = [];
  if (!RUNTIME_ID.test(draft.name)) {
    errors.push(
      "Name must match [a-z][a-z0-9_-]* and contain at most 63 characters.",
    );
  }
  if (!RUNTIME_VERSION.test(draft.version)) {
    errors.push("Version must be a non-empty immutable selector component.");
  }
  if (
    !draft.gateway &&
    !draft.workerTelemetry &&
    !draft.plannerTelemetry &&
    !draft.httpProxy
  ) {
    errors.push("Select at least one typed RuntimeConfig block.");
  }
  const selectedGateway = gateways.find(
    (resource) => gatewayKey(resource) === draft.gatewayKey,
  );
  if (draft.gateway && selectedGateway === undefined) {
    errors.push("Select one exact published LLM Gateway.");
  }
  for (const [enabled, endpoint, label] of [
    [draft.workerTelemetry, draft.workerTelemetryEndpoint, "Worker telemetry"],
    [
      draft.plannerTelemetry,
      draft.plannerTelemetryEndpoint,
      "Planner telemetry",
    ],
  ] as const) {
    if (enabled && !safeHTTPURL(endpoint)) {
      errors.push(
        `${label} endpoint must be an HTTP(S) URL without userinfo, query or fragment.`,
      );
    }
  }
  for (const [enabled, timeout, label] of [
    [draft.workerTelemetry, draft.workerTelemetryTimeout, "Worker telemetry"],
    [
      draft.plannerTelemetry,
      draft.plannerTelemetryTimeout,
      "Planner telemetry",
    ],
  ] as const) {
    if (
      enabled &&
      timeout !== "" &&
      (!Number.isInteger(Number(timeout)) || Number(timeout) < 1)
    ) {
      errors.push(`${label} flush timeout must be a positive integer.`);
    }
  }
  if (draft.httpProxy && !safeHTTPURL(draft.proxyURL)) {
    errors.push(
      "HTTP proxy URL must be HTTP(S) without userinfo, query or fragment.",
    );
  }
  if (draft.httpProxy && draft.proxyTargets.length === 0) {
    errors.push("HTTP proxy requires at least one explicit traffic target.");
  }
  if (errors.length > 0 || (draft.gateway && selectedGateway === undefined)) {
    return { errors };
  }
  const worker: NonNullable<RuntimeConfigDocument["spec"]["worker"]> = {
    ...(draft.gateway
      ? {
          llmGateway: {
            gateway: {
              gatewayId: selectedGateway!.ref.name,
              version: selectedGateway!.ref.version,
              digest: selectedGateway!.ref.digest,
            },
            ...(draft.llmCredential === ""
              ? {}
              : { credential: draft.llmCredential }),
          },
        }
      : {}),
    ...(draft.workerTelemetry
      ? {
          telemetry: telemetry(
            draft.workerTelemetryEndpoint,
            draft.workerTelemetryCredential,
            draft.workerTelemetryTimeout,
            draft.workerTelemetryCaptureContent,
          ),
        }
      : {}),
    ...(draft.httpProxy
      ? {
          httpProxy: {
            adapter: "http-proxy@1" as const,
            proxyUrl: draft.proxyURL,
            ...(draft.proxyCredential === ""
              ? {}
              : { credential: draft.proxyCredential }),
            ...(draft.proxyCA === "" ? {} : { caBundlePem: draft.proxyCA }),
            targets: [...draft.proxyTargets].sort(),
          },
        }
      : {}),
  };
  return {
    errors,
    document: {
      apiVersion: "contractor/v1alpha1",
      kind: "RuntimeConfig",
      metadata: { name: draft.name, version: draft.version },
      spec: {
        ...(Object.keys(worker).length === 0 ? {} : { worker }),
        ...(draft.plannerTelemetry
          ? {
              planner: {
                telemetry: telemetry(
                  draft.plannerTelemetryEndpoint,
                  draft.plannerTelemetryCredential,
                  draft.plannerTelemetryTimeout,
                  draft.plannerTelemetryCaptureContent,
                ),
              },
            }
          : {}),
      },
    },
  };
}

function RuntimeConfigPublishForm({
  configs,
  gateways,
}: {
  configs: RuntimeConfigResource[];
  gateways: ConfigurationResource[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState(emptyConfigDraft);
  const [errors, setErrors] = useState<string[]>([]);
  const [published, setPublished] = useState<RuntimeConfigResource>();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<RuntimeConfigDocument>("publish-runtime-config"),
  );
  const mutation = useMutation({
    mutationFn: (document: RuntimeConfigDocument) =>
      publishRuntimeConfig(api, document, keyring.keyFor(document)),
    onSuccess: async (resource) => {
      setPublished(resource);
      setDraft(emptyConfigDraft());
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeConfigs.all,
      });
    },
  });

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    mutation.reset();
    setPublished(undefined);
    const result = buildDocument(draft, gateways);
    setErrors(result.errors);
    if (result.document !== undefined) mutation.mutate(result.document);
  }

  function update<K extends keyof ConfigDraft>(key: K, value: ConfigDraft[K]) {
    setDraft((current) => ({ ...current, [key]: value }));
    setErrors([]);
  }

  const targetOptions = [
    ["llm-gateway", "LLM Gateway HTTP"],
    ["tool-http", "Tool HTTP"],
    ["tool-subprocess", "Tool subprocess environment"],
  ] as const;
  return (
    <form className="configuration-draft" onSubmit={submit} noValidate>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Immutable typed infrastructure patch</p>
          <h3>Publish RuntimeConfig</h3>
        </div>
        <span>{configs.length} loaded versions</span>
      </div>
      <p className="muted-copy">
        Values become active only through a label binding. Publishing another
        version never mutates existing Runs or allocations.
      </p>
      <div className="form-grid">
        <label>
          RuntimeConfig name
          <input
            aria-label="RuntimeConfig name"
            required
            maxLength={63}
            value={draft.name}
            onChange={(event) => update("name", event.target.value)}
          />
        </label>
        <label>
          Immutable version
          <input
            aria-label="RuntimeConfig version"
            required
            maxLength={128}
            value={draft.version}
            onChange={(event) => update("version", event.target.value)}
          />
        </label>
      </div>

      <fieldset className="runtime-config-block">
        <legend>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={draft.gateway}
              onChange={(event) => update("gateway", event.target.checked)}
            />
            Worker LLM Gateway route
          </label>
        </legend>
        <div className="form-grid">
          <label>
            Exact LLM Gateway
            <select
              disabled={!draft.gateway}
              value={draft.gatewayKey}
              onChange={(event) => update("gatewayKey", event.target.value)}
            >
              <option value="">Select exact Gateway</option>
              {gateways.map((resource) => (
                <option key={gatewayKey(resource)} value={gatewayKey(resource)}>
                  {resource.ref.name}@{resource.ref.version} · {resource.source}
                </option>
              ))}
            </select>
          </label>
          <label>
            Active LLM credential ID (optional)
            <input
              disabled={!draft.gateway}
              value={draft.llmCredential}
              onChange={(event) => update("llmCredential", event.target.value)}
            />
          </label>
        </div>
      </fieldset>

      {(
        [
          [
            "workerTelemetry",
            "Worker telemetry",
            "workerTelemetryEndpoint",
            "workerTelemetryCredential",
            "workerTelemetryTimeout",
            "workerTelemetryCaptureContent",
          ],
          [
            "plannerTelemetry",
            "Planner telemetry",
            "plannerTelemetryEndpoint",
            "plannerTelemetryCredential",
            "plannerTelemetryTimeout",
            "plannerTelemetryCaptureContent",
          ],
        ] as const
      ).map(
        ([enabled, title, endpoint, credential, timeout, captureContent]) => (
          <fieldset className="runtime-config-block" key={enabled}>
            <legend>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={draft[enabled]}
                  onChange={(event) => update(enabled, event.target.checked)}
                />
                {title} · otlp-http@1
              </label>
            </legend>
            <label className="checkbox-label">
              <input
                type="checkbox"
                disabled={!draft[enabled]}
                checked={draft[captureContent]}
                onChange={(event) =>
                  update(captureContent, event.target.checked)
                }
              />
              Capture content — trust this sink with unredacted prompts,
              responses and tool content, including any secrets they contain.
            </label>
            <div className="form-grid">
              <label>
                OTLP traces endpoint
                <input
                  disabled={!draft[enabled]}
                  placeholder="https://otel.example/v1/traces"
                  value={draft[endpoint]}
                  onChange={(event) => update(endpoint, event.target.value)}
                />
              </label>
              <label>
                Runtime credential ID (optional)
                <input
                  disabled={!draft[enabled]}
                  value={draft[credential]}
                  onChange={(event) => update(credential, event.target.value)}
                />
              </label>
              <label>
                Flush timeout seconds
                <input
                  disabled={!draft[enabled]}
                  type="number"
                  min={1}
                  step={1}
                  value={draft[timeout]}
                  onChange={(event) => update(timeout, event.target.value)}
                />
              </label>
            </div>
          </fieldset>
        ),
      )}

      <fieldset className="runtime-config-block">
        <legend>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={draft.httpProxy}
              onChange={(event) => update("httpProxy", event.target.checked)}
            />
            Worker HTTP proxy · http-proxy@1
          </label>
        </legend>
        <div className="form-grid">
          <label>
            Proxy URL
            <input
              disabled={!draft.httpProxy}
              placeholder="http://caido.internal:8080"
              value={draft.proxyURL}
              onChange={(event) => update("proxyURL", event.target.value)}
            />
          </label>
          <label>
            Runtime credential ID (optional)
            <input
              disabled={!draft.httpProxy}
              value={draft.proxyCredential}
              onChange={(event) =>
                update("proxyCredential", event.target.value)
              }
            />
          </label>
          <label className="runtime-ca-field">
            Additional CA bundle PEM (optional)
            <textarea
              disabled={!draft.httpProxy}
              value={draft.proxyCA}
              onChange={(event) => update("proxyCA", event.target.value)}
            />
          </label>
        </div>
        <div className="runtime-targets">
          {targetOptions.map(([target, label]) => (
            <label className="checkbox-label" key={target}>
              <input
                type="checkbox"
                disabled={!draft.httpProxy}
                checked={draft.proxyTargets.includes(target)}
                onChange={(event) =>
                  update(
                    "proxyTargets",
                    event.target.checked
                      ? [...draft.proxyTargets, target]
                      : draft.proxyTargets.filter((value) => value !== target),
                  )
                }
              />
              {label}
            </label>
          ))}
        </div>
      </fieldset>

      {errors.length === 0 ? null : (
        <div className="notice notice-error" role="alert">
          <strong>RuntimeConfig draft is not publishable</strong>
          <ul>
            {errors.map((error) => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      )}
      {mutation.error === null ? null : <ErrorNotice error={mutation.error} />}
      {published === undefined ? null : (
        <div className="notice notice-success" role="status">
          Published {published.ref.name}@{published.ref.version}. Bind a label
          explicitly before it affects future resolution.
        </div>
      )}
      <button type="submit" disabled={mutation.isPending}>
        {mutation.isPending ? "Publishing…" : "Publish immutable RuntimeConfig"}
      </button>
    </form>
  );
}

function BindingEditor({
  binding,
  configs,
}: {
  binding: RuntimeLabelBinding;
  configs: RuntimeConfigResource[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState(
    configs.find((resource) => resource.ref.digest === binding.config.digest)
      ? exactKey(
          configs.find(
            (resource) => resource.ref.digest === binding.config.digest,
          )!,
        )
      : "",
  );
  const [stale, setStale] = useState(false);
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{
        label: string;
        config: string;
        revision: string;
      }>("bind-runtime-label"),
  );
  const mutation = useMutation({
    mutationFn: (resource: RuntimeConfigResource) => {
      const identity = {
        label: binding.label,
        config: exactKey(resource),
        revision: binding.revision,
      };
      return putRuntimeLabel(
        api,
        binding.label,
        resource.ref,
        keyring.keyFor(identity),
        binding.revision,
      );
    },
    onSuccess: async () => {
      setStale(false);
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeLabels.all,
      });
    },
    onError: async (error) => {
      const conflict = error instanceof PublicAPIError && error.status === 412;
      setStale(conflict);
      if (!conflict) {
        await queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeLabels.all,
        });
      }
    },
  });
  const deletion = useMutation({
    mutationFn: () =>
      deleteRuntimeLabel(
        api,
        binding.label,
        binding.revision,
        `delete-runtime-label-ui-${crypto.randomUUID()}`,
      ),
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeLabels.all,
      });
    },
    onError: async (error) => {
      const conflict = error instanceof PublicAPIError && error.status === 412;
      setStale(conflict);
      if (!conflict) {
        await queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeLabels.all,
        });
      }
    },
  });
  const selectedResource = configs.find(
    (resource) => exactKey(resource) === selected,
  );
  return (
    <article
      className={`runtime-binding-card ${binding.label === "default" ? "runtime-default-binding" : ""}`}
    >
      <div>
        <strong>{binding.label}</strong>
        {binding.label === "default" ? <span>always applied</span> : null}
        <small>
          revision {binding.revision} · future resolution boundaries
        </small>
      </div>
      <select
        aria-label={`RuntimeConfig for ${binding.label}`}
        value={selected}
        onChange={(event) => {
          setSelected(event.target.value);
          setStale(false);
          mutation.reset();
        }}
      >
        <option value="">Current version is outside this loaded page</option>
        {configs.map((resource) => (
          <option key={exactKey(resource)} value={exactKey(resource)}>
            {resource.ref.name}@{resource.ref.version} ·{" "}
            {resource.ref.digest.slice(0, 18)}…
          </option>
        ))}
      </select>
      <div className="runtime-binding-actions">
        <button
          type="button"
          disabled={selectedResource === undefined || mutation.isPending}
          onClick={() =>
            selectedResource === undefined
              ? undefined
              : mutation.mutate(selectedResource)
          }
        >
          {mutation.isPending ? "Rebinding…" : "Rebind with current revision"}
        </button>
        {binding.label === "default" ? null : (
          <button
            className="danger-button"
            type="button"
            disabled={deletion.isPending}
            onClick={() => deletion.mutate()}
          >
            Remove binding
          </button>
        )}
      </div>
      {stale ? (
        <div className="notice notice-warning" role="alert">
          <strong>Binding changed in another view.</strong>
          <p>
            Reload the authoritative revision and review it before retrying.
          </p>
          <button
            className="secondary-button"
            type="button"
            onClick={() =>
              void queryClient.invalidateQueries({
                queryKey: queryKeys.operations.runtimeLabels.all,
              })
            }
          >
            Reload authoritative binding
          </button>
        </div>
      ) : mutation.error === null && deletion.error === null ? null : (
        <ErrorNotice error={mutation.error ?? deletion.error} />
      )}
    </article>
  );
}

function RuntimeBindings({
  bindings,
  configs,
}: {
  bindings: RuntimeLabelBinding[];
  configs: RuntimeConfigResource[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [label, setLabel] = useState("");
  const [selected, setSelected] = useState("");
  const [error, setError] = useState<string>();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{ label: string; config: string }>(
        "create-runtime-label",
      ),
  );
  const mutation = useMutation({
    mutationFn: ({
      name,
      resource,
    }: {
      name: string;
      resource: RuntimeConfigResource;
    }) =>
      putRuntimeLabel(
        api,
        name,
        resource.ref,
        keyring.keyFor({ label: name, config: exactKey(resource) }),
      ),
    onSuccess: async () => {
      setLabel("");
      setSelected("");
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeLabels.all,
      });
    },
  });
  const ordered = [...bindings].sort((left, right) => {
    if (left.label === "default") return -1;
    if (right.label === "default") return 1;
    return left.label.localeCompare(right.label);
  });
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.reset();
    const resource = configs.find(
      (candidate) => exactKey(candidate) === selected,
    );
    if (!RUNTIME_ID.test(label) || label === "default") {
      setError("Label must match [a-z][a-z0-9_-]* and cannot be default.");
      return;
    }
    if (resource === undefined) {
      setError("Select an exact RuntimeConfig version.");
      return;
    }
    setError(undefined);
    mutation.mutate({ name: label, resource });
  }
  return (
    <div className="panel operations-library runtime-bindings">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Revisioned Control Plane pointers</p>
          <h3>Runtime label bindings</h3>
          <p className="muted-copy">
            CAS updates affect only future Run creation or allocation pinning.
            Active allocations retain their exact snapshot.
          </p>
        </div>
      </div>
      <div className="runtime-binding-list">
        {ordered.map((binding) => (
          <BindingEditor
            key={`${binding.label}:${binding.revision}`}
            binding={binding}
            configs={configs}
          />
        ))}
      </div>
      <form
        className="inline-form runtime-label-create"
        onSubmit={submit}
        noValidate
      >
        <label>
          New label
          <input
            value={label}
            onChange={(event) => {
              setLabel(event.target.value);
              setError(undefined);
            }}
          />
        </label>
        <label>
          Exact RuntimeConfig
          <select
            value={selected}
            onChange={(event) => {
              setSelected(event.target.value);
              setError(undefined);
            }}
          >
            <option value="">Select version</option>
            {configs.map((resource) => (
              <option key={exactKey(resource)} value={exactKey(resource)}>
                {resource.ref.name}@{resource.ref.version}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "Binding…" : "Create binding"}
        </button>
      </form>
      {error === undefined ? null : (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
      {mutation.error === null ? null : <ErrorNotice error={mutation.error} />}
    </div>
  );
}

interface HeaderDraft {
  id: number;
  name: string;
  value: string;
}

function RuntimeCredentialCreateForm() {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const nextHeaderID = useRef(2);
  const [credentialId, setCredentialId] = useState("");
  const [kind, setKind] =
    useState<CreateRuntimeCredentialRequest["kind"]>("otlp-headers@1");
  const [headers, setHeaders] = useState<HeaderDraft[]>([
    { id: 1, name: "Authorization", value: "" },
  ]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [created, setCreated] = useState<RuntimeCredentialMetadata>();

  function clearSecretFields() {
    setHeaders((current) =>
      current.map((header) => ({ ...header, value: "" })),
    );
    setUsername("");
    setPassword("");
    setToken("");
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setCreated(undefined);
    if (!/^[a-z][a-z0-9_-]{0,127}$/.test(credentialId)) {
      setError(new Error("Credential ID must match [a-z][a-z0-9_-]*."));
      clearSecretFields();
      return;
    }
    let request: CreateRuntimeCredentialRequest | undefined;
    if (kind === "otlp-headers@1") {
      const material = Object.fromEntries(
        headers.map((header) => [header.name.trim(), header.value]),
      );
      if (
        headers.length === 0 ||
        Object.keys(material).length !== headers.length ||
        Object.entries(material).some(
          ([name, value]) => name === "" || value === "",
        )
      ) {
        setError(
          new Error("Provide unique non-empty OTLP header names and values."),
        );
      } else {
        request = { credentialId, kind, material: { headers: material } };
      }
    } else if (
      kind === "http-proxy-basic@1" ||
      kind === "http-origin-basic@1"
    ) {
      if (username === "" || password === "") {
        setError(new Error("Username and password are required."));
      } else {
        request = { credentialId, kind, material: { username, password } };
      }
    } else if (token === "") {
      setError(new Error("Bearer token is required."));
    } else {
      request = { credentialId, kind, material: { token } };
    }
    clearSecretFields();
    if (request === undefined) return;
    setPending(true);
    try {
      const result = await createRuntimeCredential(
        api,
        request,
        `create-runtime-credential-ui-${crypto.randomUUID()}`,
      );
      setCreated(result);
      setCredentialId("");
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeCredentials.all,
      });
    } catch (reason) {
      setError(reason);
    } finally {
      setPending(false);
    }
  }

  return (
    <form
      className="configuration-draft runtime-credential-form"
      onSubmit={(event) => void submit(event)}
      noValidate
      autoComplete="off"
    >
      <div className="section-heading">
        <div>
          <p className="eyebrow">Write-only encrypted material</p>
          <h3>Create Runtime credential</h3>
        </div>
        <span>active when present</span>
      </div>
      <p className="muted-copy">
        Secret values are erased from the form immediately on submit and are
        never readable through the API. A failed request requires re-entry.
      </p>
      <div className="form-grid">
        <label>
          Runtime credential ID
          <input
            aria-label="Runtime credential ID"
            autoComplete="off"
            value={credentialId}
            onChange={(event) => setCredentialId(event.target.value)}
          />
        </label>
        <label>
          Credential kind
          <select
            value={kind}
            onChange={(event) => {
              setKind(
                event.target.value as CreateRuntimeCredentialRequest["kind"],
              );
              clearSecretFields();
            }}
          >
            <option value="otlp-headers@1">OTLP headers</option>
            <option value="http-proxy-basic@1">HTTP proxy basic auth</option>
            <option value="http-proxy-bearer@1">HTTP proxy bearer token</option>
            <option value="caido-bearer@1">Caido bearer token</option>
            <option value="http-origin-basic@1">HTTP origin basic auth</option>
            <option value="http-origin-bearer@1">
              HTTP origin bearer token
            </option>
          </select>
        </label>
      </div>
      {kind === "otlp-headers@1" ? (
        <div className="runtime-secret-rows">
          {headers.map((header) => (
            <div className="runtime-secret-row" key={header.id}>
              <label>
                Header name
                <input
                  autoComplete="off"
                  value={header.name}
                  onChange={(event) =>
                    setHeaders((current) =>
                      current.map((candidate) =>
                        candidate.id === header.id
                          ? { ...candidate, name: event.target.value }
                          : candidate,
                      ),
                    )
                  }
                />
              </label>
              <label>
                Header value · write only
                <input
                  type="password"
                  autoComplete="new-password"
                  value={header.value}
                  onChange={(event) =>
                    setHeaders((current) =>
                      current.map((candidate) =>
                        candidate.id === header.id
                          ? { ...candidate, value: event.target.value }
                          : candidate,
                      ),
                    )
                  }
                />
              </label>
              <button
                className="secondary-button"
                type="button"
                disabled={headers.length === 1}
                onClick={() =>
                  setHeaders((current) =>
                    current.filter((candidate) => candidate.id !== header.id),
                  )
                }
              >
                Remove header
              </button>
            </div>
          ))}
          <button
            className="secondary-button"
            type="button"
            onClick={() => {
              const id = nextHeaderID.current++;
              setHeaders((current) => [
                ...current,
                { id, name: "", value: "" },
              ]);
            }}
          >
            Add header
          </button>
        </div>
      ) : kind === "http-proxy-basic@1" || kind === "http-origin-basic@1" ? (
        <div className="form-grid">
          <label>
            {kind === "http-origin-basic@1" ? "Origin" : "Proxy"} username ·
            write only
            <input
              autoComplete="off"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>
          <label>
            {kind === "http-origin-basic@1" ? "Origin" : "Proxy"} password ·
            write only
            <input
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
        </div>
      ) : (
        <label>
          Bearer token · write only
          <input
            type="password"
            autoComplete="new-password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
          />
        </label>
      )}
      {error === null ? null : <ErrorNotice error={error} />}
      {created === undefined ? null : (
        <div className="notice notice-success" role="status">
          Created safe metadata for {created.credentialId}; secret material
          cannot be read back.
        </div>
      )}
      <button type="submit" disabled={pending}>
        {pending ? "Creating…" : "Create active Runtime credential"}
      </button>
    </form>
  );
}

function RuntimeCredentialList({
  credentials,
}: {
  credentials: RuntimeCredentialMetadata[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const deletion = useMutation({
    mutationFn: (credentialId: string) =>
      deleteRuntimeCredential(
        api,
        credentialId,
        `delete-runtime-credential-ui-${crypto.randomUUID()}`,
      ),
    onSuccess: async () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeCredentials.all,
      }),
  });
  return (
    <div className="panel operations-library">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Secret-free active inventory</p>
          <h3>Runtime credentials</h3>
        </div>
        <span>{credentials.length} loaded</span>
      </div>
      {credentials.length === 0 ? (
        <p className="compact-empty">No Runtime credentials exist.</p>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Kind</th>
                <th>Created</th>
                <th>Lifecycle</th>
              </tr>
            </thead>
            <tbody>
              {credentials.map((credential) => (
                <tr key={credential.credentialId}>
                  <td>
                    <code>{credential.credentialId}</code>
                  </td>
                  <td>{credential.kind}</td>
                  <td>{formatTimestamp(credential.createdAt)}</td>
                  <td>
                    <button
                      className="danger-button"
                      type="button"
                      disabled={deletion.isPending}
                      onClick={() => deletion.mutate(credential.credentialId)}
                    >
                      Delete active credential
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {deletion.error === null ? null : <ErrorNotice error={deletion.error} />}
    </div>
  );
}

export function RuntimeConfigurationRoute() {
  const api = usePublicAPI();
  const [configCursors, setConfigCursors] = useState<Array<string | undefined>>(
    [undefined],
  );
  const configCursor = configCursors.at(-1);
  const configs = useQuery({
    queryKey: queryKeys.operations.runtimeConfigs.list(configCursor),
    queryFn: () =>
      listRuntimeConfigs(
        api,
        configCursor === undefined ? {} : { cursor: configCursor },
      ),
  });
  const labels = useQuery({
    queryKey: queryKeys.operations.runtimeLabels.list(),
    queryFn: () => listRuntimeLabels(api),
  });
  const credentials = useQuery({
    queryKey: queryKeys.operations.runtimeCredentials.list(),
    queryFn: () => listRuntimeCredentials(api),
  });
  const gateways = useQuery({
    queryKey: queryKeys.configurations.picker("llm-gateways"),
    queryFn: () => listConfigurations(api, "llm-gateways"),
  });
  const loadedConfigs = configs.data?.items ?? [];
  const inventoryError =
    configs.error ?? labels.error ?? credentials.error ?? gateways.error;
  return (
    <>
      {inventoryError === null ? null : <ErrorNotice error={inventoryError} />}
      <div className="panel operations-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Immutable safe documents</p>
            <h3>RuntimeConfig versions</h3>
            <p className="muted-copy">
              A version has no effect until default or a named label points to
              it.
            </p>
          </div>
          <span>{loadedConfigs.length} loaded</span>
        </div>
        {configs.isPending ? (
          <p className="loading-copy">Loading RuntimeConfigs…</p>
        ) : loadedConfigs.length === 0 ? (
          <p className="compact-empty">
            No RuntimeConfig versions are visible.
          </p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Digest</th>
                  <th>Source</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {loadedConfigs.map((resource) => (
                  <tr key={exactKey(resource)}>
                    <td>
                      <RuntimeRef resource={resource} />
                    </td>
                    <td>
                      <code>{resource.ref.digest.slice(0, 18)}…</code>
                    </td>
                    <td>
                      {resource.builtIn ? "built-in" : resource.createdBy}
                    </td>
                    <td>{formatTimestamp(resource.createdAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CursorControls
          label="RuntimeConfig pages"
          canGoBack={configCursors.length > 1}
          {...(configs.data?.page.hasMore === true &&
          configs.data.page.nextCursor !== undefined
            ? { nextCursor: configs.data.page.nextCursor }
            : {})}
          onBack={() =>
            setConfigCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setConfigCursors((current) => [...current, next])}
        />
      </div>
      <RuntimeBindings
        bindings={labels.data?.items ?? []}
        configs={loadedConfigs}
      />
      <RuntimeCredentialList credentials={credentials.data?.items ?? []} />
      <RuntimeCredentialCreateForm />
      <RuntimeConfigPublishForm
        configs={loadedConfigs}
        gateways={gateways.data?.items ?? []}
      />
    </>
  );
}
