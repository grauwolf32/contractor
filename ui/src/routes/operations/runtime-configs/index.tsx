import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";
import { type FormEvent, useId, useMemo, useRef, useState } from "react";
import { Link } from "react-router";

import { runtimeConfigVersionPath } from "../../../app/navigation";

import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import {
  createRuntimeCredential,
  deleteRuntimeCredential,
  deleteRuntimeLabel,
  listAllRuntimeLabels,
  listConfigurations,
  listRuntimeConfigs,
  listRuntimeCredentials,
  publishRuntimeConfig,
  putRuntimeLabel,
  type ConfigurationResource,
  type CreateRuntimeCredentialRequest,
  type RuntimeConfigAuthorDocument,
  type RuntimeConfigResource,
  type RuntimeCredentialMetadata,
  type RuntimeCredentialPage,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { ConfirmRemovalDialog } from "../../../app/confirm-removal-dialog";
import { Icon } from "../../../app/icon";
import { DeleteIcon } from "../../../app/delete-icon";
import {
  createMutationIdempotencyKey,
  MutationDraftKeyring,
} from "../../../mutations/idempotency";
import { CursorControls } from "../../../app/cursor-controls";
import {
  nextPageCursor,
  type CursorStackControls,
  useCursorStack,
} from "../../../app/pagination";
import { LoadMoreButton } from "../../../app/load-more";
import { ErrorNotice } from "../../../app/error-notice";
import { QueryView } from "../../../app/query-view";
import { RecordedTime } from "../../../app/recorded-time";
import { InUseErrorDetails } from "../in-use-details";
import { compactDigest } from "../../../app/format";
import { OpsSection, PublicationFeedback } from "../common";

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
    <span className="ops-ref">
      <Link to={runtimeConfigVersionPath(resource.ref)}>
        {resource.ref.name}@{resource.ref.version}
      </Link>
      <code title={resource.ref.digest}>
        {compactDigest(resource.ref.digest)}
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
  workerTelemetryBatchSize: string;
  workerTelemetryMaxAttempts: string;
  workerTelemetryMaxPendingSpans: string;
  workerTelemetryMaxPendingSize: string;
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
  workerTelemetryTimeout: "10",
  workerTelemetryCaptureContent: false,
  workerTelemetryBatchSize: "8",
  workerTelemetryMaxAttempts: "2",
  workerTelemetryMaxPendingSpans: "2048",
  workerTelemetryMaxPendingSize: "64",
  plannerTelemetry: false,
  plannerTelemetryEndpoint: "",
  plannerTelemetryCredential: "",
  plannerTelemetryTimeout: "10",
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
  NonNullable<RuntimeConfigAuthorDocument["spec"]["worker"]>["telemetry"]
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
): { document?: RuntimeConfigAuthorDocument; errors: string[] } {
  const errors: string[] = [];
  if (!RUNTIME_ID.test(draft.name)) {
    errors.push(
      "Name must match [a-z][a-z0-9_-]* and contain at most 63 characters.",
    );
  }
  if (!RUNTIME_VERSION.test(draft.version)) {
    errors.push("Version must not be empty.");
  }
  if (
    !draft.gateway &&
    !draft.workerTelemetry &&
    !draft.plannerTelemetry &&
    !draft.httpProxy
  ) {
    errors.push("Select at least one RuntimeConfig block.");
  }
  const selectedGateway = gateways.find(
    (resource) => gatewayKey(resource) === draft.gatewayKey,
  );
  if (draft.gateway && selectedGateway === undefined) {
    errors.push("Select one published LLM Gateway.");
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
      (!Number.isInteger(Number(timeout)) ||
        Number(timeout) < 1 ||
        Number(timeout) > 10)
    ) {
      errors.push(
        `${label} flush timeout must be an integer from 1 to 10 seconds.`,
      );
    }
  }
  if (draft.workerTelemetry) {
    for (const [value, maximum, label] of [
      [draft.workerTelemetryBatchSize, 64, "Batch size (MiB)"],
      [draft.workerTelemetryMaxAttempts, 10, "Export attempts"],
      [draft.workerTelemetryMaxPendingSpans, 2048, "Pending spans"],
      [draft.workerTelemetryMaxPendingSize, 64, "Pending size (MiB)"],
    ] as const) {
      if (
        !Number.isInteger(Number(value)) ||
        Number(value) < 1 ||
        Number(value) > maximum
      ) {
        errors.push(`${label} must be an integer from 1 to ${maximum}.`);
      }
    }
    if (
      Number(draft.workerTelemetryMaxPendingSize) <
      Number(draft.workerTelemetryBatchSize)
    ) {
      errors.push("Pending size must be at least the batch size.");
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
  const worker: NonNullable<RuntimeConfigAuthorDocument["spec"]["worker"]> = {
    ...(draft.gateway
      ? {
          llmGateway: {
            gateway: `${selectedGateway!.ref.name}@${selectedGateway!.ref.version}`,
            ...(draft.llmCredential === ""
              ? {}
              : { credential: draft.llmCredential }),
          },
        }
      : {}),
    ...(draft.workerTelemetry
      ? {
          telemetry: {
            ...telemetry(
              draft.workerTelemetryEndpoint,
              draft.workerTelemetryCredential,
              draft.workerTelemetryTimeout,
              draft.workerTelemetryCaptureContent,
            ),
            export: {
              batchSizeBytes:
                Number(draft.workerTelemetryBatchSize) * 1024 * 1024,
              maxAttempts: Number(draft.workerTelemetryMaxAttempts),
              maxPendingSpans: Number(draft.workerTelemetryMaxPendingSpans),
              maxPendingBytes:
                Number(draft.workerTelemetryMaxPendingSize) * 1024 * 1024,
            },
          },
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
  onClose,
}: {
  configs: RuntimeConfigResource[];
  onClose: () => void;
}) {
  const heading = useId();
  const initialFocus = useRef<HTMLInputElement>(null);
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const gatewayInventory = useInfiniteQuery({
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
  const gateways = useMemo(
    () => gatewayInventory.data?.pages.flatMap((page) => page.items) ?? [],
    [gatewayInventory.data],
  );
  const [draft, setDraft] = useState(emptyConfigDraft);
  const [errors, setErrors] = useState<string[]>([]);
  const [published, setPublished] = useState<RuntimeConfigResource>();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<RuntimeConfigAuthorDocument>(
        "publish-runtime-config",
      ),
  );
  const mutation = useMutation({
    mutationFn: (document: RuntimeConfigAuthorDocument) =>
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
    <Dialog
      className="project-dialog panel ops-dialog"
      labelledBy={heading}
      initialFocusRef={initialFocus}
      onRequestClose={() => {
        if (!mutation.isPending) onClose();
      }}
    >
      <form className="ops-dialog-form" onSubmit={submit} noValidate>
        <DialogHeader
          id={heading}
          eyebrow={<>New version · {configs.length} loaded versions</>}
          title="Publish RuntimeConfig"
          close={{
            label: "Close RuntimeConfig form",
            disabled: mutation.isPending,
            onClose: onClose,
          }}
        />
        <p className="ops-dialog-note">
          Values become active only through a label binding. Publishing another
          version never mutates existing Runs or allocations.
        </p>
        <div className="form-grid">
          <label>
            RuntimeConfig name
            <input
              ref={initialFocus}
              aria-label="RuntimeConfig name"
              required
              maxLength={63}
              value={draft.name}
              onChange={(event) => update("name", event.target.value)}
            />
          </label>
          <label>
            Version
            <input
              aria-label="RuntimeConfig version"
              required
              maxLength={128}
              value={draft.version}
              onChange={(event) => update("version", event.target.value)}
            />
          </label>
        </div>

        <fieldset className="ops-config-block">
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
              LLM Gateway
              <select
                disabled={!draft.gateway}
                value={draft.gatewayKey}
                onChange={(event) => update("gatewayKey", event.target.value)}
              >
                <option value="">Select Gateway</option>
                {gateways.map((resource) => (
                  <option
                    key={gatewayKey(resource)}
                    value={gatewayKey(resource)}
                  >
                    {resource.ref.name}@{resource.ref.version} ·{" "}
                    {resource.source}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Active LLM credential ID (optional)
              <input
                disabled={!draft.gateway}
                value={draft.llmCredential}
                onChange={(event) =>
                  update("llmCredential", event.target.value)
                }
              />
            </label>
          </div>
          <LoadMoreButton query={gatewayInventory} label="Load more Gateways" />
        </fieldset>

        {(
          [
            [
              "workerTelemetry",
              "Worker telemetry",
              "workerTelemetryEndpoint",
              "workerTelemetryCredential",
              "workerTelemetryCaptureContent",
            ],
            [
              "plannerTelemetry",
              "Planner telemetry",
              "plannerTelemetryEndpoint",
              "plannerTelemetryCredential",
              "plannerTelemetryCaptureContent",
            ],
          ] as const
        ).map(([enabled, title, endpoint, credential, captureContent]) => (
          <fieldset
            className="ops-config-block"
            key={enabled}
            aria-label={title}
          >
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
            </div>
          </fieldset>
        ))}

        <fieldset className="ops-config-block">
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
          </div>
          <div className="ops-config-targets">
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
                        : draft.proxyTargets.filter(
                            (value) => value !== target,
                          ),
                    )
                  }
                />
                {label}
              </label>
            ))}
          </div>
        </fieldset>

        <details className="optional-settings">
          <summary>Optional settings</summary>
          <div>
            <p className="ops-note">
              Telemetry delivery limits and proxy trust. Each field applies only
              while its block above is enabled.
            </p>
            <div className="form-grid">
              {(
                [
                  [
                    "workerTelemetryTimeout",
                    "Worker telemetry flush timeout (seconds)",
                    "workerTelemetry",
                    10,
                  ],
                  [
                    "workerTelemetryBatchSize",
                    "Worker telemetry batch size (MiB)",
                    "workerTelemetry",
                    64,
                  ],
                  [
                    "workerTelemetryMaxAttempts",
                    "Worker telemetry maximum attempts per batch",
                    "workerTelemetry",
                    10,
                  ],
                  [
                    "workerTelemetryMaxPendingSpans",
                    "Worker telemetry maximum pending spans",
                    "workerTelemetry",
                    2048,
                  ],
                  [
                    "workerTelemetryMaxPendingSize",
                    "Worker telemetry maximum pending size (MiB)",
                    "workerTelemetry",
                    64,
                  ],
                  [
                    "plannerTelemetryTimeout",
                    "Planner telemetry flush timeout (seconds)",
                    "plannerTelemetry",
                    10,
                  ],
                ] as const
              ).map(([field, label, block, maximum]) => (
                <label key={field}>
                  {label}
                  <input
                    disabled={!draft[block]}
                    type="number"
                    min={1}
                    max={maximum}
                    step={1}
                    value={draft[field]}
                    onChange={(event) => update(field, event.target.value)}
                  />
                </label>
              ))}
            </div>
            <label className="ops-config-ca">
              HTTP proxy additional CA bundle PEM (optional)
              <textarea
                disabled={!draft.httpProxy}
                value={draft.proxyCA}
                onChange={(event) => update("proxyCA", event.target.value)}
              />
            </label>
          </div>
        </details>

        <PublicationFeedback
          title="RuntimeConfig draft is not publishable"
          errors={errors}
          mutationError={mutation.error}
        />
        {gatewayInventory.error === null ? null : (
          <ErrorNotice error={gatewayInventory.error} />
        )}
        {published === undefined ? null : (
          <div className="notice notice-success" role="status">
            Published {published.ref.name}@{published.ref.version}. Bind a label
            explicitly before it affects future resolution.
          </div>
        )}
        <div className="ops-proposal" aria-label="Proposed RuntimeConfig">
          <strong>
            Proposed new version: {draft.name || "Choose a name"}@
            {draft.version || "Choose a version"}
          </strong>
          <p>Current label bindings remain in effect until you rebind them.</p>
        </div>
        <div className="project-dialog-actions">
          <button
            type="button"
            className="ui-btn"
            disabled={mutation.isPending}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="ui-btn"
            data-variant="primary"
            type="submit"
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "Publishing…" : "Publish RuntimeConfig"}
          </button>
        </div>
      </form>
    </Dialog>
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
  const [conflictRevision, setConflictRevision] = useState<string>();
  const [confirmRemoval, setConfirmRemoval] = useState(false);
  const stale = conflictRevision === binding.revision;
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
      setConflictRevision(undefined);
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeLabels.all,
      });
    },
    onError: async (error) => {
      const conflict = error instanceof PublicAPIError && error.status === 412;
      setConflictRevision(conflict ? binding.revision : undefined);
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
        createMutationIdempotencyKey("delete-runtime-label"),
      ),
    onSuccess: async () => {
      setConfirmRemoval(false);
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeLabels.all,
      });
    },
    onError: async (error) => {
      const conflict = error instanceof PublicAPIError && error.status === 412;
      setConflictRevision(conflict ? binding.revision : undefined);
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
    <article className="ops-binding-card runtime-binding-card">
      <header className="ops-binding-head">
        <strong>{binding.label}</strong>
        {binding.label === "default" ? (
          <span className="ops-scope">always applied</span>
        ) : null}
        <small>Current binding revision {binding.revision}</small>
      </header>
      <dl className="ops-glance ops-binding-compare">
        <div>
          <dt>Current version</dt>
          <dd>
            <strong>
              {binding.config.name}@{binding.config.version}
            </strong>
            <code title={binding.config.digest}>
              {compactDigest(binding.config.digest)}
            </code>
          </dd>
        </div>
        <div>
          <dt>Proposed version</dt>
          <dd>
            {selectedResource === undefined ? (
              "Choose a loaded version"
            ) : (
              <>
                <strong>
                  {selectedResource.ref.name}@{selectedResource.ref.version}
                </strong>
                <code title={selectedResource.ref.digest}>
                  {compactDigest(selectedResource.ref.digest)}
                </code>
              </>
            )}
          </dd>
        </div>
      </dl>
      <div className="ops-binding-actions">
        <select
          aria-label={`RuntimeConfig for ${binding.label}`}
          value={selected}
          onChange={(event) => {
            setSelected(event.target.value);
            mutation.reset();
          }}
        >
          <option value="">Current version is outside this loaded page</option>
          {configs.map((resource) => (
            <option key={exactKey(resource)} value={exactKey(resource)}>
              {resource.ref.name}@{resource.ref.version} ·{" "}
              {compactDigest(resource.ref.digest)}
            </option>
          ))}
        </select>
        <button
          className="ui-btn"
          data-variant="primary"
          type="button"
          disabled={
            selectedResource === undefined ||
            selectedResource.ref.digest === binding.config.digest ||
            mutation.isPending ||
            stale
          }
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
            className="ui-btn ops-icon-button"
            data-variant="danger"
            type="button"
            aria-label={`Remove binding ${binding.label}`}
            title={`Remove binding ${binding.label}`}
            aria-haspopup="dialog"
            disabled={deletion.isPending || stale}
            onClick={() => {
              deletion.reset();
              setConfirmRemoval(true);
            }}
          >
            <DeleteIcon />
          </button>
        )}
      </div>
      {stale ? (
        <div className="notice notice-warning" role="alert">
          <strong>Binding changed in another view.</strong>
          <p>Reload the saved revision and review it before retrying.</p>
          <button
            className="ui-btn"
            data-size="sm"
            type="button"
            onClick={() => {
              mutation.reset();
              deletion.reset();
              void queryClient.invalidateQueries({
                queryKey: queryKeys.operations.runtimeLabels.all,
              });
            }}
          >
            Reload binding
          </button>
        </div>
      ) : mutation.error === null && deletion.error === null ? null : (
        <>
          <ErrorNotice error={mutation.error ?? deletion.error} />
          <InUseErrorDetails error={mutation.error ?? deletion.error} />
        </>
      )}
      {confirmRemoval ? (
        <ConfirmRemovalDialog
          className="ops-confirm"
          title={`Remove binding ${binding.label}?`}
          description={
            <>
              New Runs will no longer be able to select the label{" "}
              <code>{binding.label}</code>. To restore it, create a new binding.
            </>
          }
          confirmLabel="Remove binding"
          pending={deletion.isPending}
          confirmDisabled={stale}
          error={
            deletion.error === null ? null : (
              <>
                <ErrorNotice error={deletion.error} />
                <InUseErrorDetails error={deletion.error} />
              </>
            )
          }
          onCancel={() => {
            setConfirmRemoval(false);
            deletion.reset();
          }}
          onConfirm={() => deletion.mutate()}
        />
      ) : null}
    </article>
  );
}

function RuntimeBindings({
  bindings,
  configs,
  target,
}: {
  bindings: RuntimeLabelBinding[];
  configs: RuntimeConfigResource[];
  target: RuntimeConfigResource;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [label, setLabel] = useState("");
  const [selected, setSelected] = useState(exactKey(target));
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
      setSelected(exactKey(target));
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
      setError("Select a RuntimeConfig version.");
      return;
    }
    setError(undefined);
    mutation.mutate({ name: label, resource });
  }
  return (
    <div className="ops-dialog-form">
      <p className="ops-dialog-note">
        Default and Run-selected labels are pinned when a Run is created.
        Rebinding changes new Run snapshots. Labels assigned to a Runtime Agent
        are resolved again for future allocations, including allocations for an
        existing Run. Already prepared allocations keep their pinned settings.
      </p>
      {ordered.length === 0 ? (
        <p className="ops-empty">No label bindings are loaded.</p>
      ) : null}
      <div>
        <h3 className="ops-section-title">Current label bindings</h3>
        <p className="ops-section-description">
          Review current and proposed versions before rebinding.
        </p>
      </div>
      <div className="ops-binding-list">
        {ordered.map((binding) => (
          <BindingEditor
            key={binding.label}
            binding={binding}
            configs={configs}
          />
        ))}
      </div>
      <form
        className="ops-label-create runtime-label-create"
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
          RuntimeConfig
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
        <button
          className="ui-btn"
          data-variant="primary"
          type="submit"
          disabled={mutation.isPending}
        >
          {mutation.isPending ? "Binding…" : "Create binding"}
        </button>
      </form>
      {error === undefined ? null : (
        <p className="ops-field-error" role="alert">
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

function RuntimeCredentialCreateForm({ onClose }: { onClose: () => void }) {
  const heading = useId();
  const initialFocus = useRef<HTMLInputElement>(null);
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
    if (pending) return;
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
        createMutationIdempotencyKey("create-runtime-credential"),
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
    <Dialog
      className="project-dialog panel ops-dialog"
      labelledBy={heading}
      initialFocusRef={initialFocus}
      onRequestClose={() => {
        if (!pending) onClose();
      }}
    >
      <form
        className="ops-dialog-form runtime-credential-form"
        onSubmit={(event) => void submit(event)}
        noValidate
        autoComplete="off"
      >
        <DialogHeader
          id={heading}
          eyebrow="Secret"
          title="Create Runtime credential"
          close={{
            label: "Close Runtime credential form",
            disabled: pending,
            onClose: onClose,
          }}
        />
        <p className="ops-dialog-note">
          Secret values are erased from the form immediately on submit and are
          never readable through the API. A failed request requires re-entry.
        </p>
        <div className="form-grid">
          <label>
            Runtime credential ID
            <input
              ref={initialFocus}
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
              <option value="http-proxy-bearer@1">
                HTTP proxy bearer token
              </option>
              <option value="caido-bearer@1">Caido bearer token</option>
              <option value="http-origin-basic@1">
                HTTP origin basic auth
              </option>
              <option value="http-origin-bearer@1">
                HTTP origin bearer token
              </option>
            </select>
          </label>
        </div>
        {kind === "otlp-headers@1" ? (
          <div className="ops-secret-rows">
            {headers.map((header) => (
              <div className="ops-secret-row" key={header.id}>
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
                  className="ui-btn ops-icon-button"
                  data-variant="danger"
                  type="button"
                  aria-label={`Remove header ${header.name || header.id}`}
                  title="Remove header"
                  disabled={headers.length === 1}
                  onClick={() =>
                    setHeaders((current) =>
                      current.filter((candidate) => candidate.id !== header.id),
                    )
                  }
                >
                  <DeleteIcon />
                </button>
              </div>
            ))}
            <button
              className="ui-btn"
              data-size="sm"
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
        <div className="project-dialog-actions">
          <button
            type="button"
            className="ui-btn"
            disabled={pending}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="ui-btn"
            data-variant="primary"
            type="submit"
            disabled={pending}
          >
            {pending ? "Creating…" : "Create active Runtime credential"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

function RuntimeCredentialList({
  credentials,
  controls,
  onCreate,
}: {
  credentials: UseQueryResult<RuntimeCredentialPage>;
  controls: CursorStackControls;
  onCreate: () => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [confirmRemoval, setConfirmRemoval] =
    useState<RuntimeCredentialMetadata | null>(null);
  const deletion = useMutation({
    mutationFn: (credentialId: string) =>
      deleteRuntimeCredential(
        api,
        credentialId,
        createMutationIdempotencyKey("delete-runtime-credential"),
      ),
    onSuccess: async () => {
      setConfirmRemoval(null);
      await queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeCredentials.all,
      });
    },
  });
  return (
    <OpsSection
      id="runtime-credentials-heading"
      titleAs="h3"
      title="Runtime credentials"
      description="Secrets for telemetry sinks, HTTP proxies, Caido and project live targets (HTTP origins). Values are write-only; only safe metadata is listed."
      aside={
        credentials.data === undefined
          ? undefined
          : `${credentials.data.items.length} on this page`
      }
      actions={
        <button
          className="ui-btn"
          data-size="sm"
          type="button"
          aria-haspopup="dialog"
          onClick={onCreate}
        >
          <Icon name="add-credential" />
          Add Runtime credential
        </button>
      }
    >
      <QueryView
        query={credentials}
        loading={
          <p className="ops-loading" role="status">
            Loading Runtime credentials…
          </p>
        }
        errorContext="Could not load Runtime credentials"
        onRetry={() => void credentials.refetch()}
        isEmpty={(page) => page.items.length === 0}
        empty={
          <p className="ops-empty">
            {controls.canGoBack
              ? "This page lists no Runtime credentials. Earlier pages may list more."
              : "No Runtime credentials exist."}
          </p>
        }
      >
        {(page) => (
          <div className="ops-table-wrap">
            <table className="ops-table" data-stack="">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Kind</th>
                  <th>Created</th>
                  <th>
                    <span className="ui-visually-hidden">Lifecycle</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((credential) => (
                  <tr key={credential.credentialId}>
                    <td data-label="ID">
                      <code className="ops-mono">
                        {credential.credentialId}
                      </code>
                    </td>
                    <td data-label="Kind">
                      <code className="ops-label-chip">{credential.kind}</code>
                    </td>
                    <td data-label="Created">
                      <RecordedTime value={credential.createdAt} />
                    </td>
                    <td data-label="" className="ops-cell-actions">
                      <button
                        className="ui-btn ops-icon-button"
                        data-size="sm"
                        data-variant="danger"
                        type="button"
                        aria-label={`Delete Runtime credential ${credential.credentialId}`}
                        title={`Delete Runtime credential ${credential.credentialId}`}
                        aria-haspopup="dialog"
                        disabled={deletion.isPending}
                        onClick={() => {
                          deletion.reset();
                          setConfirmRemoval(credential);
                        }}
                      >
                        <DeleteIcon />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </QueryView>
      <CursorControls label="Runtime credential pages" {...controls} />
      {confirmRemoval ? (
        <ConfirmRemovalDialog
          className="ops-confirm"
          title={`Delete Runtime credential ${confirmRemoval.credentialId}?`}
          description={
            <>
              This permanently deletes the <code>{confirmRemoval.kind}</code>{" "}
              credential <code>{confirmRemoval.credentialId}</code>. Its ID
              cannot be reused, and RuntimeConfig versions that name it cannot
              be bound again. Restore access by entering the secret under a new
              ID and publishing a new version.
            </>
          }
          confirmLabel="Delete Runtime credential"
          pending={deletion.isPending}
          error={
            deletion.error === null ? null : (
              <>
                <ErrorNotice error={deletion.error} />
                <InUseErrorDetails error={deletion.error} />
              </>
            )
          }
          onCancel={() => {
            setConfirmRemoval(null);
            deletion.reset();
          }}
          onConfirm={() => deletion.mutate(confirmRemoval.credentialId)}
        />
      ) : null}
    </OpsSection>
  );
}

function RuntimeBindingsDialog({
  target,
  configs,
  bindings,
  onClose,
}: {
  target: RuntimeConfigResource;
  configs: RuntimeConfigResource[];
  bindings: RuntimeLabelBinding[];
  onClose: () => void;
}) {
  const heading = useId();
  return (
    <Dialog
      className="project-dialog panel ops-dialog"
      labelledBy={heading}
      onRequestClose={onClose}
    >
      <DialogHeader
        id={heading}
        eyebrow={
          <>
            {target.ref.name}@{target.ref.version}
          </>
        }
        title="Runtime label bindings"
        close={{ label: "Close Runtime bindings", onClose: onClose }}
      />
      <RuntimeBindings target={target} configs={configs} bindings={bindings} />
    </Dialog>
  );
}

function BindingCell({
  resource,
  labels,
  onManage,
}: {
  resource: RuntimeConfigResource;
  labels: UseQueryResult<Awaited<ReturnType<typeof listAllRuntimeLabels>>>;
  onManage: () => void;
}) {
  const bound = labels.isSuccess
    ? labels.data.items.filter(
        (binding) => binding.config.digest === resource.ref.digest,
      )
    : [];
  return (
    <div className="ops-binding-cell">
      {labels.isSuccess ? (
        bound.map((binding) => (
          <span
            className="ops-label-chip"
            data-tone={binding.label === "default" ? "default" : undefined}
            key={binding.label}
          >
            {binding.label}
          </span>
        ))
      ) : (
        <small className="ops-muted">Labels unavailable</small>
      )}
      {labels.isSuccess && bound.length === 0 ? (
        <small className="ops-muted">No labels</small>
      ) : null}
      <button
        className={`ui-btn ops-icon-button runtime-binding-trigger ${bound.length > 0 ? "has-bindings" : ""}`}
        data-size="sm"
        type="button"
        aria-label={`Manage bindings for ${resource.ref.name}@${resource.ref.version}`}
        title={
          bound.length > 0
            ? "Label bindings active — manage bindings"
            : "Add a label binding"
        }
        aria-haspopup="dialog"
        disabled={!labels.isSuccess}
        onClick={onManage}
      >
        <Icon name="binding" />
      </button>
    </div>
  );
}

export function RuntimeConfigurationRoute() {
  const api = usePublicAPI();
  const [createDialog, setCreateDialog] = useState<"config" | "credential">();
  const [bindingTarget, setBindingTarget] = useState<RuntimeConfigResource>();
  const configPages = useCursorStack();
  const credentialPages = useCursorStack();
  const configCursor = configPages.cursor;
  const credentialCursor = credentialPages.cursor;
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
    queryFn: () => listAllRuntimeLabels(api),
  });
  const credentials = useQuery({
    queryKey: queryKeys.operations.runtimeCredentials.list(credentialCursor),
    queryFn: () =>
      listRuntimeCredentials(
        api,
        credentialCursor === undefined ? {} : { cursor: credentialCursor },
      ),
  });
  const loadedConfigs = configs.data?.items ?? [];
  return (
    <>
      {labels.error === null ? null : <ErrorNotice error={labels.error} />}
      <OpsSection
        id="runtime-config-versions-heading"
        titleAs="h3"
        title="RuntimeConfig versions"
        description="A version has no effect until default or a named label points to it."
        aside={`${loadedConfigs.length} loaded`}
        actions={
          <button
            className="ui-btn"
            data-size="sm"
            type="button"
            aria-haspopup="dialog"
            onClick={() => setCreateDialog("config")}
          >
            <Icon name="add-config" />
            Publish RuntimeConfig
          </button>
        }
      >
        <QueryView
          query={configs}
          loading={
            <p className="ops-loading" role="status">
              Loading RuntimeConfigs…
            </p>
          }
          errorContext="Could not load RuntimeConfig versions"
          onRetry={() => void configs.refetch()}
          isEmpty={(page) => page.items.length === 0}
          empty={
            <p className="ops-empty">
              {configPages.cursor === undefined
                ? "No RuntimeConfig versions are visible."
                : "This page lists no RuntimeConfig versions. Earlier pages may list more."}
            </p>
          }
        >
          {(page) => (
            <div className="ops-table-wrap">
              <table className="ops-table" data-stack="">
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Source</th>
                    <th>Created</th>
                    <th>Bindings</th>
                  </tr>
                </thead>
                <tbody>
                  {page.items.map((resource) => (
                    <tr key={exactKey(resource)}>
                      <td data-label="Version">
                        <RuntimeRef resource={resource} />
                      </td>
                      <td data-label="Source">
                        {resource.builtIn ? "built-in" : resource.createdBy}
                      </td>
                      <td data-label="Created">
                        {resource.builtIn ? (
                          "Built-in"
                        ) : (
                          <RecordedTime value={resource.createdAt} />
                        )}
                      </td>
                      <td data-label="Bindings">
                        <BindingCell
                          resource={resource}
                          labels={labels}
                          onManage={() => setBindingTarget(resource)}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </QueryView>
        <CursorControls
          label="RuntimeConfig pages"
          {...configPages.controls(configs.data?.page)}
        />
      </OpsSection>
      <div className="ops-divided">
        <RuntimeCredentialList
          credentials={credentials}
          controls={credentialPages.controls(credentials.data?.page)}
          onCreate={() => setCreateDialog("credential")}
        />
      </div>
      {createDialog === "credential" ? (
        <RuntimeCredentialCreateForm
          onClose={() => setCreateDialog(undefined)}
        />
      ) : null}
      {createDialog === "config" ? (
        <RuntimeConfigPublishForm
          configs={loadedConfigs}
          onClose={() => setCreateDialog(undefined)}
        />
      ) : null}
      {bindingTarget === undefined ? null : (
        <RuntimeBindingsDialog
          target={bindingTarget}
          configs={loadedConfigs}
          bindings={labels.data?.items ?? []}
          onClose={() => setBindingTarget(undefined)}
        />
      )}
    </>
  );
}
