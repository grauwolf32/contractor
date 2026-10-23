import {
  useInfiniteQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
import { type FormEvent, useId, useMemo, useRef, useState } from "react";

import {
  RUNTIME_CONFIG_VERSION,
  RUNTIME_INFRASTRUCTURE_ID,
} from "../../../api/contract";
import { usePublicAPI } from "../../../api/context";
import {
  listConfigurations,
  publishRuntimeConfig,
  type RuntimeConfigAuthorDocument,
  type RuntimeConfigResource,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { ErrorNotice } from "../../../app/error-notice";
import { LoadMoreButton } from "../../../app/load-more";
import { nextPageCursor } from "../../../app/pagination";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { Button, Field } from "../../../ui";
import { PublicationFeedback } from "../common";
import {
  buildDocument,
  emptyConfigDraft,
  LIMIT_FIELDS,
  PROXY_TARGETS,
  type ConfigDraft,
} from "./draft";
import { exactKey } from "./identity";

const TELEMETRY_BLOCKS = [
  {
    enabled: "workerTelemetry",
    title: "Worker telemetry",
    endpoint: "workerTelemetryEndpoint",
    credential: "workerTelemetryCredential",
    captureContent: "workerTelemetryCaptureContent",
  },
  {
    enabled: "plannerTelemetry",
    title: "Planner telemetry",
    endpoint: "plannerTelemetryEndpoint",
    credential: "plannerTelemetryCredential",
    captureContent: "plannerTelemetryCaptureContent",
  },
] as const;

/** A block's switch: the legend of its card. */
function BlockSwitch({
  checked,
  onChange,
  children,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  children: string;
}) {
  return (
    <legend>
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
        />
        {children}
      </label>
    </legend>
  );
}

export function RuntimeConfigPublishForm({
  configs,
  onClose,
}: {
  configs: readonly RuntimeConfigResource[];
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

  return (
    <Dialog
      className="ui-dialog ui-panel ops-dialog"
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
        <div className="ops-form-grid">
          <Field label="RuntimeConfig name">
            {(control) => (
              <input
                {...control}
                ref={initialFocus}
                required
                maxLength={RUNTIME_INFRASTRUCTURE_ID.maxLength}
                value={draft.name}
                onChange={(event) => update("name", event.target.value)}
              />
            )}
          </Field>
          <Field label="Version">
            {(control) => (
              <input
                {...control}
                aria-label="RuntimeConfig version"
                required
                maxLength={RUNTIME_CONFIG_VERSION.maxLength}
                value={draft.version}
                onChange={(event) => update("version", event.target.value)}
              />
            )}
          </Field>
        </div>

        <fieldset className="ops-config-block">
          <BlockSwitch
            checked={draft.gateway}
            onChange={(checked) => update("gateway", checked)}
          >
            Worker LLM Gateway route
          </BlockSwitch>
          <div className="ops-form-grid">
            <Field label="LLM Gateway">
              {(control) => (
                <select
                  {...control}
                  disabled={!draft.gateway}
                  value={draft.gatewayKey}
                  onChange={(event) => update("gatewayKey", event.target.value)}
                >
                  <option value="">Select Gateway</option>
                  {gateways.map((resource) => (
                    <option
                      key={exactKey(resource.ref)}
                      value={exactKey(resource.ref)}
                    >
                      {resource.ref.name}@{resource.ref.version} ·{" "}
                      {resource.source}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            <Field label="Active LLM credential ID" optional>
              {(control) => (
                <input
                  {...control}
                  disabled={!draft.gateway}
                  value={draft.llmCredential}
                  onChange={(event) =>
                    update("llmCredential", event.target.value)
                  }
                />
              )}
            </Field>
          </div>
          <LoadMoreButton query={gatewayInventory} label="Load more Gateways" />
        </fieldset>

        {TELEMETRY_BLOCKS.map(
          ({ enabled, title, endpoint, credential, captureContent }) => (
            <fieldset
              className="ops-config-block"
              key={enabled}
              aria-label={title}
            >
              <BlockSwitch
                checked={draft[enabled]}
                onChange={(checked) => update(enabled, checked)}
              >
                {`${title} · otlp-http@1`}
              </BlockSwitch>
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
              <div className="ops-form-grid">
                <Field label="OTLP traces endpoint">
                  {(control) => (
                    <input
                      {...control}
                      disabled={!draft[enabled]}
                      placeholder="https://otel.example/v1/traces"
                      value={draft[endpoint]}
                      onChange={(event) => update(endpoint, event.target.value)}
                    />
                  )}
                </Field>
                <Field label="Runtime credential ID" optional>
                  {(control) => (
                    <input
                      {...control}
                      disabled={!draft[enabled]}
                      value={draft[credential]}
                      onChange={(event) =>
                        update(credential, event.target.value)
                      }
                    />
                  )}
                </Field>
              </div>
            </fieldset>
          ),
        )}

        <fieldset className="ops-config-block">
          <BlockSwitch
            checked={draft.httpProxy}
            onChange={(checked) => update("httpProxy", checked)}
          >
            Worker HTTP proxy · http-proxy@1
          </BlockSwitch>
          <div className="ops-form-grid">
            <Field label="Proxy URL">
              {(control) => (
                <input
                  {...control}
                  disabled={!draft.httpProxy}
                  placeholder="http://caido.internal:8080"
                  value={draft.proxyURL}
                  onChange={(event) => update("proxyURL", event.target.value)}
                />
              )}
            </Field>
            <Field label="Runtime credential ID" optional>
              {(control) => (
                <input
                  {...control}
                  disabled={!draft.httpProxy}
                  value={draft.proxyCredential}
                  onChange={(event) =>
                    update("proxyCredential", event.target.value)
                  }
                />
              )}
            </Field>
          </div>
          <div className="ops-config-targets">
            {PROXY_TARGETS.map(([target, label]) => (
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

        <fieldset className="ops-config-block" disabled={mutation.isPending}>
          <BlockSwitch
            checked={draft.interactsh}
            onChange={(value) => update("interactsh", value)}
          >
            Worker Interactsh
          </BlockSwitch>
          {draft.interactsh && (
            <>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={draft.interactshClear}
                  onChange={(event) =>
                    update("interactshClear", event.target.checked)
                  }
                />
                Clear inherited Interactsh configuration
              </label>
              {!draft.interactshClear && (
                <>
                  <Field label="Interactsh endpoint">
                    {(control) => (
                      <input
                        {...control}
                        value={draft.interactshEndpoint}
                        onChange={(event) =>
                          update("interactshEndpoint", event.target.value)
                        }
                        placeholder="https://oast.example"
                      />
                    )}
                  </Field>
                  <Field label="Interactsh credential ID">
                    {(control) => (
                      <input
                        {...control}
                        value={draft.interactshCredential}
                        onChange={(event) =>
                          update("interactshCredential", event.target.value)
                        }
                      />
                    )}
                  </Field>
                  <Field label="Interactsh CA bundle PEM">
                    {(control) => (
                      <textarea
                        {...control}
                        value={draft.interactshCA}
                        onChange={(event) =>
                          update("interactshCA", event.target.value)
                        }
                      />
                    )}
                  </Field>
                  <Field label="Interactsh request timeout (seconds)">
                    {(control) => (
                      <input
                        {...control}
                        inputMode="numeric"
                        value={draft.interactshTimeout}
                        onChange={(event) =>
                          update("interactshTimeout", event.target.value)
                        }
                      />
                    )}
                  </Field>
                </>
              )}
            </>
          )}
        </fieldset>

        <details className="optional-settings">
          <summary>Optional settings</summary>
          <div>
            <p className="ui-caption">
              Telemetry delivery limits and proxy trust. Each field applies only
              while its block above is enabled.
            </p>
            <div className="ops-form-grid">
              {LIMIT_FIELDS.map(({ field, label, block, minimum, maximum }) => (
                <Field key={field} label={label}>
                  {(control) => (
                    <input
                      {...control}
                      disabled={!draft[block]}
                      type="number"
                      min={minimum}
                      max={maximum}
                      step={1}
                      value={draft[field]}
                      onChange={(event) => update(field, event.target.value)}
                    />
                  )}
                </Field>
              ))}
            </div>
            <Field
              className="ops-config-ca"
              label="HTTP proxy additional CA bundle PEM"
              optional
            >
              {(control) => (
                <textarea
                  {...control}
                  disabled={!draft.httpProxy}
                  value={draft.proxyCA}
                  onChange={(event) => update("proxyCA", event.target.value)}
                />
              )}
            </Field>
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
        <div className="ui-dialog-actions">
          <Button disabled={mutation.isPending} onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            type="submit"
            pending={mutation.isPending}
            pendingLabel="Publishing…"
          >
            Publish RuntimeConfig
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
