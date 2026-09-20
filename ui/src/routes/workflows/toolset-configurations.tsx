import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import {
  type FormEvent,
  useEffect,
  useId,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import {
  listRuntimeCredentials,
  type CreateRuntimeCredentialRequest,
  type RuntimeConfigAuthorDocument,
  type RuntimeLabelBinding,
} from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import type { WorkflowResource } from "../../api/workflows";
import { Dialog } from "../../app/dialog";
import {
  ToolsetConnectionCreation,
  type ToolsetCoverageState,
} from "../../run-drafts/toolset-connections";
import { ErrorNotice } from "../artifacts/common";
import {
  HeaderFields,
  mcpHeaderMaterial,
  type HeaderDraft,
} from "../operations/credentials/header-fields";
import {
  buildMCPConfig,
  emptyMCPConfigDraft,
  TOOLSET_CONFIGURATION_FORMS,
} from "../operations/runtime-configs/mcp-fields";

export function ToolsetConfigurationRequirements({
  requirements,
  coverage,
  onCreate,
}: {
  requirements: NonNullable<
    WorkflowResource["toolsetConfigurationRequirements"]
  >;
  coverage: readonly ToolsetCoverageState[];
  onCreate: () => void;
}) {
  const title = useId();
  return (
    <section
      className="run-draft-section toolset-configurations"
      aria-labelledby={title}
    >
      <h4 id={title}>Toolset connections</h4>
      <p>
        Choose Runtime profiles below. One connection is shared by all consumers
        of a toolset in this Run.
      </p>
      {requirements.map((requirement) => {
        const value = coverage.find((item) => item.ref === requirement.ref);
        const status = value?.state ?? "unavailable";
        return (
          <article className="toolset-requirement" key={requirement.ref}>
            <strong>
              {requirement.ref}
              {requirement.configurationRequired
                ? " · required"
                : " · optional"}
            </strong>
            <p>Tools: {requirement.tools.join(", ")}</p>
            <p>
              Used by{" "}
              {requirement.consumers
                .map((consumer) => `${consumer.stage}/${consumer.agent}`)
                .join(", ")}
            </p>
            <p role="status">
              {status === "configured"
                ? value?.sources[0] === "default"
                  ? "Inherited from default"
                  : `Configured by ${value?.sources.join(", ")}`
                : status === "cleared"
                  ? `Cleared by ${value?.sources.join(", ")}`
                  : status === "conflict"
                    ? `Conflicting profiles: ${value?.sources.join(", ")}`
                    : status === "missing"
                      ? "Choose a connection profile"
                      : `Loading or unavailable profiles: ${value?.sources.join(", ") ?? "default"}`}
            </p>
          </article>
        );
      })}
      {coverage
        .filter(
          (value) =>
            value.state === "conflict" &&
            !requirements.some((requirement) => requirement.ref === value.ref),
        )
        .map((value) => (
          <p className="field-error" role="alert" key={value.ref}>
            {value.ref}: conflicting profiles {value.sources.join(", ")}
          </p>
        ))}
      {requirements.some((item) => item.ref in TOOLSET_CONFIGURATION_FORMS) ? (
        <button className="secondary-button" type="button" onClick={onCreate}>
          Create or resume MCP connection
        </button>
      ) : null}
      <p className="muted-copy">
        Connection, authentication and selected tools are checked when the Run
        prepares its tools. Preparation failures appear in Run details.
      </p>
    </section>
  );
}

export function MCPConnectionDialog({
  progress,
  onStarted,
  onCreated,
  onClose,
}: {
  progress: ToolsetConnectionCreation;
  onStarted: () => boolean;
  onCreated: (binding: RuntimeLabelBinding) => void;
  onClose: () => void;
}) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const heading = useId();
  const initialFocus = useRef<HTMLInputElement>(null);
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useSyncExternalStore(progress.subscribe, progress.snapshot);
  const [label, setLabel] = useState(progress.label);
  const [draft, setDraft] = useState(() => ({
    ...emptyMCPConfigDraft(),
    credential: progress.credentialId ?? "",
  }));
  const [auth, setAuth] = useState<"none" | "existing" | "new">(
    progress.credentialId === undefined ? "none" : "existing",
  );
  const [credentialId, setCredentialId] = useState(progress.credentialId ?? "");
  const [headers, setHeaders] = useState<HeaderDraft[]>([
    { id: 1, name: "Authorization", value: "" },
  ]);
  const [error, setError] = useState<unknown>(null);
  const inventory = useInfiniteQuery({
    queryKey: queryKeys.operations.runtimeCredentials.picker,
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) =>
      listRuntimeCredentials(
        api,
        pageParam === null ? {} : { cursor: pageParam },
      ),
    getNextPageParam: (page) =>
      page.page.hasMore ? page.page.nextCursor : undefined,
    enabled:
      auth === "existing" &&
      progress.credentialId === undefined &&
      progress.config === undefined,
  });
  const Fields = TOOLSET_CONFIGURATION_FORMS["mcp@1"];

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    event.stopPropagation();
    if (progress.pending) return;
    setError(null);
    let credential: CreateRuntimeCredentialRequest | undefined;
    try {
      if (!/^[a-z][a-z0-9_-]{0,62}$/.test(label) || label === "default")
        throw new Error(
          "Connection name must start with a lowercase letter and contain at most 63 lowercase letters, numbers, underscores or hyphens. Default is reserved.",
        );
      let document: RuntimeConfigAuthorDocument | undefined;
      if (progress.config === undefined) {
        const selectedCredential =
          progress.credentialId ??
          (auth === "none"
            ? ""
            : auth === "existing"
              ? draft.credential
              : credentialId);
        if (auth !== "none" && selectedCredential === "")
          throw new Error(
            "Choose an existing MCP credential or enter a new credential ID.",
          );
        const config = buildMCPConfig({
          ...draft,
          credential: selectedCredential,
        });
        if (auth === "new" && progress.credentialId === undefined)
          credential = {
            credentialId,
            kind: "mcp-headers@1",
            material: { headers: mcpHeaderMaterial(headers) },
          };
        document = {
          apiVersion: "contractor/v1alpha1",
          kind: "RuntimeConfig",
          metadata: { name: label, version: "1" },
          spec: { worker: { toolsets: { "mcp@1": config } } },
        };
      }
      // Erase UI values before the first request. A failed credential step
      // requires the same material again to replay its saved operation key.
      setHeaders((rows) => rows.map((row) => ({ ...row, value: "" })));
      if (!onStarted()) return;
      const binding = await progress.run(api, {
        label,
        ...(document === undefined ? {} : { document }),
        ...(credential === undefined ? {} : { credential }),
      });
      await Promise.all([
        client.invalidateQueries({
          queryKey: queryKeys.operations.runtimeLabels.all,
        }),
        client.invalidateQueries({
          queryKey: queryKeys.operations.runtimeConfigs.all,
        }),
        client.invalidateQueries({
          queryKey: queryKeys.operations.runtimeCredentials.all,
        }),
      ]);
      client.setQueryData(
        queryKeys.operations.runtimeLabels.detail(binding.label),
        binding,
      );
      if (mounted.current) onCreated(binding);
    } catch (reason) {
      if (mounted.current) setError(reason);
    } finally {
      if (credential !== undefined && "headers" in credential.material)
        for (const name of Object.keys(credential.material.headers))
          delete credential.material.headers[name];
    }
  }

  return (
    <Dialog
      className="project-dialog panel runtime-configuration-dialog"
      labelledBy={heading}
      initialFocusRef={initialFocus}
      onRequestClose={() => {
        if (!progress.pending) onClose();
      }}
    >
      <form
        className="configuration-draft"
        autoComplete="off"
        noValidate
        onSubmit={(event) => void submit(event)}
      >
        <div className="project-dialog-heading">
          <h2 id={heading}>Create MCP connection</h2>
          <button
            type="button"
            className="project-dialog-close"
            aria-label="Close MCP connection form"
            disabled={progress.pending}
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <label>
          Connection name
          <input
            ref={initialFocus}
            maxLength={63}
            autoComplete="off"
            value={label}
            disabled={progress.pending || progress.binding !== undefined}
            onChange={(event) => setLabel(event.target.value)}
          />
        </label>
        {progress.credentialId === undefined ? null : (
          <p role="status">
            Credential saved: {progress.credentialId}. Secret material cannot be
            read back.
          </p>
        )}
        {progress.config !== undefined ? (
          <p role="status">
            Configuration saved: {progress.config.name}@
            {progress.config.version}. Continue to create a new label and select
            it.
          </p>
        ) : (
          <>
            <label>
              Authentication
              <select
                value={auth}
                disabled={
                  progress.pending || progress.credentialId !== undefined
                }
                onChange={(event) => {
                  setAuth(event.target.value as typeof auth);
                  setHeaders((rows) =>
                    rows.map((row) => ({ ...row, value: "" })),
                  );
                }}
              >
                <option value="none">No authentication</option>
                <option value="existing">Existing MCP credential</option>
                <option value="new">New header credential</option>
              </select>
            </label>
            <Fields
              idPrefix={heading}
              value={draft}
              onChange={setDraft}
              disabled={progress.pending}
              showCredential={
                auth === "existing" && progress.credentialId === undefined
              }
              credentials={
                inventory.data?.pages.flatMap((page) => page.items) ?? []
              }
            />
            {auth !== "existing" ? null : (
              <>
                {inventory.error === null ? null : (
                  <ErrorNotice error={inventory.error} />
                )}
                {inventory.hasNextPage ? (
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => void inventory.fetchNextPage()}
                  >
                    Load more MCP credentials
                  </button>
                ) : null}
              </>
            )}
            {auth === "new" && progress.credentialId === undefined ? (
              <>
                <label>
                  New MCP credential ID
                  <input
                    autoComplete="off"
                    maxLength={128}
                    value={credentialId}
                    disabled={progress.pending}
                    onChange={(event) => setCredentialId(event.target.value)}
                  />
                </label>
                <HeaderFields
                  rows={headers}
                  onChange={setHeaders}
                  disabled={progress.pending}
                />
              </>
            ) : null}
          </>
        )}
        {error === null ? null : (
          <>
            <ErrorNotice error={error} />
            <p>
              Retry resumes at the {progress.step} step. Re-enter header values
              if that step did not return a credential. Existing profiles are
              never retargeted by this form.
            </p>
          </>
        )}
        {progress.pending ? <p role="status">Saving {progress.step}…</p> : null}
        <p className="muted-copy">
          Secret inputs are cleared on submit. Closing or reloading the browser
          cannot recover them. Published resources remain available in{" "}
          <Link to="/runs/configuration">Runtime configuration</Link>.
        </p>
        <button type="submit" disabled={progress.pending}>
          {progress.pending
            ? "Saving connection…"
            : progress.binding !== undefined
              ? "Use saved connection"
              : "Save and select connection"}
        </button>
      </form>
    </Dialog>
  );
}
