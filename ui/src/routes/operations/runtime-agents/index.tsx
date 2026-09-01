import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import {
  deleteRuntimeAgentPrincipal,
  listRuntimeAgentPrincipals,
  listRuntimeLabels,
  replaceRuntimeAgentPrincipalLabels,
  type RuntimeAgentPrincipal,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice } from "../../artifacts/common";
import { useOperationsSnapshot } from "../context";
import { OperationsState, OptionalTimestamp, SafeReason } from "../common";

function PrincipalCard({
  principal,
  bindings,
}: {
  principal: RuntimeAgentPrincipal;
  bindings: RuntimeLabelBinding[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [labels, setLabels] = useState([...principal.labels]);
  const [stale, setStale] = useState(false);
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{
        runtimeAgentId: string;
        revision: string;
        labels: string[];
      }>("runtime-agent-labels"),
  );
  const refresh = async () => {
    await Promise.all([
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.snapshot,
      }),
    ]);
  };
  const mutation = useMutation({
    mutationFn: () => {
      const request = {
        runtimeAgentId: principal.runtimeAgentId,
        revision: principal.revision,
        labels: [...labels].sort(),
      };
      return replaceRuntimeAgentPrincipalLabels(
        api,
        principal.runtimeAgentId,
        request.labels,
        principal.revision,
        keyring.keyFor(request),
      );
    },
    onSuccess: refresh,
    onError: async (error) => {
      const conflict = error instanceof PublicAPIError && error.status === 412;
      setStale(conflict);
      if (!conflict) await refresh();
    },
  });
  const deletion = useMutation({
    mutationFn: () =>
      deleteRuntimeAgentPrincipal(
        api,
        principal.runtimeAgentId,
        principal.revision,
        `delete-runtime-principal-ui-${crypto.randomUUID()}`,
      ),
    onSuccess: refresh,
    onError: async (error) => {
      const conflict = error instanceof PublicAPIError && error.status === 412;
      setStale(conflict);
      if (!conflict) await refresh();
    },
  });
  const selectable = bindings
    .filter((binding) => binding.label !== "default")
    .sort((left, right) => left.label.localeCompare(right.label));
  const unchanged =
    JSON.stringify([...labels].sort()) === JSON.stringify(principal.labels);
  return (
    <article className="panel runtime-principal-card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Certificate public-key principal</p>
          <h4 title={principal.runtimeAgentId}>
            {principal.runtimeAgentId.slice(0, 18)}…
          </h4>
        </div>
        <OperationsState state={principal.availability} />
      </div>
      <dl className="key-value-list">
        <div>
          <dt>Principal revision</dt>
          <dd>{principal.revision}</dd>
        </div>
        <div>
          <dt>Process liveness</dt>
          <dd>
            {principal.live === undefined
              ? "offline · durable labels retained"
              : `live as ${principal.live.instanceId}`}
          </dd>
        </div>
        <div>
          <dt>Required adapters</dt>
          <dd>
            {principal.requiredRuntimeAdapters.length === 0
              ? "none"
              : principal.requiredRuntimeAdapters.map((adapter) => (
                  <code key={adapter}>{adapter}</code>
                ))}
          </dd>
        </div>
        <div>
          <dt>Missing adapters</dt>
          <dd>
            {principal.missingRuntimeAdapters.length === 0
              ? "none"
              : principal.missingRuntimeAdapters.map((adapter) => (
                  <code key={adapter}>{adapter}</code>
                ))}
          </dd>
        </div>
        {principal.live === undefined ? null : (
          <div>
            <dt>Frozen supported adapters</dt>
            <dd>
              {principal.live.supportedRuntimeAdapters.length === 0
                ? "none"
                : principal.live.supportedRuntimeAdapters.map((adapter) => (
                    <code key={adapter}>{adapter}</code>
                  ))}
            </dd>
          </div>
        )}
        {principal.live?.workspaceCapabilities === undefined ? null : (
          <div>
            <dt>Frozen workspace capability</dt>
            <dd>
              <code>{principal.live.workspaceCapabilities.storage}</code>
              {" · "}
              {principal.live.workspaceCapabilities.modes.map((mode) => (
                <code key={mode}>{mode}</code>
              ))}
            </dd>
          </div>
        )}
      </dl>
      <fieldset className="runtime-principal-labels">
        <legend>Authoritative Agent labels</legend>
        {selectable.length === 0 ? (
          <p className="compact-empty">No named Runtime labels are bound.</p>
        ) : (
          selectable.map((binding) => (
            <label className="checkbox-label" key={binding.label}>
              <input
                type="checkbox"
                checked={labels.includes(binding.label)}
                onChange={(event) => {
                  setLabels((current) =>
                    (event.target.checked
                      ? [...current, binding.label]
                      : current.filter((label) => label !== binding.label)
                    ).sort(),
                  );
                  setStale(false);
                  mutation.reset();
                }}
              />
              <span>
                <strong>{binding.label}</strong> · {binding.config.name}@
                {binding.config.version}
              </span>
            </label>
          ))
        )}
      </fieldset>
      <div className="runtime-binding-actions">
        <button
          type="button"
          disabled={unchanged || mutation.isPending}
          onClick={() => mutation.mutate()}
        >
          {mutation.isPending
            ? "Saving labels…"
            : "Replace labels with current revision"}
        </button>
        {principal.live === undefined && principal.labels.length === 0 ? (
          <button
            className="danger-button"
            type="button"
            disabled={deletion.isPending}
            onClick={() => deletion.mutate()}
          >
            Remove offline principal
          </button>
        ) : null}
      </div>
      <p className="muted-copy">
        Changes apply only to future allocations. A busy allocation retains its
        already pinned RuntimeSettings and provenance.
      </p>
      {principal.availability === "adapter_capability_mismatch" ? (
        <div className="notice notice-warning">
          Labels are durable, but this live process cannot receive matching work
          until its immutable startup adapter capability set is compatible.
        </div>
      ) : null}
      {stale ? (
        <div className="notice notice-warning" role="alert">
          <strong>Principal labels changed in another view.</strong>
          <p>
            Reload the authoritative revision and review it before retrying.
          </p>
          <button
            className="secondary-button"
            type="button"
            onClick={() => void refresh()}
          >
            Reload authoritative labels
          </button>
        </div>
      ) : mutation.error === null && deletion.error === null ? null : (
        <ErrorNotice error={mutation.error ?? deletion.error} />
      )}
    </article>
  );
}

export function RuntimeAgentListRoute() {
  const api = usePublicAPI();
  const { snapshot } = useOperationsSnapshot();
  const principals = useQuery({
    queryKey: queryKeys.operations.runtimeAgentPrincipals.list(),
    queryFn: () => listRuntimeAgentPrincipals(api),
  });
  const bindings = useQuery({
    queryKey: queryKeys.operations.runtimeLabels.list(),
    queryFn: () => listRuntimeLabels(api),
  });
  return (
    <>
      <div className="operations-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Durable configuration identities</p>
            <h3>Runtime Agent principals</h3>
            <p className="muted-copy">
              Liveness and adapter eligibility are independent from the one
              authoritative label set bound to a certificate public key.
            </p>
          </div>
          <span>{principals.data?.items.length ?? 0} loaded</span>
        </div>
        {principals.error !== null ? (
          <ErrorNotice error={principals.error} />
        ) : bindings.error !== null ? (
          <ErrorNotice error={bindings.error} />
        ) : principals.isPending || bindings.isPending ? (
          <p className="loading-copy">Loading durable Runtime principals…</p>
        ) : principals.data.items.length === 0 ? (
          <p className="compact-empty">
            No Runtime Agent certificate principal has registered yet.
          </p>
        ) : (
          <div className="runtime-principal-grid">
            {principals.data.items.map((principal) => (
              <PrincipalCard
                key={`${principal.runtimeAgentId}:${principal.revision}`}
                principal={principal}
                bindings={bindings.data.items}
              />
            ))}
          </div>
        )}
      </div>
      <div className="panel operations-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Observed process inventory</p>
            <h3>Runtime Agents</h3>
            <p className="muted-copy">
              A Runtime Agent is one deployed, long-running single-slot process.
              Worker is the temporary role it assumes for an allocation, not a
              second service.
            </p>
          </div>
          <span>{snapshot.runtimeAgents.length} current</span>
        </div>
        {snapshot.runtimeAgents.length === 0 ? (
          <div className="compact-empty">
            <strong>No Runtime Agent is currently registered.</strong>
            <p>
              The snapshot contains current process state, not durable history.
            </p>
          </div>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Runtime Agent</th>
                  <th>Observed</th>
                  <th>Authoritative slot</th>
                  <th>Last accepted heartbeat</th>
                  <th>Confirmed lease</th>
                  <th>Allocation binding</th>
                </tr>
              </thead>
              <tbody>
                {snapshot.runtimeAgents.map((agent) => {
                  const reconciled =
                    agent.currentAllocationId ===
                      agent.authoritativeAllocationId &&
                    agent.reconciliationReason === undefined;
                  return (
                    <tr key={agent.instanceId}>
                      <td>
                        <details className="inline-details">
                          <summary>{agent.instanceId}</summary>
                          <dl className="key-value-list">
                            <div>
                              <dt>Software version</dt>
                              <dd>
                                <code>{agent.softwareVersion}</code>
                              </dd>
                            </div>
                            <div>
                              <dt>Worker runtimes</dt>
                              <dd>
                                {agent.supportedRuntimes.map((runtime) => (
                                  <code key={runtime}>{runtime}</code>
                                ))}
                              </dd>
                            </div>
                            <div>
                              <dt>Sandbox profiles</dt>
                              <dd>
                                {agent.supportedSandboxProfiles.map(
                                  (sandbox) => (
                                    <code key={sandbox}>{sandbox}</code>
                                  ),
                                )}
                              </dd>
                            </div>
                            <div>
                              <dt>Toolsets</dt>
                              <dd>
                                {agent.supportedToolsets.length === 0 ? (
                                  <span className="muted-copy">
                                    No usable Toolsets reported
                                  </span>
                                ) : (
                                  <ul
                                    aria-label={`Toolsets for ${agent.instanceId}`}
                                  >
                                    {agent.supportedToolsets.map((toolset) => (
                                      <li key={toolset.ref}>
                                        <code>{toolset.ref}</code>:{" "}
                                        {toolset.tools.map((tool) => (
                                          <code key={tool}>{tool}</code>
                                        ))}
                                      </li>
                                    ))}
                                  </ul>
                                )}
                              </dd>
                            </div>
                            <div>
                              <dt>Workspace</dt>
                              <dd>
                                {agent.workspaceCapabilities === undefined ? (
                                  <span className="muted-copy">
                                    Not configured
                                  </span>
                                ) : (
                                  <>
                                    <code>
                                      {agent.workspaceCapabilities.storage}
                                    </code>
                                    {" · "}
                                    {agent.workspaceCapabilities.modes.map(
                                      (mode) => (
                                        <code key={mode}>{mode}</code>
                                      ),
                                    )}
                                    {" · "}
                                    {agent.workspaceCapabilities.limits.maxFiles.toLocaleString()}{" "}
                                    files /{" "}
                                    {agent.workspaceCapabilities.limits.maxExpandedBytes.toLocaleString()}{" "}
                                    bytes expanded
                                  </>
                                )}
                              </dd>
                            </div>
                            <div>
                              <dt>Agent-reported allocation</dt>
                              <dd>
                                <code>
                                  {agent.currentAllocationId ?? "none"}
                                </code>
                              </dd>
                            </div>
                            <div>
                              <dt>Control Plane allocation</dt>
                              <dd>
                                <code>
                                  {agent.authoritativeAllocationId ?? "none"}
                                </code>
                              </dd>
                            </div>
                            <div>
                              <dt>Reconciliation reason</dt>
                              <dd>
                                <SafeReason
                                  reason={agent.reconciliationReason}
                                />
                              </dd>
                            </div>
                          </dl>
                        </details>
                      </td>
                      <td>
                        <OperationsState state={agent.observedState} />
                      </td>
                      <td>
                        <OperationsState state={agent.slotState} />
                      </td>
                      <td>
                        <OptionalTimestamp
                          value={agent.lastAcceptedHeartbeat}
                        />
                      </td>
                      <td>
                        <OptionalTimestamp value={agent.confirmedLeaseUntil} />
                      </td>
                      <td>
                        {agent.authoritativeAllocationId === undefined ? (
                          <span className="muted-copy">Unallocated</span>
                        ) : (
                          <Link to="/operations/allocations">
                            {agent.authoritativeAllocationId}
                          </Link>
                        )}
                        <small
                          className={
                            reconciled ? "reconciled" : "reconciliation-warning"
                          }
                        >
                          {reconciled
                            ? "observations agree"
                            : "reconciliation pending"}
                        </small>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <div className="notice notice-warning operations-safety-note">
          <strong>Self-termination remains the recovery boundary.</strong>
          <p>
            A fenced or stale slot is reconciled by Control Plane and Runtime
            lease rules. This UI intentionally has no force-idle or reassignment
            command.
          </p>
        </div>
      </div>
    </>
  );
}
