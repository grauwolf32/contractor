import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import {
  deleteRuntimeAgentPrincipal,
  type AllocationObservation,
  type RuntimeAgentPrincipal,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { Icon } from "../../../app/icon";
import {
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../../artifacts/common";
import { OperationsState, OptionalTimestamp, SafeReason } from "../common";
import { AgentLabelsDialog } from "./labels-dialog";

const availabilityCopy = {
  available: { label: "Available", symbol: "✓" },
  busy: { label: "Busy", symbol: "◷" },
  offline: { label: "Offline", symbol: "○" },
  slot_unavailable: { label: "Slot unavailable", symbol: "!" },
  adapter_capability_mismatch: { label: "Missing adapter", symbol: "!" },
} as const;
const adapterNames: Record<string, string> = {
  "caido-graphql@1": "Caido",
  "otlp-http@1": "Telemetry",
  "http-proxy@1": "HTTP proxy",
};

function heartbeatAge(value: string | undefined, now: number): string {
  if (value === undefined) return "Not observed";
  const seconds = Math.max(0, Math.floor((now - Date.parse(value)) / 1000));
  if (!Number.isFinite(seconds)) return "Not observed";
  if (seconds < 5) return "Just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function CapabilityRefs({ values }: { values: string[] }) {
  return values.length === 0 ? (
    <span className="muted-copy">None</span>
  ) : (
    <span className="runtime-agent-exact-refs">
      {values.map((value) => (
        <code key={value}>{value}</code>
      ))}
    </span>
  );
}

export function AgentCard({
  principal,
  bindings,
  allocations,
  now,
}: {
  principal: RuntimeAgentPrincipal;
  bindings: RuntimeLabelBinding[] | undefined;
  allocations: AllocationObservation[];
  now: number;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const heading = useId();
  const diagnostics = useRef<HTMLDetailsElement>(null);
  const [editing, setEditing] = useState(false);
  const [copyStatus, setCopyStatus] = useState<string>();
  const live = principal.live;
  const availability = availabilityCopy[principal.availability];
  const shortId = `${principal.runtimeAgentId.slice(0, 6)}…${principal.runtimeAgentId.slice(-4)}`;
  const allocation =
    live === undefined
      ? undefined
      : allocations.find(
          (value) =>
            value.allocationId === live.authoritativeAllocationId &&
            value.runtimeAgentInstanceId === live.instanceId,
        );
  const reconciling =
    live !== undefined &&
    (live.currentAllocationId !== live.authoritativeAllocationId ||
      live.reconciliationReason !== undefined);
  const deletion = useMutation({
    mutationFn: () =>
      deleteRuntimeAgentPrincipal(
        api,
        principal.runtimeAgentId,
        principal.revision,
        `delete-runtime-principal-ui-${crypto.randomUUID()}`,
      ),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.snapshot,
        }),
      ]);
    },
    onError: () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
      }),
  });
  async function copyId() {
    try {
      await navigator.clipboard.writeText(principal.runtimeAgentId);
      setCopyStatus("Agent ID copied.");
    } catch {
      if (diagnostics.current) diagnostics.current.open = true;
      setCopyStatus(
        "Copy unavailable. Select the full Agent ID in diagnostics.",
      );
    }
  }
  return (
    <article className="panel runtime-principal-card" aria-labelledby={heading}>
      <div className="runtime-agent-summary">
        <div className="runtime-agent-heading">
          <span className="runtime-agent-avatar">
            <Icon name="operations" />
          </span>
          <div className="runtime-agent-identity">
            <div className="runtime-agent-name">
              <h4 id={heading} title={principal.runtimeAgentId}>
                Agent · {shortId}
              </h4>
              <button
                className="runtime-agent-copy"
                type="button"
                title="Copy Agent ID"
                aria-label="Copy Agent ID"
                onClick={() => void copyId()}
              >
                <Icon name="copy" />
              </button>
            </div>
            <p className="runtime-agent-connection">
              <span
                className={`runtime-agent-dot ${live === undefined ? "is-offline" : ""}`}
                aria-hidden="true"
              />
              {live === undefined ? (
                "No live process"
              ) : (
                <>
                  Online <span aria-hidden="true">·</span> v
                  {live.softwareVersion}
                </>
              )}
            </p>
          </div>
          <span
            className={`runtime-agent-status status-${principal.availability.replaceAll("_", "-")}`}
          >
            <span aria-hidden="true">{availability.symbol}</span>
            {availability.label}
          </span>
        </div>
        {copyStatus === undefined ? null : (
          <p className="runtime-agent-copy-status" role="status">
            {copyStatus}
          </p>
        )}
        <dl className="runtime-agent-metrics">
          <div>
            <dt>Slot</dt>
            <dd>
              {live === undefined ? (
                "No process"
              ) : live.slotState === "idle" ? (
                <>
                  0 / 1 <small>occupied</small>
                </>
              ) : live.slotState === "fenced" ? (
                "Fenced"
              ) : (
                <>
                  1 / 1 <small>{live.slotState}</small>
                </>
              )}
            </dd>
          </div>
          <div>
            <dt>Last heartbeat</dt>
            <dd>
              {live?.lastAcceptedHeartbeat === undefined ? (
                "Not observed"
              ) : (
                <time
                  dateTime={live.lastAcceptedHeartbeat}
                  title={formatTimestamp(live.lastAcceptedHeartbeat)}
                >
                  {heartbeatAge(live.lastAcceptedHeartbeat, now)}
                </time>
              )}
            </dd>
          </div>
          <div>
            <dt>Workspace</dt>
            <dd>
              {live?.workspaceCapabilities === undefined ? (
                "Not observed"
              ) : (
                <>
                  <span className="runtime-agent-storage">
                    {live.workspaceCapabilities.storage}
                  </span>
                  <small>{live.workspaceCapabilities.modes.join(" · ")}</small>
                </>
              )}
            </dd>
          </div>
        </dl>
        {live?.authoritativeAllocationId === undefined ? null : (
          <div className="runtime-agent-allocation">
            <Icon name="runs" />
            <div>
              {allocation === undefined ? null : (
                <Link to={`/runs/${encodeURIComponent(allocation.runId)}`}>
                  {allocation.logicalWorker} · Open Run
                </Link>
              )}
              <Link
                to={`/operations/allocations#${encodeURIComponent(live.authoritativeAllocationId)}`}
              >
                View allocation <span aria-hidden="true">↗</span>
              </Link>
            </div>
          </div>
        )}
        {principal.availability === "offline" ? (
          <p className="runtime-agent-offline-note">
            Offline · saved labels are retained for the next registration.
          </p>
        ) : null}
        {principal.missingRuntimeAdapters.length > 0 ? (
          <div className="notice notice-warning runtime-agent-warning">
            <strong>
              Missing {principal.missingRuntimeAdapters.join(", ")}
            </strong>
            <p>
              Assigned labels require adapters this process did not advertise at
              startup.
            </p>
          </div>
        ) : null}
        {principal.availability === "slot_unavailable" || reconciling ? (
          <div className="notice notice-warning runtime-agent-warning">
            <strong>
              {reconciling
                ? "State reconciliation pending"
                : "Slot unavailable"}
            </strong>
            <p>
              {live === undefined ? (
                "No current slot observation."
              ) : (
                <>
                  Process: {live.observedState}. Server slot: {live.slotState}.
                </>
              )}
            </p>
            {live?.reconciliationReason === undefined ? null : (
              <SafeReason reason={live.reconciliationReason} />
            )}
          </div>
        ) : null}
        <div className="runtime-agent-label-heading">
          <span>Agent labels</span>
          <button
            className="runtime-agent-edit"
            type="button"
            disabled={bindings === undefined || deletion.isPending}
            aria-haspopup="dialog"
            onClick={() => setEditing(true)}
          >
            <Icon name="settings" />
            Edit labels
          </button>
        </div>
        <div className="runtime-agent-label-chips">
          {principal.labels.length === 0 ? (
            <span className="muted-copy">No labels assigned</span>
          ) : (
            principal.labels.map((label) => (
              <span className="runtime-agent-label-chip" key={label}>
                {label}
              </span>
            ))
          )}
        </div>
        {live === undefined ||
        live.supportedRuntimeAdapters.length === 0 ? null : (
          <div
            className="runtime-agent-capabilities"
            aria-label="Supported adapters"
          >
            {live.supportedRuntimeAdapters.map((adapter) => (
              <span key={adapter} title={adapter}>
                {adapterNames[adapter] ?? adapter}
              </span>
            ))}
          </div>
        )}
      </div>
      <details className="runtime-agent-diagnostics" ref={diagnostics}>
        <summary>Capabilities and diagnostics</summary>
        <dl className="key-value-list">
          <div>
            <dt>Agent ID</dt>
            <dd>
              <code>{principal.runtimeAgentId}</code>
            </dd>
          </div>
          <div>
            <dt>Labels revision</dt>
            <dd>{principal.revision}</dd>
          </div>
          <div>
            <dt>Required adapters</dt>
            <dd>
              <CapabilityRefs values={principal.requiredRuntimeAdapters} />
            </dd>
          </div>
          {live === undefined ? null : (
            <>
              <div>
                <dt>Process ID</dt>
                <dd>
                  <code>{live.instanceId}</code>
                </dd>
              </div>
              <div>
                <dt>Observed process / server slot</dt>
                <dd>
                  <OperationsState state={live.observedState} />{" "}
                  <OperationsState state={live.slotState} />
                </dd>
              </div>
              <div>
                <dt>Last accepted heartbeat</dt>
                <dd>
                  <OptionalTimestamp value={live.lastAcceptedHeartbeat} />
                </dd>
              </div>
              <div>
                <dt>Confirmed lease until</dt>
                <dd>
                  <OptionalTimestamp value={live.confirmedLeaseUntil} />
                </dd>
              </div>
              <div>
                <dt>Supported adapters at startup</dt>
                <dd>
                  <CapabilityRefs values={live.supportedRuntimeAdapters} />
                </dd>
              </div>
              <div>
                <dt>Worker runtimes</dt>
                <dd>
                  <CapabilityRefs values={live.supportedRuntimes} />
                </dd>
              </div>
              <div>
                <dt>Sandbox profiles</dt>
                <dd>
                  <CapabilityRefs values={live.supportedSandboxProfiles} />
                </dd>
              </div>
              {live.workspaceCapabilities === undefined ? null : (
                <div>
                  <dt>Workspace limits</dt>
                  <dd>
                    {live.workspaceCapabilities.limits.maxFiles.toLocaleString()}{" "}
                    files ·{" "}
                    {formatBytes(
                      live.workspaceCapabilities.limits.maxExpandedBytes,
                    )}{" "}
                    expanded
                  </dd>
                </div>
              )}
              <div>
                <dt>Agent-reported allocation</dt>
                <dd>
                  <code>{live.currentAllocationId ?? "None"}</code>
                </dd>
              </div>
              <div>
                <dt>Control Plane allocation</dt>
                <dd>
                  <code>{live.authoritativeAllocationId ?? "None"}</code>
                </dd>
              </div>
              <div>
                <dt>Reconciliation reason</dt>
                <dd>
                  <SafeReason reason={live.reconciliationReason} />
                </dd>
              </div>
              <div>
                <dt>Toolsets</dt>
                <dd>
                  {live.supportedToolsets.length === 0 ? (
                    "None reported"
                  ) : (
                    <ul className="runtime-agent-toolsets">
                      {live.supportedToolsets.map((toolset) => (
                        <li key={toolset.ref}>
                          <code>{toolset.ref}</code>
                          <CapabilityRefs values={toolset.tools} />
                        </li>
                      ))}
                    </ul>
                  )}
                </dd>
              </div>
            </>
          )}
        </dl>
        {live === undefined && principal.labels.length === 0 ? (
          <button
            className="danger-button runtime-agent-remove"
            type="button"
            disabled={deletion.isPending}
            onClick={() => deletion.mutate()}
          >
            {deletion.isPending ? "Removing…" : "Remove offline principal"}
          </button>
        ) : null}
      </details>
      {deletion.error === null ? null : <ErrorNotice error={deletion.error} />}
      {editing && bindings !== undefined ? (
        <AgentLabelsDialog
          principal={principal}
          bindings={bindings}
          onClose={() => setEditing(false)}
        />
      ) : null}
    </article>
  );
}
