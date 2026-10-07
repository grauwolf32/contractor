import { useId, useState } from "react";
import { Link } from "react-router";

import type {
  AllocationObservation,
  RuntimeAgentPrincipal,
  RuntimeLabelBinding,
} from "../../../api/operations";
import { Icon } from "../../../app/icon";
import { formatBytes, formatTimestamp } from "../../../app/format";
import { IdChip, StatusChip, TechnicalDetails } from "../../../ui";
import {
  Glance,
  OperationsState,
  OptionalTimestamp,
  SafeReason,
} from "../common";
import { agentDisplayName, availabilityCopy, relativeAge } from "./identity";
import { AgentLabelsDialog } from "./labels-dialog";

const adapterNames: Record<string, string> = {
  "caido-graphql@1": "Caido",
  "otlp-http@1": "Telemetry",
  "http-proxy@1": "HTTP proxy",
};

export function CapabilityRefs({ values }: { values: string[] }) {
  return values.length === 0 ? (
    <span className="ops-muted">None</span>
  ) : (
    <span className="ops-chips">
      {values.map((value) => (
        <code className="ops-label-chip" key={value}>
          {value}
        </code>
      ))}
    </span>
  );
}

/** Saved Agent labels as chips, or a line saying there are none. */
export function AgentLabelChips({ labels }: { labels: readonly string[] }) {
  return (
    <div className="ops-chips runtime-agent-label-chips">
      {labels.length === 0 ? (
        <span className="ops-note">No labels assigned</span>
      ) : (
        labels.map((label) => (
          <span className="ops-label-chip" key={label}>
            {label}
          </span>
        ))
      )}
    </div>
  );
}

function SlotValue({
  slotState,
}: {
  slotState: NonNullable<RuntimeAgentPrincipal["live"]>["slotState"];
}) {
  if (slotState === "idle") {
    return (
      <>
        0 / 1 <small>occupied</small>
      </>
    );
  }
  if (slotState === "fenced") return <>Fenced</>;
  return (
    <>
      1 / 1 <small>{slotState}</small>
    </>
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
  const heading = useId();
  const [editing, setEditing] = useState(false);
  const live = principal.live;
  const availability = availabilityCopy[principal.availability];
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
  return (
    <article
      className="ops-agent-card runtime-principal-card"
      aria-labelledby={heading}
    >
      <div className="ops-agent-main">
        <header className="ops-agent-head">
          <div className="ops-agent-identity">
            <div className="ops-agent-name">
              <h3 id={heading}>{agentDisplayName(principal)}</h3>
              <IdChip value={principal.runtimeAgentId} label="Agent ID" />
            </div>
            <p className="ops-agent-connection">
              <span
                className="ops-dot"
                data-offline={live === undefined ? "" : undefined}
                aria-hidden="true"
              />
              {live === undefined ? (
                "No live process observed"
              ) : (
                <>
                  Online <span aria-hidden="true">·</span> v
                  {live.softwareVersion}
                </>
              )}
            </p>
          </div>
          <StatusChip tone={availability.tone}>{availability.label}</StatusChip>
        </header>

        {live === undefined ? (
          <p className="ops-agent-no-process">No live process observed</p>
        ) : (
          <Glance
            className="ops-agent-metrics"
            items={[
              ["Slot", <SlotValue key="slot" slotState={live.slotState} />],
              [
                "Last heartbeat",
                live.lastAcceptedHeartbeat === undefined ? (
                  "Not observed"
                ) : (
                  <time
                    key="heartbeat"
                    dateTime={live.lastAcceptedHeartbeat}
                    title={formatTimestamp(live.lastAcceptedHeartbeat)}
                  >
                    {relativeAge(live.lastAcceptedHeartbeat, now)}
                  </time>
                ),
              ],
              [
                "Workspace",
                live.workspaceCapabilities === undefined ? (
                  "Not observed"
                ) : (
                  <span key="workspace">
                    {live.workspaceCapabilities.storage === "local"
                      ? "Local"
                      : "Memory"}
                    <small>
                      {live.workspaceCapabilities.modes.join(" · ")}
                    </small>
                  </span>
                ),
              ],
            ]}
          />
        )}

        {live?.authoritativeAllocationId === undefined ? null : (
          <div className="ops-agent-allocation">
            <Icon name="runs" />
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
        )}
        {principal.availability === "offline" ? (
          <p className="ops-note">
            Offline · saved labels are retained for the next registration.
          </p>
        ) : null}
        {principal.missingRuntimeAdapters.length > 0 ? (
          <div className="ops-agent-warning">
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
          <div className="ops-agent-warning">
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

        <div className="ops-agent-labels">
          <div className="ops-agent-labels-head">
            <span>Agent labels</span>
            <button
              className="ui-btn"
              data-size="xs"
              type="button"
              disabled={bindings === undefined}
              aria-haspopup="dialog"
              onClick={() => setEditing(true)}
            >
              <Icon name="settings" />
              Edit labels
            </button>
          </div>
          <AgentLabelChips labels={principal.labels} />
        </div>
        {live === undefined ||
        live.supportedRuntimeAdapters.length === 0 ? null : (
          <div className="ops-chips" aria-label="Supported adapters">
            {live.supportedRuntimeAdapters.map((adapter) => (
              <span className="ops-adapter-chip" key={adapter} title={adapter}>
                {adapterNames[adapter] ?? adapter}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="ops-tech">
        <TechnicalDetails summary="Capabilities and diagnostics">
          <dl className="ops-facts">
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
                  <dd className="ops-chips">
                    <OperationsState state={live.observedState} />
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
                      <ul className="ops-toolsets">
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
        </TechnicalDetails>
      </div>
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
