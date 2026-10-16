import { useId, useState } from "react";
import { Link } from "react-router";

import type {
  AllocationObservation,
  RuntimeAgentPrincipal,
  RuntimeLabelBinding,
} from "../../../api/operations";
import { Icon } from "../../../app/icon";
import {
  formatAge,
  formatBytes,
  formatNumber,
  formatTimestamp,
} from "../../../app/format";
import {
  Button,
  Facts,
  IdChip,
  StatusChip,
  TechnicalDetails,
} from "../../../ui";
import { OperationsState, OptionalTimestamp, SafeReason } from "../common";
import { agentDisplayName, availabilityCopy } from "./identity";
import { AgentLabelsDialog } from "./labels-dialog";

const adapterNames: Record<string, string> = {
  "caido-graphql@1": "Caido",
  "otlp-http@1": "Telemetry",
  "http-proxy@1": "HTTP proxy",
};

function CapabilityRefs({ values }: { values: string[] }) {
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
function AgentLabelChips({ labels }: { labels: readonly string[] }) {
  return (
    <div className="ops-chips runtime-agent-label-chips">
      {labels.length === 0 ? (
        <span className="ui-caption">No labels assigned</span>
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

/** The diagnostics of a live process, for the Technical details. */
function liveFacts(live: NonNullable<RuntimeAgentPrincipal["live"]>) {
  return [
    { term: "Process ID", value: <code>{live.instanceId}</code> },
    {
      term: "Observed process / server slot",
      value: (
        <span className="ops-chips">
          <OperationsState state={live.observedState} />
          <OperationsState state={live.slotState} />
        </span>
      ),
    },
    {
      term: "Last accepted heartbeat",
      value: <OptionalTimestamp value={live.lastAcceptedHeartbeat} />,
    },
    {
      term: "Confirmed lease until",
      value: <OptionalTimestamp value={live.confirmedLeaseUntil} />,
    },
    {
      term: "Supported adapters at startup",
      value: <CapabilityRefs values={live.supportedRuntimeAdapters} />,
    },
    {
      term: "Worker runtimes",
      value: <CapabilityRefs values={live.supportedRuntimes} />,
    },
    {
      term: "Sandbox profiles",
      value: <CapabilityRefs values={live.supportedSandboxProfiles} />,
    },
    {
      term: "Workspace limits",
      value: live.workspaceCapabilities !== undefined && (
        <>
          {formatNumber(live.workspaceCapabilities.limits.maxFiles)} files ·{" "}
          {formatBytes(live.workspaceCapabilities.limits.maxExpandedBytes)}{" "}
          expanded
        </>
      ),
    },
    {
      term: "Agent-reported allocation",
      value: <code>{live.currentAllocationId ?? "None"}</code>,
    },
    {
      term: "Control Plane allocation",
      value: <code>{live.authoritativeAllocationId ?? "None"}</code>,
    },
    {
      term: "Reconciliation reason",
      value: <SafeReason reason={live.reconciliationReason} />,
    },
    {
      term: "Toolsets",
      value:
        live.supportedToolsets.length === 0 ? (
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
        ),
    },
  ];
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
          <Facts
            className="ops-glance ops-agent-metrics"
            items={[
              {
                term: "Slot",
                value: <SlotValue slotState={live.slotState} />,
              },
              {
                term: "Last heartbeat",
                value:
                  live.lastAcceptedHeartbeat === undefined ? (
                    "Not observed"
                  ) : (
                    <time
                      dateTime={live.lastAcceptedHeartbeat}
                      title={formatTimestamp(live.lastAcceptedHeartbeat)}
                    >
                      {formatAge(live.lastAcceptedHeartbeat, now)}
                    </time>
                  ),
              },
              {
                term: "Workspace",
                value:
                  live.workspaceCapabilities === undefined ? (
                    "Not observed"
                  ) : (
                    <span>
                      {live.workspaceCapabilities.storage === "local"
                        ? "Local"
                        : "Memory"}
                      <small>
                        {live.workspaceCapabilities.modes.join(" · ")}
                      </small>
                    </span>
                  ),
              },
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
          <p className="ui-caption">
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
            <Button
              size="xs"
              icon={<Icon name="settings" />}
              disabled={bindings === undefined}
              aria-haspopup="dialog"
              onClick={() => setEditing(true)}
            >
              Edit labels
            </Button>
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
          <Facts
            layout="rows"
            items={[
              {
                term: "Agent ID",
                value: <code>{principal.runtimeAgentId}</code>,
              },
              { term: "Labels revision", value: principal.revision },
              {
                term: "Required adapters",
                value: (
                  <CapabilityRefs values={principal.requiredRuntimeAdapters} />
                ),
              },
              ...(live === undefined ? [] : liveFacts(live)),
            ]}
          />
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
