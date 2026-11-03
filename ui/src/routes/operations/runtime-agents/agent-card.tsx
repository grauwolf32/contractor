import { useId, useState } from "react";
import { Link } from "react-router";

import type {
  AllocationObservation,
  RuntimeAgentPrincipal,
  RuntimeLabelBinding,
} from "../../../api/operations";
import { Icon } from "../../../app/icon";
import { formatAge, formatBytes, formatNumber } from "../../../app/format";
import { Button, Facts, IdChip, StatusChip } from "../../../ui";
import {
  EntryHeading,
  OperationsState,
  OptionalTimestamp,
  SafeReason,
} from "../common";
import { agentDisplayName, availabilityCopy } from "./identity";
import { AgentLabelsDialog } from "./labels-dialog";

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

/** The complete live process record, visible when its inventory row is open. */
function liveFacts(live: NonNullable<RuntimeAgentPrincipal["live"]>) {
  return [
    { term: "Connection", value: "Online" },
    { term: "Software", value: `v${live.softwareVersion}` },
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
      wide: true,
      value:
        live.supportedToolsets.length === 0 ? (
          "None reported"
        ) : (
          <ul
            className="ops-toolsets"
            aria-label={`Toolsets for ${live.instanceId}`}
          >
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
      <details className="ops-agent-disclosure">
        <summary>
          <EntryHeading
            icon="operations"
            title={<h3 id={heading}>{agentDisplayName(principal)}</h3>}
            subtitle={live?.instanceId ?? "No current process"}
          >
            <StatusChip tone={availability.tone}>
              {availability.label}
            </StatusChip>
          </EntryHeading>
        </summary>
        <div className="ops-resource-body">
          {principal.missingRuntimeAdapters.length === 0 ? null : (
            <div className="ops-agent-warning">
              <strong>
                Missing {principal.missingRuntimeAdapters.join(", ")}
              </strong>
              <p>
                Assigned labels require adapters this process did not advertise
                at startup.
              </p>
            </div>
          )}
          {principal.availability === "slot_unavailable" || reconciling ? (
            <div className="ops-agent-warning">
              <strong>
                {reconciling
                  ? "State reconciliation pending"
                  : "Slot unavailable"}
              </strong>
              <p>
                {live === undefined
                  ? "No current slot observation."
                  : `Process: ${live.observedState}. Server slot: ${live.slotState}.`}
              </p>
              {live?.reconciliationReason === undefined ? null : (
                <SafeReason reason={live.reconciliationReason} />
              )}
            </div>
          ) : null}
          <Facts
            className="ops-agent-facts"
            layout="rows"
            items={[
              {
                term: "Agent ID",
                value: (
                  <IdChip value={principal.runtimeAgentId} label="Agent ID" />
                ),
              },
              {
                term: "Saved labels",
                value: <AgentLabelChips labels={principal.labels} />,
              },
              { term: "Labels revision", value: principal.revision },
              {
                term: "Required adapters",
                value: (
                  <CapabilityRefs values={principal.requiredRuntimeAdapters} />
                ),
              },
              ...(live === undefined
                ? []
                : [
                    {
                      term: "Slot occupancy",
                      value: <SlotValue slotState={live.slotState} />,
                    },
                    {
                      term: "Heartbeat age",
                      value:
                        live.lastAcceptedHeartbeat === undefined ? (
                          "Not observed"
                        ) : (
                          <time dateTime={live.lastAcceptedHeartbeat}>
                            {formatAge(live.lastAcceptedHeartbeat, now)}
                          </time>
                        ),
                    },
                    {
                      term: "Workspace",
                      value:
                        live.workspaceCapabilities === undefined
                          ? "Not observed"
                          : `${live.workspaceCapabilities.storage} · ${live.workspaceCapabilities.modes.join(" · ")}`,
                    },
                    ...liveFacts(live),
                  ]),
            ]}
          />
          <div className="ops-agent-actions">
            <Button
              size="sm"
              icon={<Icon name="settings" />}
              disabled={bindings === undefined}
              aria-haspopup="dialog"
              onClick={() => setEditing(true)}
            >
              Edit labels
            </Button>
            {allocation === undefined ? null : (
              <Link to={`/runs/${encodeURIComponent(allocation.runId)}`}>
                {allocation.logicalWorker} · Open Run
              </Link>
            )}
            {live?.authoritativeAllocationId === undefined ? null : (
              <Link
                to={`/operations/allocations#${encodeURIComponent(live.authoritativeAllocationId)}`}
              >
                Inspect allocation →
              </Link>
            )}
          </div>
        </div>
      </details>
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
