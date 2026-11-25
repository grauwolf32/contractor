import { observationsAgree } from "./capacity";
import { Link, useLocation } from "react-router";

import { Icon } from "../../../app/icon";
import { formatNumber } from "../../../app/format";
import { Facts, IdChip, TechnicalDetails } from "../../../ui";
import { useOperationsSnapshot } from "../context";
import type { RuntimeAgentObservation } from "../../../api/operations";
import {
  DisclosureChevron,
  OperationsState,
  OptionalTimestamp,
  SafeReason,
} from "../common";

/** What a registered process advertised, and its allocation binding. */
function ProcessFacts({ agent }: { agent: RuntimeAgentObservation }) {
  const workspace = agent.workspaceCapabilities;
  return (
    <Facts
      layout="rows"
      items={[
        {
          term: "Software version",
          value: <code>{agent.softwareVersion}</code>,
        },
        {
          term: "Worker runtimes",
          value: (
            <span className="ops-chips">
              {agent.supportedRuntimes.map((runtime) => (
                <code key={runtime}>{runtime}</code>
              ))}
            </span>
          ),
        },
        {
          term: "Sandbox profiles",
          value: (
            <span className="ops-chips">
              {agent.supportedSandboxProfiles.map((sandbox) => (
                <code key={sandbox}>{sandbox}</code>
              ))}
            </span>
          ),
        },
        {
          term: "Toolsets",
          value:
            agent.supportedToolsets.length === 0 ? (
              <span className="ops-muted">No usable Toolsets reported</span>
            ) : (
              <ul
                className="ops-toolsets"
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
            ),
        },
        {
          term: "Workspace",
          value:
            workspace === undefined ? (
              <span className="ops-muted">Not configured</span>
            ) : (
              <>
                <code>{workspace.storage}</code>
                {" · "}
                {workspace.modes.map((mode) => (
                  <code key={mode}>{mode}</code>
                ))}
                {" · "}
                {formatNumber(workspace.limits.maxFiles)} files /{" "}
                {formatNumber(workspace.limits.maxExpandedBytes)} bytes expanded
              </>
            ),
        },
        {
          term: "Agent-reported allocations",
          value: (
            <code>
              {agent.observedAllocations
                .map((item) => `${item.allocationId} (${item.observedState})`)
                .join(", ") || "none"}
            </code>
          ),
        },
        {
          term: "Control Plane allocations",
          value: (
            <code>{agent.authoritativeAllocationIds.join(", ") || "none"}</code>
          ),
        },
        {
          term: "Reconciliation reason",
          value: <SafeReason reason={agent.reconciliationReason} />,
        },
      ]}
    />
  );
}

/** One snapshot process; the full capability and lease record stays on demand. */
export function ProcessRow({
  agent,
  targeted = false,
}: {
  agent: RuntimeAgentObservation;
  targeted?: boolean;
}) {
  return (
    <details
      className="ops-process-row"
      id={`process-${agent.instanceId}`}
      open={targeted || undefined}
    >
      <summary>
        <span className="ops-process-name">
          <Icon name="operations" />
          <strong>{agent.instanceId}</strong>
        </span>
        <OperationsState state={agent.state} />
        <span className="ops-process-context">
          {`${agent.allocationSlots - agent.availableSlots} / ${agent.allocationSlots} occupied · ${agent.availableSlots} free · ${agent.fencedSlots} fenced`}
        </span>
        <DisclosureChevron />
      </summary>
      <div className="ops-process-details">
        <Facts
          layout="rows"
          items={[
            {
              term: "Process ID",
              value: <IdChip value={agent.instanceId} label="process ID" />,
            },
            {
              term: "Observed process",
              value: <OperationsState state={agent.observedState} />,
            },
            {
              term: "Assigned slot",
              value: <OperationsState state={agent.state} />,
            },
            {
              term: "Last accepted heartbeat",
              value: <OptionalTimestamp value={agent.lastAcceptedHeartbeat} />,
            },
            {
              term: "Confirmed lease until",
              value: <OptionalTimestamp value={agent.confirmedLeaseUntil} />,
            },
          ]}
        />
        <ProcessFacts agent={agent} />
        {agent.authoritativeAllocationIds.map((id) => (
          <Link
            key={id}
            className="ops-details-link"
            to={`/operations/allocations#${encodeURIComponent(id)}`}
          >
            Inspect allocation {id} →
          </Link>
        ))}
      </div>
    </details>
  );
}

/**
 * The registered processes of the current snapshot with their lease facts,
 * kept behind a disclosure: the cards above are the main view.
 */
export function ProcessInventory() {
  const { snapshot } = useOperationsSnapshot();
  const count = snapshot.runtimeAgents.length;
  const { hash } = useLocation();
  return (
    <section
      className="ops-section ops-inventory"
      aria-label="Process inventory"
    >
      <TechnicalDetails
        summary="Process inventory and lease diagnostics"
        defaultOpen={hash.startsWith("#process-")}
        description={`${count} current · A Runtime Agent is one deployed, long-running process with 1–4 allocation slots.`}
      >
        <div className="ops-stack">
          <div className="ops-section-head">
            <div className="ops-section-heading">
              <h3 className="ops-section-title">Registered processes</h3>
            </div>
            <span className="ops-section-aside">{count} current</span>
          </div>
          {count === 0 ? (
            <div className="ops-empty">
              <strong>No Runtime Agent is currently registered.</strong>
              <p>The snapshot shows current process state only.</p>
            </div>
          ) : (
            <div className="ops-table-wrap">
              <table className="ops-table" data-stack="">
                <thead>
                  <tr>
                    <th>Runtime Agent</th>
                    <th>Observed</th>
                    <th>Assigned slot</th>
                    <th>Last accepted heartbeat</th>
                    <th>Confirmed lease</th>
                    <th>Allocation binding</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshot.runtimeAgents.map((agent) => {
                    const reconciled = observationsAgree(agent);
                    return (
                      <tr
                        key={agent.instanceId}
                        id={`process-${agent.instanceId}`}
                      >
                        <td data-label="Runtime Agent">
                          <details
                            className="ops-inventory-agent"
                            open={
                              hash ===
                                `#process-${encodeURIComponent(agent.instanceId)}` ||
                              undefined
                            }
                          >
                            <summary>{agent.instanceId}</summary>
                            <ProcessFacts agent={agent} />
                          </details>
                        </td>
                        <td data-label="Observed">
                          <OperationsState state={agent.observedState} />
                        </td>
                        <td data-label="Assigned slot">
                          <OperationsState state={agent.state} />
                        </td>
                        <td data-label="Last accepted heartbeat">
                          <OptionalTimestamp
                            value={agent.lastAcceptedHeartbeat}
                          />
                        </td>
                        <td data-label="Confirmed lease">
                          <OptionalTimestamp
                            value={agent.confirmedLeaseUntil}
                          />
                        </td>
                        <td data-label="Allocation binding">
                          {agent.authoritativeAllocationIds.length === 0 ? (
                            <span className="ops-muted">Unallocated</span>
                          ) : (
                            agent.authoritativeAllocationIds.map((id) => (
                              <Link
                                key={id}
                                className="ui-mono"
                                to={`/operations/allocations#${encodeURIComponent(id)}`}
                              >
                                {id}
                              </Link>
                            ))
                          )}
                          <small>
                            {agent.availableSlots} / {agent.allocationSlots}{" "}
                            free · {agent.fencedSlots} fenced
                          </small>
                          <small
                            className="ops-reconcile"
                            data-pending={reconciled ? undefined : ""}
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
          <div className="notice notice-warning">
            <strong>Self-termination remains the recovery boundary.</strong>
            <p>
              A fenced or stale slot is reconciled by Control Plane and Runtime
              lease rules. This UI intentionally has no force-idle or
              reassignment command.
            </p>
          </div>
        </div>
      </TechnicalDetails>
    </section>
  );
}
