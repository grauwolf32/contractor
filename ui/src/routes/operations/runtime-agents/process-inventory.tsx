import { Link } from "react-router";

import { TechnicalDetails } from "../../../ui";
import { useOperationsSnapshot } from "../context";
import { OperationsState, OptionalTimestamp, SafeReason } from "../common";

/**
 * The registered processes of the current snapshot with their lease facts,
 * kept behind a disclosure: the cards above are the main view.
 */
export function ProcessInventory() {
  const { snapshot } = useOperationsSnapshot();
  const count = snapshot.runtimeAgents.length;
  return (
    <section
      className="ops-section ops-inventory"
      aria-label="Process inventory"
    >
      <TechnicalDetails
        summary="Process inventory and lease diagnostics"
        description={`${count} current · A Runtime Agent is one deployed, long-running single-slot process.`}
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
                    const reconciled =
                      agent.currentAllocationId ===
                        agent.authoritativeAllocationId &&
                      agent.reconciliationReason === undefined;
                    return (
                      <tr key={agent.instanceId}>
                        <td data-label="Runtime Agent">
                          <details className="ops-inventory-agent">
                            <summary>{agent.instanceId}</summary>
                            <dl className="ops-facts">
                              <div>
                                <dt>Software version</dt>
                                <dd>
                                  <code>{agent.softwareVersion}</code>
                                </dd>
                              </div>
                              <div>
                                <dt>Worker runtimes</dt>
                                <dd className="ops-chips">
                                  {agent.supportedRuntimes.map((runtime) => (
                                    <code key={runtime}>{runtime}</code>
                                  ))}
                                </dd>
                              </div>
                              <div>
                                <dt>Sandbox profiles</dt>
                                <dd className="ops-chips">
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
                                    <span className="ops-muted">
                                      No usable Toolsets reported
                                    </span>
                                  ) : (
                                    <ul
                                      className="ops-toolsets"
                                      aria-label={`Toolsets for ${agent.instanceId}`}
                                    >
                                      {agent.supportedToolsets.map(
                                        (toolset) => (
                                          <li key={toolset.ref}>
                                            <code>{toolset.ref}</code>:{" "}
                                            {toolset.tools.map((tool) => (
                                              <code key={tool}>{tool}</code>
                                            ))}
                                          </li>
                                        ),
                                      )}
                                    </ul>
                                  )}
                                </dd>
                              </div>
                              <div>
                                <dt>Workspace</dt>
                                <dd>
                                  {agent.workspaceCapabilities === undefined ? (
                                    <span className="ops-muted">
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
                        <td data-label="Observed">
                          <OperationsState state={agent.observedState} />
                        </td>
                        <td data-label="Assigned slot">
                          <OperationsState state={agent.slotState} />
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
                          {agent.authoritativeAllocationId === undefined ? (
                            <span className="ops-muted">Unallocated</span>
                          ) : (
                            <Link
                              className="ops-mono"
                              to={`/operations/allocations#${encodeURIComponent(agent.authoritativeAllocationId)}`}
                            >
                              {agent.authoritativeAllocationId}
                            </Link>
                          )}
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
