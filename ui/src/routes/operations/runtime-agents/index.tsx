import { Link } from "react-router";

import { useOperationsSnapshot } from "../context";
import { OperationsState, OptionalTimestamp, SafeReason } from "../common";

export function RuntimeAgentListRoute() {
  const { snapshot } = useOperationsSnapshot();
  return (
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
                              {agent.supportedSandboxProfiles.map((sandbox) => (
                                <code key={sandbox}>{sandbox}</code>
                              ))}
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
                            <dt>Agent-reported allocation</dt>
                            <dd>
                              <code>{agent.currentAllocationId ?? "none"}</code>
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
                              <SafeReason reason={agent.reconciliationReason} />
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
                      <OptionalTimestamp value={agent.lastAcceptedHeartbeat} />
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
  );
}
