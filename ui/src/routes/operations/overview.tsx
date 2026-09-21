import { Link } from "react-router";

import { useOperationsSnapshot } from "./context";
import "./overview.css";

export function OperationsOverviewRoute() {
  const { snapshot } = useOperationsSnapshot();
  const mismatches = snapshot.runtimeAgents.filter(
    (agent) =>
      agent.reconciliationReason !== undefined ||
      agent.currentAllocationId !== agent.authoritativeAllocationId,
  ).length;
  return (
    <div className="operations-overview-grid">
      <section
        className="panel operations-readiness"
        aria-labelledby="operations-readiness-heading"
      >
        <p className="eyebrow">Authoritative process snapshot</p>
        <h3 id="operations-readiness-heading">Execution readiness</h3>
        <dl className="metadata-grid">
          <div>
            <dt>Observed idle slots</dt>
            <dd>
              {
                snapshot.runtimeAgents.filter(
                  (agent) => agent.slotState === "idle",
                ).length
              }
            </dd>
          </div>
          <div>
            <dt>Reserved or busy</dt>
            <dd>
              {
                snapshot.runtimeAgents.filter(
                  (agent) =>
                    agent.slotState === "reserved" ||
                    agent.slotState === "busy",
                ).length
              }
            </dd>
          </div>
          <div>
            <dt>Draining or fenced</dt>
            <dd>
              {
                snapshot.runtimeAgents.filter(
                  (agent) =>
                    agent.slotState === "draining" ||
                    agent.slotState === "fenced",
                ).length
              }
            </dd>
          </div>
        </dl>
        <p>
          Idle slots do not establish compatible capacity for a particular Run.
          Placement depends on its requirements and each Runtime Agent’s
          capabilities and resolved settings.
        </p>
        {snapshot.runtimeAgents.length === 0 ? (
          <p className="notice">
            No Runtime processes are present in this snapshot. Inspect agent
            readiness before starting execution.
          </p>
        ) : null}
        <ul
          className="operations-readiness-links"
          aria-label="Readiness shortcuts"
        >
          <li>
            <Link to="/operations/runtime-agents">
              Inspect Runtime Agents →
            </Link>
          </li>
          <li>
            <Link to="/runs/configuration">Runtime configuration →</Link>
          </li>
          <li>
            <Link to="/runs">Inspect Run wait reasons →</Link>
          </li>
        </ul>
      </section>
      <Link
        className="panel operations-summary-card"
        to="/operations/runtime-agents"
      >
        <p className="eyebrow">Deployed processes</p>
        <strong>{snapshot.runtimeAgents.length}</strong>
        <h3>Runtime Agents</h3>
        <p>Each is one long-running, single-slot process.</p>
      </Link>
      <Link
        className="panel operations-summary-card"
        to="/operations/allocations"
      >
        <p className="eyebrow">Temporary roles</p>
        <strong>{snapshot.allocations.length}</strong>
        <h3>Allocations</h3>
        <p>A Worker exists only as the role bound to one allocation.</p>
      </Link>
      <Link
        className="panel operations-summary-card"
        to="/operations/runtime-agents"
      >
        <p className="eyebrow">Reconciliation</p>
        <strong>{mismatches}</strong>
        <h3>Visible mismatches</h3>
        <p>Observed and authoritative facts remain deliberately separate.</p>
      </Link>
    </div>
  );
}
