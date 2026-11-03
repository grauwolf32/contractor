import { Link, useLocation } from "react-router";
import type { RuntimeAgentObservation } from "../../api/operations";
import type { StatusTone } from "../../app/status-tone";
import { Icon } from "../../app/icon";
import { plural } from "../../app/format";
import { StatusGlyph } from "../../ui";
import { OpsSection } from "./common";
import { useOperationsSnapshot } from "./context";
import { ProcessRow } from "./runtime-agents/process-inventory";
import { AllocationEntry } from "./allocations";

const SLOT_STATES: readonly {
  state: RuntimeAgentObservation["slotState"];
  label: string;
  tone: StatusTone;
}[] = [
  { state: "idle", label: "Idle", tone: "idle" },
  { state: "busy", label: "Busy", tone: "progress" },
  { state: "reserved", label: "Reserved", tone: "progress" },
  { state: "draining", label: "Draining", tone: "warning" },
  { state: "fenced", label: "Fenced", tone: "blocked" },
];

export function OperationsOverviewRoute() {
  const { snapshot } = useOperationsSnapshot();
  const { hash } = useLocation();
  const agents = snapshot.runtimeAgents;
  const mismatches = agents.filter(
    (agent) =>
      agent.reconciliationReason !== undefined ||
      agent.currentAllocationId !== agent.authoritativeAllocationId,
  );
  const counts = SLOT_STATES.map((slot) => ({
    ...slot,
    count: agents.filter((agent) => agent.slotState === slot.state).length,
  }));
  const draining = counts.find((slot) => slot.state === "draining")!.count;
  const summary = counts
    .filter((slot) => slot.count > 0)
    .map((slot) => `${slot.count} ${slot.label.toLowerCase()}`)
    .join(", ");
  return (
    <div className="ops-overview ops-stack">
      <ul className="ops-tiles" aria-label="Snapshot counts">
        <li>
          <Link className="ops-tile" to="/operations/runtime-agents">
            <span className="ops-tile-label">
              <Icon name="operations" /> Runtime processes
            </span>
            <span className="ops-tile-value">
              <strong>{agents.length}</strong>
            </span>
            <span className="ui-caption">One slot per deployed process</span>
          </Link>
        </li>
        <li>
          <Link className="ops-tile" to="/operations/allocations">
            <span className="ops-tile-label">
              <Icon name="binding" /> Current allocations
            </span>
            <span className="ops-tile-value">
              <strong>{snapshot.allocations.length}</strong>
            </span>
            <span className="ui-caption">Temporary Runtime roles</span>
          </Link>
        </li>
        <li>
          <Link
            className="ops-tile"
            data-warning={mismatches.length > 0 || undefined}
            to="/operations/runtime-agents"
          >
            <span className="ops-tile-label">
              <Icon name="issues" /> Reconciliation
            </span>
            <span
              className="ops-tile-value"
              data-tone={mismatches.length > 0 ? "warning" : undefined}
            >
              <strong>{mismatches.length}</strong>
            </span>
            <span className="ui-caption">
              Observed allocation differs from assignment
            </span>
          </Link>
        </li>
      </ul>
      <div className="ops-readiness-grid">
        <OpsSection
          className="ops-readiness-panel"
          id="operations-readiness-heading"
          title="Execution slots"
          aside={plural(agents.length, "slot", "slots")}
        >
          {agents.length === 0 ? (
            <p className="ops-empty">
              No Runtime processes are present in this snapshot. Inspect agent
              readiness before starting execution.
            </p>
          ) : (
            <div
              className="ops-slot-track"
              role="img"
              aria-label={`${plural(agents.length, "Runtime Agent slot", "Runtime Agent slots")}: ${summary}`}
            >
              {counts
                .filter((slot) => slot.count > 0)
                .map((slot) => (
                  <span
                    key={slot.state}
                    data-state={slot.state}
                    style={{ flexGrow: slot.count }}
                  />
                ))}
            </div>
          )}
          <ul className="ops-legend" aria-label="Slots by state">
            {counts.map((slot) => (
              <li key={slot.state}>
                <span className="ops-slot-pill" data-state={slot.state}>
                  {slot.label}
                </span>{" "}
                <strong className="ops-legend-count">{slot.count}</strong>
              </li>
            ))}
          </ul>
          <p className="ops-capacity-note">
            Idle slots do not guarantee capacity for a particular Run.
          </p>
        </OpsSection>
        <OpsSection
          className="ops-inspection-panel"
          id="operations-inspection-heading"
          title="Needs inspection"
        >
          {mismatches.length === 0 ? (
            <p className="ui-caption">
              No reconciliation mismatch in this snapshot.
            </p>
          ) : (
            mismatches.slice(0, 4).map((agent) => (
              <div key={agent.instanceId} className="ops-inspection-item">
                <StatusGlyph tone="warning" size={16} />
                <div>
                  <Link
                    to={`${agents.slice(0, 6).includes(agent) ? "" : "/operations/runtime-agents"}#process-${encodeURIComponent(agent.instanceId)}`}
                  >
                    {agent.instanceId}
                  </Link>
                  <p>
                    {agent.reconciliationReason?.code.replaceAll("_", " ") ??
                      (agent.slotState === "reserved"
                        ? "Reserved — awaiting a Runtime allocation observation."
                        : "Observed allocation differs from its assignment.")}
                  </p>
                </div>
              </div>
            ))
          )}
          {draining === 0 ? null : (
            <p className="ui-caption">
              {plural(draining, "draining slot", "draining slots")} · inspect
              current allocations.
            </p>
          )}
          <Link className="ops-details-link" to="/operations/runtime-agents">
            Inspect Runtime Agents →
          </Link>
        </OpsSection>
      </div>
      <OpsSection
        id="operations-processes-heading"
        title="Runtime processes"
        aside={`${agents.length} current`}
        actions={<Link to="/operations/runtime-agents">Runtime Agents →</Link>}
      >
        <div className="ops-process-list">
          {agents.slice(0, 6).map((agent) => (
            <ProcessRow
              key={agent.instanceId}
              agent={agent}
              targeted={
                hash === `#process-${encodeURIComponent(agent.instanceId)}`
              }
            />
          ))}
        </div>
        {agents.length > 6 ? (
          <Link to="/operations/runtime-agents">
            View all {agents.length} processes →
          </Link>
        ) : null}
      </OpsSection>
      <OpsSection
        id="operations-current-allocations-heading"
        title="Current allocations"
        aside={`${snapshot.allocations.length} current`}
        actions={
          <Link to="/operations/allocations/completed">
            Completed history →
          </Link>
        }
      >
        <p className="ui-caption">
          Runtime “prepared” confirms a role was prepared; it does not prove a
          Worker call is currently executing.
        </p>
        {snapshot.allocations.length === 0 ? (
          <p className="ops-empty">No current allocations in this snapshot.</p>
        ) : (
          <ul className="ops-allocation-list">
            {snapshot.allocations.slice(0, 6).map((allocation) => (
              <AllocationEntry
                key={allocation.allocationId}
                allocation={allocation}
                targeted={false}
              />
            ))}
          </ul>
        )}
        {snapshot.allocations.length > 6 ? (
          <Link to="/operations/allocations">
            View all {snapshot.allocations.length} allocations →
          </Link>
        ) : null}
      </OpsSection>
      <ul className="ops-shortcuts" aria-label="Readiness shortcuts">
        <li>
          <Link to="/operations/configuration">Runtime configuration →</Link>
        </li>
        <li>
          <Link to="/runs">Inspect Run wait reasons →</Link>
        </li>
        <li>
          <Link to="/operations/performance">Performance →</Link>
        </li>
      </ul>
    </div>
  );
}
