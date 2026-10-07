import { Link } from "react-router";

import type { RuntimeAgentObservation } from "../../api/operations";
import type { StatusTone } from "../../app/status-tone";
import { ProgressSegments, StatusGlyph } from "../../ui";
import { useOperationsSnapshot } from "./context";

type SlotState = RuntimeAgentObservation["slotState"];

/** Slot states in display order, each with its word and tone. */
const SLOT_STATES: readonly {
  state: SlotState;
  label: string;
  tone: StatusTone;
}[] = [
  { state: "idle", label: "Idle", tone: "idle" },
  { state: "busy", label: "Busy", tone: "progress" },
  { state: "reserved", label: "Reserved", tone: "progress" },
  { state: "draining", label: "Draining", tone: "warning" },
  { state: "fenced", label: "Fenced", tone: "blocked" },
];

function plural(count: number, one: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`;
}

/** A count that links to the section it summarizes. */
function OverviewTile({
  to,
  label,
  count,
  noun,
  text,
  warn = false,
}: {
  to: string;
  label: string;
  count: number;
  noun: readonly [one: string, many: string];
  text: string;
  warn?: boolean;
}) {
  // The spaces between the parts keep the link's accessible name readable;
  // the flex layout does not render them.
  return (
    <li>
      <Link className="ops-tile" to={to}>
        <span className="ops-tile-label">{label}</span>{" "}
        <span
          className="ops-tile-value"
          data-tone={warn ? "warning" : undefined}
        >
          {warn ? <StatusGlyph tone="warning" /> : null}
          <strong>{count}</strong> {count === 1 ? noun[0] : noun[1]}
        </span>{" "}
        <span className="ops-tile-text">{text}</span>
      </Link>
    </li>
  );
}

export function OperationsOverviewRoute() {
  const { snapshot } = useOperationsSnapshot();
  const agents = snapshot.runtimeAgents;
  const mismatches = agents.filter(
    (agent) =>
      agent.reconciliationReason !== undefined ||
      agent.currentAllocationId !== agent.authoritativeAllocationId,
  ).length;
  const counts = SLOT_STATES.map((slot) => ({
    ...slot,
    count: agents.filter((agent) => agent.slotState === slot.state).length,
  }));
  const ordered = counts.flatMap((slot) =>
    Array.from({ length: slot.count }, () => ({
      tone: slot.tone,
      label: slot.label,
    })),
  );
  const summary = counts
    .filter((slot) => slot.count > 0)
    .map((slot) => `${slot.count} ${slot.label.toLowerCase()}`)
    .join(", ");
  return (
    <div className="ops-stack">
      <section
        className="ops-section ops-readiness"
        aria-labelledby="operations-readiness-heading"
      >
        <header className="ops-section-head">
          <div className="ops-section-heading">
            <h2 id="operations-readiness-heading" className="ops-section-title">
              Execution readiness
            </h2>
            <p className="ops-section-description">
              Slots reported by the Runtime Agents in this snapshot.
            </p>
          </div>
        </header>
        <div className="ops-slot-line">
          <span className="ops-slot-total">
            {plural(agents.length, "slot", "slots")}
          </span>
          {agents.length === 0 ? null : (
            <ProgressSegments
              label={`${plural(agents.length, "Runtime Agent slot", "Runtime Agent slots")}: ${summary}`}
              segments={ordered}
            />
          )}
        </div>
        <ul className="ops-legend" aria-label="Slots by state">
          {counts.map((slot) => (
            <li key={slot.state}>
              <span
                className="ops-legend-swatch"
                data-tone={slot.tone}
                aria-hidden="true"
              />
              <span>{slot.label}</span>{" "}
              <span className="ops-legend-count">{slot.count}</span>
            </li>
          ))}
        </ul>
        <p className="ops-capacity-note">
          <StatusGlyph tone="info" size={15} />
          <span>
            Idle slots do not guarantee capacity for a particular Run.
          </span>
        </p>
        {agents.length === 0 ? (
          <div className="notice notice-warning" role="status">
            No Runtime processes are present in this snapshot. Inspect agent
            readiness before starting execution.
          </div>
        ) : null}
        <ul className="ops-shortcuts" aria-label="Readiness shortcuts">
          <li>
            <Link to="/operations/runtime-agents">
              Inspect Runtime Agents →
            </Link>
          </li>
          <li>
            <Link to="/operations/configuration">Runtime configuration →</Link>
          </li>
          <li>
            <Link to="/runs">Inspect Run wait reasons →</Link>
          </li>
        </ul>
      </section>

      <section
        className="ops-section"
        aria-labelledby="operations-glance-heading"
      >
        <header className="ops-section-head">
          <div className="ops-section-heading">
            <h2 id="operations-glance-heading" className="ops-section-title">
              Processes and roles
            </h2>
          </div>
        </header>
        <ul className="ops-tiles">
          <OverviewTile
            to="/operations/runtime-agents"
            label="Runtime Agents"
            count={agents.length}
            noun={["deployed process", "deployed processes"]}
            text="Each is one long-running, single-slot process."
          />
          <OverviewTile
            to="/operations/allocations"
            label="Allocations"
            count={snapshot.allocations.length}
            noun={["temporary role", "temporary roles"]}
            text="A Worker exists only as the role bound to one allocation."
          />
          <OverviewTile
            to="/operations/runtime-agents"
            label="Reconciliation"
            count={mismatches}
            noun={["visible mismatch", "visible mismatches"]}
            text="Runtime Agents whose observed state differs from their assignment."
            warn={mismatches > 0}
          />
        </ul>
      </section>
    </div>
  );
}
