import { useOperationsSnapshot } from "./context";

export function OperationsOverviewRoute() {
  const { snapshot } = useOperationsSnapshot();
  const mismatches = snapshot.runtimeAgents.filter(
    (agent) =>
      agent.reconciliationReason !== undefined ||
      agent.currentAllocationId !== agent.authoritativeAllocationId,
  ).length;
  return (
    <div className="operations-overview-grid">
      <article className="panel operations-summary-card">
        <p className="eyebrow">Deployed processes</p>
        <strong>{snapshot.runtimeAgents.length}</strong>
        <h3>Runtime Agents</h3>
        <p>Each is one long-running, single-slot process.</p>
      </article>
      <article className="panel operations-summary-card">
        <p className="eyebrow">Temporary roles</p>
        <strong>{snapshot.allocations.length}</strong>
        <h3>Allocations</h3>
        <p>A Worker exists only as the role bound to one allocation.</p>
      </article>
      <article className="panel operations-summary-card">
        <p className="eyebrow">Reconciliation</p>
        <strong>{mismatches}</strong>
        <h3>Visible mismatches</h3>
        <p>Observed and authoritative facts remain deliberately separate.</p>
      </article>
      <div className="panel operations-snapshot-record">
        <h3>Snapshot identity</h3>
        <dl className="key-value-list">
          <div>
            <dt>Generation</dt>
            <dd>
              <code>{snapshot.cursor.generation}</code>
            </dd>
          </div>
          <div>
            <dt>Revision</dt>
            <dd>
              <code>{snapshot.cursor.revision}</code>
            </dd>
          </div>
        </dl>
        <p className="muted-copy">
          A Server restart changes generation. Any missed revision causes a full
          REST resynchronization.
        </p>
      </div>
    </div>
  );
}
