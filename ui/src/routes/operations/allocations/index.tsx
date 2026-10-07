import { Link, useLocation } from "react-router";

import type { AllocationObservation } from "../../../api/operations";
import { IdChip } from "../../../ui";
import { useOperationsSnapshot } from "../context";
import {
  ConfigurationRefLink,
  DisclosureChevron,
  MetricsSummary,
  OperationsState,
  OpsSection,
  SafeReason,
} from "../common";
import { exactConfigurationRef } from "../references";
import { AllocationViewTabs } from "./tabs";

function AllocationEntry({
  allocation,
  targeted,
}: {
  allocation: AllocationObservation;
  targeted: boolean;
}) {
  const config = allocation.executionConfig;
  return (
    <li>
      <details
        className="ops-allocation allocation-card"
        id={allocation.allocationId}
        open={targeted || undefined}
        data-targeted={targeted ? "" : undefined}
      >
        <summary>
          <span className="ops-allocation-title">
            <DisclosureChevron />
            <strong>{allocation.logicalWorker}</strong>{" "}
            <code className="ops-digest">{allocation.allocationId}</code>
          </span>{" "}
          <span className="ops-allocation-phases">
            <OperationsState
              prefix="Control Plane"
              state={allocation.authoritativePhase}
            />
            <OperationsState
              prefix="Runtime"
              state={allocation.observedPhase}
            />
          </span>
        </summary>
        <div className="ops-allocation-body">
          <div>
            <h3>Bindings</h3>
            <dl className="ops-facts">
              <div>
                <dt>Allocation</dt>
                <dd>
                  <IdChip
                    value={allocation.allocationId}
                    label="allocation ID"
                  />
                </dd>
              </div>
              <div>
                <dt>Workflow Run</dt>
                <dd>
                  <Link to={`/runs/${encodeURIComponent(allocation.runId)}`}>
                    {allocation.runId}
                  </Link>
                </dd>
              </div>
              <div>
                <dt>StageExecution</dt>
                <dd>
                  <code>{allocation.stageExecutionId}</code>
                </dd>
              </div>
              <div>
                <dt>Runtime Agent process</dt>
                <dd>
                  <code>{allocation.runtimeAgentInstanceId}</code>
                </dd>
              </div>
              <div>
                <dt>Temporary logical Worker</dt>
                <dd>
                  <code>{allocation.logicalWorker}</code>
                </dd>
              </div>
              <div>
                <dt>AgentTemplate</dt>
                <dd>
                  <ConfigurationRefLink
                    value={exactConfigurationRef(allocation.agentTemplate)}
                  />
                </dd>
              </div>
              <div>
                <dt>ModelPolicy</dt>
                <dd>
                  {config.modelPolicy === undefined ? (
                    <span>No model</span>
                  ) : (
                    <ConfigurationRefLink
                      value={exactConfigurationRef(config.modelPolicy)}
                    />
                  )}
                </dd>
              </div>
              <div>
                <dt>LLM Gateway</dt>
                <dd>
                  {config.llmGateway === undefined ? (
                    <span className="ops-muted">
                      {config.modelPolicy === undefined
                        ? "Not required"
                        : "Awaiting route resolution"}
                    </span>
                  ) : (
                    <ConfigurationRefLink
                      value={exactConfigurationRef(config.llmGateway)}
                    />
                  )}
                </dd>
              </div>
              <div>
                <dt>Credential</dt>
                <dd>
                  {config.credential === undefined ? (
                    <span className="ops-muted">
                      {config.modelPolicy === undefined
                        ? "Not required"
                        : "Unauthenticated Gateway access"}
                    </span>
                  ) : (
                    <Link
                      to={`/operations/credentials/${encodeURIComponent(config.credential.credentialId)}`}
                    >
                      {config.credential.credentialId}
                    </Link>
                  )}
                </dd>
              </div>
              <div>
                <dt>Bounded reason</dt>
                <dd>
                  <SafeReason reason={allocation.reason} />
                </dd>
              </div>
              <div>
                <dt>Exhausted execution-safety dimension</dt>
                <dd>
                  <code>{allocation.exhaustedDimension ?? "none"}</code>
                </dd>
              </div>
            </dl>
          </div>
          <div>
            <h3>Safe aggregate metrics</h3>
            <MetricsSummary metrics={allocation.metrics} />
            {!allocation.metrics.reportsComplete ? (
              <p className="ops-note">
                Runtime has not submitted its final report; zero counters are
                incomplete observations, not proof of no work.
              </p>
            ) : null}
          </div>
        </div>
      </details>
    </li>
  );
}

export function AllocationListRoute() {
  const { snapshot } = useOperationsSnapshot();
  const { hash } = useLocation();
  return (
    <div className="ops-stack">
      <AllocationViewTabs />
      <OpsSection
        id="current-allocations-heading"
        title="Current allocations"
        description="Allocated means the temporary Worker role was prepared; it does not prove that an A2A model call is executing now."
        aside={`${snapshot.allocations.length} unreleased`}
      >
        {snapshot.allocations.length === 0 ? (
          <div className="ops-empty">
            <strong>No active allocation exists.</strong>
            <p>Released allocations leave this current-state projection.</p>
          </div>
        ) : (
          <ul className="ops-allocation-list">
            {snapshot.allocations.map((allocation) => (
              <AllocationEntry
                key={allocation.allocationId}
                allocation={allocation}
                targeted={
                  hash === `#${encodeURIComponent(allocation.allocationId)}`
                }
              />
            ))}
          </ul>
        )}
        <p className="ops-note">
          Lifecycle actions are Scheduler-owned. Operations cannot finish,
          abort, release, or reassign an allocation.
        </p>
      </OpsSection>
    </div>
  );
}
