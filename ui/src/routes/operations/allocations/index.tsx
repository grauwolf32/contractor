import { Link, useLocation } from "react-router";

import type { AllocationObservation } from "../../../api/operations";
import { Facts, IdChip } from "../../../ui";
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
            <Facts
              layout="rows"
              items={[
                {
                  term: "Allocation",
                  value: (
                    <IdChip
                      value={allocation.allocationId}
                      label="allocation ID"
                    />
                  ),
                },
                {
                  term: "Workflow Run",
                  value: (
                    <Link to={`/runs/${encodeURIComponent(allocation.runId)}`}>
                      {allocation.runId}
                    </Link>
                  ),
                },
                {
                  term: "StageExecution",
                  value: <code>{allocation.stageExecutionId}</code>,
                },
                {
                  term: "Runtime Agent process",
                  value: <code>{allocation.runtimeAgentInstanceId}</code>,
                },
                {
                  term: "Temporary logical Worker",
                  value: <code>{allocation.logicalWorker}</code>,
                },
                {
                  term: "AgentTemplate",
                  value: (
                    <ConfigurationRefLink
                      value={exactConfigurationRef(allocation.agentTemplate)}
                    />
                  ),
                },
                {
                  term: "ModelPolicy",
                  value:
                    config.modelPolicy === undefined ? (
                      <span>No model</span>
                    ) : (
                      <ConfigurationRefLink
                        value={exactConfigurationRef(config.modelPolicy)}
                      />
                    ),
                },
                {
                  term: "LLM Gateway",
                  value:
                    config.llmGateway === undefined ? (
                      <span className="ops-muted">
                        {config.modelPolicy === undefined
                          ? "Not required"
                          : "Awaiting route resolution"}
                      </span>
                    ) : (
                      <ConfigurationRefLink
                        value={exactConfigurationRef(config.llmGateway)}
                      />
                    ),
                },
                {
                  term: "Credential",
                  value:
                    config.credential === undefined ? (
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
                    ),
                },
                {
                  term: "Bounded reason",
                  value: <SafeReason reason={allocation.reason} />,
                },
                {
                  term: "Exhausted execution-safety dimension",
                  value: <code>{allocation.exhaustedDimension ?? "none"}</code>,
                },
              ]}
            />
          </div>
          <div>
            <h3>Safe aggregate metrics</h3>
            <MetricsSummary metrics={allocation.metrics} />
            {!allocation.metrics.reportsComplete ? (
              <p className="ui-caption">
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
        <p className="ui-caption">
          Lifecycle actions are Scheduler-owned. Operations cannot finish,
          abort, release, or reassign an allocation.
        </p>
      </OpsSection>
    </div>
  );
}
