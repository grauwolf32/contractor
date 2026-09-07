import { Link } from "react-router";

import { useOperationsSnapshot } from "../context";
import {
  ConfigurationRefLink,
  MetricsSummary,
  OperationsState,
  SafeReason,
} from "../common";
import { exactConfigurationRef } from "../references";
import { AllocationViewTabs } from "./tabs";

export function AllocationListRoute() {
  const { snapshot } = useOperationsSnapshot();
  return (
    <div className="operations-library">
      <AllocationViewTabs />
      <section className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">
              Authoritative ownership + Runtime observation
            </p>
            <h3>Current allocations</h3>
            <p className="muted-copy">
              Allocated means the temporary Worker role was prepared; it does
              not prove that an A2A model call is executing now.
            </p>
          </div>
          <span>{snapshot.allocations.length} unreleased</span>
        </div>
        {snapshot.allocations.length === 0 ? (
          <div className="compact-empty">
            <strong>No active allocation exists.</strong>
            <p>Released allocations leave this current-state projection.</p>
          </div>
        ) : (
          <div className="allocation-card-list">
            {snapshot.allocations.map((allocation) => (
              <details
                className="allocation-card"
                key={allocation.allocationId}
              >
                <summary>
                  <span>
                    <strong>{allocation.logicalWorker}</strong>
                    <code>{allocation.allocationId}</code>
                  </span>
                  <span>
                    <OperationsState state={allocation.authoritativePhase} />
                    <OperationsState state={allocation.observedPhase} />
                  </span>
                </summary>
                <div className="allocation-card-body">
                  <dl className="key-value-list allocation-bindings">
                    <div>
                      <dt>Workflow Run</dt>
                      <dd>
                        <Link
                          to={`/runs/${encodeURIComponent(allocation.runId)}`}
                        >
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
                          value={exactConfigurationRef(
                            allocation.agentTemplate,
                          )}
                        />
                      </dd>
                    </div>
                    <div>
                      <dt>ModelPolicy</dt>
                      <dd>
                        <ConfigurationRefLink
                          value={exactConfigurationRef(
                            allocation.executionConfig.modelPolicy,
                          )}
                        />
                      </dd>
                    </div>
                    <div>
                      <dt>LLM Gateway</dt>
                      <dd>
                        {allocation.executionConfig.llmGateway === undefined ? (
                          <span className="muted-copy">
                            Awaiting route resolution
                          </span>
                        ) : (
                          <ConfigurationRefLink
                            value={exactConfigurationRef(
                              allocation.executionConfig.llmGateway,
                            )}
                          />
                        )}
                      </dd>
                    </div>
                    <div>
                      <dt>Credential</dt>
                      <dd>
                        {allocation.executionConfig.credential === undefined ? (
                          <span className="muted-copy">
                            Unauthenticated Gateway access
                          </span>
                        ) : (
                          <Link
                            to={`/operations/credentials/${encodeURIComponent(allocation.executionConfig.credential.credentialId)}`}
                          >
                            {allocation.executionConfig.credential.credentialId}
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
                  <div className="attempt-block">
                    <h4>Safe aggregate metrics</h4>
                    <MetricsSummary metrics={allocation.metrics} />
                    {!allocation.metrics.reportsComplete ? (
                      <p className="muted-copy">
                        Runtime has not submitted its final report; zero
                        counters are incomplete observations, not proof of no
                        work.
                      </p>
                    ) : null}
                  </div>
                </div>
              </details>
            ))}
          </div>
        )}
        <p className="muted-copy">
          Lifecycle actions are Scheduler-owned. Operations cannot finish,
          abort, release, or reassign an allocation.
        </p>
      </section>
    </div>
  );
}
