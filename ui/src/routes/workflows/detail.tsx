import { useQuery } from "@tanstack/react-query";
import { type ReactNode } from "react";
import { Link, useLocation, useParams } from "react-router";

import type { components } from "../../api/generated/public";
import { usePublicAPI } from "../../api/context";
import { agentPath } from "../../api/agents";
import { queryKeys } from "../../api/query-keys";
import {
  CONFIG_ID_PATTERN,
  CONFIG_VERSION_PATTERN,
  getWorkflow,
  type WorkflowResource,
} from "../../api/workflows";
import { ErrorNotice } from "../artifacts/common";
import { locationDestination } from "../catalog/navigation";
import { WorkflowOverview } from "./overview";
import { workflowSelector } from "./presentation";

import "../primary-actions.css";

type ConsumerConfig = components["schemas"]["ConsumerExecutionConfig"];
type ResolvedConfig = components["schemas"]["ResolvedStageExecutionConfig"];
type SuccessTransition = components["schemas"]["WorkflowSucceededTransition"];
type FailureTransition = components["schemas"]["WorkflowFailureTransition"];

function compactDigest(digest: string): string {
  return `${digest.slice(0, 15)}…${digest.slice(-8)}`;
}

function ConsumerConfigView({
  name,
  config,
}: {
  name: string;
  config: ConsumerConfig;
}) {
  if (config.modelPolicy === undefined) {
    return (
      <li>
        <strong>{name}</strong>
        <span>Tool execution · no model</span>
      </li>
    );
  }
  return (
    <li>
      <strong>{name}</strong>
      <span>
        Model {config.modelPolicy.policyId}@{config.modelPolicy.version}
      </span>
      <span>
        {config.llmGateway === undefined
          ? "Gateway resolved during Runtime placement"
          : `Gateway ${config.llmGateway.gatewayId}@${config.llmGateway.version}`}
      </span>
      <span>Credential {config.credential?.credentialId ?? "none"}</span>
    </li>
  );
}

function ResolvedConfigView({ config }: { config: ResolvedConfig }) {
  return (
    <ul className="consumer-config-list">
      {config.planner === undefined ? null : (
        <ConsumerConfigView name="Planner" config={config.planner} />
      )}
      {Object.entries(config.agents)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([name, agent]) => (
          <ConsumerConfigView
            key={name}
            name={`Agent ${name}`}
            config={agent}
          />
        ))}
    </ul>
  );
}

function TerminalTransition({
  transition,
}: {
  transition:
    components["schemas"]["WorkflowNextOrFailTransition"] | SuccessTransition;
}): ReactNode {
  if (transition.kind === "next") {
    return (
      <span>
        next → <code>{transition.nextStage}</code>
      </span>
    );
  }
  return <span>{transition.kind}</span>;
}

function FailureTransitionView({
  transition,
}: {
  transition: FailureTransition;
}): ReactNode {
  if (transition.kind === "retry") {
    return (
      <span>
        retry up to {transition.maxAttempts}{" "}
        {transition.maxAttempts === 1 ? "attempt" : "attempts"}, then{" "}
        <TerminalTransition transition={transition.then} />
      </span>
    );
  }
  if (transition.kind === "escalate") {
    const ref = transition.executionConfig.ref;
    return (
      <div className="escalation-contract">
        <span>
          escalate up to {transition.maxAttempts}{" "}
          {transition.maxAttempts === 1 ? "attempt" : "attempts"} using{" "}
          {ref === undefined ? (
            "an inline resolved profile"
          ) : (
            <code>
              {ref.configId}@{ref.version}
            </code>
          )}
          , then <TerminalTransition transition={transition.then} />
        </span>
        <ResolvedConfigView config={transition.executionConfig.effective} />
      </div>
    );
  }
  return <TerminalTransition transition={transition} />;
}

function StageContract({
  name,
  workflow,
}: {
  name: string;
  workflow: WorkflowResource;
}) {
  const location = useLocation();
  const stage = workflow.stages[name];
  if (stage === undefined) {
    return null;
  }
  return (
    <details className="stage-contract" open={name === workflow.entryStage}>
      <summary>
        <span>
          <code>{name}</code>
          {name === workflow.entryStage ? <strong>entry</strong> : null}
        </span>
        <span>
          {stage.planner.plannerId}@{stage.planner.version} ·{" "}
          {Object.keys(stage.agents).length} logical Worker
          {Object.keys(stage.agents).length === 1 ? "" : "s"}
        </span>
      </summary>
      <div className="stage-contract-body">
        <div>
          <h4>Immutable objective</h4>
          <p>{stage.objective}</p>
        </div>
        <dl className="stage-reference-grid">
          <div>
            <dt>Planner</dt>
            <dd>
              <code>
                {stage.planner.plannerId}@{stage.planner.version}
              </code>
            </dd>
          </div>
          <div>
            <dt>Instructions</dt>
            <dd>
              <code>{stage.instructions.ref}</code>
              <small title={stage.instructions.digest}>
                {compactDigest(stage.instructions.digest)}
              </small>
            </dd>
          </div>
        </dl>
        <div>
          <h4>Logical Workers</h4>
          <ul className="agent-contract-list">
            {Object.entries(stage.agents)
              .sort(([left], [right]) => left.localeCompare(right))
              .map(([logicalName, binding]) => (
                <li key={logicalName}>
                  <strong>{logicalName}</strong>
                  <Link
                    to={agentPath(
                      binding.template.templateId,
                      binding.template.version,
                    )}
                    state={{
                      returnTo: locationDestination(location),
                      returnLabel: workflowSelector(workflow),
                      returnState: location.state,
                    }}
                  >
                    {binding.template.templateId}@{binding.template.version}
                  </Link>
                  <span>namespace {binding.namespace}</span>
                  {(binding.skills ?? []).length === 0 ? (
                    <span className="muted-copy">No global Skills</span>
                  ) : (
                    <span className="workflow-skill-links">
                      Skills:{" "}
                      {(binding.skills ?? []).map((skill) => (
                        <Link
                          key={`${skill.namespace}/${skill.name}`}
                          to={`/artifacts/${encodeURIComponent(skill.namespace)}/${encodeURIComponent(skill.name)}`}
                        >
                          {skill.namespace}/{skill.name}
                        </Link>
                      ))}
                    </span>
                  )}
                </li>
              ))}
          </ul>
        </div>
        <div>
          <h4>Resolved base execution config</h4>
          <ResolvedConfigView config={stage.executionConfig} />
        </div>
        <div>
          <h4>Scheduler transitions</h4>
          <dl className="transition-grid">
            <div>
              <dt>succeeded</dt>
              <dd>
                <TerminalTransition transition={stage.on.succeeded} />
              </dd>
            </div>
            <div>
              <dt>failed</dt>
              <dd>
                <FailureTransitionView transition={stage.on.failed} />
              </dd>
            </div>
            <div>
              <dt>interrupted</dt>
              <dd>
                <FailureTransitionView transition={stage.on.interrupted} />
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </details>
  );
}

export function WorkflowDetailRoute() {
  const api = usePublicAPI();
  const { name = "", version = "" } = useParams();
  const valid =
    CONFIG_ID_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.workflows.detail(name, version),
    queryFn: ({ signal }) => getWorkflow(api, name, version, signal),
    enabled: valid,
  });
  if (!valid) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Workflow route is invalid")} />
        <Link to="/catalog/workflows">Return to Workflows</Link>
      </section>
    );
  }

  if (query.isPending)
    return (
      <p className="loading-copy" aria-live="polite">
        Loading exact Workflow contract…
      </p>
    );
  if (!query.data)
    return (
      <section className="route-page">
        <ErrorNotice error={query.error} />
        <Link to="/catalog/workflows">Return to Workflows</Link>
      </section>
    );

  return (
    <WorkflowOverview
      key={workflowSelector(query.data)}
      workflow={query.data}
      refreshing={query.isFetching}
      refresh={() => void query.refetch()}
      error={query.error}
    >
      <details className="workflow-technical-details">
        <summary>Agents, execution settings and transitions</summary>
        <div className="workflow-technical-details-body workflow-stages">
          {Object.keys(query.data.stages)
            .sort()
            .map((stageName) => (
              <StageContract
                key={stageName}
                name={stageName}
                workflow={query.data}
              />
            ))}
        </div>
      </details>
    </WorkflowOverview>
  );
}
