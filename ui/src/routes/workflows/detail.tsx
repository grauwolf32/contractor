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
import { ErrorNotice } from "../../app/error-notice";
import { locationDestination } from "../catalog/navigation";
import { WorkflowOverview } from "./overview";
import { workflowSelector } from "./presentation";

import { compactDigest } from "../../app/format";
import { artifactDetailPath } from "../artifacts/paths";

type ConsumerConfig = components["schemas"]["ConsumerExecutionConfig"];
type ResolvedConfig = components["schemas"]["ResolvedStageExecutionConfig"];
type SuccessTransition = components["schemas"]["WorkflowSucceededTransition"];
type FailureTransition = components["schemas"]["WorkflowFailureTransition"];

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
    <ul className="workflow-consumer-list">
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
      <div className="workflow-escalation">
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
    <details className="workflow-stage" open={name === workflow.entryStage}>
      <summary>
        <span className="workflow-stage-name">
          <code>{name}</code>
          {name === workflow.entryStage ? (
            <span className="workflow-entry-label">Entry</span>
          ) : null}
        </span>
        <span className="workflow-stage-summary">
          {stage.planner.plannerId}@{stage.planner.version} ·{" "}
          {Object.keys(stage.agents).length} logical Worker
          {Object.keys(stage.agents).length === 1 ? "" : "s"}
        </span>
      </summary>
      <div className="workflow-stage-body">
        <div>
          <h4>Objective</h4>
          <p>{stage.objective}</p>
        </div>
        <dl className="workflow-stage-facts">
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
          <ul className="workflow-agent-list">
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
                          to={artifactDetailPath(
                            { kind: "user" },
                            {
                              namespace: skill.namespace,
                              name: skill.name,
                            },
                          )}
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
          <dl className="workflow-stage-facts">
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
      <div className="library-detail">
        <ErrorNotice error={new Error("Workflow route is invalid")} />
        <Link className="library-back" to="/catalog/workflows">
          Return to Workflows
        </Link>
      </div>
    );
  }

  if (query.isPending)
    return (
      <p className="library-muted" role="status">
        Loading Workflow contract…
      </p>
    );
  if (!query.data)
    return (
      <div className="library-detail">
        <ErrorNotice error={query.error} />
        <Link className="library-back" to="/catalog/workflows">
          Return to Workflows
        </Link>
      </div>
    );

  return (
    <WorkflowOverview
      key={workflowSelector(query.data)}
      workflow={query.data}
      refreshing={query.isFetching}
      refresh={() => void query.refetch()}
      error={query.error}
    >
      <Link className="ui-btn" to="/catalog/studio?kind=Workflow">
        Open Node Studio
      </Link>
      <details className="workflow-technical-details">
        <summary>
          <svg
            className="workflow-technical-chevron"
            width="13"
            height="13"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            focusable="false"
          >
            <path d="M9.5 6l6 6-6 6" />
          </svg>
          Agents, execution settings and transitions
        </summary>
        <div className="workflow-technical-details-body">
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
