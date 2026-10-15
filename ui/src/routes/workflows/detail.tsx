import { useQuery } from "@tanstack/react-query";
import { type ReactNode } from "react";
import { Link, useLocation, useParams } from "react-router";

import type { components } from "../../api/generated/public";
import { usePublicAPI } from "../../api/context";
import { agentPath } from "../../api/agents";
import { queryKeys } from "../../api/query-keys";
import {
  CONFIG_NAME_PATTERN,
  CONFIG_VERSION_PATTERN,
  getWorkflow,
  type WorkflowResource,
} from "../../api/workflows";
import { ErrorNotice } from "../../app/error-notice";
import { compactDigest, plural } from "../../app/format";
import { QueryView } from "../../app/query-view";
import { ButtonLink, Facts, TechnicalDetails } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { LibraryBackLink } from "../catalog/library-parts";
import { locationDestination } from "../catalog/navigation";
import { ConsumerConfigList } from "./consumer-config";
import { WorkflowOverview } from "./overview";
import { workflowSelector } from "./presentation";

type SuccessTransition = components["schemas"]["WorkflowSucceededTransition"];
type FailureTransition = components["schemas"]["WorkflowFailureTransition"];

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
        retry up to {plural(transition.maxAttempts, "attempt")}, then{" "}
        <TerminalTransition transition={transition.then} />
      </span>
    );
  }
  if (transition.kind === "escalate") {
    const ref = transition.executionConfig.ref;
    return (
      <div className="workflow-escalation">
        <span>
          escalate up to {plural(transition.maxAttempts, "attempt")} using{" "}
          {ref === undefined ? (
            "an inline resolved profile"
          ) : (
            <code>
              {ref.configId}@{ref.version}
            </code>
          )}
          , then <TerminalTransition transition={transition.then} />
        </span>
        <ConsumerConfigList
          config={transition.executionConfig.effective}
          workerLabel={(name) => `Agent ${name}`}
          className="workflow-consumer-list"
        />
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
          {plural(Object.keys(stage.agents).length, "logical Worker")}
        </span>
      </summary>
      <div className="workflow-stage-body">
        <div>
          <h4>Objective</h4>
          <p>{stage.objective}</p>
        </div>
        <Facts
          layout="list"
          size="sm"
          className="workflow-stage-facts"
          items={[
            {
              term: "Planner",
              value: (
                <code>
                  {stage.planner.plannerId}@{stage.planner.version}
                </code>
              ),
            },
            {
              term: "Instructions",
              value: (
                <>
                  <code>{stage.instructions.ref}</code>
                  <small title={stage.instructions.digest}>
                    {compactDigest(stage.instructions.digest)}
                  </small>
                </>
              ),
            },
          ]}
        />
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
          <ConsumerConfigList
            config={stage.executionConfig}
            workerLabel={(name) => `Agent ${name}`}
            className="workflow-consumer-list"
          />
        </div>
        <div>
          <h4>Scheduler transitions</h4>
          <Facts
            layout="list"
            size="sm"
            className="workflow-stage-facts"
            items={[
              {
                term: "succeeded",
                value: <TerminalTransition transition={stage.on.succeeded} />,
              },
              {
                term: "failed",
                value: <FailureTransitionView transition={stage.on.failed} />,
              },
              {
                term: "interrupted",
                value: (
                  <FailureTransitionView transition={stage.on.interrupted} />
                ),
              },
            ]}
          />
        </div>
      </div>
    </details>
  );
}

export function WorkflowDetailRoute() {
  const api = usePublicAPI();
  const { name = "", version = "" } = useParams();
  const valid =
    CONFIG_NAME_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.workflows.detail(name, version),
    queryFn: ({ signal }) => getWorkflow(api, name, version, signal),
    enabled: valid,
  });
  return (
    <div className="ui-stack workflow-overview-page">
      <LibraryBackLink
        fallback={{
          returnTo: "/catalog/workflows",
          returnLabel: "All workflows",
        }}
      />
      {valid ? (
        <QueryView
          query={query}
          loading={
            <p className="library-muted" role="status">
              Loading Workflow contract…
            </p>
          }
        >
          {(workflow) => (
            <WorkflowOverview
              key={workflowSelector(workflow)}
              workflow={workflow}
              refreshing={query.isFetching}
              refresh={() => void query.refetch()}
            >
              <ButtonLink to="/catalog/studio?kind=Workflow">
                Open Node Studio
              </ButtonLink>
              <TechnicalDetails
                summary="Agents, execution settings and transitions"
                className="workflow-technical-details"
              >
                <div className="ui-stack">
                  {Object.keys(workflow.stages)
                    .sort()
                    .map((stageName) => (
                      <StageContract
                        key={stageName}
                        name={stageName}
                        workflow={workflow}
                      />
                    ))}
                </div>
              </TechnicalDetails>
            </WorkflowOverview>
          )}
        </QueryView>
      ) : (
        <ErrorNotice error={new Error("Workflow route is invalid")} />
      )}
    </div>
  );
}
