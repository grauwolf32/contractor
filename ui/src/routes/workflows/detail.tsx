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
import { catalogReturnState, locationDestination } from "../catalog/navigation";
import { WorkflowRunForm } from "./run-form";
import {
  workflowDescription,
  workflowDisplayName,
  workflowSelector,
} from "./presentation";

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

function SlotContract({
  title,
  slots,
}: {
  title: string;
  slots: WorkflowResource["inputs"];
}) {
  return (
    <div className="panel contract-slots">
      <p className="eyebrow">{title}</p>
      {Object.entries(slots).length === 0 ? (
        <p className="compact-empty">None declared.</p>
      ) : (
        <ul>
          {Object.entries(slots)
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([name, slot]) => (
              <li key={name}>
                <code>{name}</code>
                <span>{slot.required ? "required" : "optional"}</span>
                <span>{slot.mediaTypes.join(", ")}</span>
              </li>
            ))}
        </ul>
      )}
    </div>
  );
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
  const location = useLocation();
  const { name = "", version = "" } = useParams();
  const valid =
    CONFIG_ID_PATTERN.test(name) && CONFIG_VERSION_PATTERN.test(version);
  const query = useQuery({
    queryKey: queryKeys.workflows.detail(name, version),
    queryFn: ({ signal }) => getWorkflow(api, name, version, signal),
    enabled: valid,
  });
  const back = catalogReturnState(location.state, {
    returnTo: "/catalog/workflows",
    returnLabel: "All Workflows",
  });

  function focusRunSetup(): void {
    const setup = document.getElementById("workflow-run-setup");
    setup?.scrollIntoView?.({ block: "start", behavior: "smooth" });
    setup?.focus({ preventScroll: true });
  }

  if (!valid) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Workflow route is invalid")} />
        <Link to="/catalog/workflows">Return to Workflows</Link>
      </section>
    );
  }

  return (
    <section className="route-page workflow-page">
      <header className="route-header-row">
        <div>
          <Link
            className="back-link"
            to={back.returnTo}
            state={back.returnState}
          >
            ← {back.returnLabel}
          </Link>
          <p className="eyebrow">Exact published contract</p>
          <h2>
            {query.data
              ? workflowDisplayName(query.data)
              : `${name}@${version}`}
          </h2>
          {query.data ? (
            <code className="catalog-exact-selector">
              {workflowSelector(query.data)}
            </code>
          ) : null}
          <p className="lede">
            {query.data
              ? workflowDescription(query.data)
              : "Loading the authored purpose and exact input contract…"}
          </p>
        </div>
        <div className="workflow-header-actions primary-action-cluster">
          <button
            type="button"
            aria-controls="workflow-run-setup"
            disabled={query.data === undefined}
            onClick={focusRunSetup}
          >
            Configure Run
          </button>
          <button
            className="secondary-button"
            type="button"
            disabled={query.isFetching}
            onClick={() => void query.refetch()}
          >
            {query.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      {query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading exact Workflow contract…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <>
          <section
            className="workflow-run-primary"
            id="workflow-run-setup"
            tabIndex={-1}
          >
            <WorkflowRunForm
              key={`${query.data.ref.name}@${query.data.ref.version}`}
              workflow={query.data}
            />
          </section>

          <details className="workflow-technical-details">
            <summary>
              <span>
                <strong>Technical Workflow contract</strong>
                <small>
                  Inputs, outputs, Stage graph, Workers and resolved execution
                  settings
                </small>
              </span>
              <span>
                {Object.keys(query.data.stages).length} Stages ·{" "}
                {Object.keys(query.data.inputs).length} inputs ·{" "}
                {Object.keys(query.data.outputs).length} outputs
              </span>
            </summary>
            <div className="workflow-technical-details-body">
              <div className="workflow-contract-grid">
                <div className="panel contract-slots">
                  <p className="eyebrow">String parameters</p>
                  {Object.entries(query.data.parameters).length === 0 ? (
                    <p className="compact-empty">None declared.</p>
                  ) : (
                    <ul>
                      {Object.entries(query.data.parameters)
                        .sort(([left], [right]) => left.localeCompare(right))
                        .map(([slot, contract]) => (
                          <li key={slot}>
                            <code>{slot}</code>
                            <span>
                              {contract.required ? "required" : "optional"}{" "}
                              string
                            </span>
                          </li>
                        ))}
                    </ul>
                  )}
                </div>
                <SlotContract
                  title="Artifact inputs"
                  slots={query.data.inputs}
                />
                <SlotContract
                  title="Declared outputs"
                  slots={query.data.outputs}
                />
              </div>

              <div className="workflow-stages">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">Scheduler graph</p>
                    <h3>Stages and escalation</h3>
                  </div>
                  <span>{Object.keys(query.data.stages).length} Stages</span>
                </div>
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
            </div>
          </details>
        </>
      )}
    </section>
  );
}
