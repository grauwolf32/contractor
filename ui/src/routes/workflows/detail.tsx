import { useQuery } from "@tanstack/react-query";
import { type ReactNode } from "react";
import { Link, useParams } from "react-router";

import type { components } from "../../api/generated/public";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  CONFIG_ID_PATTERN,
  CONFIG_VERSION_PATTERN,
  getWorkflow,
  type WorkflowResource,
} from "../../api/workflows";
import { ErrorNotice } from "../artifacts/common";
import { WorkflowRunForm } from "./run-form";

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
        Gateway {config.llmGateway.gatewayId}@{config.llmGateway.version}
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
                  <span>
                    {binding.template.templateId}@{binding.template.version}
                  </span>
                  <span>namespace {binding.namespace}</span>
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
    queryFn: () => getWorkflow(api, name, version),
    enabled: valid,
  });

  if (!valid) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Workflow route is invalid")} />
        <Link to="/workflows">Return to Workflows</Link>
      </section>
    );
  }

  return (
    <section className="route-page workflow-page">
      <header className="route-header-row">
        <div>
          <Link className="back-link" to="/workflows">
            ← All Workflows
          </Link>
          <p className="eyebrow">Exact published contract</p>
          <h2>
            {name}@{version}
          </h2>
          <p className="lede">
            Review the immutable contract, then create a Run whose local draft
            contains only declared strings, exact Artifact revisions, and
            optional published refs.
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={query.isFetching}
          onClick={() => void query.refetch()}
        >
          {query.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      {query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading exact Workflow contract…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <>
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
                          {contract.required ? "required" : "optional"} string
                        </span>
                      </li>
                    ))}
                </ul>
              )}
            </div>
            <SlotContract title="Artifact inputs" slots={query.data.inputs} />
            <SlotContract title="Declared outputs" slots={query.data.outputs} />
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

          <WorkflowRunForm
            key={`${query.data.ref.name}@${query.data.ref.version}`}
            workflow={query.data}
          />
        </>
      )}
    </section>
  );
}
