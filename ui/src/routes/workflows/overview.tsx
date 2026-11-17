import { inputLabel } from "../checks/start/check-types";
import { libraryReadableName } from "../catalog/presentation";
import { type ReactNode } from "react";
import { Link, useLocation, useNavigate } from "react-router";
import { agentPath } from "../../api/agents";
import type { WorkflowResource } from "../../api/workflows";
import { plural } from "../../app/format";
import { Icon } from "../../app/icon";
import { RefreshButton } from "../../app/refresh-button";
import { Button, TechnicalDetails } from "../../ui";
import { formatLabel } from "../artifacts/kinds";
import { LibraryDetailHeading, LibraryTag } from "../catalog/browser";
import { locationDestination } from "../catalog/navigation";
import { groupWorkflowVersions, hasNumericVersions } from "./families";
import { useWorkflowInventory } from "./inventory";
import { workflowSelector } from "./presentation";
import { WorkflowRunDrawer } from "./run-drawer";
import { WorkflowRunForm } from "./run-form";
import { WorkflowVersionSelect, WorkflowVersionTag } from "./version-select";
import "./cards.css";
import "./overview.css";

function OverviewSlots({
  title,
  slots,
  output = false,
}: {
  title: string;
  slots: WorkflowResource["outputs"];
  output?: boolean;
}) {
  return (
    <section
      className="library-contract-panel"
      aria-label={title}
      data-output={output}
    >
      <header>
        <h3>
          <Icon name={output ? "download" : "projects"} />
          {title}
        </h3>
        <span>
          {plural(Object.keys(slots).length, output ? "output" : "file")}
        </span>
      </header>
      {Object.entries(slots)
        .sort(
          ([a, x], [b, y]) =>
            Number(!!y.primary) - Number(!!x.primary) ||
            Number(y.required) - Number(x.required) ||
            a.localeCompare(b),
        )
        .map(([name, slot]) => (
          <div className="library-contract-row" key={name}>
            <Icon name={name === "source" ? "projects" : "docs"} />
            <div>
              <strong>{libraryReadableName(inputLabel(name))}</strong>
              <small>{slot.mediaTypes.map(formatLabel).join(" · ")}</small>
              <LibraryTag tone={slot.primary ? "green" : undefined}>
                {slot.primary
                  ? "Primary result"
                  : slot.required
                    ? "Required"
                    : "Optional"}
              </LibraryTag>
            </div>
          </div>
        ))}
      {Object.keys(slots).length === 0 ? (
        <p className="library-empty-contract">None declared.</p>
      ) : null}
    </section>
  );
}

export function WorkflowOverview({
  workflow,
  refreshing,
  refresh,
  children,
}: {
  workflow: WorkflowResource;
  refreshing: boolean;
  refresh: () => void;
  children: ReactNode;
}) {
  const inventory = useWorkflowInventory(false);
  const location = useLocation();
  const navigate = useNavigate();
  const family = groupWorkflowVersions([
    ...(inventory.data ?? []),
    workflow,
  ]).find((f) => f.name === workflow.ref.name)!;
  const known =
    inventory.isSuccess &&
    inventory.data.some(
      (w) =>
        w.ref.name === workflow.ref.name &&
        w.ref.version === workflow.ref.version,
    );
  const numeric = known && hasNumericVersions(family.versions);
  const latest = family.versions[0]?.ref.version === workflow.ref.version;
  const selector = workflowSelector(workflow);
  const exactPath = `/catalog/workflows/${encodeURIComponent(workflow.ref.name)}/${encodeURIComponent(workflow.ref.version)}`;
  const stages = Object.entries(workflow.stages).sort(
    ([a], [b]) =>
      Number(b === workflow.entryStage) - Number(a === workflow.entryStage) ||
      a.localeCompare(b),
  );
  const parameters = Object.entries(workflow.parameters).sort(
    ([a, x], [b, y]) =>
      Number(y.required) - Number(x.required) || a.localeCompare(b),
  );
  const requiredInputs = Object.values(workflow.inputs).filter(
    (i) => i.required,
  ).length;
  const requiredParameters = parameters.filter(([, v]) => v.required).length;
  const open = location.hash === "#workflow-run-setup";
  return (
    <>
      <article className="library-design-detail library-workflow-detail">
        <LibraryDetailHeading
          title={
            workflow.presentation?.displayName ??
            libraryReadableName(workflow.ref.name)
          }
          kind="Published Workflow"
          description={workflow.presentation?.description}
          selector={selector}
          identityLabel="workflow version"
          icon="workflow"
          version={
            <WorkflowVersionSelect
              workflow={workflow}
              versions={family.versions}
              onChange={(v) =>
                void navigate(
                  `/catalog/workflows/${encodeURIComponent(workflow.ref.name)}/${encodeURIComponent(v)}${location.search}`,
                  { state: location.state },
                )
              }
            />
          }
          meta={
            numeric ? (
              <WorkflowVersionTag latest={latest} />
            ) : (
              <LibraryTag tone="blue">Published</LibraryTag>
            )
          }
          actions={
            <RefreshButton
              isFetching={refreshing || inventory.isFetching}
              onRefresh={() => {
                refresh();
                void inventory.refetch();
              }}
              size="sm"
            />
          }
        />
        {inventory.isError ? (
          <div className="library-note" data-tone="warning" role="status">
            <p>
              Other published versions could not be loaded. This version remains
              available.
            </p>
            <Button size="sm" onClick={() => void inventory.refetch()}>
              Retry versions
            </Button>
          </div>
        ) : null}
        <div className="library-detail-grid">
          <div className="library-detail-main">
            <section
              aria-label="Inputs and results"
              className="library-contract-pair"
            >
              <OverviewSlots title="Inputs" slots={workflow.inputs} />
              <OverviewSlots
                title="Declared results"
                slots={workflow.outputs}
                output
              />
            </section>
            <section className="library-design-section">
              <header className="library-design-section-header">
                <h3>Stages & agents</h3>
                <span>{plural(stages.length, "declared stage")}</span>
              </header>
              <ol className="library-stages">
                {stages.map(([name, stage]) => (
                  <li className="library-stage-card" key={name}>
                    <span
                      className="library-stage-icon"
                      data-tone={stage.operation ? "green" : "blue"}
                    >
                      <Icon
                        name={
                          stage.operation
                            ? "settings"
                            : name === workflow.entryStage
                              ? "search"
                              : "workflow"
                        }
                      />
                    </span>
                    <div>
                      <h4>
                        {libraryReadableName(name)}
                        {name === workflow.entryStage ? (
                          <LibraryTag tone="blue">Entry</LibraryTag>
                        ) : null}
                        {stage.operation ? (
                          <LibraryTag tone="green">Operation</LibraryTag>
                        ) : null}
                      </h4>
                      <p>{stage.objective}</p>
                      {stage.operation ? (
                        <code className="library-operation-ref">
                          {stage.operation.ref}
                        </code>
                      ) : (
                        <div className="library-agent-bindings">
                          {Object.entries(stage.agents).map(
                            ([logical, binding]) => (
                              <Link
                                key={logical}
                                className="library-agent-binding"
                                to={agentPath(
                                  binding.template.templateId,
                                  binding.template.version,
                                )}
                                state={{
                                  returnTo: locationDestination({
                                    ...location,
                                    pathname: exactPath,
                                  }),
                                  returnLabel: selector,
                                  returnState: location.state,
                                }}
                              >
                                <Icon name="agent" />
                                <span>
                                  {libraryReadableName(
                                    binding.template.templateId,
                                  )}{" "}
                                  <small>
                                    v{binding.template.version} · {logical}
                                  </small>
                                </span>
                                <span aria-hidden="true">↗</span>
                              </Link>
                            ),
                          )}
                        </div>
                      )}
                      <div className="library-stage-edges">
                        {Object.entries(stage.on).flatMap(
                          ([outcome, transition]) =>
                            transition.kind === "next"
                              ? [
                                  <span key={outcome}>
                                    {outcome} →{" "}
                                    <code>{transition.nextStage}</code>
                                  </span>,
                                ]
                              : [],
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
              <p className="ui-caption">
                Select an Agent to inspect its instructions and tools.
              </p>
            </section>
            {parameters.length ? (
              <details
                className="library-parameters"
                open={requiredParameters > 0}
              >
                <summary>
                  String parameters{" "}
                  <span>
                    {requiredParameters} required ·{" "}
                    {parameters.length - requiredParameters} optional
                  </span>
                </summary>
                <dl>
                  {parameters.map(([name, slot]) => (
                    <div key={name}>
                      <dt>
                        <code>{name}</code>
                      </dt>
                      <dd>{slot.required ? "Required" : "Optional"} string</dd>
                    </div>
                  ))}
                </dl>
              </details>
            ) : null}
            <TechnicalDetails
              summary="Exact input and output contract"
              className="library-design-technical"
            >
              <dl className="library-exact-contract">
                {[
                  ...Object.entries(workflow.inputs).map(([name, slot]) => ({
                    name,
                    slot,
                    kind: "Input",
                  })),
                  ...Object.entries(workflow.outputs).map(([name, slot]) => ({
                    name,
                    slot,
                    kind: "Output",
                  })),
                ].map(({ name, slot, kind }) => (
                  <div key={`${kind}:${name}`}>
                    <dt>
                      {kind} <code>{name}</code>
                    </dt>
                    <dd>
                      {slot.required ? "Required" : "Optional"} ·{" "}
                      {slot.mediaTypes.join(", ")}
                    </dd>
                  </div>
                ))}
              </dl>
            </TechnicalDetails>
            <div className="library-design-technical-actions">{children}</div>
          </div>
          <aside
            className="library-context-card library-run-card"
            aria-label="Run this version"
          >
            <span className="library-eyebrow">Run this version</span>
            <h3>Prepare your inputs</h3>
            <p>
              Choose exact material revisions and review parameters before
              starting.
            </p>
            <dl className="library-context-facts">
              <div>
                <dt>Required files</dt>
                <dd>{requiredInputs}</dd>
              </div>
              <div>
                <dt>Required parameters</dt>
                <dd>{requiredParameters}</dd>
              </div>
              <div>
                <dt>Declared outputs</dt>
                <dd>{Object.keys(workflow.outputs).length}</dd>
              </div>
            </dl>
            <Button
              variant="primary"
              aria-haspopup="dialog"
              onClick={() =>
                void navigate(
                  {
                    pathname: exactPath,
                    search: location.search,
                    hash: "#workflow-run-setup",
                  },
                  { state: location.state },
                )
              }
            >
              Configure Run <Icon name="arrow-right" />
            </Button>
            <small>Standalone · inputs from your library</small>
          </aside>
        </div>
      </article>
      {open ? (
        <WorkflowRunDrawer
          workflow={workflow}
          onClose={() =>
            void navigate(
              {
                pathname: location.pathname,
                search: location.search,
                hash: "",
              },
              { replace: true, state: location.state },
            )
          }
        >
          {(onSubmittingChange) => (
            <WorkflowRunForm
              workflow={workflow}
              presentation="drawer"
              onSubmittingChange={onSubmittingChange}
            />
          )}
        </WorkflowRunDrawer>
      ) : null}
    </>
  );
}
