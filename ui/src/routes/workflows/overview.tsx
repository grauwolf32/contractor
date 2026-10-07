import { useId, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router";

import type { WorkflowResource } from "../../api/workflows";
import { ErrorNotice } from "../../app/error-notice";
import { RefreshButton } from "../../app/refresh-button";
import { DetailHeader, DetailPane, IdChip, StatusChip } from "../../ui";
import { LibraryBackLink } from "../catalog/library-parts";
import { groupWorkflowVersions } from "./families";
import { workflowFormats } from "./formats";
import { useWorkflowInventory } from "./inventory";
import { workflowDisplayName, workflowSelector } from "./presentation";
import { WorkflowRunDrawer } from "./run-drawer";
import { WorkflowRunForm } from "./run-form";
import "./cards.css";
import "./overview.css";

function OverviewSlots({
  title,
  slots,
}: {
  title: string;
  slots: WorkflowResource["outputs"];
}) {
  const heading = useId();
  return (
    <section className="workflow-overview-slots" aria-labelledby={heading}>
      <h4 id={heading}>{title}</h4>
      {Object.keys(slots).length === 0 ? (
        <p className="library-muted">None declared.</p>
      ) : (
        <dl>
          {Object.entries(slots)
            .sort(
              ([a, x], [b, y]) =>
                Number(!!y.primary) - Number(!!x.primary) ||
                Number(y.required) - Number(x.required) ||
                a.localeCompare(b),
            )
            .map(([name, slot]) => (
              <div key={name}>
                <dt>
                  <code>{name}</code>
                  <span>{slot.required ? "Required" : "Optional"}</span>
                  {slot.primary ? (
                    <span className="workflow-primary-output">Primary</span>
                  ) : null}
                </dt>
                <dd>
                  {slot.mediaTypes.map((type) => (
                    <span key={type}>
                      {workflowFormats[type] ? (
                        <>{workflowFormats[type]} · </>
                      ) : null}
                      <code>{type}</code>
                    </span>
                  ))}
                </dd>
              </div>
            ))}
        </dl>
      )}
    </section>
  );
}

function plural(count: number, singular: string, many: string): string {
  return `${count} ${count === 1 ? singular : many}`;
}

export function WorkflowOverview({
  workflow,
  refreshing,
  refresh,
  error,
  children,
}: {
  workflow: WorkflowResource;
  refreshing: boolean;
  refresh: () => void;
  error: Error | null;
  children: ReactNode;
}) {
  const inventory = useWorkflowInventory();
  const location = useLocation();
  const navigate = useNavigate();
  const titleId = useId();
  const contractId = useId();
  const stagesId = useId();
  const family = groupWorkflowVersions([
    ...(inventory.data ?? []),
    workflow,
  ]).find((item) => item.name === workflow.ref.name)!;
  const knownVersion =
    inventory.isSuccess &&
    inventory.data.some(
      (item) =>
        item.ref.name === workflow.ref.name &&
        item.ref.version === workflow.ref.version,
    );
  const numericVersions =
    knownVersion &&
    family.versions.every((item) => /^\d+(?:\.\d+)*$/.test(item.ref.version));
  const latest = family.versions[0]?.ref.version === workflow.ref.version;
  const selector = workflowSelector(workflow);
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
    (slot) => slot.required,
  ).length;
  const requiredParameters = parameters.filter(
    ([, slot]) => slot.required,
  ).length;
  const open = location.hash === "#workflow-run-setup";

  return (
    <div className="library-detail workflow-overview-page">
      <LibraryBackLink
        fallback={{
          returnTo: "/catalog/workflows",
          returnLabel: "All workflows",
        }}
      />
      <article className="library-sheet" aria-labelledby={titleId}>
        <DetailPane
          header={
            <DetailHeader
              title={<span id={titleId}>{workflowDisplayName(workflow)}</span>}
              status={
                numericVersions ? (
                  <StatusChip tone={latest ? "info" : "neutral"} size="sm">
                    {latest ? "Latest" : "Earlier version"}
                  </StatusChip>
                ) : undefined
              }
              meta={
                <>
                  <IdChip
                    value={selector}
                    display={selector}
                    label="workflow version"
                  />
                  <label className="workflow-card-select">
                    <span>Version</span>
                    <select
                      aria-label={`Version of ${workflow.ref.name}`}
                      value={workflow.ref.version}
                      onChange={(event) =>
                        void navigate(
                          `/catalog/workflows/${encodeURIComponent(workflow.ref.name)}/${encodeURIComponent(event.target.value)}`,
                          { state: location.state },
                        )
                      }
                    >
                      {family.versions.map((item) => (
                        <option key={item.ref.version} value={item.ref.version}>
                          {item.ref.version}
                        </option>
                      ))}
                    </select>
                  </label>
                  <span>Published Workflow</span>
                </>
              }
              actions={
                <RefreshButton
                  className="ui-btn"
                  isFetching={refreshing || inventory.isFetching}
                  onRefresh={() => {
                    refresh();
                    void inventory.refetch();
                  }}
                  label="Refresh"
                />
              }
            />
          }
        >
          {workflow.presentation?.description ? (
            <p className="workflow-overview-lede">
              {workflow.presentation.description}
            </p>
          ) : null}
          {error ? <ErrorNotice error={error} /> : null}
          {inventory.isError ? (
            <div className="library-note" data-tone="warning" role="status">
              <p>
                Other published versions could not be loaded. This version
                remains available.
              </p>
              <div>
                <button
                  type="button"
                  className="ui-btn"
                  data-size="sm"
                  onClick={() => void inventory.refetch()}
                >
                  Retry versions
                </button>
              </div>
            </div>
          ) : null}
          <div className="workflow-overview-grid">
            <div className="workflow-overview-content">
              <section
                className="library-block workflow-overview-contract"
                aria-labelledby={contractId}
              >
                <header className="library-block-header">
                  <h3 id={contractId} className="library-block-title">
                    Inputs and results
                  </h3>
                  <span className="library-muted">Contract</span>
                </header>
                <div
                  className={`workflow-overview-io${stages.length === 1 ? " is-single-stage" : ""}`}
                >
                  <OverviewSlots title="Input files" slots={workflow.inputs} />
                  {stages.length === 1 ? (
                    <div
                      className="workflow-overview-flow"
                      aria-label={`Single stage: ${workflow.entryStage}`}
                    >
                      <span aria-hidden="true">→</span>
                      <code>{workflow.entryStage}</code>
                    </div>
                  ) : null}
                  <OverviewSlots
                    title="Declared outputs"
                    slots={workflow.outputs}
                  />
                </div>
              </section>
              <section
                className="library-block workflow-overview-stages"
                aria-labelledby={stagesId}
              >
                <header className="library-block-header">
                  <h3 id={stagesId} className="library-block-title">
                    How it runs
                  </h3>
                  <span className="library-muted">
                    {stages.length} {stages.length === 1 ? "Stage" : "Stages"}
                  </span>
                </header>
                <ol className="workflow-stage-overview">
                  {stages.map(([name, stage]) => (
                    <li key={name}>
                      <div>
                        <code>{name}</code>
                        {name === workflow.entryStage ? (
                          <span className="workflow-entry-label">Entry</span>
                        ) : null}
                      </div>
                      <p>{stage.objective}</p>
                      <small>Published stage objective</small>
                    </li>
                  ))}
                </ol>
                {children}
              </section>
              {parameters.length > 0 ? (
                <details
                  className="workflow-overview-parameters"
                  open={requiredParameters > 0}
                >
                  <summary>
                    String parameters{" "}
                    <span>
                      {requiredParameters
                        ? `${requiredParameters} required · `
                        : ""}
                      {parameters.length - requiredParameters} optional
                    </span>
                  </summary>
                  <dl>
                    {parameters.map(([name, slot]) => (
                      <div key={name}>
                        <dt>
                          <code>{name}</code>
                        </dt>
                        <dd>
                          {slot.required ? "Required" : "Optional"} string
                        </dd>
                      </div>
                    ))}
                  </dl>
                </details>
              ) : null}
            </div>
            <aside
              className="workflow-launch-card"
              aria-label="Run this version"
            >
              <p className="workflow-launch-kind">Run this version</p>
              <h3 className="workflow-launch-selector">{selector}</h3>
              <p>Standalone · inputs from your library</p>
              <dl>
                <div>
                  <dt>Required inputs</dt>
                  <dd>
                    {plural(requiredInputs, "file", "files")}
                    {requiredParameters
                      ? ` · ${plural(requiredParameters, "parameter", "parameters")}`
                      : ""}
                  </dd>
                </div>
                <div>
                  <dt>Outputs</dt>
                  <dd>{Object.keys(workflow.outputs).length} declared</dd>
                </div>
              </dl>
              <button
                type="button"
                className="ui-btn"
                data-variant="primary"
                aria-haspopup="dialog"
                onClick={() =>
                  void navigate(
                    {
                      pathname: location.pathname,
                      search: location.search,
                      hash: "#workflow-run-setup",
                    },
                    { state: location.state },
                  )
                }
              >
                Configure Run <span aria-hidden="true">→</span>
              </button>
              <small>Review the input revisions before starting.</small>
            </aside>
          </div>
        </DetailPane>
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
    </div>
  );
}
