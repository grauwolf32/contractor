import { useId, useState, type ReactNode } from "react";
import { Link, useLocation, useNavigate } from "react-router";
import type { WorkflowResource } from "../../api/workflows";
import { ActionMenu } from "../../app/action-menu";
import { Dialog } from "../../app/dialog";
import { Icon } from "../../app/icon";
import { ErrorNotice } from "../artifacts/common";
import { catalogReturnState } from "../catalog/navigation";
import { groupWorkflowVersions } from "./families";
import { workflowFormats } from "./formats";
import { useWorkflowInventory } from "./inventory";
import { workflowDisplayName, workflowSelector } from "./presentation";
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
  return (
    <section className="workflow-overview-slots" aria-label={title}>
      <h4>{title}</h4>
      {Object.keys(slots).length === 0 ? (
        <p className="compact-empty">None declared.</p>
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

function RunDrawer({
  workflow,
  onClose,
}: {
  workflow: WorkflowResource;
  onClose: () => void;
}) {
  const heading = useId();
  const description = useId();
  const [submitting, setSubmitting] = useState(false);
  return (
    <Dialog
      className="workflow-run-drawer"
      backdropClassName="workflow-run-backdrop"
      labelledBy={heading}
      describedBy={description}
      onRequestClose={() => {
        if (!submitting) onClose();
      }}
    >
      <header className="workflow-drawer-heading">
        <div>
          <p className="eyebrow">Standalone Run</p>
          <h2 id={heading}>Configure Run</h2>
          <code>{workflowSelector(workflow)}</code>
        </div>
        <button
          className="project-dialog-close"
          type="button"
          aria-label="Close Run setup"
          disabled={submitting}
          onClick={onClose}
        >
          ×
        </button>
      </header>
      <p className="workflow-drawer-description" id={description}>
        Inputs come from your library (UserScope). Your draft is kept in this
        tab when you close this panel.
      </p>
      <div id="workflow-run-setup" className="workflow-drawer-body">
        <WorkflowRunForm
          workflow={workflow}
          presentation="drawer"
          onSubmittingChange={setSubmitting}
        />
      </div>
    </Dialog>
  );
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
  const back = catalogReturnState(location.state, {
    returnTo: "/catalog/workflows",
    returnLabel: "Workflows",
  });
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
    <section className="route-page workflow-overview-page">
      <nav className="workflow-breadcrumbs" aria-label="Breadcrumb">
        <Link to="/catalog">Catalog</Link>
        <span aria-hidden="true">/</span>
        <Link to={back.returnTo} state={back.returnState}>
          {back.returnLabel}
        </Link>
        <span aria-hidden="true">/</span>
        <span aria-current="page">{workflow.ref.name}</span>
      </nav>
      <header className="workflow-overview-heading">
        <div className="workflow-overview-identity">
          <div className="workflow-overview-title">
            <span className="workflow-card-icon">
              <Icon name="catalog" />
            </span>
            <h2>{workflowDisplayName(workflow)}</h2>
          </div>
          <div className="workflow-card-version">
            <label>
              Version{" "}
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
            {numericVersions ? (
              <span className={latest ? "workflow-version-latest" : ""}>
                {latest ? "Latest" : "Earlier version"}
              </span>
            ) : null}
            <span>Published Workflow</span>
          </div>
          {workflow.presentation?.description ? (
            <p className="lede">{workflow.presentation.description}</p>
          ) : null}
        </div>
        <ActionMenu label="Workflow actions">
          <button
            type="button"
            className="secondary-button"
            disabled={refreshing || inventory.isFetching}
            onClick={() => {
              refresh();
              void inventory.refetch();
            }}
          >
            {refreshing || inventory.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </ActionMenu>
      </header>
      {error ? <ErrorNotice error={error} /> : null}
      {inventory.isError ? (
        <p className="notice notice-warning" role="status">
          Other published versions could not be loaded. This exact version
          remains available.{" "}
          <button
            type="button"
            className="secondary-button"
            onClick={() => void inventory.refetch()}
          >
            Retry versions
          </button>
        </p>
      ) : null}
      <div className="workflow-overview-grid">
        <div className="workflow-overview-content">
          <section className="panel workflow-overview-contract">
            <div className="section-heading">
              <h3>Inputs and results</h3>
              <span>Exact contract</span>
            </div>
            <div
              className={`workflow-overview-io${stages.length === 1 ? " is-single-stage" : ""}`}
            >
              <OverviewSlots title="Artifact inputs" slots={workflow.inputs} />
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
          <section className="panel workflow-overview-stages">
            <div className="section-heading">
              <h3>How it runs</h3>
              <span>
                {stages.length} {stages.length === 1 ? "Stage" : "Stages"}
              </span>
            </div>
            <ul className="workflow-stage-overview">
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
            </ul>
            {children}
          </section>
          {parameters.length > 0 ? (
            <details
              className="panel workflow-overview-parameters"
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
                    <dd>{slot.required ? "Required" : "Optional"} string</dd>
                  </div>
                ))}
              </dl>
            </details>
          ) : null}
        </div>
        <aside
          className="panel workflow-launch-card"
          aria-label="Run this version"
        >
          <p className="eyebrow">Run this version</p>
          <h3>{workflowSelector(workflow)}</h3>
          <p>Standalone · inputs from your library</p>
          <dl>
            <div>
              <dt>Required inputs</dt>
              <dd>
                {requiredInputs}{" "}
                {requiredInputs === 1 ? "Artifact" : "Artifacts"}
                {requiredParameters
                  ? ` · ${requiredParameters} ${requiredParameters === 1 ? "parameter" : "parameters"}`
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
          <small>Review the exact input revisions before starting.</small>
        </aside>
      </div>
      {open ? (
        <RunDrawer
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
        />
      ) : null}
    </section>
  );
}
