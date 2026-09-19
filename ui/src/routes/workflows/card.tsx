import { useId } from "react";
import { Link } from "react-router";
import type { WorkflowSummary } from "../../api/workflows";
import { Icon } from "../../app/icon";
import type { WorkflowCompatibility } from "../projects/recommendations";
import { workflowDisplayName, workflowSelector } from "./presentation";
import { workflowFormats } from "./formats";
import "./cards.css";

function Slots({ slots }: { slots: WorkflowSummary["outputs"] }) {
  return Object.keys(slots).length === 0 ? (
    <p className="workflow-card-empty">None declared</p>
  ) : (
    <dl className="workflow-card-slots">
      {Object.entries(slots)
        .sort(
          ([a, x], [b, y]) =>
            Number(!!y.primary) - Number(!!x.primary) || a.localeCompare(b),
        )
        .map(([name, slot]) => (
          <div key={name}>
            <dt>
              <code>{name}</code>{" "}
              <small>{slot.required ? "required" : "optional"}</small>
              {slot.primary ? (
                <small className="workflow-primary-output"> primary</small>
              ) : null}
            </dt>
            <dd title={slot.mediaTypes.join(", ")}>
              {slot.mediaTypes
                .map((value) => workflowFormats[value] ?? value)
                .join(" · ")}
            </dd>
          </div>
        ))}
    </dl>
  );
}

export function WorkflowCard({
  workflow,
  versions,
  onVersion,
  matching,
  onConfigure,
  returnTo,
  returnState,
}: {
  workflow: WorkflowSummary;
  versions: WorkflowSummary[];
  onVersion: (version: string) => void;
  matching?: WorkflowCompatibility;
  onConfigure?: () => void;
  returnTo?: string;
  returnState?: unknown;
}) {
  const heading = useId();
  const selector = workflowSelector(workflow);
  const route = `/catalog/workflows/${encodeURIComponent(workflow.ref.name)}/${encodeURIComponent(workflow.ref.version)}`;
  const state = { returnTo, returnLabel: "Workflow search", returnState };
  const latest = versions.every((w) => /^\d+(?:\.\d+)*$/.test(w.ref.version));
  const ambiguous =
    matching &&
    Object.values(matching.candidates).some((items) => items.length > 1);
  const status = !matching
    ? undefined
    : matching.suppressed
      ? "Primary result exists"
      : !matching.compatible
        ? "Missing inputs"
        : ambiguous
          ? "Choose inputs"
          : "Format matches found";
  const parameters = Object.values(workflow.parameters);
  return (
    <article
      className={`workflow-card ${matching ? "workflow-card-project" : "workflow-card-catalog"}`}
      aria-labelledby={heading}
    >
      <header className="workflow-card-heading">
        <span className="workflow-card-icon">
          <Icon name="catalog" />
        </span>
        <div>
          <h3 id={heading}>
            {matching ? (
              workflowDisplayName(workflow)
            ) : (
              <Link to={route} state={state}>
                {workflowDisplayName(workflow)}
              </Link>
            )}
          </h3>
          <div className="workflow-card-version">
            <label>
              Version{" "}
              <select
                aria-label={`Version of ${workflow.ref.name}`}
                value={workflow.ref.version}
                onChange={(event) => onVersion(event.target.value)}
              >
                {versions.map((version) => (
                  <option key={version.ref.version} value={version.ref.version}>
                    {version.ref.version}
                  </option>
                ))}
              </select>
            </label>
            {latest ? (
              <span
                className={
                  workflow.ref.version === versions[0]?.ref.version
                    ? "workflow-version-latest"
                    : ""
                }
              >
                {workflow.ref.version === versions[0]?.ref.version
                  ? "Latest"
                  : "Earlier version"}
              </span>
            ) : null}
            <span>
              {versions.length} version{versions.length === 1 ? "" : "s"}
            </span>
          </div>
        </div>
      </header>
      {status ? (
        <p
          className={`workflow-card-status ${matching?.compatible && !ambiguous ? "is-found" : "is-missing"}`}
        >
          {status}
          {matching?.compatible ? (
            <small>
              Matched by file type. Review contents and parameters before
              running.
            </small>
          ) : null}
        </p>
      ) : null}
      {workflow.presentation?.description ? (
        <p className="workflow-card-description">
          {workflow.presentation.description}
        </p>
      ) : (
        <p className="workflow-card-description muted-copy">
          Purpose is not described in this published version.
        </p>
      )}
      {matching ? (
        <>
          <div className="workflow-card-inputs">
            {Object.entries(workflow.inputs)
              .filter(([, slot]) => slot.required)
              .map(([name]) => {
                const count = matching.candidates[name]?.length ?? 0;
                return (
                  <span
                    key={name}
                    className={count === 1 ? "" : "needs-attention"}
                  >
                    {count === 1 ? "✓" : "!"} {name}
                    {count === 0
                      ? " · missing"
                      : count > 1
                        ? ` · choose 1 of ${count}`
                        : ""}
                  </span>
                );
              })}
            {Object.values(workflow.inputs).every((slot) => !slot.required) ? (
              <span>No required Artifact inputs</span>
            ) : null}
          </div>
          <p className="workflow-card-result">
            Primary outputs{" "}
            <code>{matching.primaryOutputs.join(", ") || "None declared"}</code>
          </p>
        </>
      ) : (
        <>
          <div className="workflow-card-contract">
            <section>
              <h4>Inputs</h4>
              <Slots slots={workflow.inputs} />
            </section>
            <section>
              <h4>Outputs</h4>
              <Slots slots={workflow.outputs} />
            </section>
          </div>
          <p className="workflow-card-parameters">
            {parameters.length
              ? `${parameters.length} parameter${parameters.length === 1 ? "" : "s"} · ${parameters.filter((p) => p.required).length} required`
              : "No parameters"}
          </p>
        </>
      )}
      <footer className="workflow-card-footer">
        <details>
          <summary>
            {matching ? "Inputs & contract" : "Technical details"}
          </summary>
          <code>{selector}</code>
          <p>
            Entry stage: <code>{workflow.entryStage}</code>
          </p>
          {Object.entries(workflow.inputs).map(([name, slot]) => (
            <p key={name}>
              <code>{name}</code> · {slot.required ? "required" : "optional"}
              <br />
              {slot.mediaTypes.join(", ")}
              {matching
                ? ` · ${matching.candidates[name]?.length ?? 0} matching artifacts`
                : ""}
            </p>
          ))}
          {Object.entries(workflow.parameters).map(([name, slot]) => (
            <p key={name}>
              <code>{name}</code> · {slot.required ? "required" : "optional"}{" "}
              string parameter
            </p>
          ))}
        </details>
        {matching ? (
          <button
            className="secondary-button workflow-card-action"
            type="button"
            aria-label={
              matching.suppressed
                ? `Run again ${selector}`
                : `Configure ${selector}`
            }
            onClick={onConfigure}
          >
            {matching.suppressed
              ? "Run again"
              : matching.compatible
                ? "Configure run"
                : "Review inputs"}
            <span aria-hidden="true"> →</span>
          </button>
        ) : (
          <Link className="workflow-card-action" to={route} state={state}>
            View workflow <span aria-hidden="true">→</span>
          </Link>
        )}
      </footer>
    </article>
  );
}
