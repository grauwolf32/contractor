import { useId } from "react";
import { Link } from "react-router";

import type { WorkflowSummary } from "../../api/workflows";
import { IdChip, StatusChip, StatusGlyph, TechnicalDetails } from "../../ui";
import type { StatusTone } from "../../ui";
import type { WorkflowCompatibility } from "../projects/recommendations";
import { workflowFormats } from "./formats";
import { workflowDisplayName, workflowSelector } from "./presentation";
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

/** Whether a project's materials fit the Workflow, in the "format matches" sense. */
function matchingStatus(
  matching: WorkflowCompatibility,
  ambiguous: boolean,
): { label: string; tone: StatusTone } {
  if (matching.suppressed)
    return { label: "Primary result exists", tone: "info" };
  if (!matching.compatible) return { label: "Missing inputs", tone: "warning" };
  if (ambiguous) return { label: "Choose inputs", tone: "review" };
  return { label: "Format matches found", tone: "success" };
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
  const numeric = versions.every((w) => /^\d+(?:\.\d+)*$/.test(w.ref.version));
  const latest = workflow.ref.version === versions[0]?.ref.version;
  const ambiguous =
    matching !== undefined &&
    Object.values(matching.candidates).some((items) => items.length > 1);
  const status =
    matching === undefined ? undefined : matchingStatus(matching, ambiguous);
  const parameters = Object.values(workflow.parameters);
  const requiredInputs = Object.entries(workflow.inputs).filter(
    ([, slot]) => slot.required,
  );
  return (
    <article
      className={`workflow-card ${matching ? "workflow-card-project" : "workflow-card-catalog"}`}
      aria-labelledby={heading}
    >
      <header className="workflow-card-heading">
        <h3 id={heading}>
          {matching ? (
            workflowDisplayName(workflow)
          ) : (
            <Link to={route} state={state}>
              {workflowDisplayName(workflow)}
            </Link>
          )}
        </h3>
        {status === undefined ? null : (
          <StatusChip tone={status.tone} size="sm">
            {status.label}
          </StatusChip>
        )}
      </header>
      <div className="workflow-card-version">
        <IdChip value={selector} display={selector} label="workflow version" />
        <label className="workflow-card-select">
          <span>Version</span>
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
        {numeric ? (
          <span
            className="workflow-card-tag"
            data-tone={latest ? "accent" : undefined}
          >
            {latest ? "Latest" : "Earlier version"}
          </span>
        ) : null}
        <span className="workflow-card-count">
          {versions.length} version{versions.length === 1 ? "" : "s"}
        </span>
      </div>
      {workflow.presentation?.description ? (
        <p className="workflow-card-description">
          {workflow.presentation.description}
        </p>
      ) : null}
      {matching ? (
        <>
          <ul className="workflow-card-inputs" aria-label="Required inputs">
            {requiredInputs.map(([name]) => {
              const count = matching.candidates[name]?.length ?? 0;
              return (
                <li key={name} data-state={count === 1 ? "found" : "attention"}>
                  {count === 1 ? (
                    <StatusGlyph tone="success" size={14} label="Found" />
                  ) : (
                    <StatusGlyph
                      tone={count === 0 ? "warning" : "review"}
                      size={14}
                    />
                  )}
                  <code>{name}</code>
                  {count === 0
                    ? " · missing"
                    : count > 1
                      ? ` · choose 1 of ${count}`
                      : ""}
                </li>
              );
            })}
            {requiredInputs.length === 0 ? (
              <li data-state="found">No required inputs</li>
            ) : null}
          </ul>
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
        <TechnicalDetails
          summary={matching ? "Inputs & contract" : "Technical details"}
        >
          <dl className="workflow-card-technical">
            <div>
              <dt>Entry stage</dt>
              <dd>
                <code>{workflow.entryStage}</code>
              </dd>
            </div>
            {Object.entries(workflow.inputs).map(([name, slot]) => (
              <div key={`input:${name}`}>
                <dt>
                  Input <code>{name}</code> ·{" "}
                  {slot.required ? "required" : "optional"}
                </dt>
                <dd>
                  {slot.mediaTypes.join(", ")}
                  {matching
                    ? ` · ${matching.candidates[name]?.length ?? 0} matching materials`
                    : ""}
                </dd>
              </div>
            ))}
            {Object.entries(workflow.parameters).map(([name, slot]) => (
              <div key={`parameter:${name}`}>
                <dt>
                  Parameter <code>{name}</code>
                </dt>
                <dd>{slot.required ? "Required" : "Optional"} string</dd>
              </div>
            ))}
          </dl>
        </TechnicalDetails>
        {matching ? (
          <button
            className="ui-btn workflow-card-action"
            data-size="sm"
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
            <span aria-hidden="true">→</span>
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
