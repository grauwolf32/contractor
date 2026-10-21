import type { Audit } from "../../../api/audits";
import { ContextLink } from "../../../app/context-navigation";
import { formatNumber, plural } from "../../../app/format";
import {
  readableValue,
  type ItemKind,
  itemNoun,
} from "../../../app/vocabulary";
import { Facts, StatusChip, TechnicalDetails } from "../../../ui";
import { exactArtifactLink } from "./artifact-links";
import { AuditMaterial } from "./material";

function materialLabel(name: string): string {
  if (name === "source" || name === "sources") return "Source code";
  if (name === "api" || name === "openapi") return "API spec";
  if (name === "settings") return "Scan settings";
  if (name.includes("validation")) return "Validation report";
  return readableValue(name);
}

/** A phase strip follows Server state; it never invents percentage progress. */
export function CheckJourney({
  audit,
  kind,
}: {
  audit: Audit;
  kind: ItemKind;
}) {
  const prepared = audit.phase === "inventory" || audit.phase === "rounds";
  const report =
    audit.phase === "rounds" &&
    ["finalizing", "completed"].includes(audit.state);
  const current = report ? 2 : audit.phase === "rounds" ? 1 : 0;
  const steps = [
    audit.preparation === undefined ? "Select materials" : "Prepare materials",
    `Check ${itemNoun(kind, 2)}`,
    "Review & report",
  ];
  return (
    <ol className="checks-journey" aria-label="Check stages">
      {steps.map((label, index) => (
        <li
          key={label}
          aria-current={index === current ? "step" : undefined}
          data-complete={index === 0 ? prepared : index < current}
        >
          <span className="checks-journey-number" aria-hidden="true">
            {index + 1}
          </span>
          <span>
            {label}
            <small>
              {index < current
                ? "Finished"
                : index === current
                  ? "Current stage"
                  : "Next"}
            </small>
          </span>
        </li>
      ))}
    </ol>
  );
}

export function CheckPreparation({
  audit,
  kind,
}: {
  audit: Audit;
  kind: ItemKind;
}) {
  if (audit.preparation === undefined) return null;
  const pending = audit.phase !== "rounds";
  const stopped = ["failed", "cancelled", "cancelling"].includes(audit.state);
  return (
    <section className="checks-preparation" aria-label="Material preparation">
      <div className="checks-section-heading">
        <h3>
          {pending
            ? stopped
              ? "Preparation stopped"
              : audit.phase === "inventory"
                ? "Building the worklist"
                : "Preparing materials"
            : "Preparation complete"}
        </h3>
        <p className="ui-caption">
          {pending
            ? `The ${itemNoun(kind, 2)} and coverage are unavailable until a valid worklist is accepted.`
            : audit.generatedInventory === undefined
              ? "Accepted preparation outputs are saved; inventory details are unavailable."
              : `${plural(audit.generatedInventory.itemCount, itemNoun(kind, 1))} accepted from the saved preparation output.`}
        </p>
      </div>
      <ul className="checks-preparation-roles">
        {Object.entries(audit.preparation.roles).map(([name, role]) => (
          <li key={name}>
            <div className="ui-toolbar" data-justify="between">
              <strong>
                {readableValue(name).replace(/\bapi\b/giu, "API")}
              </strong>
              <StatusChip
                tone={
                  role.status === "accepted"
                    ? "info"
                    : role.status === "failed"
                      ? "blocked"
                      : role.status === "running"
                        ? "progress"
                        : "idle"
                }
              >
                {role.status === "accepted"
                  ? "Accepted"
                  : role.status === "running"
                    ? "Preparing"
                    : role.status === "failed"
                      ? "Failed"
                      : "Waiting"}
              </StatusChip>
            </div>
            <p className="ui-caption">
              {role.attempts === 0
                ? "Not started"
                : `Attempt ${role.attempts} of ${role.maxAttempts}`}
            </p>
            {role.runId === undefined ? null : (
              <ContextLink
                returnLabel="Check preparation"
                to={`/runs/${encodeURIComponent(role.runId)}`}
              >
                Inspect preparation
              </ContextLink>
            )}
          </li>
        ))}
      </ul>
      {audit.generatedInventory === undefined ? null : (
        <p className="ui-caption">
          Accepted worklist: {formatNumber(audit.generatedInventory.itemCount)}{" "}
          {itemNoun(kind, 2)}. Execution completion and security coverage are
          assessed separately.
        </p>
      )}
    </section>
  );
}

/** Ordinary inputs remain distinct from accepted generated materials. */
export function CheckMaterials({ audit }: { audit: Audit }) {
  const outputs = Object.values(audit.preparation?.roles ?? {}).flatMap(
    (role) => Object.entries(role.outputs),
  );
  return (
    <section className="checks-materials" aria-label="Check materials">
      <h3>Materials</h3>
      <p className="ui-caption">
        Original inputs and accepted generated outputs, pinned to saved
        versions.
      </p>
      <ul>
        {Object.entries(audit.baseline?.inputs ?? audit.inputs).map(
          ([name, artifact]) => (
            <li key={name}>
              <ContextLink
                returnLabel="Check"
                to={exactArtifactLink(audit.projectId, artifact)}
              >
                {materialLabel(name)}
              </ContextLink>
              <span className="ui-caption">Original input</span>
            </li>
          ),
        )}
        {outputs.map(([name, output]) => (
          <li key={`${output.executionId}/${name}`}>
            <AuditMaterial
              auditId={audit.auditId}
              receipt={`prepare:${output.executionId}:${name}`}
              artifact={output.artifact}
              label={materialLabel(name)}
            />
            <span className="ui-caption">Accepted preparation output</span>
          </li>
        ))}
      </ul>
      {audit.generatedInventory === undefined ? null : (
        <>
          <p>
            <strong>
              {formatNumber(audit.generatedInventory.itemCount)} accepted items
            </strong>
          </p>
          {audit.generatedInventory.gaps.length === 0 ? null : (
            <ul>
              {audit.generatedInventory.gaps.map((gap) => (
                <li key={gap}>{gap}</li>
              ))}
            </ul>
          )}
          <TechnicalDetails summary="Material provenance">
            <Facts
              size="sm"
              items={[
                {
                  term: "Source content",
                  value: (
                    <code>{audit.generatedInventory.sourceContentDigest}</code>
                  ),
                },
                {
                  term: "Worklist identity",
                  value: (
                    <code>
                      {audit.generatedInventory.canonicalInventoryDigest}
                    </code>
                  ),
                },
              ]}
            />
            <AuditMaterial
              auditId={audit.auditId}
              receipt="initial-inventory"
              artifact={audit.generatedInventory.snapshot}
              label="Inventory record"
            />
          </TechnicalDetails>
        </>
      )}
    </section>
  );
}
