import { Link } from "react-router";

import type { Audit } from "../../../api/audits";
import { formatBytes, formatTimestamp } from "../../artifacts/common";
import { AuditProgress } from "./progress";
import { ExactArtifactLink } from "./shared";

function compactDigest(digest: string): string {
  return digest.length <= 28
    ? digest
    : `${digest.slice(0, 15)}…${digest.slice(-8)}`;
}

function StringList({
  values,
  empty = "none",
}: {
  values: string[];
  empty?: string;
}) {
  if (values.length === 0) return <span className="muted-copy">{empty}</span>;
  return (
    <ul className="audit-string-list">
      {values.map((value, index) => (
        <li key={`${index}-${value}`}>{value}</li>
      ))}
    </ul>
  );
}

export function AuditOverview({ audit }: { audit: Audit }) {
  const baseline = audit.baseline;
  return (
    <div className="audit-detail-stack">
      {audit.stopReason === undefined ||
      (audit.state === "paused" &&
        audit.stopReason.code === "deadline_exhausted") ? null : (
        <section className="notice notice-error audit-stop-reason" role="alert">
          <strong>{audit.stopReason.code}</strong>
          <p>{audit.stopReason.message}</p>
        </section>
      )}
      <AuditProgress audit={audit} />
      <details
        className="panel audit-section-panel audit-setup-details"
        open={audit.state === "draft"}
      >
        <summary>Profile and scope</summary>
        <div className="section-heading">
          <div>
            <p className="eyebrow">Audit setup</p>
            <h3>Exact setup</h3>
          </div>
          <Link
            className="audit-open-link"
            to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/coverage`}
          >
            View checks & results →
          </Link>
        </div>
        <dl className="metadata-grid">
          <div>
            <dt>Profile digest</dt>
            <dd>
              <code>{audit.profile.digest}</code>
            </dd>
          </div>
          <div>
            <dt>Current revision</dt>
            <dd>
              <code>{audit.revision}</code>
            </dd>
          </div>
          <div>
            <dt>Dispatch</dt>
            <dd>{audit.dispatchState}</dd>
          </div>
          <div>
            <dt>Evidence hold</dt>
            <dd>{audit.holdState}</dd>
          </div>
          <div>
            <dt>Objective</dt>
            <dd>{audit.scope.objective ?? "Not set"}</dd>
          </div>
          <div>
            <dt>Target</dt>
            <dd>{audit.scope.target ?? "Not set"}</dd>
          </div>
          <div>
            <dt>Authorization scope</dt>
            <dd>{audit.scope.authorizationScope ?? "Not set"}</dd>
          </div>
          <div>
            <dt>Runtime labels</dt>
            <dd>
              {audit.runtimeLabels.length === 0
                ? "default"
                : audit.runtimeLabels.join(", ")}
            </dd>
          </div>
        </dl>
      </details>
      <section className="panel audit-section-panel">
        <p className="eyebrow">Immutable selections</p>
        <h3>Inputs</h3>
        <div className="audit-artifact-list">
          {Object.entries(audit.inputs).map(([name, artifact]) => (
            <ExactArtifactLink
              key={name}
              projectId={audit.projectId}
              artifact={artifact}
              label={name}
              projectReadable
            />
          ))}
        </div>
      </section>
      <section className="panel audit-section-panel">
        <p className="eyebrow">Bounded execution</p>
        <h3>Limits and consumption</h3>
        <dl className="metadata-grid audit-limit-grid">
          <div>
            <dt>Rounds</dt>
            <dd>{audit.limits.maxRounds}</dd>
          </div>
          <div>
            <dt>Batch size</dt>
            <dd>{audit.limits.batchSize}</dd>
          </div>
          <div>
            <dt>Items</dt>
            <dd>{audit.limits.maxItemsTotal}</dd>
          </div>
          <div>
            <dt>Attempts/item</dt>
            <dd>{audit.limits.maxItemRunAttempts}</dd>
          </div>
          <div>
            <dt>Runs submitted</dt>
            <dd>
              {audit.submittedRunCount}/{audit.limits.maxSubmittedRuns}
            </dd>
          </div>
          <div>
            <dt>Runs outstanding</dt>
            <dd>{audit.outstandingRunCount}</dd>
          </div>
          <div>
            <dt>Evidence retained</dt>
            <dd>
              {formatBytes(audit.retainedEvidenceBytes)} /{" "}
              {formatBytes(audit.limits.maxEvidenceBytes)}
            </dd>
          </div>
          <div>
            <dt>Deadline</dt>
            <dd>
              {audit.deadlineAt === undefined
                ? audit.state === "draft"
                  ? "Set when starting"
                  : "No time limit"
                : audit.stopReason?.code === "deadline_exhausted"
                  ? "Time limit reached"
                  : audit.state === "paused"
                    ? "Paused — remaining time is saved"
                    : formatTimestamp(audit.deadlineAt)}
            </dd>
          </div>
        </dl>
      </section>
      {baseline === undefined ? (
        <section className="panel audit-section-panel compact-empty">
          <strong>Baseline is not pinned yet.</strong>
          <p>
            Starting this draft validates inputs and records the exact
            inventory.
          </p>
        </section>
      ) : (
        <details className="panel audit-section-panel audit-setup-details">
          <summary>Baseline and exact standards</summary>
          <dl className="metadata-grid">
            <div>
              <dt>Source content</dt>
              <dd>
                <code>{baseline.inventory.sourceContentDigest}</code>
              </dd>
            </div>
            <div>
              <dt>Canonical inventory</dt>
              <dd>
                <code>{baseline.inventory.canonicalInventoryDigest}</code>
              </dd>
            </div>
            <div>
              <dt>Worklist</dt>
              <dd>
                <ExactArtifactLink
                  projectId={audit.projectId}
                  artifact={baseline.inventory.worklist}
                />
              </dd>
            </div>
            <div>
              <dt>Skills</dt>
              <dd>{baseline.skills.length}</dd>
            </div>
            <div>
              <dt>Standards</dt>
              <dd>{baseline.standards.length}</dd>
            </div>
            <div>
              <dt>Runtime configs</dt>
              <dd>{1 + baseline.runtimeConfig.labels.length}</dd>
            </div>
          </dl>
          {baseline.standards.length === 0 ? null : (
            <div
              className="audit-gap-block"
              data-testid="audit-baseline-standards"
            >
              <h4>Exact standards</h4>
              <ul className="audit-string-list">
                {baseline.standards.map((standard) => (
                  <li
                    key={`${standard.reference.scheme}@${standard.reference.version}`}
                  >
                    <strong>{standard.title}</strong>{" "}
                    <code>
                      {standard.reference.scheme}@{standard.reference.version}
                    </code>{" "}
                    · <code>{compactDigest(standard.retained.digest)}</code> ·{" "}
                    <a
                      href={standard.source.url}
                      rel="noreferrer"
                      target="_blank"
                    >
                      source
                    </a>{" "}
                    · {standard.license.id}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {baseline.inventory.standardSelection === undefined ? null : (
            <div
              className="audit-gap-block"
              data-testid="audit-baseline-standard-selection"
            >
              <h4>Selected denominator</h4>
              <p>{baseline.inventory.standardSelection.scope}</p>
              <p>
                Level {baseline.inventory.standardSelection.levels.join(", ")} ·{" "}
                {baseline.inventory.standardSelection.entryIds.length} exact
                requirements
              </p>
              <StringList
                values={baseline.inventory.standardSelection.entryIds}
              />
            </div>
          )}
          <div className="audit-gap-block">
            <h4>Inventory gaps</h4>
            <StringList
              values={baseline.inventory.gaps}
              empty="No inventory gaps."
            />
          </div>
        </details>
      )}
    </div>
  );
}
