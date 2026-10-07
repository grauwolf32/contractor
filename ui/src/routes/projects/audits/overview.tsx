import { useId, type ReactNode } from "react";

import type { Audit, AuditWorkspace } from "../../../api/audits";
import {
  compactDigest,
  formatBytes,
  formatTimestamp,
} from "../../../app/format";
import { checkStateLabel } from "../../../app/vocabulary";
import { IdChip, TechnicalDetails } from "../../../ui";
import { SourceLink } from "../../catalog/audit-preset-checks";
import { auditProfileLabel } from "./labels";
import { ExactArtifactLink } from "./shared";
import { describeStopReason } from "./stop-reason";

function StringList({
  values,
  empty = "None",
}: {
  values: readonly string[];
  empty?: string;
}) {
  if (values.length === 0) return <span className="checks-quiet">{empty}</span>;
  return (
    <ul className="checks-string-list">
      {values.map((value, index) => (
        <li key={`${index}-${value}`}>{value}</li>
      ))}
    </ul>
  );
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div>
      <dt>{term}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function TechSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  const heading = useId();
  return (
    <section className="checks-tech-section" aria-labelledby={heading}>
      <h3 id={heading}>{title}</h3>
      {children}
    </section>
  );
}

function deadlineText(audit: Audit): string {
  if (audit.deadlineAt === undefined)
    return audit.state === "draft" ? "Set when it starts" : "No time limit";
  if (audit.stopReason?.code === "deadline_exhausted")
    return "Time limit reached";
  if (audit.state === "paused") return "Paused: the remaining time is kept";
  return formatTimestamp(audit.deadlineAt);
}

/**
 * Everything the check records about its setup, limits, inputs and pinned
 * baseline, behind one Technical details disclosure: digests, revisions,
 * rounds and counters are for admins and debugging. Drafts open it, since
 * their inputs are still to be checked before starting.
 */
export function CheckTechnicalDetails({
  audit,
  workspace,
  open = false,
}: {
  audit: Audit;
  workspace: AuditWorkspace | undefined;
  /** Start open (drafts always do). */
  open?: boolean;
}) {
  const baseline = audit.baseline;
  const stop = describeStopReason(audit);
  const inputs = Object.entries(audit.inputs);
  return (
    <div id="technical-details" className="checks-tech">
      <TechnicalDetails
        description="Setup, inputs, limits and the pinned baseline, for admins and debugging."
        defaultOpen={open || audit.state === "draft"}
      >
        <TechSection title="Profile and scope">
          <dl className="checks-facts">
            <Fact term="Check type">
              {auditProfileLabel(audit)}{" "}
              <IdChip
                value={`${audit.profile.name}@${audit.profile.version}`}
                display={`${audit.profile.name}@${audit.profile.version}`}
                label="check type version"
              />
            </Fact>
            <Fact term="Profile digest">
              <code className="checks-mono">{audit.profile.digest}</code>
            </Fact>
            <Fact term="Current revision">
              <code className="checks-mono">{audit.revision}</code>
            </Fact>
            <Fact term="Dispatch">{audit.dispatchState}</Fact>
            <Fact term="Evidence hold">{audit.holdState}</Fact>
            <Fact term="Objective">{audit.scope.objective ?? "Not set"}</Fact>
            <Fact term="Target">{audit.scope.target ?? "Not set"}</Fact>
            <Fact term="Authorization scope">
              {audit.scope.authorizationScope ?? "Not set"}
            </Fact>
            <Fact term="Runtime labels">
              {audit.runtimeLabels.length === 0
                ? "default"
                : audit.runtimeLabels.join(", ")}
            </Fact>
            <Fact term="Created">{formatTimestamp(audit.createdAt)}</Fact>
            {audit.startedAt === undefined ? null : (
              <Fact term="Started">{formatTimestamp(audit.startedAt)}</Fact>
            )}
            {audit.finishedAt === undefined ? null : (
              <Fact term="Ended">{formatTimestamp(audit.finishedAt)}</Fact>
            )}
            <Fact term="Updated">{formatTimestamp(audit.updatedAt)}</Fact>
            {stop === null ? null : (
              <Fact term="Stop reason">
                {stop.label === undefined ? null : (
                  <strong>{stop.label}.</strong>
                )}
                {/* The time limit is explained in words; its message is not. */}
                {stop.deadline ? null : <> {stop.message}</>}
              </Fact>
            )}
          </dl>
        </TechSection>
        <TechSection title="Inputs">
          {inputs.length === 0 ? (
            <p className="checks-quiet">No inputs selected.</p>
          ) : (
            <div className="checks-artifact-list">
              {inputs.map(([name, artifact]) => (
                <ExactArtifactLink
                  key={name}
                  projectId={audit.projectId}
                  artifact={artifact}
                  label={name}
                  projectReadable
                />
              ))}
            </div>
          )}
        </TechSection>
        <TechSection title="Limits and consumption">
          <dl className="checks-facts">
            <Fact term="Rounds">{audit.limits.maxRounds}</Fact>
            <Fact term="Batch size">{audit.limits.batchSize}</Fact>
            <Fact term="Items">{audit.limits.maxItemsTotal}</Fact>
            <Fact term="Attempts per item">
              {audit.limits.maxItemRunAttempts}
            </Fact>
            <Fact term="Runs submitted">
              {audit.submittedRunCount}/{audit.limits.maxSubmittedRuns}
            </Fact>
            <Fact term="Runs outstanding">{audit.outstandingRunCount}</Fact>
            <Fact term="Evidence retained">
              {formatBytes(audit.retainedEvidenceBytes)} /{" "}
              {formatBytes(audit.limits.maxEvidenceBytes)}
            </Fact>
            <Fact term="Deadline">{deadlineText(audit)}</Fact>
          </dl>
        </TechSection>
        {workspace === undefined ? null : (
          <TechSection title="Progress snapshot">
            <dl className="checks-facts">
              <Fact term="Revision">{workspace.auditRevision}</Fact>
              <Fact term="As of">{formatTimestamp(workspace.asOf)}</Fact>
              <Fact term="Round">{workspace.roundId ?? "No round yet"}</Fact>
              <Fact term="Execution">
                {checkStateLabel(workspace.executionState).label}
              </Fact>
              <Fact term="Outstanding runs">{workspace.outstandingRuns}</Fact>
              <Fact term="Done / total">
                {workspace.completedChecks} / {workspace.totalChecks}
              </Fact>
              <Fact term="Issues found">{workspace.issues}</Fact>
              <Fact term="Need follow-up">{workspace.gaps}</Fact>
              <Fact term="Not checked yet">{workspace.unchecked}</Fact>
              <Fact term="Possible issues">{workspace.findings}</Fact>
              <Fact term="Not reviewed yet">
                {workspace.unreviewedFindings}
              </Fact>
              <Fact term="Decisions waiting">{workspace.pendingReviews}</Fact>
            </dl>
            <p className="checks-quiet">
              Done counts concluded results and explicit exclusions of the
              current round. Finished work does not mean complete coverage,
              confirmed issues or an accepted report.
            </p>
          </TechSection>
        )}
        {baseline === undefined ? (
          <TechSection title="Baseline and standards">
            <p>
              <strong>The baseline is not pinned yet.</strong> Starting this
              draft validates its inputs and records the exact inventory.
            </p>
          </TechSection>
        ) : (
          <TechSection title="Baseline and standards">
            <dl className="checks-facts">
              <Fact term="Source content">
                <code className="checks-mono">
                  {baseline.inventory?.sourceContentDigest ??
                    "Pending preparation"}
                </code>
              </Fact>
              <Fact term="Inventory">
                <code className="checks-mono">
                  {baseline.inventory?.canonicalInventoryDigest ??
                    "Pending inventory"}
                </code>
              </Fact>
              <Fact term="Worklist">
                {baseline.inventory === undefined ? (
                  "Not created yet"
                ) : (
                  <ExactArtifactLink
                    projectId={audit.projectId}
                    artifact={baseline.inventory.worklist}
                  />
                )}
              </Fact>
              <Fact term="Skills">{baseline.skills.length}</Fact>
              <Fact term="Standards">{baseline.standards.length}</Fact>
              <Fact term="Runtime configs">
                {1 + baseline.runtimeConfig.labels.length}
              </Fact>
            </dl>
            {baseline.standards.length === 0 ? null : (
              <div
                className="checks-tech-block"
                data-testid="audit-baseline-standards"
              >
                <h4>Standards</h4>
                <ul className="checks-string-list">
                  {baseline.standards.map((standard) => (
                    <li
                      key={`${standard.reference.scheme}@${standard.reference.version}`}
                    >
                      <strong>{standard.title}</strong>{" "}
                      <code className="checks-mono">
                        {standard.reference.scheme}@{standard.reference.version}
                      </code>{" "}
                      ·{" "}
                      <code className="checks-mono">
                        {compactDigest(standard.retained.digest)}
                      </code>{" "}
                      ·{" "}
                      <SourceLink url={standard.source.url}>source</SourceLink>{" "}
                      · {standard.license.id}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {baseline.inventory?.standardSelection === undefined ? null : (
              <div
                className="checks-tech-block"
                data-testid="audit-baseline-standard-selection"
              >
                <h4>Selected requirements</h4>
                <p>{baseline.inventory.standardSelection.scope}</p>
                <p>
                  Level {baseline.inventory.standardSelection.levels.join(", ")}{" "}
                  · {baseline.inventory.standardSelection.entryIds.length}{" "}
                  requirements
                </p>
                <StringList
                  values={baseline.inventory.standardSelection.entryIds}
                />
              </div>
            )}
            {baseline.inventory === undefined ? null : (
              <div className="checks-tech-block">
                <h4>Inventory gaps</h4>
                <StringList
                  values={baseline.inventory.gaps}
                  empty="No inventory gaps."
                />
              </div>
            )}
          </TechSection>
        )}
      </TechnicalDetails>
    </div>
  );
}
