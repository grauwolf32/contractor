import { useId, type ReactNode } from "react";

import type { Audit, AuditWorkspace } from "../../../api/audits";
import {
  compactDigest,
  formatBytes,
  formatTimestamp,
} from "../../../app/format";
import { checkStateLabel } from "../../../app/vocabulary";
import { Facts, IdChip, TechnicalDetails } from "../../../ui";
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
  if (values.length === 0) return <span className="ui-caption">{empty}</span>;
  return (
    <ul className="checks-string-list">
      {values.map((value, index) => (
        <li key={`${index}-${value}`}>{value}</li>
      ))}
    </ul>
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
  const inventory = baseline?.inventory ?? audit.generatedInventory;
  const stop = describeStopReason(audit);
  const inputs = Object.entries(audit.inputs);
  return (
    <div id="technical-details" className="checks-tech">
      <TechnicalDetails
        description="Setup, inputs, limits and the pinned baseline, for admins and debugging."
        defaultOpen={open || audit.state === "draft"}
      >
        <TechSection title="Profile and scope">
          <Facts
            size="sm"
            items={[
              {
                term: "Check type",
                value: (
                  <>
                    {auditProfileLabel(audit)}{" "}
                    <IdChip
                      value={`${audit.profile.name}@${audit.profile.version}`}
                      display={`${audit.profile.name}@${audit.profile.version}`}
                      label="check type version"
                    />
                  </>
                ),
              },
              {
                term: "Profile digest",
                value: <code className="ui-mono">{audit.profile.digest}</code>,
              },
              {
                term: "Current revision",
                value: <code className="ui-mono">{audit.revision}</code>,
              },
              { term: "Dispatch", value: audit.dispatchState },
              { term: "Evidence hold", value: audit.holdState },
              { term: "Objective", value: audit.scope.objective ?? "Not set" },
              { term: "Target", value: audit.scope.target ?? "Not set" },
              {
                term: "Authorization scope",
                value: audit.scope.authorizationScope ?? "Not set",
              },
              {
                term: "Runtime labels",
                value:
                  audit.runtimeLabels.length === 0
                    ? "default"
                    : audit.runtimeLabels.join(", "),
              },
              { term: "Created", value: formatTimestamp(audit.createdAt) },
              {
                term: "Started",
                value:
                  audit.startedAt === undefined
                    ? undefined
                    : formatTimestamp(audit.startedAt),
              },
              {
                term: "Ended",
                value:
                  audit.finishedAt === undefined
                    ? undefined
                    : formatTimestamp(audit.finishedAt),
              },
              { term: "Updated", value: formatTimestamp(audit.updatedAt) },
              {
                term: "Stop reason",
                value:
                  stop === null ? undefined : (
                    <>
                      {stop.label === undefined ? null : (
                        <strong>{stop.label}.</strong>
                      )}
                      {/* The time limit is explained in words; its message is not. */}
                      {stop.deadline ? null : <> {stop.message}</>}
                    </>
                  ),
              },
            ]}
          />
        </TechSection>
        <TechSection title="Inputs">
          {inputs.length === 0 ? (
            <p className="ui-caption">No inputs selected.</p>
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
          <Facts
            size="sm"
            items={[
              { term: "Rounds", value: audit.limits.maxRounds },
              { term: "Batch size", value: audit.limits.batchSize },
              { term: "Items", value: audit.limits.maxItemsTotal },
              {
                term: "Attempts per item",
                value: audit.limits.maxItemRunAttempts,
              },
              {
                term: "Runs submitted",
                value: (
                  <>
                    {audit.submittedRunCount}/{audit.limits.maxSubmittedRuns}
                  </>
                ),
              },
              { term: "Runs outstanding", value: audit.outstandingRunCount },
              {
                term: "Evidence retained",
                value: (
                  <>
                    {formatBytes(audit.retainedEvidenceBytes)} /{" "}
                    {formatBytes(audit.limits.maxEvidenceBytes)}
                  </>
                ),
              },
              { term: "Deadline", value: deadlineText(audit) },
            ]}
          />
        </TechSection>
        {workspace === undefined ? null : (
          <TechSection title="Progress snapshot">
            <Facts
              size="sm"
              items={[
                { term: "Revision", value: workspace.auditRevision },
                { term: "As of", value: formatTimestamp(workspace.asOf) },
                { term: "Round", value: workspace.roundId ?? "No round yet" },
                {
                  term: "Execution",
                  value: checkStateLabel(workspace.executionState).label,
                },
                { term: "Outstanding runs", value: workspace.outstandingRuns },
                {
                  term: "Done / total",
                  value: (
                    <>
                      {workspace.completedChecks} / {workspace.totalChecks}
                    </>
                  ),
                },
                { term: "Issues found", value: workspace.issues },
                { term: "Need follow-up", value: workspace.gaps },
                { term: "Not checked yet", value: workspace.unchecked },
                { term: "Possible issues", value: workspace.findings },
                {
                  term: "Not reviewed yet",
                  value: workspace.unreviewedFindings,
                },
                { term: "Decisions waiting", value: workspace.pendingReviews },
              ]}
            />
            <p className="ui-caption">
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
            <Facts
              size="sm"
              items={[
                {
                  term: "Source content",
                  value: (
                    <code className="ui-mono">
                      {inventory?.sourceContentDigest ?? "Pending preparation"}
                    </code>
                  ),
                },
                {
                  term: "Inventory",
                  value: (
                    <code className="ui-mono">
                      {inventory?.canonicalInventoryDigest ??
                        "Pending inventory"}
                    </code>
                  ),
                },
                {
                  term: "Worklist",
                  value:
                    inventory === undefined ? (
                      "Not created yet"
                    ) : (
                      <ExactArtifactLink
                        projectId={audit.projectId}
                        artifact={inventory.worklist}
                      />
                    ),
                },
                {
                  term: "Active checks",
                  value:
                    baseline.activeChecks === "automatic"
                      ? "Automatically"
                      : baseline.activeChecks === "approval-required"
                        ? "Ask before each active check"
                        : "Disabled",
                },
                { term: "Skills", value: baseline.skills.length },
                { term: "Standards", value: baseline.standards.length },
                {
                  term: "Runtime configs",
                  value: 1 + baseline.runtimeConfig.labels.length,
                },
              ]}
            />
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
                      <code className="ui-mono">
                        {standard.reference.scheme}@{standard.reference.version}
                      </code>{" "}
                      ·{" "}
                      <code className="ui-mono">
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
            {inventory === undefined ? null : (
              <div className="checks-tech-block">
                <h4>Inventory gaps</h4>
                <StringList
                  values={inventory.gaps}
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
