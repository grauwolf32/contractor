import { useId, type ReactNode } from "react";
import { Link, type To } from "react-router";

import type { Audit, AuditFinding } from "../../../api/audits";
import { ContextLink } from "../../../app/context-navigation";
import {
  capitalize,
  findingStateLabel,
  itemNoun,
  severityLabel,
} from "../../../app/vocabulary";
import {
  ActivityLog,
  Kbd,
  MethodChip,
  StatusChip,
  StatusGlyph,
} from "../../../ui";
import { weaknessReferences } from "../../decisions/text";
import { itemActivity } from "./activity-model";
import { statusDescription } from "./assessments";
import { formatShortDateTime, formatSpan } from "./check-format";
import type { CheckEntry } from "./check-model";
import { ItemTechnicalDetails } from "./executions";
import { PathText } from "./path-text";
import { AuditMarkdown } from "./shared";

const EVIDENCE_LABELS: Readonly<Record<string, string>> = {
  artifact: "Artifact",
  observation: "Observation",
  "tool-result": "Tool result",
  "manual-attestation": "Manual review",
  "runtime-metric": "Runtime measurement",
  implementation: "Implementation",
  tests: "Tests",
  "source-analysis": "Source analysis",
  "configuration-review": "Configuration review",
  "documentation-review": "Documentation review",
  "active-test": "Active test",
  "manual-review": "Manual review",
  "operation-resolution": "Operation resolution",
};

function readable(value: string): string {
  return EVIDENCE_LABELS[value] ?? value;
}

function Section({
  title,
  aside,
  children,
}: {
  title: string;
  aside?: ReactNode;
  children: ReactNode;
}) {
  const heading = useId();
  return (
    <section className="checks-item-section" aria-labelledby={heading}>
      <div className="checks-item-section-heading">
        <h3 id={heading}>{title}</h3>
        {aside === undefined ? null : (
          <span className="checks-quiet">{aside}</span>
        )}
      </div>
      {children}
    </section>
  );
}

/** Why a result is not complete, by status. */
function gapsHeading(entry: CheckEntry): string {
  switch (entry.row.coverage.status) {
    case "traced-partial":
      return "Why only partially";
    case "blocked":
      return "Why it is blocked";
    case "inconclusive":
      return "Why it is inconclusive";
    case "unmapped":
      return "Why it could not be mapped";
    default:
      return "Limitations and missing evidence";
  }
}

function PossibleIssue({
  audit,
  finding,
}: {
  audit: Audit;
  finding: AuditFinding;
}) {
  const document = finding.firstProposal.document;
  const state = findingStateLabel(finding.state);
  const weaknesses = weaknessReferences(document);
  return (
    <li className="checks-issue">
      <StatusGlyph tone={state.tone} />
      <div className="checks-issue-main">
        <Link
          to={`/issues/${encodeURIComponent(audit.auditId)}/${encodeURIComponent(finding.findingId)}`}
        >
          {document.title}
        </Link>
        <p className="checks-issue-meta">
          {weaknesses.length === 0 ? null : (
            <span>
              {weaknesses.map((weakness) => weakness.requirement_id).join(", ")}
            </span>
          )}
          <span>
            Severity {severityLabel(finding.analystSeverity).toLowerCase()}
          </span>
          <span>Found {formatShortDateTime(finding.createdAt)}</span>
        </p>
      </div>
      <StatusChip tone={state.tone} size="sm">
        {state.label}
      </StatusChip>
    </li>
  );
}

/** The work done on an item: span of its attempts and what is left. */
function Attempts({ audit, entry }: { audit: Audit; entry: CheckEntry }) {
  const attempts = entry.item?.attempts ?? [];
  const first = attempts[0];
  const last = attempts.at(-1);
  if (first === undefined || last === undefined) return null;
  const verb = entry.kind === "endpoint" ? "Traced" : "Checked";
  const running = last.collectedAt === undefined;
  const max = audit.limits.maxItemRunAttempts;
  const failed =
    last.terminalOutcome === "failed" ||
    last.terminalOutcome === "submission-failed" ||
    last.collectionDisposition === "execution-failed";
  return (
    <>
      <span className="checks-quiet">
        {running
          ? `Started ${formatShortDateTime(last.createdAt)}`
          : `${verb} ${formatSpan(first.createdAt, last.collectedAt)}`}
      </span>
      {attempts.length > 1 ? (
        <span className="checks-quiet">
          Retried automatically · attempt {last.itemAttempt} of {max}
        </span>
      ) : null}
      {last.runId === undefined ? null : (
        <ContextLink
          returnLabel="Check"
          to={`/runs/${encodeURIComponent(last.runId)}`}
          className="checks-run-link"
        >
          {last.runDeleted ? "Deleted run" : "Open the run"}
        </ContextLink>
      )}
      {failed && !running && attempts.length >= max ? (
        <span className="checks-quiet">
          No attempts are left. The run shows what went wrong and the recovery
          the Server offers there.
        </span>
      ) : null}
    </>
  );
}

/**
 * One endpoint, requirement or scenario of the check: its result and why it
 * is incomplete, the possible issues found on it, the evidence the AI used,
 * what happened to it, and its technical details. Items retry by
 * themselves; there is no retry action here.
 */
export function CheckItemView({
  audit,
  entry,
  place,
  area,
  previous,
  next,
}: {
  audit: Audit;
  entry: CheckEntry;
  /** Position in the shown list and its length, when the item is listed. */
  place: { index: number; count: number } | undefined;
  /** "mechanic area", or the standard the requirement comes from. */
  area: string | undefined;
  previous: To | undefined;
  next: To | undefined;
}) {
  const { row } = entry;
  const singular = itemNoun(entry.kind, 1);
  const noun = capitalize(singular);
  const conclusion = row.details?.resultSummary || row.coverage.rationale;
  const description = statusDescription(row.coverage.status);
  const completed = new Set(row.coverage.completed);
  const evidence = [
    ...new Set([...row.coverage.requested, ...row.coverage.completed]),
  ];
  const supporting = row.details?.evidence ?? [];
  const activity = itemActivity(entry);
  const objective = row.details?.objective ?? "";
  const operationLine =
    entry.operation === undefined
      ? undefined
      : `${entry.operation.method} ${entry.operation.path}`;
  const task =
    operationLine !== undefined && objective.startsWith(operationLine)
      ? objective.slice(operationLine.length).trim()
      : objective.trim();
  return (
    <article
      className="checks-item"
      aria-labelledby={`check-${row.itemId}-title`}
    >
      <header className="checks-item-header">
        <div className="checks-item-place">
          <span>
            {place === undefined
              ? noun
              : `${noun} ${place.index + 1} of ${place.count}`}
            {area === undefined ? "" : ` · ${area}`}
          </span>
          <span className="checks-item-steps">
            {previous === undefined ? (
              <button
                type="button"
                className="ui-btn"
                data-size="sm"
                data-icon-only=""
                aria-label={`Previous ${singular}`}
                aria-keyshortcuts="K"
                disabled
              >
                <span aria-hidden="true">↑</span>
              </button>
            ) : (
              <Link
                className="ui-btn"
                data-size="sm"
                data-icon-only=""
                to={previous}
                aria-label={`Previous ${singular}`}
                aria-keyshortcuts="K"
                title="Previous (K)"
              >
                <span aria-hidden="true">↑</span>
              </Link>
            )}
            {next === undefined ? (
              <button
                type="button"
                className="ui-btn"
                data-size="sm"
                data-icon-only=""
                aria-label={`Next ${singular}`}
                aria-keyshortcuts="J"
                disabled
              >
                <span aria-hidden="true">↓</span>
              </button>
            ) : (
              <Link
                className="ui-btn"
                data-size="sm"
                data-icon-only=""
                to={next}
                aria-label={`Next ${singular}`}
                aria-keyshortcuts="J"
                title="Next (J)"
              >
                <span aria-hidden="true">↓</span>
              </Link>
            )}
            <span className="checks-key-hint" aria-hidden="true">
              <Kbd>K</Kbd> <Kbd>J</Kbd>
            </span>
          </span>
        </div>
        <h2 className="checks-item-title" id={`check-${row.itemId}-title`}>
          {entry.operation === undefined ? (
            <>
              <span className="checks-item-key">{entry.key}</span>
              {entry.summary === "" || entry.kind === "endpoint" ? null : (
                <>
                  {" "}
                  <span className="checks-item-name">{entry.summary}</span>
                </>
              )}
            </>
          ) : (
            <>
              <MethodChip method={entry.operation.method} />{" "}
              <span className="checks-item-path">
                <PathText path={entry.operation.path} />
              </span>
            </>
          )}
        </h2>
        <div className="checks-item-status">
          <StatusChip tone={entry.status.tone}>{entry.status.label}</StatusChip>
          <Attempts audit={audit} entry={entry} />
        </div>
      </header>
      <Section title="Conclusion">
        {conclusion ? (
          <AuditMarkdown source={conclusion} />
        ) : (
          <p className="checks-quiet">
            {row.coverage.status === "not-tested"
              ? `This ${singular} has no result yet.`
              : "No written conclusion is available."}
          </p>
        )}
        {description === undefined ? null : (
          <p className="checks-status-note">{description}</p>
        )}
        {row.coverage.rationale && row.coverage.rationale !== conclusion ? (
          <AuditMarkdown source={row.coverage.rationale} />
        ) : null}
        {row.coverage.gaps.length === 0 ? null : (
          <div className="checks-gaps">
            <strong>{gapsHeading(entry)}</strong>
            <ul>
              {row.coverage.gaps.map((gap, index) => (
                <li key={index}>{gap}</li>
              ))}
            </ul>
          </div>
        )}
      </Section>
      {entry.findings.length === 0 ? null : (
        <Section
          title={
            entry.findings.length === 1
              ? `Possible issue on this ${singular}`
              : `Possible issues on this ${singular}`
          }
        >
          <ul className="checks-issues">
            {entry.findings.map((finding) => (
              <PossibleIssue
                key={finding.findingId}
                audit={audit}
                finding={finding}
              />
            ))}
          </ul>
        </Section>
      )}
      <div className="checks-item-columns">
        <Section
          title="What the AI looked at"
          aside={
            supporting.length === 0
              ? undefined
              : `${supporting.length} ${supporting.length === 1 ? "piece" : "pieces"} of evidence`
          }
        >
          <ul
            className="checks-evidence-checklist"
            aria-label="Evidence coverage"
          >
            {evidence.length === 0 ? (
              <li className="checks-quiet">No evidence requirements listed.</li>
            ) : (
              evidence.map((value) => (
                <li
                  key={value}
                  data-collected={completed.has(value) ? "" : undefined}
                >
                  <StatusGlyph tone={completed.has(value) ? "done" : "idle"} />
                  {readable(value)}{" "}
                  <span className="checks-quiet">
                    · {completed.has(value) ? "collected" : "missing"}
                  </span>
                </li>
              ))
            )}
          </ul>
          {supporting.length === 0 ? null : (
            <ul className="checks-evidence">
              {supporting.map((piece) => (
                <li key={piece.id}>
                  <strong>{readable(piece.kind)}</strong>
                  <AuditMarkdown source={piece.summary} />
                  <code className="checks-mono checks-quiet">{piece.id}</code>
                </li>
              ))}
            </ul>
          )}
        </Section>
        <Section title={`Activity on this ${singular}`}>
          {activity.length === 0 ? (
            <p className="checks-quiet">Nothing has happened yet.</p>
          ) : (
            <ActivityLog
              aria-label={`Activity on this ${singular}`}
              entries={activity}
            />
          )}
        </Section>
      </div>
      {task === "" && (row.details?.methods.length ?? 0) === 0 ? null : (
        <Section title="What the AI was asked">
          {task === "" ? null : <AuditMarkdown source={task} />}
          {row.details?.methods.length ? (
            <p className="checks-quiet">
              Method: {row.details.methods.map(readable).join(", ")}
            </p>
          ) : null}
        </Section>
      )}
      <ItemTechnicalDetails audit={audit} entry={entry} />
    </article>
  );
}
