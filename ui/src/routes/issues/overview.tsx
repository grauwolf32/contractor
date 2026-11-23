import { useId, type ReactNode } from "react";
import type { ProjectFinding, ProjectFindingSummary } from "../../api/audits";
import { Icon } from "../../app/icon";

import { findingStateLabel, severityLabel } from "../../app/vocabulary";
import { MethodChip, StatusChip } from "../../ui";
import { RecordedTime } from "../../app/recorded-time";

import { DecisionMarkdown } from "../decisions/markdown";
import {
  httpOperation,
  splitImpact,
  standardReferences,
  weaknessReferences,
} from "../decisions/text";

import { EvidencePanel } from "./evidence-panel";

export function IssueIdentity({
  finding,
}: {
  finding: ProjectFinding | ProjectFindingSummary;
}) {
  const document =
    "firstReceiptId" in finding
      ? finding.claim!
      : {
          title: finding.title,
          subject:
            finding.subjectKind === "" && finding.subjectKey === ""
              ? null
              : { kind: finding.subjectKind, key: finding.subjectKey },
          severity_suggestion: finding.severitySuggestion,
        };
  const operation =
    document.subject === null ? undefined : httpOperation(document.subject.key);
  const state = findingStateLabel(finding.state);
  return (
    <>
      <h2 className="issues-workspace-title">{document.title}</h2>
      <div className="issues-subject">
        {operation === undefined ? (
          <>
            {document.subject?.kind === "file" ? (
              <span className="issues-subject-kind">
                <Icon name="document-code" size={12} />
                FILE
              </span>
            ) : (
              <span>Subject</span>
            )}
            <code>{document.subject?.key ?? "Not identified"}</code>
          </>
        ) : (
          <>
            <MethodChip method={operation.method} />
            <code>{operation.path}</code>
          </>
        )}
      </div>
      <div className="issues-identity-status">
        <StatusChip tone={state.tone} size="sm">
          {state.label}
        </StatusChip>
        <StatusChip tone="info" size="sm">
          Verification: see independent sources
        </StatusChip>
        <span className="issues-ai-suggestion">
          {finding.analystSeverity === null
            ? document.severity_suggestion === ""
              ? "AI severity: not suggested"
              : `AI suggests ${severityLabel(document.severity_suggestion)} — a suggestion, not a decision`
            : `Analyst severity: ${severityLabel(finding.analystSeverity)}`}
        </span>
      </div>
    </>
  );
}
function SideSection({
  title,
  tone,
  children,
}: {
  title: string;
  tone?: "danger" | "warning" | "info";
  children: ReactNode;
}) {
  const id = useId();
  return (
    <section className="issues-side-section" aria-labelledby={id}>
      <h3 id={id} className="issues-section-label" data-tone={tone}>
        <Icon name="question-circle" size={13} />
        {title}
      </h3>
      {children}
    </section>
  );
}
function TextList({ values }: { values: readonly string[] }) {
  return (
    <ul className="issues-side-points">
      {values.map((value, index) => (
        <li key={index}>{value}</li>
      ))}
    </ul>
  );
}

/** Route-owned composition; shared Inbox/Audit summaries retain their owner. */
export function IssueRecordedDecision({
  finding,
}: {
  finding: ProjectFinding;
}) {
  const state = findingStateLabel(finding.state);
  return finding.analystVerdict === null ? null : (
    <section
      className="issues-recorded-review"
      aria-label="Recorded decision"
      data-tone={state.tone}
    >
      <header>
        <span className="issues-section-label">Recorded decision</span>
        <StatusChip size="sm" tone={state.tone}>
          {state.label}
        </StatusChip>
        <span className="ui-caption">
          by {finding.actorId} ·{" "}
          {finding.assessedAt === null ? null : (
            <RecordedTime value={finding.assessedAt} />
          )}
        </span>
      </header>
      {finding.rationale === null ? null : (
        <DecisionMarkdown source={finding.rationale} />
      )}
    </section>
  );
}
export function IssueOverview({
  finding,
  supporting,
}: {
  finding: ProjectFinding;
  supporting?: ReactNode;
}) {
  const document = finding.claim!;
  const { found, impact } = splitImpact(document.description);
  const refs = [
    ...weaknessReferences(document),
    ...standardReferences(document),
  ];
  const limitations = document.limitations.filter(
    (value) => value.trim() !== "",
  );
  const preconditions = document.preconditions.filter(
    (value) => value.trim() !== "",
  );
  return (
    <>
      <div className="issues-summary-grid">
        <div className="issues-summary-main">
          <section className="issues-narrative" aria-label="What the AI found">
            <span className="issues-proposal-icon">
              <Icon name="issues" size={17} />
            </span>
            <div>
              <h3 className="issues-section-label">What the AI found</h3>
              <DecisionMarkdown source={found} />
            </div>
          </section>
          <EvidencePanel finding={finding} />
        </div>
        <aside className="issues-summary-side" aria-label="Review context">
          {impact === undefined ? null : (
            <SideSection title="If real, the impact" tone="danger">
              <DecisionMarkdown source={impact} />
            </SideSection>
          )}
          <SideSection title="Verification" tone="info">
            <p>
              Each source retains its own check results and conditions. Open
              History to inspect them.
            </p>
            <p className="ui-caption">
              Verification is separate from the analyst decision.
            </p>
          </SideSection>
          {limitations.length === 0 && preconditions.length === 0 ? null : (
            <SideSection title="Limitations" tone="warning">
              {limitations.length === 0 ? null : (
                <TextList values={limitations} />
              )}
              {preconditions.length === 0 ? null : (
                <>
                  <h4 className="issues-section-label issues-preconditions">
                    Preconditions
                  </h4>
                  <TextList values={preconditions} />
                </>
              )}
            </SideSection>
          )}
          {refs.length === 0 ? null : (
            <SideSection title="Standard references">
              <ul className="issues-side-points">
                {refs.map((ref, index) => (
                  <li key={index}>
                    {ref.requirement_id}{" "}
                    <span className="ui-caption">
                      · {ref.scheme}@{ref.version}
                    </span>
                  </li>
                ))}
              </ul>
            </SideSection>
          )}
          {supporting}
        </aside>
      </div>
    </>
  );
}
