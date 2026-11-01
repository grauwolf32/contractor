import { useId, type ReactNode } from "react";
import type { AuditFinding } from "../../api/audits";
import { Icon } from "../../app/icon";
import type { StatusTone } from "../../app/status-tone";
import { findingStateLabel, severityLabel } from "../../app/vocabulary";
import { MethodChip, StatusChip } from "../../ui";
import { DecisionRecord } from "../decisions";
import { decisionOutcome } from "../decisions/model";
import { DecisionMarkdown } from "../decisions/markdown";
import {
  httpOperation,
  splitImpact,
  standardReferences,
  weaknessReferences,
} from "../decisions/text";
import { CorrelationMark } from "./correlation-mark";
import { assessmentLabel } from "./evidence";
import { EvidencePanel } from "./evidence-panel";

function assessmentTone(finding: AuditFinding): StatusTone {
  const assessment = finding.currentAssessment?.semanticAssessment;
  if (assessment === undefined) return "idle";
  if (assessment === "supported" || assessment === "satisfied") return "done";
  return assessment === "blocked" || assessment === "violated"
    ? "warning"
    : "info";
}
function verification(finding: AuditFinding): string {
  return finding.currentAssessment === undefined
    ? "Not verified yet"
    : assessmentLabel(finding.currentAssessment.semanticAssessment);
}

export function IssueIdentity({ finding }: { finding: AuditFinding }) {
  const document = finding.firstProposal.document;
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
        <StatusChip tone={assessmentTone(finding)} size="sm">
          Verification: {verification(finding)}
        </StatusChip>
        <span className="issues-ai-suggestion">
          {finding.analystSeverity === undefined
            ? document.severity_suggestion === ""
              ? "AI severity: not suggested"
              : `AI suggests ${severityLabel(document.severity_suggestion)} — a suggestion, not a decision`
            : `Analyst severity: ${severityLabel(finding.analystSeverity)}`}
        </span>
        <CorrelationMark finding={finding} />
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
export function IssueRecordedDecision({ finding }: { finding: AuditFinding }) {
  return finding.analystDecision === undefined ? null : (
    <section
      className="issues-recorded-review"
      aria-label="Recorded decision"
      data-tone={decisionOutcome(finding.analystDecision).tone}
    >
      <p className="issues-section-label">Recorded decision</p>
      <DecisionRecord decision={finding.analystDecision} />
    </section>
  );
}
export function IssueOverview({
  finding,
  supporting,
}: {
  finding: AuditFinding;
  supporting?: ReactNode;
}) {
  const document = finding.firstProposal.document;
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
            <p>{verification(finding)}</p>
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
