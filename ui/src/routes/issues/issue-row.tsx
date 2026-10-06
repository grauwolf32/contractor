import { Fragment, type ReactNode } from "react";
import type { To } from "react-router";

import type { AuditFinding } from "../../api/audits";
import { formatTimestamp } from "../../app/format";
import { findingStateLabel, severityLabel } from "../../app/vocabulary";
import { ListRow, MethodChip, StatusGlyph } from "../../ui";
import { httpOperation, weaknessReferences } from "../decisions/text";

import "./issues.css";

const foundFormat = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

/** "Oct 4, 23:56", with the full timestamp on hover. */
export function FoundTime({ value }: { value: string }) {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return <span>{value}</span>;
  return (
    <time dateTime={value} title={formatTimestamp(value)}>
      {foundFormat.format(date)}
    </time>
  );
}

/** A path that wraps after its slashes rather than inside a segment. */
function BreakablePath({ path }: { path: string }) {
  const segments = path.split("/");
  return (
    <span className="issues-row-path">
      {segments.map((segment, position) => (
        // Segments are positional.
        <Fragment key={position}>
          {segment}
          {position < segments.length - 1 ? (
            <>
              /<wbr />
            </>
          ) : null}
        </Fragment>
      ))}
    </span>
  );
}

/**
 * The analyst's rating, and apart from it the AI's suggestion while the
 * analyst has not rated: the two are never mixed (S19:1791-1793).
 */
function severityParts(finding: AuditFinding): ReactNode[] {
  const suggestion = finding.firstProposal.document.severity_suggestion;
  return [
    <span className="issues-row-severity">
      {finding.analystSeverity === undefined
        ? "Severity not set"
        : `Severity: ${severityLabel(finding.analystSeverity)}`}
    </span>,
    finding.analystSeverity !== undefined || suggestion === "" ? null : (
      <span className="issues-row-suggestion">
        AI suggestion: {severityLabel(suggestion)}
      </span>
    ),
  ];
}

function separated(parts: readonly ReactNode[]): ReactNode {
  return parts
    .filter((part) => part !== null && part !== undefined && part !== "")
    .map((part, position) => (
      // Parts are positional.
      <span key={position} className="issues-row-part">
        {position > 0 ? (
          <span className="ui-row-sep" aria-hidden="true">
            ·{" "}
          </span>
        ) : null}
        {part}
      </span>
    ));
}

export interface IssueRowProps {
  finding: AuditFinding;
  to: To;
  selected?: boolean | undefined;
  /** First part of the context line: the project or the check. */
  context?: ReactNode;
  id?: string | undefined;
  onSelect?: (() => void) | undefined;
}

/**
 * One possible issue in a list: state glyph, title, the endpoint when the
 * subject is an HTTP operation, context (project or check), weakness, when
 * it was found, then its state and severity in words.
 */
export function IssueRow({
  finding,
  to,
  selected = false,
  context,
  id,
  onSelect,
}: IssueRowProps) {
  const document = finding.firstProposal.document;
  const state = findingStateLabel(finding.state);
  const operation =
    document.subject === null ? undefined : httpOperation(document.subject.key);
  const weakness = weaknessReferences(document)[0]?.requirement_id;
  return (
    <ListRow
      to={to}
      onSelect={onSelect}
      selected={selected}
      id={id}
      glyph={<StatusGlyph tone={state.tone} />}
      title={document.title}
      meta={
        <span className="issues-row-meta">
          {operation === undefined ? null : (
            <span className="issues-row-endpoint">
              <MethodChip method={operation.method} />{" "}
              <BreakablePath path={operation.path} />
            </span>
          )}
          <span className="issues-row-line">
            {separated([
              context === undefined || context === null ? null : (
                <span className="issues-row-context">{context}</span>
              ),
              weakness,
              <FoundTime value={finding.createdAt} />,
            ])}
          </span>
          <span className="issues-row-line">
            {separated([
              <span className="issues-row-state">{state.label}</span>,
              ...severityParts(finding),
            ])}
          </span>
        </span>
      }
    />
  );
}
