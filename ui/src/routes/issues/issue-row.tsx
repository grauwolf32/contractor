import { Fragment, type ReactNode } from "react";
import type { To } from "react-router";

import type { ProjectFindingSummary } from "../../api/audits";
import { formatShortDateTime, formatTimestamp } from "../../app/format";
import { findingStateLabel, severityLabel } from "../../app/vocabulary";
import { ListRow, MethodChip, StatusGlyph, StatusChip } from "../../ui";
import { httpOperation } from "../decisions/text";

import "./issues.css";

/** "Oct 4, 23:56", with the full timestamp on hover. */
function FoundTime({ value }: { value: string }) {
  if (Number.isNaN(Date.parse(value))) return <span>{value}</span>;
  return (
    <time dateTime={value} title={formatTimestamp(value)}>
      {formatShortDateTime(value)}
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
function severityParts(
  finding: ProjectFindingSummary,
  badges: boolean,
): ReactNode[] {
  function rating(
    text: string,
    value: NonNullable<ProjectFindingSummary["analystSeverity"]>,
  ) {
    if (!badges) return text;
    const tone =
      value === "critical" || value === "high"
        ? "blocked"
        : value === "medium"
          ? "warning"
          : value === "low"
            ? "info"
            : "neutral";
    return (
      <StatusChip tone={tone} glyph={false} size="sm">
        {text}
      </StatusChip>
    );
  }
  const suggestion = finding.severitySuggestion;
  return [
    <span className="issues-row-severity">
      {finding.analystSeverity === null
        ? "Severity not set"
        : rating(
            `Severity: ${severityLabel(finding.analystSeverity)}`,
            finding.analystSeverity,
          )}
    </span>,
    finding.analystSeverity !== null || suggestion === "" ? null : (
      <span className="issues-row-suggestion">
        {rating(`AI suggestion: ${severityLabel(suggestion)}`, suggestion)}
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

interface IssueRowProps {
  finding: ProjectFindingSummary;
  to: To;
  selected?: boolean | undefined;
  /** First part of the context line: the project or the check. */
  context?: ReactNode;
  id?: string | undefined;
  onSelect?: (() => void) | undefined;
  trailing?: ReactNode;
  showCreated?: boolean | undefined;
  severityBadges?: boolean | undefined;
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
  trailing,
  showCreated = true,
  severityBadges = false,
}: IssueRowProps) {
  const document = finding;
  const state = findingStateLabel(finding.state);
  const operation =
    document.subjectKey === "" ? undefined : httpOperation(document.subjectKey);
  return (
    <ListRow
      to={to}
      onSelect={onSelect}
      selected={selected}
      trailing={trailing}
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
              showCreated ? <FoundTime value={finding.createdAt} /> : null,
            ])}
          </span>
          <span className="issues-row-line">
            {separated([
              <span className="issues-row-state">{state.label}</span>,
              ...severityParts(finding, severityBadges),
            ])}
          </span>
        </span>
      }
    />
  );
}
