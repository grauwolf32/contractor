import type { UseQueryResult } from "@tanstack/react-query";
import { Fragment, lazy, Suspense, useId, type ReactNode } from "react";
import { Link } from "react-router";

import type { Audit, AuditReport, ExactArtifactRef } from "../../api/audits";
import { PublicAPIError } from "../../api/error";
import { writeConflict } from "../../app/conflict";

import { compactDigest, formatBytes } from "../../app/format";
import { Icon } from "../../app/icon";
import { QueryView } from "../../app/query-view";
import { reportStatusLabel } from "../../app/vocabulary";
import {
  Button,
  Facts,
  StatusChip,
  StatusGlyph,
  TechnicalDetails,
  type StatusTone,
} from "../../ui";

import { saveBlob } from "../../app/download";

import "./report.css";

const ReportMarkdown = lazy(() => import("./markdown"));

const CERTIFICATION_NOTICE = "not a security or compliance certification";

/** The summary already says the report is not a certification. */
function summaryDisclaimsCertification(summary: string | undefined): boolean {
  return (
    summary !== undefined &&
    summary.replace(/\s+/g, " ").toLowerCase().includes(CERTIFICATION_NOTICE)
  );
}

/** Saves report content generated in the browser from the report response. */
function downloadReportFile(
  name: string,
  mediaType: string,
  content: string,
): void {
  saveBlob(new Blob([content], { type: mediaType }), name);
}

/** The report's status word and tone. */
export function ReportStatusChip({
  status,
  size,
}: {
  status: AuditReport["status"];
  size?: "sm" | "md" | undefined;
}) {
  const { label, tone } = reportStatusLabel(status);
  return (
    <StatusChip tone={tone} size={size}>
      {label}
    </StatusChip>
  );
}

function ReportNotice({
  tone,
  title,
  children,
  role,
}: {
  tone: StatusTone;
  title: ReactNode;
  children?: ReactNode;
  role?: "status" | undefined;
}) {
  return (
    <div className="reports-notice" data-tone={tone} role={role}>
      <StatusGlyph tone={tone} />
      <div className="reports-notice-text">
        <p className="reports-notice-title">{title}</p>
        {children}
      </div>
    </div>
  );
}

/** What the report's status means, in the words the check page always used. */
function StatusNotice({
  status,
  audit,
}: {
  status: AuditReport["status"];
  audit: Audit;
}) {
  switch (status) {
    case "pending":
      return (
        <ReportNotice
          tone="idle"
          title="The check has not reached report generation."
        >
          <p>A report that is not ready yet is not a successful assessment.</p>
        </ReportNotice>
      );
    case "unavailable":
      if (audit.state === "completed")
        return (
          <ReportNotice
            tone="neutral"
            title="This check completed without a summary report."
          >
            <p>
              Review the saved task results and their limitations on the check
              page.
            </p>
          </ReportNotice>
        );
      return (
        <ReportNotice tone="neutral" title="No report is available.">
          <p>Review the coverage of this check for explicit gaps.</p>
        </ReportNotice>
      );
    case "ready":
      return null;
  }
}

/** JSON and Markdown files built in the browser from the report response. */
function ReportDownloads({
  auditId,
  report,
}: {
  auditId: string;
  report: AuditReport;
}) {
  const { machine, summary } = report;
  if (machine === undefined && summary === undefined) return null;
  return (
    <div className="ui-toolbar">
      {machine === undefined ? null : (
        <Button
          size="sm"
          icon={<Icon name="download" size={15} strokeWidth={2} />}
          onClick={() =>
            downloadReportFile(
              `${auditId}-report.json`,
              "application/json",
              JSON.stringify(machine, null, 2),
            )
          }
        >
          Download JSON
        </Button>
      )}
      {summary === undefined ? null : (
        <Button
          size="sm"
          icon={<Icon name="download" size={15} strokeWidth={2} />}
          onClick={() =>
            downloadReportFile(`${auditId}-report.md`, "text/markdown", summary)
          }
        >
          Download summary
        </Button>
      )}
    </div>
  );
}

function exactRef(ref: ExactArtifactRef): string {
  return `${ref.namespace}/${ref.name}@${ref.revision}`;
}

type ReportArtifact = NonNullable<AuditReport["machineArtifact"]>;

interface ReportFile {
  label: string;
  artifact: ReportArtifact;
}

/** Media type, size and digest (shortened; the full digest on hover). */
function fileFacts(artifact: ReportArtifact): ReactNode[] {
  const facts: ReactNode[] = [];
  if (artifact.mediaType !== undefined) facts.push(artifact.mediaType);
  if (artifact.sizeBytes !== undefined)
    facts.push(formatBytes(artifact.sizeBytes));
  facts.push(
    <code className="ui-mono" title={artifact.digest}>
      {compactDigest(artifact.digest)}
    </code>,
  );
  return facts;
}

/** An exact reference over a line of details. */
function FileValue({ code, details }: { code: string; details: ReactNode }) {
  return (
    <span className="reports-file">
      <code className="ui-mono">{code}</code>
      <span>{details}</span>
    </span>
  );
}

/** Exact retained files, off the main path. */
function ReportFiles({ report }: { report: AuditReport }) {
  const files: ReportFile[] = [];
  if (report.machineArtifact !== undefined)
    files.push({ label: "Machine report", artifact: report.machineArtifact });
  if (report.summaryArtifact !== undefined)
    files.push({ label: "Summary report", artifact: report.summaryArtifact });
  if (files.length === 0) return null;
  return (
    <TechnicalDetails description="The exact retained report files.">
      <Facts
        layout="list"
        size="sm"
        items={[
          ...files.map(({ label, artifact }) => ({
            term: label,
            value: (
              <FileValue
                code={exactRef(artifact.ref)}
                details={fileFacts(artifact).map((part, position) => (
                  // Facts are positional.
                  <Fragment key={position}>
                    {position > 0 ? " · " : null}
                    {part}
                  </Fragment>
                ))}
              />
            ),
          })),
        ]}
      />
    </TechnicalDetails>
  );
}

function loadErrorHint(error: Error): string {
  if (writeConflict(error) === "conflict")
    return "The check changed while its report was read. Try again to load the current report.";
  if (error instanceof PublicAPIError && error.status === 404)
    return "The check or its report is no longer available. It may have been deleted.";
  return "Try again to load the report.";
}

interface ReportLinks {
  taskResults: string;
  coverageGaps: string;
}

interface ReportContentProps {
  links: ReportLinks;
  audit: Audit;
  /** The report read (useAuditReport). */
  query: Pick<
    UseQueryResult<AuditReport>,
    "data" | "error" | "isFetching" | "refetch"
  >;
}

/**
 * A check's report: status messages, the summary as Markdown, downloads,
 * the certification notice and the exact files.
 */
export function ReportContent({ audit, query, links }: ReportContentProps) {
  return (
    <div className="reports-body">
      <QueryView
        query={query}
        loading={
          <p className="ui-caption" role="status">
            Loading the report…
          </p>
        }
        errorContext="Could not load the report"
        errorHint={loadErrorHint}
      >
        {(report) => (
          <LoadedReportContent audit={audit} report={report} links={links} />
        )}
      </QueryView>
    </div>
  );
}

function LoadedReportContent({
  audit,
  report,
  links,
}: {
  audit: Audit;
  report: AuditReport;
  links: ReportLinks;
}) {
  const summaryId = useId();
  const readable = report.status === "ready";
  return (
    <>
      <StatusNotice status={report.status} audit={audit} />
      <p className="reports-intro">
        {readable ? (
          <span>
            Read the conclusion and its limitations. Full coverage and confirmed
            issues are separate from the technical outcome of the check.
          </span>
        ) : null}
        {report.status === "unavailable" ? (
          <Link to={links.taskResults}>View task results →</Link>
        ) : (
          <Link to={links.coverageGaps}>Review coverage gaps →</Link>
        )}
      </p>
      {readable ? (
        <section className="ui-stack" aria-labelledby={summaryId}>
          <div className="reports-summary-head">
            <h3 id={summaryId} className="reports-heading">
              Summary
            </h3>
            <ReportDownloads auditId={audit.auditId} report={report} />
          </div>
          {report.summary === undefined ? null : (
            <div className="reports-document">
              <Suspense
                fallback={<p className="ui-caption">Loading the summary…</p>}
              >
                <ReportMarkdown source={report.summary} />
              </Suspense>
            </div>
          )}
        </section>
      ) : null}
      {/* The Server's summaries end with this notice; say it only once. */}
      {readable && !summaryDisclaimsCertification(report.summary) ? (
        <p className="reports-disclaimer">
          <StatusGlyph tone="info" />
          <span>
            A report describes this bounded check. It is not a security or
            compliance certification.
          </span>
        </p>
      ) : null}
      <ReportFiles report={report} />
    </>
  );
}
