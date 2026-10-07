import type { UseQueryResult } from "@tanstack/react-query";
import { Fragment, lazy, Suspense, useId, type ReactNode } from "react";
import { Link } from "react-router";

import type { Audit, AuditReport, ExactArtifactRef } from "../../api/audits";
import { PublicAPIError } from "../../api/error";
import { ErrorNotice } from "../../app/error-notice";
import { compactDigest, formatBytes } from "../../app/format";
import { reportStatusLabel, reviewStateLabel } from "../../app/vocabulary";
import {
  StatusChip,
  StatusGlyph,
  TechnicalDetails,
  type StatusTone,
} from "../../ui";
import { ReportDecision } from "../decisions";
import { checkPath } from "./filters";
import {
  downloadReportFile,
  summaryDisclaimsCertification,
  type ReportAcceptance,
} from "./report-data";

import "./reports.css";

const ReportMarkdown = lazy(() => import("./report-markdown"));

/** The report's status word and tone; a proposed report never reads as ready. */
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
function StatusNotice({ status }: { status: AuditReport["status"] }) {
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
      return (
        <ReportNotice tone="neutral" title="No accepted report is available.">
          <p>Review the coverage of this check for explicit gaps.</p>
        </ReportNotice>
      );
    case "proposed":
      return (
        <ReportNotice
          tone="review"
          title="This report is awaiting owner acceptance."
        >
          <p>Review the contents below before making a decision.</p>
        </ReportNotice>
      );
    case "ready":
      return null;
  }
}

function DownloadIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19.5h14" />
    </svg>
  );
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
    <div className="reports-downloads">
      {machine === undefined ? null : (
        <button
          type="button"
          className="ui-btn"
          data-size="sm"
          onClick={() =>
            downloadReportFile(
              `${auditId}-report.json`,
              "application/json",
              JSON.stringify(machine, null, 2),
            )
          }
        >
          <DownloadIcon />
          Download JSON
        </button>
      )}
      {summary === undefined ? null : (
        <button
          type="button"
          className="ui-btn"
          data-size="sm"
          onClick={() =>
            downloadReportFile(`${auditId}-report.md`, "text/markdown", summary)
          }
        >
          <DownloadIcon />
          Download summary
        </button>
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
    <code title={artifact.digest}>{compactDigest(artifact.digest)}</code>,
  );
  return facts;
}

/** Exact retained files and the acceptance request, off the main path. */
function ReportFiles({ report }: { report: AuditReport }) {
  const files: ReportFile[] = [];
  if (report.machineArtifact !== undefined)
    files.push({ label: "Machine report", artifact: report.machineArtifact });
  if (report.summaryArtifact !== undefined)
    files.push({ label: "Summary report", artifact: report.summaryArtifact });
  const review = report.review;
  if (files.length === 0 && review === undefined) return null;
  return (
    <TechnicalDetails
      description={
        review === undefined
          ? "The exact retained report files."
          : "The exact retained report files and the acceptance request."
      }
    >
      <dl className="reports-facts">
        {files.map(({ label, artifact }) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>
              <code>{exactRef(artifact.ref)}</code>
              <span>
                {fileFacts(artifact).map((part, position) => (
                  // Facts are positional.
                  <Fragment key={position}>
                    {position > 0 ? " · " : null}
                    {part}
                  </Fragment>
                ))}
              </span>
            </dd>
          </div>
        ))}
        {review === undefined ? null : (
          <div>
            <dt>Acceptance request</dt>
            <dd>
              <code>{review.requestId}</code>
              <span>
                {reviewStateLabel(review.state).label} · revision{" "}
                {review.revision}
              </span>
            </dd>
          </div>
        )}
      </dl>
    </TechnicalDetails>
  );
}

function loadErrorHint(error: Error): string {
  if (error instanceof PublicAPIError && error.status === 409)
    return "The check changed while its report was read. Try again to load the current report.";
  if (error instanceof PublicAPIError && error.status === 404)
    return "The check or its report is no longer available. It may have been deleted.";
  return "Try again to load the report.";
}

export interface ReportContentProps {
  audit: Audit;
  /** The report read (useAuditReport). */
  query: Pick<
    UseQueryResult<AuditReport>,
    "data" | "error" | "isFetching" | "refetch"
  >;
  acceptance: ReportAcceptance;
}

/**
 * A check's report: status messages, the summary as Markdown, downloads,
 * the certification notice and the exact files. The acceptance decision is
 * placed by the page (ReportAcceptanceDecision).
 */
export function ReportContent({
  audit,
  query,
  acceptance,
}: ReportContentProps) {
  const summaryId = useId();
  const report = query.data;
  const retry = () => void query.refetch();
  if (report === undefined) {
    if (query.error === null)
      return (
        <p className="reports-quiet" role="status">
          Loading the report…
        </p>
      );
    return (
      <div className="reports-body">
        <ErrorNotice
          error={query.error}
          context="Could not load the report"
          onRetry={retry}
          retryPending={query.isFetching}
        />
        <p className="reports-quiet">{loadErrorHint(query.error)}</p>
      </div>
    );
  }
  const readable = report.status === "proposed" || report.status === "ready";
  return (
    <div className="reports-body">
      {acceptance.mismatch ? (
        <ReportNotice
          tone="warning"
          role="status"
          title="The requested report review is unavailable or no longer current."
        >
          <p>This report cannot be used to decide that review.</p>
        </ReportNotice>
      ) : null}
      {query.error === null ? null : (
        <ReportNotice
          tone="warning"
          role="status"
          title="Could not refresh; showing the last loaded data."
        >
          <p>{query.error.message}</p>
          <button
            type="button"
            className="ui-btn"
            data-size="xs"
            disabled={query.isFetching}
            onClick={retry}
          >
            {query.isFetching ? "Loading…" : "Try again"}
          </button>
        </ReportNotice>
      )}
      <StatusNotice status={report.status} />
      <p className="reports-intro">
        {readable ? (
          <span>
            Read the conclusion and its limitations. Full coverage and confirmed
            issues are separate from the technical outcome of the check.
          </span>
        ) : null}
        <Link to={`${checkPath(audit, "coverage")}?result=uncertain`}>
          Review coverage gaps →
        </Link>
      </p>
      {readable ? (
        <section className="reports-summary" aria-labelledby={summaryId}>
          <div className="reports-summary-head">
            <h3 id={summaryId} className="reports-heading">
              Summary
            </h3>
            <ReportDownloads auditId={audit.auditId} report={report} />
          </div>
          {report.summary === undefined ? null : (
            <div className="reports-document">
              <Suspense
                fallback={<p className="reports-quiet">Loading the summary…</p>}
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
    </div>
  );
}

/**
 * ReportDecision for the report's acceptance request, when the page has one
 * to show; why the request cannot be read, when that read failed; null
 * otherwise (see showsAcceptance).
 */
export function ReportAcceptanceDecision({
  auditId,
  acceptance,
}: {
  auditId: string;
  acceptance: ReportAcceptance;
}) {
  if (acceptance.error !== null)
    return (
      <ErrorNotice
        error={acceptance.error}
        context="Could not load the acceptance request"
        onRetry={acceptance.retry}
        retryPending={acceptance.retrying}
      />
    );
  if (acceptance.review === undefined) return null;
  return (
    <ReportDecision
      auditId={auditId}
      review={acceptance.review}
      report={acceptance.report}
      onDecided={acceptance.onDecided}
      onRecording={acceptance.onRecording}
    />
  );
}
