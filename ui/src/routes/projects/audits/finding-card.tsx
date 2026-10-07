import { useQuery } from "@tanstack/react-query";
import { useId, useState } from "react";

import {
  listAuditFindingProvenance,
  type Audit,
  type AuditFinding,
  type AuditReviewRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { formatBytes } from "../../../app/format";
import { RecordedTime } from "../../../app/recorded-time";
import { severityLabel, verdictLabel } from "../../../app/vocabulary";
import { IdChip, StatusChip, TechnicalDetails } from "../../../ui";
import { FindingDecision, FindingSummary } from "../../decisions";
import {
  assessmentLabel,
  checkSources,
  provenanceKindLabel,
} from "../../issues/evidence";
import { exactArtifactLink } from "./artifact-links";
import { ExactArtifactLink } from "./shared";

import "../../issues/issues.css";

type Heading = "h3" | "h4";

/**
 * Where a possible issue came from: the proposal, check attempts and direct
 * verifications behind its current assessment. Read only when the user asks
 * ("Show provenance"), pinned to the check and finding revisions shown.
 */
export function FindingProvenance({
  audit,
  finding,
}: {
  audit: Audit;
  finding: AuditFinding;
}) {
  const api = usePublicAPI();
  const panelId = useId();
  const [expanded, setExpanded] = useState(false);
  const provenance = useQuery({
    queryKey: queryKeys.audits.provenance(
      audit.auditId,
      finding.findingId,
      audit.revision,
      finding.revision,
    ),
    queryFn: () =>
      listAuditFindingProvenance(api, audit.auditId, finding.findingId, {
        auditRevision: audit.revision,
        findingRevision: finding.revision,
      }),
    enabled: expanded,
    retry: false,
  });
  return (
    <div className="issues-provenance">
      <button
        className="ui-btn"
        data-size="xs"
        type="button"
        aria-expanded={expanded}
        aria-controls={panelId}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? "Hide provenance" : "Show provenance"}
      </button>
      <div id={panelId} hidden={!expanded}>
        {!expanded ? null : provenance.data === undefined ? (
          provenance.error === null ? (
            <p className="issues-quiet" role="status">
              Loading provenance…
            </p>
          ) : (
            <div className="issues-notice" data-tone="error" role="alert">
              <p>Provenance could not be loaded: {provenance.error.message}</p>
              <button
                type="button"
                className="ui-btn"
                data-size="xs"
                disabled={provenance.isFetching}
                onClick={() => void provenance.refetch()}
              >
                Try again
              </button>
            </div>
          )
        ) : provenance.data.items.length === 0 ? (
          <p className="issues-quiet">No retained provenance records.</p>
        ) : (
          <>
            <ol className="issues-provenance-list">
              {provenance.data.items.map((record) => (
                <li key={record.recordId}>
                  <p className="issues-provenance-head">
                    <StatusChip tone="neutral" size="sm" glyph={false}>
                      {provenanceKindLabel(record.kind)}
                    </StatusChip>
                    <strong>{record.origin.workflow.name}</strong>
                    <code>@{record.origin.workflow.version}</code>
                    <span className="issues-quiet">
                      <RecordedTime value={record.createdAt} />
                    </span>
                  </p>
                  <p className="issues-provenance-line">
                    {record.origin.logicalAgentName} · Run {record.origin.runId}
                    {record.origin.runDeleted ? " · Run deleted" : ""}
                  </p>
                  {record.assessment === undefined ? null : (
                    <p className="issues-provenance-line">
                      Assessment{" "}
                      {assessmentLabel(record.assessment.semanticAssessment)} ·{" "}
                      {record.supportsCurrentAssessment
                        ? "current"
                        : "historical"}
                    </p>
                  )}
                  {record.attempt === undefined ? null : (
                    <div className="issues-provenance-attempt">
                      <span>
                        Attempt {record.attempt.itemAttempt} ·{" "}
                        {record.attempt.role}/{record.attempt.workflowRole} ·{" "}
                        {record.attempt.state}
                      </span>
                      <span>
                        {record.attempt.terminalOutcome ?? "execution pending"}{" "}
                        ·{" "}
                        {record.attempt.collectionDisposition ??
                          "not collected"}
                        {record.attempt.runDeleted ? " · Run deleted" : ""}
                      </span>
                      {record.attempt.runProvenance?.workflow ===
                      undefined ? null : (
                        <span>
                          Verification workflow{" "}
                          <strong>
                            {record.attempt.runProvenance.workflow.name}@
                            {record.attempt.runProvenance.workflow.version}
                          </strong>
                        </span>
                      )}
                      <span>
                        Inventory entry {record.attempt.itemOrigin.entryKey}
                      </span>
                      {record.attempt.itemOrigin.standard ===
                      undefined ? null : (
                        <span>
                          Causal standard mapping{" "}
                          <strong>
                            {record.attempt.itemOrigin.standard.scheme}@
                            {record.attempt.itemOrigin.standard.version}/
                            {record.attempt.itemOrigin.standard.mappingKey}
                          </strong>
                        </span>
                      )}
                      <ExactArtifactLink
                        projectId={audit.projectId}
                        artifact={record.attempt.task}
                        label="Task"
                        projectReadable
                      />
                      {record.attempt.result === undefined ? null : (
                        <ExactArtifactLink
                          projectId={audit.projectId}
                          artifact={record.attempt.result}
                          label="Result"
                          projectReadable
                        />
                      )}
                    </div>
                  )}
                </li>
              ))}
            </ol>
            {provenance.data.page.hasMore ? (
              <p className="issues-quiet">
                More provenance records exist than shown here.
              </p>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

/** The check's source materials, linked to their exact revisions. */
export function FindingSources({
  audit,
  heading = "h3",
  returnLabel,
}: {
  audit: Audit;
  heading?: Heading | undefined;
  returnLabel: string;
}) {
  const headingId = useId();
  const sources = checkSources(audit);
  if (sources.length === 0) return null;
  const Heading = heading;
  return (
    <section className="issues-sources" aria-labelledby={headingId}>
      <Heading id={headingId} className="issues-heading">
        Sources used by this check
      </Heading>
      <ul className="issues-file-list">
        {sources.map((artifact) => (
          <li key={JSON.stringify(artifact.ref)}>
            <ContextLink
              returnLabel={returnLabel}
              className="issues-file-name"
              to={exactArtifactLink(audit.projectId, artifact)}
              title={`Revision ${artifact.ref.revision}`}
            >
              {artifact.ref.namespace}/{artifact.ref.name}
            </ContextLink>
            <span className="issues-quiet">
              {artifact.sizeBytes === undefined
                ? null
                : `${formatBytes(artifact.sizeBytes)} · `}
              Source revision
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

function analystRating(finding: AuditFinding): string {
  if (finding.analystVerdict === undefined) return "Not reviewed";
  const verdict = verdictLabel(finding.analystVerdict).label;
  return finding.analystSeverity === undefined
    ? verdict
    : `${verdict} · ${severityLabel(finding.analystSeverity)}`;
}

/**
 * Internal facts of a possible issue behind "Technical details": the AI's
 * severity suggestion next to the analyst's rating (never mixed), the
 * workflow that proposed it, its verification, the original of a
 * duplicate, and its identifiers.
 */
export function FindingTechnicalDetails({
  audit,
  finding,
  pendingReview,
  showCheckId = false,
}: {
  audit: Audit;
  finding: AuditFinding;
  /** The open review request, when the page knows it. */
  pendingReview?: AuditReviewRequest | null | undefined;
  /** Show the check ID (off where the check header already shows it). */
  showCheckId?: boolean | undefined;
}) {
  const proposal = finding.firstProposal;
  const document = proposal.document;
  const workflow = `${proposal.origin.workflow.name}@${proposal.origin.workflow.version}`;
  return (
    <TechnicalDetails description="Ratings, identifiers and where this possible issue came from.">
      <dl className="issues-facts">
        <div>
          <dt>AI suggestion</dt>
          <dd>
            {document.severity_suggestion === ""
              ? "None"
              : severityLabel(document.severity_suggestion)}
          </dd>
        </div>
        <div>
          <dt>Analyst rating</dt>
          <dd>{analystRating(finding)}</dd>
        </div>
        <div>
          <dt>Source workflow</dt>
          <dd>
            <IdChip
              value={workflow}
              display={workflow}
              label="source workflow version"
            />
          </dd>
        </div>
        <div>
          <dt>Verification</dt>
          <dd>
            {finding.currentAssessment === undefined
              ? "Not verified yet"
              : assessmentLabel(finding.currentAssessment.semanticAssessment)}
          </dd>
        </div>
        {finding.duplicateTargetId === undefined ? null : (
          <div>
            <dt>Duplicate of</dt>
            <dd>
              <IdChip
                value={finding.duplicateTargetId}
                label="original possible issue ID"
              />
            </dd>
          </div>
        )}
        <div>
          <dt>Possible issue ID</dt>
          <dd>
            <IdChip value={finding.findingId} label="possible issue ID" />
          </dd>
        </div>
        {showCheckId ? (
          <div>
            <dt>Check ID</dt>
            <dd>
              <IdChip value={audit.auditId} label="check ID" />
            </dd>
          </div>
        ) : null}
        <div>
          <dt>Subject</dt>
          <dd>
            {document.subject === null ? (
              "Not specified"
            ) : (
              <>
                <span className="issues-mono">{document.subject.key}</span>{" "}
                <span className="issues-quiet">{document.subject.kind}</span>
              </>
            )}
          </dd>
        </div>
        {pendingReview === undefined || pendingReview === null ? null : (
          <div>
            <dt>Review ID</dt>
            <dd>
              <IdChip value={pendingReview.requestId} label="review ID" />
            </dd>
          </div>
        )}
        <div>
          <dt>Revision</dt>
          <dd>{finding.revision}</dd>
        </div>
      </dl>
    </TechnicalDetails>
  );
}

/**
 * One possible issue in full inside its check (the `?finding=` deep link of
 * the check's possible issues section): the summary, the decision, the
 * sources the check used, technical details and provenance.
 */
export function AuditFindingCard({
  audit,
  finding,
  pendingReview,
  reviewLoading = false,
  reviewError = null,
  reviewActionLabel = "Retry review status",
  onRetryReview,
}: {
  audit: Audit;
  finding: AuditFinding;
  /**
   * The finding's open review request; null when it has none. Omitted, the
   * decision reads the finding's pending reviews itself.
   */
  pendingReview?: AuditReviewRequest | null | undefined;
  reviewLoading?: boolean | undefined;
  /** Blocks deciding and says why, e.g. a review that no longer matches. */
  reviewError?: unknown;
  reviewActionLabel?: string | undefined;
  onRetryReview?: (() => void) | undefined;
}) {
  return (
    <div
      className="issues-card"
      id={`finding-${audit.auditId}-${finding.findingId}`}
    >
      <FindingSummary auditId={audit.auditId} finding={finding} titleAs="h3" />
      <div
        className="issues-card-decision"
        id={`decision-${audit.auditId}-${finding.findingId}`}
      >
        {reviewLoading ? (
          <p className="issues-quiet" role="status">
            Loading review status…
          </p>
        ) : reviewError !== null && reviewError !== undefined ? (
          <div className="issues-notice" data-tone="error" role="alert">
            <p>
              {reviewError instanceof Error
                ? reviewError.message
                : "The review status could not be loaded."}
            </p>
            {onRetryReview === undefined ? null : (
              <button
                className="ui-btn"
                data-size="xs"
                type="button"
                onClick={onRetryReview}
              >
                {reviewActionLabel}
              </button>
            )}
          </div>
        ) : (
          <div className="decisions-inline">
            {/* The check page binds its own keys, so no single-key shortcuts. */}
            <FindingDecision
              auditId={audit.auditId}
              finding={finding}
              pendingReview={pendingReview}
              shortcuts={false}
            />
          </div>
        )}
      </div>
      <FindingSources audit={audit} heading="h4" returnLabel="Possible issue" />
      <FindingTechnicalDetails
        audit={audit}
        finding={finding}
        pendingReview={pendingReview}
      />
      <FindingProvenance audit={audit} finding={finding} />
    </div>
  );
}
