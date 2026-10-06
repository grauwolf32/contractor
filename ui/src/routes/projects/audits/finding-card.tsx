import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import {
  listAuditFindingProvenance,
  type Audit,
  type AuditFinding,
  type AuditReviewRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { ErrorNotice } from "../../../app/error-notice";
import { formatBytes } from "../../../app/format";
import { QueryView } from "../../../app/query-view";
import { StateBadge } from "../../runs/components";
import { FindingDecision } from "../../decisions";
import { exactArtifactLink } from "./artifact-links";
import { FindingLocations } from "./finding-locations";
import { auditProfileLabel } from "./labels";
import { AuditMarkdown, ExactArtifactLink } from "./shared";

function FindingProvenanceView({
  audit,
  finding,
}: {
  audit: Audit;
  finding: AuditFinding;
}) {
  const api = usePublicAPI();
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
    <div className="audit-finding-provenance">
      <button
        className="text-button"
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? "Hide provenance" : "Show provenance"}
      </button>
      {!expanded ? null : (
        <QueryView
          query={provenance}
          loading={<p className="loading-copy">Loading exact provenance…</p>}
          onRetry={() => void provenance.refetch()}
        >
          {(data) => (
            <ol className="audit-provenance-list">
              {data.items.map((record) => (
                <li key={record.recordId}>
                  <div>
                    <StateBadge state={record.kind} />
                    <strong>{record.origin.workflow.name}</strong>
                    <code>@{record.origin.workflow.version}</code>
                  </div>
                  <span>
                    {record.origin.logicalAgentName} · {record.origin.runId}
                    {record.origin.runDeleted ? " · Run deleted" : ""}
                  </span>
                  {record.assessment === undefined ? null : (
                    <span>
                      assessment {record.assessment.semanticAssessment} ·{" "}
                      {record.supportsCurrentAssessment
                        ? "current"
                        : "historical"}
                    </span>
                  )}
                  {record.attempt === undefined ? null : (
                    <div className="audit-provenance-attempt">
                      <span>
                        attempt {record.attempt.itemAttempt} ·{" "}
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
                          verification Workflow{" "}
                          <strong>
                            {record.attempt.runProvenance.workflow.name}@
                            {record.attempt.runProvenance.workflow.version}
                          </strong>
                        </span>
                      )}
                      <span>
                        inventory entry {record.attempt.itemOrigin.entryKey}
                      </span>
                      {record.attempt.itemOrigin.standard ===
                      undefined ? null : (
                        <span>
                          causal standard mapping{" "}
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
              {data.items.length === 0 ? (
                <li>No retained provenance records.</li>
              ) : null}
            </ol>
          )}
        </QueryView>
      )}
    </div>
  );
}

export function AuditFindingCard({
  audit,
  finding,
  pendingReview,
  showAudit = false,
  reviewLoading = false,
  reviewError = null,
  reviewActionLabel = "Retry review status",
  onRetryReview,
}: {
  audit: Audit;
  finding: AuditFinding;
  /**
   * Unused: the decision's duplicate picker reads the check's possible issues
   * itself. Issues removes it together with its call sites in
   * audit-findings.tsx and findings.tsx (files owned by Issues).
   */
  findings?: AuditFinding[];
  /** The finding's open review request; omitted when it has none. */
  pendingReview?: AuditReviewRequest;
  showAudit?: boolean;
  reviewLoading?: boolean;
  reviewError?: unknown;
  reviewActionLabel?: string;
  onRetryReview?: () => void;
}) {
  const sources = [
    ...new Map(
      Object.entries(audit.inputs)
        .filter(
          ([name, artifact]) =>
            name === "source" || artifact.ref.namespace === "sources",
        )
        .map(([, artifact]) => [JSON.stringify(artifact.ref), artifact]),
    ).values(),
  ];
  return (
    <article
      className="panel audit-finding-card"
      id={`finding-${audit.auditId}-${finding.findingId}`}
    >
      <div className="section-heading">
        <div>
          {showAudit ? (
            <ContextLink
              returnLabel="Project findings"
              className="audit-finding-origin"
              to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/findings`}
            >
              {auditProfileLabel(audit)}{" "}
              <span>· {audit.auditId.slice(-8)}</span>
            </ContextLink>
          ) : null}
          <h3>{finding.firstProposal.document.title}</h3>
        </div>
        <StateBadge state={finding.state} />
      </div>
      <div
        className="audit-finding-decision"
        id={`decision-${audit.auditId}-${finding.findingId}`}
      >
        {reviewLoading ? (
          <p className="loading-copy">Loading review status…</p>
        ) : reviewError !== null ? (
          <div className="audit-finding-review-actions">
            <ErrorNotice error={reviewError} />
            <button
              className="secondary-button"
              type="button"
              onClick={onRetryReview}
            >
              {reviewActionLabel}
            </button>
          </div>
        ) : (
          <div className="decisions-inline">
            {/* Several cards share the page, so no single-key shortcuts. */}
            <FindingDecision
              auditId={audit.auditId}
              finding={finding}
              pendingReview={pendingReview ?? null}
              shortcuts={false}
            />
          </div>
        )}
      </div>
      <div className="audit-finding-description">
        <AuditMarkdown source={finding.firstProposal.document.description} />
      </div>
      <FindingLocations document={finding.firstProposal.document} />
      {sources.length === 0 ? null : (
        <section
          className="audit-finding-sources"
          aria-label="Source artifacts"
        >
          <h4>Sources used by this Audit</h4>
          <ul>
            {sources.map((artifact) => (
              <li key={JSON.stringify(artifact.ref)}>
                <ContextLink
                  returnLabel="Finding"
                  returnHash={`#finding-${audit.auditId}-${finding.findingId}`}
                  to={exactArtifactLink(audit.projectId, artifact)}
                  title={`Revision ${artifact.ref.revision}`}
                >
                  {artifact.ref.namespace}/{artifact.ref.name}
                </ContextLink>
                <small>
                  {artifact.sizeBytes === undefined
                    ? null
                    : `${formatBytes(artifact.sizeBytes)} · `}
                  Source revision
                </small>
              </li>
            ))}
          </ul>
        </section>
      )}
      <dl className="metadata-grid">
        <div>
          <dt>Model suggestion</dt>
          <dd>
            {finding.firstProposal.document.severity_suggestion || "none"}
          </dd>
        </div>
        <div>
          <dt>Analyst rating</dt>
          <dd>
            {finding.analystVerdict === undefined
              ? "Unreviewed"
              : `${finding.analystVerdict}${
                  finding.analystSeverity === undefined
                    ? ""
                    : ` · ${finding.analystSeverity}`
                }`}
          </dd>
        </div>
        <div>
          <dt>Source Workflow</dt>
          <dd>
            {finding.firstProposal.origin.workflow.name}@
            {finding.firstProposal.origin.workflow.version}
          </dd>
        </div>
        <div>
          <dt>Current verification</dt>
          <dd>
            {finding.currentAssessment?.semanticAssessment ?? "not accepted"}
          </dd>
        </div>
      </dl>
      {finding.duplicateTargetId === undefined ? null : (
        <p className="muted-copy">
          Duplicate of <code>{finding.duplicateTargetId}</code>
        </p>
      )}
      <details className="audit-record-details">
        <summary>Finding details</summary>
        <dl className="metadata-grid">
          <div>
            <dt>Finding ID</dt>
            <dd>
              <code>{finding.findingId}</code>
            </dd>
          </div>
          <div>
            <dt>Subject</dt>
            <dd>
              {finding.firstProposal.document.subject?.key ?? "Not specified"}
            </dd>
          </div>
          {pendingReview === undefined ? null : (
            <div>
              <dt>Review ID</dt>
              <dd>
                <code>{pendingReview.requestId}</code>
              </dd>
            </div>
          )}
        </dl>
      </details>
      <FindingProvenanceView audit={audit} finding={finding} />
    </article>
  );
}
