import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  createAuditFindingReview,
  decideAuditFinding,
  listAuditFindingProvenance,
  type Audit,
  type AuditAnalystVerdict,
  type AuditFinding,
  type AuditFindingSeverity,
  type AuditReviewRequest,
  type DecideAuditFindingRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice, formatBytes } from "../../artifacts/common";
import { StateBadge } from "../../runs/components";
import { exactArtifactLink } from "./artifact-links";
import { AuditMutationNotice } from "./controls";
import { FINDING_SEVERITIES } from "./finding-options";
import { FindingLocations } from "./finding-locations";
import { auditProfileLabel } from "./labels";
import { AuditMarkdown, ExactArtifactLink } from "./shared";

const FINDING_VERDICTS: readonly {
  value: AuditAnalystVerdict;
  label: string;
}[] = [
  { value: "true_positive", label: "True positive" },
  { value: "false_positive", label: "False positive" },
  { value: "needs_evidence", label: "Needs evidence" },
  { value: "duplicate", label: "Duplicate" },
  { value: "reopen", label: "Reopen" },
];

function FindingReviewControls({
  audit,
  finding,
  pendingReview,
  findings,
}: {
  audit: Audit;
  finding: AuditFinding;
  pendingReview?: AuditReviewRequest;
  findings: AuditFinding[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [verdict, setVerdict] = useState<AuditAnalystVerdict>("true_positive");
  const [severity, setSeverity] = useState<AuditFindingSeverity>(
    finding.analystSeverity ?? "medium",
  );
  const [rationale, setRationale] = useState("");
  const duplicateCandidates = findings.filter(
    (candidate) => candidate.findingId !== finding.findingId,
  );
  const [duplicateTargetId, setDuplicateTargetId] = useState(
    duplicateCandidates[0]?.findingId ?? "",
  );
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<Record<string, string | number | undefined>>(
        "audit-finding-review",
      ),
  );

  // The Audit detail key prefixes its findings and reviews queries.
  async function invalidate(): Promise<void> {
    await Promise.all([
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.detail(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.projects.audits.all(audit.projectId),
      }),
    ]);
  }

  const createReview = useMutation({
    mutationFn: () => {
      const draft = {
        operation: "create",
        auditId: audit.auditId,
        findingId: finding.findingId,
        revision: finding.revision,
      };
      return createAuditFindingReview(api, {
        auditId: audit.auditId,
        findingId: finding.findingId,
        expectedRevision: finding.revision,
        idempotencyKey: keyring.keyFor(draft),
      });
    },
    onSuccess: invalidate,
    onError: invalidate,
  });
  const decide = useMutation({
    mutationFn: () => {
      if (pendingReview === undefined) {
        throw new Error("The finding review is no longer pending");
      }
      const decision: DecideAuditFindingRequest =
        verdict === "true_positive"
          ? { verdict, rationale: rationale.trim(), severity }
          : verdict === "duplicate"
            ? { verdict, rationale: rationale.trim(), duplicateTargetId }
            : { verdict, rationale: rationale.trim() };
      const draft = {
        operation: "decide",
        auditId: audit.auditId,
        requestId: pendingReview.requestId,
        revision: pendingReview.revision,
        verdict,
        severity:
          decision.verdict === "true_positive" ? decision.severity : undefined,
        rationale: decision.rationale,
        duplicateTargetId:
          decision.verdict === "duplicate"
            ? decision.duplicateTargetId
            : undefined,
      };
      return decideAuditFinding(api, {
        auditId: audit.auditId,
        requestId: pendingReview.requestId,
        expectedRevision: pendingReview.revision,
        idempotencyKey: keyring.keyFor(draft),
        decision,
      });
    },
    onSuccess: async () => {
      setRationale("");
      await invalidate();
    },
    onError: invalidate,
  });

  if (pendingReview === undefined) {
    return (
      <div className="audit-finding-review-actions">
        <button
          className="secondary-button"
          type="button"
          disabled={createReview.isPending}
          onClick={() => createReview.mutate()}
        >
          {createReview.isPending
            ? "Opening review…"
            : finding.analystVerdict === undefined
              ? "Review finding"
              : "Correct analyst rating"}
        </button>
        {createReview.error === null ? null : (
          <AuditMutationNotice error={createReview.error} />
        )}
      </div>
    );
  }

  const invalidDuplicate =
    verdict === "duplicate" && duplicateTargetId.length === 0;
  return (
    <form
      className="audit-finding-review-form"
      onSubmit={(event) => {
        event.preventDefault();
        decide.mutate();
      }}
    >
      <p className="eyebrow">
        Finding review · revision {pendingReview.subjectRevision}
      </p>
      <div className="audit-review-fields">
        <label>
          Decision
          <select
            value={verdict}
            onChange={(event) =>
              setVerdict(event.target.value as AuditAnalystVerdict)
            }
          >
            {FINDING_VERDICTS.map((candidate) => (
              <option key={candidate.value} value={candidate.value}>
                {candidate.label}
              </option>
            ))}
          </select>
        </label>
        {verdict === "true_positive" ? (
          <label>
            Severity
            <select
              value={severity}
              onChange={(event) =>
                setSeverity(event.target.value as AuditFindingSeverity)
              }
            >
              {FINDING_SEVERITIES.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {candidate}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {verdict === "duplicate" ? (
          <label>
            Linked finding
            <select
              aria-label="Linked finding"
              value={duplicateTargetId}
              onChange={(event) => setDuplicateTargetId(event.target.value)}
            >
              {duplicateCandidates.length === 0 ? (
                <option value="">No other finding</option>
              ) : null}
              {duplicateCandidates.map((candidate) => (
                <option key={candidate.findingId} value={candidate.findingId}>
                  {candidate.firstProposal.document.title} ·{" "}
                  {candidate.findingId}
                </option>
              ))}
            </select>
            <span>Or enter a finding ID from another page in this Audit</span>
            <input
              aria-label="Linked finding ID"
              value={duplicateTargetId}
              onChange={(event) => setDuplicateTargetId(event.target.value)}
            />
          </label>
        ) : null}
      </div>
      <label className="audit-review-rationale">
        Analyst rationale
        <textarea
          required
          rows={2}
          value={rationale}
          onChange={(event) => setRationale(event.target.value)}
          placeholder="Explain the evidence for this decision. Markdown is supported."
        />
      </label>
      <button
        type="submit"
        disabled={
          decide.isPending || rationale.trim() === "" || invalidDuplicate
        }
      >
        {decide.isPending ? "Recording decision…" : "Record decision"}
      </button>
      {decide.error === null ? null : (
        <AuditMutationNotice error={decide.error} />
      )}
    </form>
  );
}

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
      {!expanded ? null : provenance.isPending ? (
        <p className="loading-copy">Loading exact provenance…</p>
      ) : provenance.error !== null ? (
        <ErrorNotice error={provenance.error} />
      ) : (
        <ol className="audit-provenance-list">
          {provenance.data.items.map((record) => (
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
                  {record.supportsCurrentAssessment ? "current" : "historical"}
                </span>
              )}
              {record.attempt === undefined ? null : (
                <div className="audit-provenance-attempt">
                  <span>
                    attempt {record.attempt.itemAttempt} · {record.attempt.role}
                    /{record.attempt.workflowRole} · {record.attempt.state}
                  </span>
                  <span>
                    {record.attempt.terminalOutcome ?? "execution pending"} ·{" "}
                    {record.attempt.collectionDisposition ?? "not collected"}
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
                  {record.attempt.itemOrigin.standard === undefined ? null : (
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
          {provenance.data.items.length === 0 ? (
            <li>No retained provenance records.</li>
          ) : null}
        </ol>
      )}
    </div>
  );
}

export function AuditFindingCard({
  audit,
  finding,
  findings,
  pendingReview,
  showAudit = false,
  reviewLoading = false,
  reviewError = null,
  onRetryReview,
}: {
  audit: Audit;
  finding: AuditFinding;
  findings: AuditFinding[];
  pendingReview?: AuditReviewRequest;
  showAudit?: boolean;
  reviewLoading?: boolean;
  reviewError?: unknown;
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
              Retry review status
            </button>
          </div>
        ) : (
          <FindingReviewControls
            audit={audit}
            finding={finding}
            findings={findings}
            {...(pendingReview === undefined ? {} : { pendingReview })}
          />
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
