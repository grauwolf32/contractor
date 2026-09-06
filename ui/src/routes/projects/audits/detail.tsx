import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router";

import {
  AUDIT_ID_PATTERN,
  auditMutationAudit,
  auditNeedsPolling,
  createAuditFindingReview,
  decideAuditAction,
  decideAuditFinding,
  getAudit,
  getAuditReport,
  listAuditCoverage,
  listAuditFindingProvenance,
  listAuditFindings,
  listAuditItems,
  listAuditReviews,
  mutateAudit,
  type Audit,
  type AuditAnalystVerdict,
  type AuditCoverageRow,
  type AuditFinding,
  type AuditFindingSeverity,
  type AuditMutationAction,
  type AuditReviewAction,
  type AuditReviewRequest,
  type DecideAuditFindingRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import { getProject, PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import {
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../../artifacts/common";
import { StateBadge } from "../../runs/components";

import "./styles.css";

type AuditSection =
  | "overview"
  | "coverage"
  | "findings"
  | "checks"
  | "reviews"
  | "runs"
  | "report";
type AuditExactArtifact = Audit["inputs"][string];

const SECTIONS: readonly { id: AuditSection; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "coverage", label: "Coverage" },
  { id: "findings", label: "Findings" },
  { id: "checks", label: "Checks" },
  { id: "reviews", label: "Reviews" },
  { id: "runs", label: "Runs" },
  { id: "report", label: "Report" },
];

function exactArtifactLink(projectId: string, artifact: AuditExactArtifact) {
  return `/projects/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(artifact.ref.namespace)}/${encodeURIComponent(artifact.ref.name)}?revision=${encodeURIComponent(artifact.ref.revision)}`;
}

function ExactArtifactLink({
  projectId,
  artifact,
  label,
  projectReadable = false,
}: {
  projectId: string;
  artifact: AuditExactArtifact;
  label?: string;
  projectReadable?: boolean;
}) {
  const content = (
    <>
      {label === undefined ? null : <strong>{label}</strong>}
      <code>
        {artifact.ref.namespace}/{artifact.ref.name}@{artifact.ref.revision}
      </code>
    </>
  );
  return projectReadable ? (
    <Link
      className="artifact-ref-link"
      to={exactArtifactLink(projectId, artifact)}
      title={artifact.digest}
    >
      {content}
    </Link>
  ) : (
    <span className="artifact-ref-link" title={artifact.digest}>
      {content}
    </span>
  );
}

function StringList({
  values,
  empty = "none",
}: {
  values: string[];
  empty?: string;
}) {
  if (values.length === 0) return <span className="muted-copy">{empty}</span>;
  return (
    <ul className="audit-string-list">
      {values.map((value, index) => (
        <li key={`${index}-${value}`}>{value}</li>
      ))}
    </ul>
  );
}

function AuditMutationNotice({ error }: { error: unknown }) {
  return (
    <>
      <ErrorNotice error={error} reconcileWrite />
      {error instanceof PublicAPIError && error.status === 412 ? (
        <p className="muted-copy" role="status">
          The Audit revision changed. The page has refreshed authoritative
          state; review it before retrying the action.
        </p>
      ) : null}
    </>
  );
}

function AuditControls({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{
        action: AuditMutationAction;
        auditId: string;
        revision: number;
      }>("mutate-audit"),
  );
  const mutation = useMutation({
    mutationFn: (action: AuditMutationAction) => {
      const draft = {
        action,
        auditId: audit.auditId,
        revision: audit.revision,
      };
      return mutateAudit(api, action, {
        auditId: audit.auditId,
        expectedRevision: audit.revision,
        idempotencyKey: keyring.keyFor(draft),
      });
    },
    onSuccess: async (result) => {
      const updated = auditMutationAudit(result);
      queryClient.setQueryData(
        queryKeys.audits.detail(updated.auditId),
        updated,
      );
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.projects.audits.all(updated.projectId),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.audits.items(updated.auditId),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.audits.coverage(updated.auditId),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.audits.report(updated.auditId),
        }),
      ]);
    },
    onError: async () => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.audits.detail(audit.auditId),
      });
    },
  });
  const buttons: Array<{
    action: AuditMutationAction;
    label: string;
    dangerous?: boolean;
  }> = [];
  if (audit.state === "draft")
    buttons.push({ action: "start", label: "Start Audit" });
  if (audit.state === "active" || audit.state === "waiting_review") {
    buttons.push({ action: "pause", label: "Pause new Runs" });
  }
  if (audit.state === "paused")
    buttons.push({ action: "resume", label: "Resume" });
  if (
    audit.state === "active" ||
    audit.state === "waiting_review" ||
    audit.state === "paused" ||
    audit.state === "finalizing"
  ) {
    buttons.push({ action: "cancel", label: "Cancel", dangerous: true });
  }
  if (
    audit.state === "draft" ||
    audit.state === "completed" ||
    audit.state === "cancelled" ||
    audit.state === "failed"
  ) {
    buttons.push({ action: "delete", label: "Delete", dangerous: true });
  }
  return (
    <div className="audit-controls">
      {buttons.map((button) => (
        <button
          key={button.action}
          className={button.dangerous ? "danger-button" : "secondary-button"}
          type="button"
          disabled={mutation.isPending}
          onClick={() => mutation.mutate(button.action)}
        >
          {mutation.isPending && mutation.variables === button.action
            ? `${button.label}…`
            : button.label}
        </button>
      ))}
      {mutation.error === null ? null : (
        <div className="audit-control-error">
          <AuditMutationNotice error={mutation.error} />
        </div>
      )}
    </div>
  );
}

function AuditOverview({ audit }: { audit: Audit }) {
  const baseline = audit.baseline;
  return (
    <div className="audit-detail-stack">
      <section className="panel audit-section-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Pinned contract</p>
            <h3>Profile and scope</h3>
          </div>
          <code title={audit.profile.digest}>
            {audit.profile.name}@{audit.profile.version}
          </code>
        </div>
        <dl className="metadata-grid">
          <div>
            <dt>Profile digest</dt>
            <dd>
              <code>{audit.profile.digest}</code>
            </dd>
          </div>
          <div>
            <dt>Current revision</dt>
            <dd>
              <code>{audit.revision}</code>
            </dd>
          </div>
          <div>
            <dt>Dispatch</dt>
            <dd>{audit.dispatchState}</dd>
          </div>
          <div>
            <dt>Evidence hold</dt>
            <dd>{audit.holdState}</dd>
          </div>
          <div>
            <dt>Objective</dt>
            <dd>{audit.scope.objective ?? "Not set"}</dd>
          </div>
          <div>
            <dt>Target</dt>
            <dd>{audit.scope.target ?? "Not set"}</dd>
          </div>
          <div>
            <dt>Authorization scope</dt>
            <dd>{audit.scope.authorizationScope ?? "Not set"}</dd>
          </div>
          <div>
            <dt>Runtime labels</dt>
            <dd>
              {audit.runtimeLabels.length === 0
                ? "default"
                : audit.runtimeLabels.join(", ")}
            </dd>
          </div>
        </dl>
      </section>
      <section className="panel audit-section-panel">
        <p className="eyebrow">Immutable selections</p>
        <h3>Inputs</h3>
        <div className="audit-artifact-list">
          {Object.entries(audit.inputs).map(([name, artifact]) => (
            <ExactArtifactLink
              key={name}
              projectId={audit.projectId}
              artifact={artifact}
              label={name}
              projectReadable
            />
          ))}
        </div>
      </section>
      <section className="panel audit-section-panel">
        <p className="eyebrow">Bounded execution</p>
        <h3>Limits and consumption</h3>
        <dl className="metadata-grid audit-limit-grid">
          <div>
            <dt>Rounds</dt>
            <dd>{audit.limits.maxRounds}</dd>
          </div>
          <div>
            <dt>Batch size</dt>
            <dd>{audit.limits.batchSize}</dd>
          </div>
          <div>
            <dt>Items</dt>
            <dd>{audit.limits.maxItemsTotal}</dd>
          </div>
          <div>
            <dt>Attempts/item</dt>
            <dd>{audit.limits.maxItemRunAttempts}</dd>
          </div>
          <div>
            <dt>Runs submitted</dt>
            <dd>
              {audit.submittedRunCount}/{audit.limits.maxSubmittedRuns}
            </dd>
          </div>
          <div>
            <dt>Runs outstanding</dt>
            <dd>{audit.outstandingRunCount}</dd>
          </div>
          <div>
            <dt>Evidence retained</dt>
            <dd>
              {formatBytes(audit.retainedEvidenceBytes)} /{" "}
              {formatBytes(audit.limits.maxEvidenceBytes)}
            </dd>
          </div>
          <div>
            <dt>Deadline</dt>
            <dd>
              {audit.deadlineAt === undefined
                ? "Not started"
                : formatTimestamp(audit.deadlineAt)}
            </dd>
          </div>
        </dl>
      </section>
      {baseline === undefined ? (
        <section className="panel audit-section-panel compact-empty">
          <strong>Baseline is not pinned yet.</strong>
          <p>
            Starting this draft validates inputs and records the exact
            inventory.
          </p>
        </section>
      ) : (
        <section className="panel audit-section-panel">
          <p className="eyebrow">Start-time snapshot</p>
          <h3>Baseline</h3>
          <dl className="metadata-grid">
            <div>
              <dt>Source content</dt>
              <dd>
                <code>{baseline.inventory.sourceContentDigest}</code>
              </dd>
            </div>
            <div>
              <dt>Canonical inventory</dt>
              <dd>
                <code>{baseline.inventory.canonicalInventoryDigest}</code>
              </dd>
            </div>
            <div>
              <dt>Worklist</dt>
              <dd>
                <ExactArtifactLink
                  projectId={audit.projectId}
                  artifact={baseline.inventory.worklist}
                />
              </dd>
            </div>
            <div>
              <dt>Skills</dt>
              <dd>{baseline.skills.length}</dd>
            </div>
            <div>
              <dt>Runtime configs</dt>
              <dd>{1 + baseline.runtimeConfig.labels.length}</dd>
            </div>
          </dl>
          <div className="audit-gap-block">
            <h4>Inventory gaps</h4>
            <StringList
              values={baseline.inventory.gaps}
              empty="No inventory gaps."
            />
          </div>
        </section>
      )}
      {audit.stopReason === undefined ? null : (
        <section className="notice notice-error audit-stop-reason" role="alert">
          <strong>{audit.stopReason.code}</strong>
          <p>{audit.stopReason.message}</p>
        </section>
      )}
    </div>
  );
}

function useAuditItems(
  audit: Audit,
  api: ReturnType<typeof usePublicAPI>,
  enabled: boolean,
) {
  return useQuery({
    queryKey: queryKeys.audits.items(audit.auditId),
    queryFn: () => listAuditItems(api, audit.auditId),
    enabled,
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
}

function AuditChecks({
  audit,
  api,
}: {
  audit: Audit;
  api: ReturnType<typeof usePublicAPI>;
}) {
  const items = useAuditItems(audit, api, true);
  if (items.isPending) return <p className="loading-copy">Loading checks…</p>;
  if (items.error !== null) return <ErrorNotice error={items.error} />;
  if (items.data.items.length === 0) {
    return (
      <div className="empty-state panel">
        <h3>No checks materialized</h3>
        <p>Start the Audit to pin its worklist.</p>
      </div>
    );
  }
  return (
    <div className="audit-check-list">
      {items.data.items.map((item) => (
        <article className="panel audit-check-card" key={item.itemId}>
          <div className="section-heading">
            <div>
              <p className="eyebrow">
                {item.kind} · #{item.ordinal + 1}
              </p>
              <h3>{item.subjectKey}</h3>
            </div>
            <StateBadge state={item.state} />
          </div>
          <dl className="metadata-grid">
            <div>
              <dt>Item key</dt>
              <dd>
                <code>{item.itemKey}</code>
              </dd>
            </div>
            <div>
              <dt>Workflow role</dt>
              <dd>{item.workflowRole}</dd>
            </div>
            <div>
              <dt>Disposition</dt>
              <dd>{item.finalDisposition ?? "pending"}</dd>
            </div>
            <div>
              <dt>Origin</dt>
              <dd>
                {item.origin.entryKey}
                {item.origin.entryVersion === undefined
                  ? ""
                  : `@${item.origin.entryVersion}`}
              </dd>
            </div>
          </dl>
          <div className="audit-artifact-list">
            <ExactArtifactLink
              projectId={audit.projectId}
              artifact={item.task}
              label="Task package"
            />
            {item.acceptedResult === undefined ? null : (
              <ExactArtifactLink
                projectId={audit.projectId}
                artifact={item.acceptedResult}
                label="Accepted result"
              />
            )}
          </div>
          <h4>Attempts</h4>
          {item.attempts.length === 0 ? (
            <p className="muted-copy">No Run submitted.</p>
          ) : (
            <ol className="audit-attempt-list">
              {item.attempts.map((attempt) => (
                <li key={attempt.executionItemId}>
                  <span>
                    Attempt {attempt.itemAttempt} · {attempt.state}
                  </span>
                  <span>
                    {attempt.terminalOutcome ?? "not terminal"} ·{" "}
                    {attempt.collectionDisposition ?? "not collected"}
                  </span>
                  {attempt.runId === undefined ? (
                    <span>No child Run</span>
                  ) : (
                    <Link to={`/runs/${encodeURIComponent(attempt.runId)}`}>
                      {attempt.runDeleted
                        ? "Deleted Run provenance"
                        : attempt.runId}
                    </Link>
                  )}
                </li>
              ))}
            </ol>
          )}
        </article>
      ))}
    </div>
  );
}

function CoverageRow({ audit, row }: { audit: Audit; row: AuditCoverageRow }) {
  return (
    <tr>
      <td data-label="Check">
        <strong>{row.subjectKey}</strong>
        <small>{row.itemKey}</small>
      </td>
      <td data-label="Assessment">
        <StateBadge state={row.coverage.status} />
      </td>
      <td data-label="Requested">
        <StringList values={row.coverage.requested} />
      </td>
      <td data-label="Completed">
        <StringList values={row.coverage.completed} />
      </td>
      <td data-label="Gaps">
        <StringList values={row.coverage.gaps} empty="none" />
      </td>
      <td data-label="Result">
        {row.result === undefined ? (
          "—"
        ) : (
          <ExactArtifactLink
            projectId={audit.projectId}
            artifact={row.result}
          />
        )}
      </td>
    </tr>
  );
}

function AuditCoverage({
  audit,
  api,
}: {
  audit: Audit;
  api: ReturnType<typeof usePublicAPI>;
}) {
  const coverage = useQuery({
    queryKey: queryKeys.audits.coverage(audit.auditId),
    queryFn: () => listAuditCoverage(api, audit.auditId),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  if (coverage.isPending)
    return <p className="loading-copy">Loading coverage…</p>;
  if (coverage.error !== null) return <ErrorNotice error={coverage.error} />;
  if (coverage.data.items.length === 0) {
    return (
      <div className="empty-state panel">
        <h3>No coverage yet</h3>
        <p>
          This is not a passing result. Checks have not produced an assessment.
        </p>
      </div>
    );
  }
  return (
    <section className="panel audit-section-panel">
      <p className="eyebrow">Assessment, not Run status</p>
      <h3>Coverage matrix</h3>
      <div className="table-scroll">
        <table className="responsive-table audit-coverage-table">
          <thead>
            <tr>
              <th>Check</th>
              <th>Assessment</th>
              <th>Requested</th>
              <th>Completed</th>
              <th>Gaps</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {coverage.data.items.map((row) => (
              <CoverageRow audit={audit} row={row} key={row.itemId} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

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

const FINDING_SEVERITIES: readonly AuditFindingSeverity[] = [
  "informational",
  "low",
  "medium",
  "high",
  "critical",
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

  async function invalidate(): Promise<void> {
    await Promise.all([
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.detail(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.findings(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.reviews(audit.auditId),
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
        throw new Error("The exact finding review is no longer pending");
      }
      const decision: DecideAuditFindingRequest = {
        verdict,
        rationale: rationale.trim(),
        ...(verdict === "true_positive" ? { severity } : {}),
        ...(verdict === "duplicate" ? { duplicateTargetId } : {}),
      };
      const draft = {
        operation: "decide",
        auditId: audit.auditId,
        requestId: pendingReview.requestId,
        revision: pendingReview.revision,
        verdict,
        severity: decision.severity,
        rationale: decision.rationale,
        duplicateTargetId: decision.duplicateTargetId,
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
            ? "Opening exact review…"
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
        Exact review {pendingReview.requestId} · finding revision{" "}
        {pendingReview.subjectRevision}
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
            Canonical finding
            <select
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
          </label>
        ) : null}
      </div>
      <label>
        Analyst rationale
        <textarea
          required
          rows={3}
          value={rationale}
          onChange={(event) => setRationale(event.target.value)}
          placeholder="Record the evidence-based reason for this decision."
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

function AuditFindings({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const findings = useQuery({
    queryKey: queryKeys.audits.findings(audit.auditId),
    queryFn: () => listAuditFindings(api, audit.auditId),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  const reviews = useQuery({
    queryKey: queryKeys.audits.reviews(audit.auditId),
    queryFn: () => listAuditReviews(api, audit.auditId),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  if (findings.isPending || reviews.isPending) {
    return <p className="loading-copy">Loading findings…</p>;
  }
  if (findings.error !== null) return <ErrorNotice error={findings.error} />;
  if (reviews.error !== null) return <ErrorNotice error={reviews.error} />;
  if (findings.data.items.length === 0) {
    return (
      <div className="empty-state panel">
        <h3>No finding candidates</h3>
        <p>A successful Run alone does not create or confirm a finding.</p>
      </div>
    );
  }
  const pending = new Map(
    reviews.data.items
      .filter((request) => request.state === "pending")
      .map((request) => [request.findingId, request]),
  );
  return (
    <div className="audit-finding-list">
      {findings.data.items.map((finding) => (
        <article className="panel audit-finding-card" key={finding.findingId}>
          <div className="section-heading">
            <div>
              <p className="eyebrow">{finding.findingId}</p>
              <h3>{finding.firstProposal.document.title}</h3>
            </div>
            <StateBadge state={finding.state} />
          </div>
          <p>{finding.firstProposal.document.description}</p>
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
                {finding.currentAssessment?.semanticAssessment ??
                  "not accepted"}
              </dd>
            </div>
          </dl>
          {finding.duplicateTargetId === undefined ? null : (
            <p className="muted-copy">
              Duplicate of <code>{finding.duplicateTargetId}</code>
            </p>
          )}
          <FindingProvenanceView audit={audit} finding={finding} />
          <FindingReviewControls
            audit={audit}
            finding={finding}
            findings={findings.data.items}
            {...(pending.has(finding.findingId)
              ? { pendingReview: pending.get(finding.findingId)! }
              : {})}
          />
        </article>
      ))}
    </div>
  );
}

function ActionReviewControls({
  audit,
  review,
}: {
  audit: Audit;
  review: AuditReviewRequest;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [rationale, setRationale] = useState("");
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<Record<string, string | number>>(
        "audit-action-review",
      ),
  );
  async function invalidate(): Promise<void> {
    await Promise.all([
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.detail(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.reviews(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.items(audit.auditId),
      }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.audits.report(audit.auditId),
      }),
    ]);
  }
  const decision = useMutation({
    mutationFn: (action: AuditReviewAction) => {
      const trimmed = rationale.trim();
      const draft = {
        auditId: audit.auditId,
        requestId: review.requestId,
        revision: review.revision,
        action,
        rationale: trimmed,
      };
      return decideAuditAction(api, {
        auditId: audit.auditId,
        requestId: review.requestId,
        expectedRevision: review.revision,
        idempotencyKey: keyring.keyFor(draft),
        action,
        rationale: trimmed,
      });
    },
    onSuccess: async () => {
      setRationale("");
      await invalidate();
    },
    onError: invalidate,
  });
  const unavailable = rationale.trim().length === 0 || decision.isPending;
  return (
    <div className="audit-finding-review-actions">
      <label>
        Rationale
        <textarea
          value={rationale}
          maxLength={65_536}
          rows={3}
          onChange={(event) => setRationale(event.target.value)}
          placeholder="Explain why this exact action or report is accepted or rejected."
        />
      </label>
      <div className="button-row">
        <button
          type="button"
          disabled={unavailable}
          onClick={() => decision.mutate("approve")}
        >
          Approve exact subject
        </button>
        <button
          className="secondary-button"
          type="button"
          disabled={unavailable}
          onClick={() => decision.mutate("reject")}
        >
          Reject
        </button>
      </div>
      {decision.error === null ? null : (
        <AuditMutationNotice error={decision.error} />
      )}
    </div>
  );
}

function AuditReviews({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const reviews = useQuery({
    queryKey: queryKeys.audits.reviews(audit.auditId),
    queryFn: () => listAuditReviews(api, audit.auditId),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  if (reviews.isPending)
    return <p className="loading-copy">Loading reviews…</p>;
  if (reviews.error !== null) return <ErrorNotice error={reviews.error} />;
  return (
    <section className="panel audit-section-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Exact human authority and immutable history</p>
          <h3>Human reviews</h3>
        </div>
        <span>{reviews.data.items.length} recorded</span>
      </div>
      {reviews.data.items.length === 0 ? (
        <p className="muted-copy">No human review has been opened.</p>
      ) : (
        <ol className="audit-review-history">
          {reviews.data.items.map((review) => (
            <li key={review.requestId}>
              <div>
                <strong>{review.findingId ?? review.subjectId}</strong>
                <StateBadge state={review.state} />
              </div>
              <span>
                {review.kind} · subject revision {review.subjectRevision} ·
                requested {formatTimestamp(review.createdAt)}
              </span>
              {review.decision === undefined ? (
                <>
                  <span>No decision recorded.</span>
                  {review.state === "pending" &&
                  review.subjectKind !== "finding" ? (
                    <ActionReviewControls audit={audit} review={review} />
                  ) : null}
                </>
              ) : (
                <>
                  <span>
                    {review.decision.action ?? review.decision.verdict}
                    {review.decision.severity === undefined
                      ? ""
                      : ` · ${review.decision.severity}`}
                    {" · "}
                    {review.decision.actorId}
                  </span>
                  <p>{review.decision.rationale}</p>
                </>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function AuditRuns({
  audit,
  api,
}: {
  audit: Audit;
  api: ReturnType<typeof usePublicAPI>;
}) {
  const items = useAuditItems(audit, api, true);
  const attempts = useMemo(
    () =>
      items.data?.items.flatMap((item) =>
        item.attempts.map((attempt) => ({ item, attempt })),
      ) ?? [],
    [items.data],
  );
  if (items.isPending)
    return <p className="loading-copy">Loading child Runs…</p>;
  if (items.error !== null) return <ErrorNotice error={items.error} />;
  return (
    <section className="panel audit-section-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Ordinary Workflow executions</p>
          <h3>Child Runs</h3>
        </div>
        <Link to="/runs">Global Runs →</Link>
      </div>
      {attempts.length === 0 ? (
        <div className="compact-empty">
          <strong>No child Runs submitted.</strong>
        </div>
      ) : (
        <div className="table-scroll">
          <table className="responsive-table">
            <thead>
              <tr>
                <th>Check</th>
                <th>Attempt</th>
                <th>Run</th>
                <th>Technical outcome</th>
                <th>Collection</th>
              </tr>
            </thead>
            <tbody>
              {attempts.map(({ item, attempt }) => (
                <tr key={attempt.executionItemId}>
                  <td data-label="Check">{item.subjectKey}</td>
                  <td data-label="Attempt">{attempt.itemAttempt}</td>
                  <td data-label="Run">
                    {attempt.runId === undefined ? (
                      "—"
                    ) : (
                      <Link to={`/runs/${encodeURIComponent(attempt.runId)}`}>
                        {attempt.runId}
                      </Link>
                    )}
                  </td>
                  <td data-label="Technical outcome">
                    {attempt.terminalOutcome ?? "running"}
                  </td>
                  <td data-label="Collection">
                    {attempt.collectionDisposition ?? attempt.state}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function AuditReportView({
  audit,
  api,
}: {
  audit: Audit;
  api: ReturnType<typeof usePublicAPI>;
}) {
  const report = useQuery({
    queryKey: queryKeys.audits.report(audit.auditId),
    queryFn: () => getAuditReport(api, audit.auditId),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  if (report.isPending) return <p className="loading-copy">Loading report…</p>;
  if (report.error !== null) return <ErrorNotice error={report.error} />;
  function downloadReport(
    name: string,
    mediaType: string,
    content: string,
  ): void {
    const url = URL.createObjectURL(new Blob([content], { type: mediaType }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section className="panel audit-section-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Server-generated projection</p>
          <h3>Audit report</h3>
        </div>
        <StateBadge state={report.data.status} />
      </div>
      {report.data.status === "pending" ? (
        <p>
          The Audit has not reached report generation. Pending is not a
          successful assessment.
        </p>
      ) : report.data.status === "unavailable" ? (
        <div className="notice notice-error">
          <strong>No accepted report is available.</strong>
          <p>Inspect checks and collection dispositions for explicit gaps.</p>
        </div>
      ) : (
        <>
          {report.data.status === "proposed" ? (
            <div className="notice">
              <strong>This exact report is awaiting owner acceptance.</strong>
              <p>
                Review its frozen contents, then approve or reject it in the{" "}
                <Link
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/reviews`}
                >
                  Reviews section
                </Link>
                .
              </p>
            </div>
          ) : null}
          <div className="audit-artifact-list">
            {report.data.machineArtifact === undefined ? null : (
              <div className="audit-download-row">
                <ExactArtifactLink
                  projectId={audit.projectId}
                  artifact={report.data.machineArtifact}
                  label="Machine report"
                />
                {report.data.machine === undefined ? null : (
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() =>
                      downloadReport(
                        `${audit.auditId}-report.json`,
                        "application/json",
                        JSON.stringify(report.data.machine, null, 2),
                      )
                    }
                  >
                    Download exact JSON
                  </button>
                )}
              </div>
            )}
            {report.data.summaryArtifact === undefined ? null : (
              <div className="audit-download-row">
                <ExactArtifactLink
                  projectId={audit.projectId}
                  artifact={report.data.summaryArtifact}
                  label="Summary report"
                />
                {report.data.summary === undefined ? null : (
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() =>
                      downloadReport(
                        `${audit.auditId}-summary.txt`,
                        "text/plain",
                        report.data.summary!,
                      )
                    }
                  >
                    Download exact summary
                  </button>
                )}
              </div>
            )}
          </div>
          {report.data.summary === undefined ? null : (
            <div className="audit-report-summary">
              <h4>Summary</h4>
              <p>{report.data.summary}</p>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function AuditSectionContent({
  audit,
  section,
  api,
}: {
  audit: Audit;
  section: AuditSection;
  api: ReturnType<typeof usePublicAPI>;
}) {
  switch (section) {
    case "overview":
      return <AuditOverview audit={audit} />;
    case "coverage":
      return <AuditCoverage audit={audit} api={api} />;
    case "findings":
      return <AuditFindings audit={audit} />;
    case "checks":
      return <AuditChecks audit={audit} api={api} />;
    case "reviews":
      return <AuditReviews audit={audit} />;
    case "runs":
      return <AuditRuns audit={audit} api={api} />;
    case "report":
      return <AuditReportView audit={audit} api={api} />;
  }
}

export function ProjectAuditDetailRoute() {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { projectId = "", auditId = "", section: rawSection } = useParams();
  const validRoute =
    PROJECT_ID_PATTERN.test(projectId) && AUDIT_ID_PATTERN.test(auditId);
  const section = SECTIONS.some((candidate) => candidate.id === rawSection)
    ? (rawSection as AuditSection)
    : "overview";
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validRoute,
  });
  const audit = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    queryFn: () => getAudit(api, auditId),
    enabled: validRoute,
    refetchInterval: (query) =>
      query.state.data !== undefined &&
      auditNeedsPolling(query.state.data.state)
        ? 1_000
        : false,
    refetchOnReconnect: true,
    refetchOnWindowFocus: true,
    retry: (count, error) =>
      !(error instanceof PublicAPIError && error.status === 404) && count < 2,
  });
  useEffect(() => {
    if (audit.error instanceof PublicAPIError && audit.error.status === 404) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.projects.audits.all(projectId),
      });
      void navigate(`/projects/${encodeURIComponent(projectId)}/audits`, {
        replace: true,
      });
    }
  }, [audit.error, navigate, projectId, queryClient]);

  let content: ReactNode;
  if (!validRoute)
    content = (
      <ErrorNotice error={new Error("Project Audit route is invalid")} />
    );
  else if (audit.isPending)
    content = <p className="loading-copy">Loading Audit…</p>;
  else if (audit.error !== null) content = <ErrorNotice error={audit.error} />;
  else if (audit.data.projectId !== projectId)
    content = (
      <ErrorNotice error={new Error("Audit does not belong to this Project")} />
    );
  else
    content = (
      <AuditSectionContent audit={audit.data} section={section} api={api} />
    );

  return (
    <section className="route-page audit-page audit-detail-page">
      <header className="route-header-row">
        <div>
          <Link
            className="back-link"
            to={`/projects/${encodeURIComponent(projectId)}/audits`}
          >
            ← {project.data?.name ?? "Project"} Audits
          </Link>
          <p className="eyebrow">Authoritative Audit execution</p>
          <h2>{audit.data?.auditId ?? auditId}</h2>
          <p className="lede">
            Run outcome, collected result and security assessment remain
            separate throughout this view.
          </p>
        </div>
        {audit.data === undefined ? null : (
          <div className="audit-header-state">
            <StateBadge state={audit.data.state} />
            <span>revision {audit.data.revision}</span>
          </div>
        )}
      </header>
      {project.error === null ? null : <ErrorNotice error={project.error} />}
      {audit.data === undefined ? null : <AuditControls audit={audit.data} />}
      <nav className="audit-section-navigation" aria-label="Audit sections">
        {SECTIONS.map((candidate) => {
          const target =
            candidate.id === "overview"
              ? `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}`
              : `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}/${candidate.id}`;
          return (
            <Link
              key={candidate.id}
              className={section === candidate.id ? "active" : ""}
              aria-current={section === candidate.id ? "page" : undefined}
              to={target}
            >
              {candidate.label}
            </Link>
          );
        })}
      </nav>
      {content}
    </section>
  );
}
