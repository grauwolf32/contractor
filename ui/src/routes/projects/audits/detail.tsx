import { MobileSectionPicker } from "../../../app/mobile-section-picker";
import { useAuditCoverage } from "./coverage-data";
import { useAuditProjectionRefresh } from "./projection-refresh";
import { auditCheckTitle } from "./check-title";
import { AuditQueueError, AuditQueuePage } from "./queue";
import { useAuditQueue } from "./queue-state";
import { AuditProgress } from "./progress";
import { ContextLink, ReturnLink } from "../../../app/context-navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  Link,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router";

import {
  AUDIT_ID_PATTERN,
  auditNeedsPolling,
  createAuditFindingReview,
  decideAuditAction,
  decideAuditFinding,
  getAudit,
  getAuditReport,
  getAuditFinding,
  getAuditReview,
  type AuditFindingState,
  listAuditFindingProvenance,
  listAuditFindings,
  listAuditItems,
  listAuditReviews,
  type Audit,
  type AuditAnalystVerdict,
  type AuditFinding,
  type AuditFindingSeverity,
  type AuditReviewAction,
  type AuditReviewRequest,
  type DecideAuditFindingRequest,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { collectAuditPages } from "../../../api/audit-collections";
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
import { AuditAnchor, AuditMarkdown } from "./shared";
import { auditProfileLabel } from "./labels";

import { AuditControls, AuditMutationNotice } from "./controls";
import { AuditCoverage } from "./coverage";

import "./styles.css";

const MarkdownArtifactPreview = lazy(
  () => import("../../artifacts/previews/markdown"),
);

type AuditSection =
  | "overview"
  | "coverage"
  | "findings"
  | "checks"
  | "reviews"
  | "runs"
  | "report";
type AuditExactArtifact = Audit["inputs"][string];

function compactDigest(digest: string): string {
  return digest.length <= 28
    ? digest
    : `${digest.slice(0, 15)}…${digest.slice(-8)}`;
}

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
    <ContextLink
      returnLabel="Audit"
      className="artifact-ref-link"
      to={exactArtifactLink(projectId, artifact)}
      title={artifact.digest}
    >
      {content}
    </ContextLink>
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

function AuditOverview({ audit }: { audit: Audit }) {
  const baseline = audit.baseline;
  return (
    <div className="audit-detail-stack">
      {audit.stopReason === undefined ||
      audit.stopReason.code === "deadline_exhausted" ? null : (
        <section className="notice notice-error audit-stop-reason" role="alert">
          <strong>{audit.stopReason.code}</strong>
          <p>{audit.stopReason.message}</p>
        </section>
      )}
      <AuditProgress audit={audit} />
      <details
        className="panel audit-section-panel audit-setup-details"
        open={audit.state === "draft"}
      >
        <summary>Profile and scope</summary>
        <div className="section-heading">
          <div>
            <p className="eyebrow">Audit setup</p>
            <h3>Exact setup</h3>
          </div>
          <Link
            className="audit-open-link"
            to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/coverage`}
          >
            View checks & results →
          </Link>
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
      </details>
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
                ? audit.state === "draft"
                  ? "Set when starting"
                  : "No time limit"
                : audit.stopReason?.code === "deadline_exhausted"
                  ? "Time limit reached"
                  : audit.state === "paused"
                    ? "Paused — remaining time is saved"
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
        <details className="panel audit-section-panel audit-setup-details">
          <summary>Baseline and exact standards</summary>
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
              <dt>Standards</dt>
              <dd>{baseline.standards.length}</dd>
            </div>
            <div>
              <dt>Runtime configs</dt>
              <dd>{1 + baseline.runtimeConfig.labels.length}</dd>
            </div>
          </dl>
          {baseline.standards.length === 0 ? null : (
            <div
              className="audit-gap-block"
              data-testid="audit-baseline-standards"
            >
              <h4>Exact standards</h4>
              <ul className="audit-string-list">
                {baseline.standards.map((standard) => (
                  <li
                    key={`${standard.reference.scheme}@${standard.reference.version}`}
                  >
                    <strong>{standard.title}</strong>{" "}
                    <code>
                      {standard.reference.scheme}@{standard.reference.version}
                    </code>{" "}
                    · <code>{compactDigest(standard.retained.digest)}</code> ·{" "}
                    <a
                      href={standard.source.url}
                      rel="noreferrer"
                      target="_blank"
                    >
                      source
                    </a>{" "}
                    · {standard.license.id}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {baseline.inventory.standardSelection === undefined ? null : (
            <div
              className="audit-gap-block"
              data-testid="audit-baseline-standard-selection"
            >
              <h4>Selected denominator</h4>
              <p>{baseline.inventory.standardSelection.scope}</p>
              <p>
                Level {baseline.inventory.standardSelection.levels.join(", ")} ·{" "}
                {baseline.inventory.standardSelection.entryIds.length} exact
                requirements
              </p>
              <StringList
                values={baseline.inventory.standardSelection.entryIds}
              />
            </div>
          )}
          <div className="audit-gap-block">
            <h4>Inventory gaps</h4>
            <StringList
              values={baseline.inventory.gaps}
              empty="No inventory gaps."
            />
          </div>
        </details>
      )}
    </div>
  );
}

function useAuditItems(
  audit: Audit,
  api: ReturnType<typeof usePublicAPI>,
  enabled: boolean,
) {
  const queryKey = queryKeys.audits.allItems(audit.auditId);
  const query = useQuery({
    queryKey,
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditItems(
          api,
          audit.auditId,
          cursor === undefined ? {} : { cursor },
        ),
      ),
    enabled,
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  useAuditProjectionRefresh(audit, queryKey, enabled);
  return query;
}

function AuditChecks({
  audit,
  api,
}: {
  audit: Audit;
  api: ReturnType<typeof usePublicAPI>;
}) {
  const items = useAuditItems(audit, api, true);
  const { hash } = useLocation();
  const coverage = useAuditCoverage(audit);
  if (items.isPending) return <p className="loading-copy">Loading checks…</p>;
  if (items.error !== null) return <ErrorNotice error={items.error} />;
  if (items.data.length === 0) {
    return (
      <div className="empty-state panel">
        <h3>No checks materialized</h3>
        <p>Start the Audit to pin its worklist.</p>
      </div>
    );
  }
  return (
    <div className="audit-check-list">
      <AuditAnchor />
      {items.data.map((item) => (
        <article
          className="panel audit-check-card"
          key={item.itemId}
          id={`check-${item.itemId}`}
        >
          <div className="section-heading">
            <div>
              <p className="eyebrow">
                {item.kind} · #{item.ordinal + 1}
              </p>
              <h3 title={item.subjectKey}>
                {auditCheckTitle(
                  coverage.data?.find((row) => row.itemId === item.itemId) ??
                    item,
                )}
              </h3>
            </div>
            <StateBadge state={item.state} />
          </div>
          <p className="muted-copy">
            {item.workflowRole} ·{" "}
            {item.finalDisposition ?? "Pending assessment"} ·{" "}
            {item.attempts.length} attempts
          </p>
          <details
            className="audit-record-details"
            open={hash === `#check-${item.itemId}`}
          >
            <summary>Attempts, artifacts and exact identity</summary>
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
              {item.origin.standard === undefined ? null : (
                <div>
                  <dt>Causal standard mapping</dt>
                  <dd>
                    <code>
                      {item.origin.standard.scheme}@
                      {item.origin.standard.version}/
                      {item.origin.standard.mappingKey}
                    </code>
                  </dd>
                </div>
              )}
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
                      <ContextLink
                        returnLabel="Audit"
                        to={`/runs/${encodeURIComponent(attempt.runId)}`}
                      >
                        {attempt.runDeleted
                          ? "Deleted Run provenance"
                          : attempt.runId}
                      </ContextLink>
                    )}
                  </li>
                ))}
              </ol>
            )}
          </details>
        </article>
      ))}
    </div>
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
        queryKey: queryKeys.projects.audits.all(audit.projectId),
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
            Canonical finding
            <select
              aria-label="Canonical finding"
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
            <span>
              Or enter an exact finding ID from another page in this Audit
            </span>
            <input
              aria-label="Canonical finding ID"
              value={duplicateTargetId}
              onChange={(event) => setDuplicateTargetId(event.target.value)}
            />
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

function FindingInQueue({
  audit,
  finding,
  siblings,
}: {
  audit: Audit;
  finding: AuditFinding;
  siblings: AuditFinding[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const reviews = useQuery({
    queryKey: [
      ...queryKeys.audits.reviews(audit.auditId, finding.findingId),
      "pending",
    ],
    queryFn: () =>
      listAuditReviews(api, audit.auditId, {
        finding: finding.findingId,
        state: "pending",
      }),
  });
  const pending = reviews.data?.items.find(
    (review) => review.state === "pending",
  );
  return (
    <AuditFindingCard
      audit={audit}
      finding={finding}
      findings={siblings}
      {...(pending === undefined ? {} : { pendingReview: pending })}
      reviewLoading={reviews.isPending}
      reviewError={
        reviews.error ??
        (pending !== undefined && pending.subjectRevision !== finding.revision
          ? new Error(
              "Finding evidence changed. Refresh the context before deciding.",
            )
          : null)
      }
      onRetryReview={() =>
        void queryClient.invalidateQueries({
          queryKey: queryKeys.audits.detail(audit.auditId),
        })
      }
    />
  );
}

function ExactFinding({
  audit,
  findingId,
  reviewId,
}: {
  audit: Audit;
  findingId: string;
  reviewId: string | null;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const finding = useQuery({
    queryKey: [
      ...queryKeys.audits.detail(audit.auditId),
      "findings",
      findingId,
      "exact",
    ],
    queryFn: () => getAuditFinding(api, audit.auditId, findingId),
  });
  const review = useQuery({
    queryKey: [...queryKeys.audits.detail(audit.auditId), "reviews", reviewId],
    queryFn: () => getAuditReview(api, audit.auditId, reviewId!),
    enabled: reviewId !== null,
  });
  const refresh = () =>
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
    });
  if (finding.error !== null || review.error !== null)
    return (
      <AuditQueueError
        error={(finding.error ?? review.error)!}
        onRefresh={refresh}
      />
    );
  if (finding.isPending || (reviewId !== null && review.isPending))
    return <p role="status">Loading exact finding and review…</p>;
  const stale =
    review.data !== undefined &&
    (review.data.subjectKind !== "finding" ||
      review.data.findingId !== findingId ||
      (review.data.state === "pending" &&
        review.data.subjectRevision !== finding.data.revision));
  return (
    <div className="audit-finding-list">
      <AuditAnchor />
      {stale ? (
        <div className="notice notice-error">
          The requested review no longer matches this finding revision. Refresh
          the context before making a decision.
          <button type="button" className="secondary-button" onClick={refresh}>
            Refresh context
          </button>
        </div>
      ) : null}
      {stale || review.data?.state === "pending" ? (
        <AuditFindingCard
          audit={audit}
          finding={finding.data}
          findings={[finding.data]}
          {...(!stale && review.data?.state === "pending"
            ? { pendingReview: review.data }
            : {})}
          reviewError={
            stale ? new Error("Review subject revision changed") : null
          }
        />
      ) : null}
      {!stale && review.data?.state !== "pending" ? (
        <FindingInQueue
          audit={audit}
          finding={finding.data}
          siblings={[finding.data]}
        />
      ) : null}
    </div>
  );
}

function AuditFindings({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const queue = useAuditQueue();
  const state = queue.params.get("state") ?? "";
  const verdict = queue.params.get("verdict") ?? "";
  const severity = queue.params.get("severity") ?? "";
  const exactId = queue.params.get("finding");
  const findings = useQuery({
    queryKey: [
      ...queryKeys.audits.allFindings(audit.auditId),
      queue.params.toString(),
    ],
    queryFn: () =>
      listAuditFindings(api, audit.auditId, {
        ...queue.request,
        ...([
          "proposed",
          "confirmed",
          "rejected",
          "duplicate",
          "needs-evidence",
        ].includes(state)
          ? { state: state as AuditFindingState }
          : {}),
        ...(["unreviewed", "true_positive", "false_positive"].includes(verdict)
          ? {
              verdict: verdict as
                "unreviewed" | "true_positive" | "false_positive",
            }
          : {}),
        ...(FINDING_SEVERITIES.includes(severity as AuditFindingSeverity)
          ? { severity: severity as AuditFindingSeverity }
          : {}),
      }),
    enabled: exactId === null,
  });
  const refresh = () => {
    queue.refresh();
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
    });
  };
  if (exactId !== null)
    return (
      <ExactFinding
        audit={audit}
        findingId={exactId}
        reviewId={queue.params.get("review")}
      />
    );
  return (
    <section className="audit-finding-list">
      <AuditAnchor />
      <div className="section-heading">
        <h3>Findings in this audit</h3>
        <ContextLink
          returnLabel="Audit findings"
          to={`/projects/${encodeURIComponent(audit.projectId)}/findings`}
        >
          View all project findings →
        </ContextLink>
      </div>
      <div className="audit-review-fields">
        <label>
          Finding disposition
          <select
            value={state}
            onChange={(event) => queue.change("state", event.target.value)}
          >
            <option value="">All dispositions</option>
            {[
              "proposed",
              "confirmed",
              "rejected",
              "duplicate",
              "needs-evidence",
            ].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          Analyst verdict
          <select
            value={verdict}
            onChange={(event) => queue.change("verdict", event.target.value)}
          >
            <option value="">All verdicts</option>
            <option value="unreviewed">Unreviewed</option>
            <option value="true_positive">True positive</option>
            <option value="false_positive">False positive</option>
          </select>
        </label>
        <label>
          Analyst severity
          <select
            value={severity}
            onChange={(event) => queue.change("severity", event.target.value)}
          >
            <option value="">All severities</option>
            {FINDING_SEVERITIES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="muted-copy">
        Severity filters apply to analyst decisions. Model proposals remain
        separate from accepted findings.
      </p>
      {findings.isPending ? (
        <p role="status">Loading findings…</p>
      ) : findings.error !== null ? (
        <AuditQueueError error={findings.error} onRefresh={refresh} />
      ) : (
        <>
          <AuditQueuePage
            page={findings.data}
            currentRevision={audit.revision}
            queue={queue}
            onRefresh={refresh}
          />
          {findings.data.items.length === 0 ? (
            <div className="empty-state panel">
              <h3>No matching finding candidates</h3>
              <p>
                A successful Run alone does not create or confirm a finding.
              </p>
            </div>
          ) : (
            findings.data.items.map((finding) => (
              <FindingInQueue
                key={finding.findingId}
                audit={audit}
                finding={finding}
                siblings={findings.data.items}
              />
            ))
          )}
        </>
      )}
    </section>
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
      <a
        className="audit-decision-jump"
        href={`#decision-${audit.auditId}-${finding.findingId}`}
      >
        Review decision ↓
      </a>
      <div className="audit-finding-description">
        <AuditMarkdown source={finding.firstProposal.document.description} />
      </div>
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
                  title={`Exact revision ${artifact.ref.revision}`}
                >
                  {artifact.ref.namespace}/{artifact.ref.name}
                </ContextLink>
                <small>
                  {artifact.sizeBytes === undefined
                    ? null
                    : `${formatBytes(artifact.sizeBytes)} · `}
                  Exact source revision
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
            <dd>{finding.firstProposal.document.subject.key}</dd>
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
    </article>
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
          placeholder="Explain your decision. Markdown is supported."
        />
      </label>
      <div className="button-row">
        {review.requestedActions.includes("approve") ? (
          <button
            type="button"
            disabled={unavailable}
            onClick={() => decision.mutate("approve")}
          >
            Approve exact subject
          </button>
        ) : null}
        {review.requestedActions.includes("reject") ? (
          <button
            className="secondary-button"
            type="button"
            disabled={unavailable}
            onClick={() => decision.mutate("reject")}
          >
            Reject
          </button>
        ) : null}
        {review.requestedActions.includes("not_applicable") ? (
          <button
            className="secondary-button"
            type="button"
            disabled={unavailable}
            onClick={() => decision.mutate("not_applicable")}
          >
            Mark not applicable
          </button>
        ) : null}
      </div>
      {decision.error === null ? null : (
        <AuditMutationNotice error={decision.error} />
      )}
    </div>
  );
}

function AuditReviews({ audit }: { audit: Audit }) {
  const queue = useAuditQueue();
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const state = queue.params.get("state") ?? "";
  const pendingOnly = state === "pending";
  const reviews = useQuery({
    queryKey: [
      ...queryKeys.audits.allReviews(audit.auditId),
      queue.params.toString(),
    ],
    queryFn: () =>
      listAuditReviews(api, audit.auditId, {
        ...queue.request,
        ...(["pending", "decided", "expired"].includes(state)
          ? { state: state as AuditReviewRequest["state"] }
          : {}),
      }),
  });
  const refresh = () => {
    queue.refresh();
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
    });
  };
  const items = useQuery({
    queryKey: queryKeys.audits.allItems(audit.auditId),
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditItems(
          api,
          audit.auditId,
          cursor === undefined ? {} : { cursor },
        ),
      ),
    enabled:
      reviews.data?.items.some(
        (review) => review.subjectKind === "audit-item-action",
      ) ?? false,
    refetchOnReconnect: true,
  });
  const subjects = new Map(
    items.data?.map((item) => [item.itemId, item.subjectKey]),
  );
  const reviewLabels: Record<string, string> = {
    "requirement-applicability": "Requirement applicability",
    "active-check-approval": "Active check approval",
    "finding-triage": "Finding review",
    "report-acceptance": "Report acceptance",
  };
  if (reviews.isPending)
    return <p className="loading-copy">Loading reviews…</p>;
  if (reviews.error !== null)
    return <AuditQueueError error={reviews.error} onRefresh={refresh} />;
  const visibleReviews = reviews.data.items;
  return (
    <section className="panel audit-section-panel">
      <AuditAnchor />
      <div className="section-heading">
        <div>
          <p className="eyebrow">Decisions and review history</p>
          <h3>Human reviews</h3>
        </div>
      </div>
      <label className="audit-review-filter">
        Review state
        <select
          value={state || "all"}
          onChange={(event) => queue.change("state", event.target.value)}
        >
          <option value="all">All reviews</option>
          <option value="pending">Pending decisions</option>
          <option value="decided">Decided</option>
          <option value="expired">Expired</option>
        </select>
      </label>
      <AuditQueuePage
        page={reviews.data}
        currentRevision={audit.revision}
        queue={queue}
        onRefresh={refresh}
      />
      {items.error === null ? null : (
        <div className="notice">
          <p>
            Check names could not be loaded. Review identifiers are shown below.
          </p>
          <button
            className="secondary-button"
            type="button"
            onClick={() => void items.refetch()}
          >
            Retry check details
          </button>
        </div>
      )}
      {visibleReviews.length === 0 ? (
        <p className="muted-copy">
          {pendingOnly
            ? "No pending decisions. Choose All reviews to inspect previous decisions."
            : "No human review has been opened."}
        </p>
      ) : (
        <ol className="audit-review-history">
          {visibleReviews.map((review) => (
            <li
              className="audit-review-card"
              key={review.requestId}
              id={`review-${review.requestId}`}
            >
              <div className="audit-review-heading">
                <div>
                  <p className="eyebrow">
                    {reviewLabels[review.kind] ??
                      review.kind.replaceAll("-", " ")}
                  </p>
                  <h4>
                    {subjects.get(review.subjectId) ??
                      (review.subjectKind === "finding"
                        ? "Finding assessment"
                        : review.subjectKind === "audit-report"
                          ? "Audit report"
                          : "Review requested")}
                  </h4>
                </div>
                <StateBadge state={review.state} />
              </div>
              <p className="audit-review-meta">
                Requested {formatTimestamp(review.createdAt)}
              </p>
              {subjects.has(review.subjectId) ? (
                <ContextLink
                  returnLabel="Audit reviews"
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/checks#check-${encodeURIComponent(review.subjectId)}`}
                >
                  View check →
                </ContextLink>
              ) : null}
              {review.subjectKind === "finding" ? (
                <ContextLink
                  returnLabel="Audit reviews"
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/findings?finding=${encodeURIComponent(review.findingId ?? review.subjectId)}&review=${encodeURIComponent(review.requestId)}`}
                >
                  Review finding →
                </ContextLink>
              ) : null}
              {review.subjectKind === "audit-report" ? (
                <ContextLink
                  returnLabel="Audit reviews"
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/report?review=${encodeURIComponent(review.requestId)}`}
                >
                  Review report →
                </ContextLink>
              ) : null}
              <details className="audit-record-details">
                <summary>Review details</summary>
                <dl className="metadata-grid">
                  <div>
                    <dt>Type</dt>
                    <dd>{review.kind}</dd>
                  </div>
                  <div>
                    <dt>Subject revision</dt>
                    <dd>{review.subjectRevision}</dd>
                  </div>
                  <div>
                    <dt>Review ID</dt>
                    <dd>
                      <code>{review.requestId}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Subject ID</dt>
                    <dd>
                      <code>{review.findingId ?? review.subjectId}</code>
                    </dd>
                  </div>
                </dl>
              </details>
              {review.decision === undefined ? (
                <>
                  {review.state === "pending" ? null : (
                    <p className="muted-copy">No decision recorded.</p>
                  )}
                  {review.state === "pending" &&
                  review.subjectKind === "audit-item-action" ? (
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
                  <AuditMarkdown source={review.decision.rationale} />
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
  const coverage = useAuditCoverage(audit);
  const attempts = useMemo(
    () =>
      items.data?.flatMap((item) =>
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
                  <td data-label="Check">
                    <ContextLink
                      returnLabel="Audit Runs"
                      to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/checks#check-${encodeURIComponent(item.itemId)}`}
                      title={item.subjectKey}
                    >
                      {auditCheckTitle(
                        coverage.data?.find(
                          (row) => row.itemId === item.itemId,
                        ) ?? item,
                      )}
                    </ContextLink>
                  </td>
                  <td data-label="Attempt">{attempt.itemAttempt}</td>
                  <td data-label="Run">
                    {attempt.runId === undefined ? (
                      "—"
                    ) : (
                      <ContextLink
                        returnLabel="Audit"
                        to={`/runs/${encodeURIComponent(attempt.runId)}`}
                        title={attempt.runId}
                        aria-label={attempt.runId}
                      >
                        {attempt.runId.length > 24
                          ? `${attempt.runId.slice(0, 12)}…${attempt.runId.slice(-8)}`
                          : attempt.runId}
                      </ContextLink>
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
  const [params] = useSearchParams();
  const requestedReview = params.get("review");
  const queryKey = queryKeys.audits.report(audit.auditId);
  const report = useQuery({
    queryKey,
    queryFn: () => getAuditReport(api, audit.auditId),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  useAuditProjectionRefresh(audit, queryKey);
  if (report.isPending) return <p className="loading-copy">Loading report…</p>;
  if (report.error !== null)
    return (
      <AuditQueueError
        error={report.error}
        onRefresh={() => void report.refetch()}
      />
    );
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
    <section className="panel audit-section-panel audit-report-view">
      <div className="section-heading">
        <div>
          <h3>Audit report</h3>
        </div>
        <StateBadge state={report.data.status} />
      </div>
      <p className="muted-copy">
        Read the conclusion and its limitations. Full coverage and accepted
        findings are separate from execution status.
      </p>
      <Link
        to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/coverage?result=uncertain`}
      >
        Review coverage gaps →
      </Link>
      {requestedReview !== null &&
      report.data.review?.requestId !== requestedReview ? (
        <div className="notice notice-error" role="status">
          The requested report review is unavailable or no longer current. This
          report cannot be used to decide that review.
        </div>
      ) : null}
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
                Review the exact frozen contents below before making a decision.
              </p>
            </div>
          ) : null}
          {report.data.summary === undefined ? null : (
            <div className="audit-report-summary">
              <h4>Summary</h4>
              {report.data.summaryArtifact?.mediaType === "text/markdown" ? (
                <Suspense fallback={<p>Loading Markdown preview…</p>}>
                  <MarkdownArtifactPreview source={report.data.summary} />
                </Suspense>
              ) : (
                <p style={{ whiteSpace: "pre-wrap" }}>{report.data.summary}</p>
              )}
            </div>
          )}
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
                        `${audit.auditId}-report.${report.data.summaryArtifact?.mediaType === "text/markdown" ? "md" : "txt"}`,
                        report.data.summaryArtifact?.mediaType ?? "text/plain",
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
        </>
      )}
      {report.data.status === "proposed" &&
      report.data.review?.state === "pending" &&
      (requestedReview === null ||
        requestedReview === report.data.review.requestId) ? (
        <ActionReviewControls audit={audit} review={report.data.review} />
      ) : null}
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
      return <AuditCoverage audit={audit} />;
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

function AuditIdentity({ audit }: { audit: Audit }) {
  const [copyStatus, setCopyStatus] = useState("");
  return (
    <div className="audit-identity">
      <time dateTime={audit.createdAt}>{formatTimestamp(audit.createdAt)}</time>
      <code>{audit.auditId}</code>
      <button
        type="button"
        className="secondary-button"
        onClick={() => {
          void navigator.clipboard.writeText(audit.auditId).then(
            () => setCopyStatus("Copied Audit ID"),
            () =>
              setCopyStatus(
                "Copy unavailable. Select the Audit ID to copy it manually.",
              ),
          );
        }}
      >
        Copy Audit ID
      </button>
      <span role="status">{copyStatus}</span>
    </div>
  );
}

export function ProjectAuditDetailRoute() {
  const location = useLocation();
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
          <ReturnLink
            to={`/projects/${encodeURIComponent(projectId)}/audits`}
            label={`${project.data?.name ?? "Project"} Audits`}
          />
          <p className="eyebrow">{project.data?.name ?? "Audit"}</p>
          <h2>
            {audit.data === undefined ? "Audit" : auditProfileLabel(audit.data)}
          </h2>
          {audit.data === undefined ? (
            <code>{auditId}</code>
          ) : (
            <AuditIdentity audit={audit.data} />
          )}
        </div>
        {audit.data === undefined ? null : (
          <div className="audit-header-state">
            <StateBadge state={audit.data.state} />
            <span>revision {audit.data.revision}</span>
          </div>
        )}
      </header>
      {project.error === null ? null : <ErrorNotice error={project.error} />}
      {audit.data === undefined ? null : (
        <AuditControls audit={audit.data} projectName={project.data?.name} />
      )}
      <nav
        className="audit-section-navigation section-navigation"
        aria-label="Audit sections"
      >
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
              state={location.state}
            >
              {candidate.label}
            </Link>
          );
        })}
      </nav>
      <MobileSectionPicker
        label="Audit section"
        value={
          section === "overview"
            ? `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}`
            : `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}/${section}`
        }
        options={SECTIONS.map((candidate) => ({
          to: `/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(auditId)}${candidate.id === "overview" ? "" : `/${candidate.id}`}`,
          label: candidate.label,
        }))}
        state={location.state}
      />
      {content}
    </section>
  );
}
