import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  lazy,
  Suspense,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
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
import { collectAuditPages } from "../../../api/audit-collections";
import { PublicAPIError } from "../../../api/error";
import { getProject, PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { Dialog } from "../../../app/dialog";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import {
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../../artifacts/common";
import { StateBadge } from "../../runs/components";
import { AuditAnchor, AuditMarkdown } from "./shared";
import { auditProfileLabel } from "./labels";

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

type DestructiveAuditAction = Extract<AuditMutationAction, "cancel" | "delete">;

function auditActionAllowed(
  audit: Audit,
  action: DestructiveAuditAction,
): boolean {
  if (action === "cancel") {
    return (
      audit.state === "active" ||
      audit.state === "waiting_review" ||
      audit.state === "paused" ||
      audit.state === "finalizing"
    );
  }
  return (
    audit.state === "draft" ||
    audit.state === "completed" ||
    audit.state === "cancelled" ||
    audit.state === "failed"
  );
}

function AuditMutationDialog({
  action,
  audit,
  projectName,
  error,
  pending,
  onClose,
  onConfirm,
}: {
  action: DestructiveAuditAction;
  audit: Audit;
  projectName: string | undefined;
  error: unknown;
  pending: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const heading = useId();
  const description = useId();
  const safeAction = useRef<HTMLButtonElement>(null);
  const allowed = auditActionAllowed(audit, action);
  const cancelling = action === "cancel";
  return (
    <Dialog
      className="project-dialog panel audit-mutation-dialog"
      labelledBy={heading}
      describedBy={description}
      initialFocusRef={safeAction}
      onRequestClose={onClose}
      role="alertdialog"
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Destructive Audit action</p>
          <h2 id={heading}>
            {cancelling ? "Cancel this Audit?" : "Delete this Audit?"}
          </h2>
        </div>
        <button
          className="project-dialog-close"
          type="button"
          aria-label="Close Audit confirmation"
          disabled={pending}
          onClick={onClose}
        >
          ×
        </button>
      </div>
      <p id={description}>
        {cancelling
          ? "Cancellation closes dispatch and begins bounded cancellation, collection and release of child Runs. The Audit may remain cancelling while that cleanup finishes; retained evidence is not deleted."
          : "Deletion is asynchronous. The Server closes dispatch, drains and collects owned Runs, releases retained evidence, and then purges Audit-managed artifacts and records."}
      </p>
      <dl className="metadata-grid audit-mutation-identity">
        <div>
          <dt>Project</dt>
          <dd>
            {projectName ?? audit.projectId} <code>{audit.projectId}</code>
          </dd>
        </div>
        <div>
          <dt>Audit</dt>
          <dd>
            <code>{audit.auditId}</code>
          </dd>
        </div>
        <div>
          <dt>Profile</dt>
          <dd>
            <code>
              {audit.profile.name}@{audit.profile.version}
            </code>
          </dd>
        </div>
        <div>
          <dt>Current state</dt>
          <dd>
            {audit.state} · revision {audit.revision}
          </dd>
        </div>
      </dl>
      {!allowed ? (
        <div className="notice notice-warning" role="status">
          <strong>This action is no longer available.</strong>
          <p>
            Authoritative state is now <code>{audit.state}</code>. Close this
            confirmation and review the refreshed Audit.
          </p>
        </div>
      ) : null}
      {error === null ? null : <AuditMutationNotice error={error} />}
      <div className="inline-actions audit-mutation-actions">
        <button
          ref={safeAction}
          type="button"
          className="secondary-button"
          disabled={pending}
          onClick={onClose}
        >
          Keep Audit unchanged
        </button>
        <button
          type="button"
          className="danger-button"
          disabled={pending || !allowed}
          onClick={onConfirm}
        >
          {pending
            ? cancelling
              ? "Cancelling…"
              : "Starting deletion…"
            : cancelling
              ? "Confirm cancellation"
              : "Begin Audit deletion"}
        </button>
      </div>
    </Dialog>
  );
}

function AuditControls({
  audit,
  projectName,
}: {
  audit: Audit;
  projectName: string | undefined;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [confirmation, setConfirmation] = useState<DestructiveAuditAction>();
  const destructiveRequestInFlight = useRef(false);
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
      setConfirmation(undefined);
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
    onSettled: () => {
      destructiveRequestInFlight.current = false;
    },
  });
  function closeConfirmation(): void {
    if (mutation.isPending) return;
    mutation.reset();
    setConfirmation(undefined);
  }
  function confirmDestructiveAction(): void {
    if (
      confirmation === undefined ||
      mutation.isPending ||
      destructiveRequestInFlight.current ||
      !auditActionAllowed(audit, confirmation)
    ) {
      return;
    }
    destructiveRequestInFlight.current = true;
    mutation.reset();
    mutation.mutate(confirmation);
  }
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
          onClick={() => {
            if (button.dangerous) {
              mutation.reset();
              setConfirmation(button.action as DestructiveAuditAction);
            } else {
              mutation.mutate(button.action);
            }
          }}
        >
          {mutation.isPending && mutation.variables === button.action
            ? `${button.label}…`
            : button.label}
        </button>
      ))}
      {confirmation === undefined && mutation.error !== null ? (
        <div className="audit-control-error">
          <AuditMutationNotice error={mutation.error} />
        </div>
      ) : null}
      {confirmation === undefined ? null : (
        <AuditMutationDialog
          action={confirmation}
          audit={audit}
          projectName={projectName}
          error={mutation.error}
          pending={mutation.isPending}
          onClose={closeConfirmation}
          onConfirm={confirmDestructiveAction}
        />
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
    queryKey: queryKeys.audits.allItems(audit.auditId),
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
            {item.origin.standard === undefined ? null : (
              <div>
                <dt>Causal standard mapping</dt>
                <dd>
                  <code>
                    {item.origin.standard.scheme}@{item.origin.standard.version}
                    /{item.origin.standard.mappingKey}
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

function AuditFindings({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const findings = useQuery({
    queryKey: queryKeys.audits.allFindings(audit.auditId),
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditFindings(
          api,
          audit.auditId,
          cursor === undefined ? {} : { cursor },
        ),
      ),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  const reviews = useQuery({
    queryKey: queryKeys.audits.allReviews(audit.auditId),
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditReviews(
          api,
          audit.auditId,
          cursor === undefined ? {} : { cursor },
        ),
      ),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  if (findings.isPending || reviews.isPending) {
    return <p className="loading-copy">Loading findings…</p>;
  }
  if (findings.error !== null) return <ErrorNotice error={findings.error} />;
  if (reviews.error !== null) return <ErrorNotice error={reviews.error} />;
  if (findings.data.length === 0) {
    return (
      <div className="empty-state panel">
        <h3>No finding candidates</h3>
        <p>A successful Run alone does not create or confirm a finding.</p>
        <Link to={`/projects/${encodeURIComponent(audit.projectId)}/findings`}>
          View all project findings →
        </Link>
      </div>
    );
  }
  const pending = new Map(
    reviews.data
      .filter((request) => request.state === "pending")
      .map((request) => [request.findingId, request]),
  );
  return (
    <div className="audit-finding-list">
      <AuditAnchor />
      <div className="section-heading">
        <h3>Findings in this audit</h3>
        <Link to={`/projects/${encodeURIComponent(audit.projectId)}/findings`}>
          View all project findings →
        </Link>
      </div>
      {findings.data.map((finding) => (
        <AuditFindingCard
          key={finding.findingId}
          audit={audit}
          finding={finding}
          findings={findings.data}
          {...(pending.has(finding.findingId)
            ? { pendingReview: pending.get(finding.findingId)! }
            : {})}
        />
      ))}
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
  return (
    <article
      className="panel audit-finding-card"
      id={`finding-${audit.auditId}-${finding.findingId}`}
    >
      <div className="section-heading">
        <div>
          {showAudit ? (
            <Link
              className="audit-finding-origin"
              to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/findings`}
            >
              {auditProfileLabel(audit)}{" "}
              <span>· {audit.auditId.slice(-8)}</span>
            </Link>
          ) : null}
          <h3>{finding.firstProposal.document.title}</h3>
        </div>
        <StateBadge state={finding.state} />
      </div>
      <AuditMarkdown source={finding.firstProposal.document.description} />
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
  const api = usePublicAPI();
  const reviews = useQuery({
    queryKey: queryKeys.audits.allReviews(audit.auditId),
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditReviews(
          api,
          audit.auditId,
          cursor === undefined ? {} : { cursor },
        ),
      ),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
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
      reviews.data?.some(
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
  if (reviews.error !== null) return <ErrorNotice error={reviews.error} />;
  return (
    <section className="panel audit-section-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Decisions and review history</p>
          <h3>Human reviews</h3>
        </div>
        <span>
          {reviews.data.filter((review) => review.state === "pending").length}{" "}
          pending · {reviews.data.length} total
        </span>
      </div>
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
      {reviews.data.length === 0 ? (
        <p className="muted-copy">No human review has been opened.</p>
      ) : (
        <ol className="audit-review-history">
          {reviews.data.map((review) => (
            <li className="audit-review-card" key={review.requestId}>
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
                <Link
                  to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/checks#check-${encodeURIComponent(review.subjectId)}`}
                >
                  View check →
                </Link>
              ) : null}
              {review.subjectKind === "finding" ? (
                <Link
                  to={`/projects/${encodeURIComponent(audit.projectId)}/findings?audit=${encodeURIComponent(audit.auditId)}#finding-${encodeURIComponent(audit.auditId)}-${encodeURIComponent(review.findingId ?? review.subjectId)}`}
                >
                  Review finding →
                </Link>
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
          <p className="eyebrow">{project.data?.name ?? "Audit"}</p>
          <h2>
            {audit.data === undefined ? "Audit" : auditProfileLabel(audit.data)}
          </h2>
          <p className="audit-identity">
            <code>{audit.data?.auditId ?? auditId}</code>
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
      {audit.data === undefined ? null : (
        <AuditControls audit={audit.data} projectName={project.data?.name} />
      )}
      <nav className="audit-section-navigation" aria-label="Audit sections">
        {SECTIONS.map((candidate) => {
          const target =
            candidate.id === "findings"
              ? `/projects/${encodeURIComponent(projectId)}/findings`
              : candidate.id === "overview"
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
