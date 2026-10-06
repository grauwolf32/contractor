import {
  useMutation,
  useQuery,
  useQueryClient,
  type Query,
  type QueryKey,
} from "@tanstack/react-query";
import { useEffect, useId, useRef, type ReactNode } from "react";
import { Link } from "react-router";

import {
  getAuditReport,
  getAuditWorkspace,
  type AuditReport,
  type AuditWorkspace,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import {
  CROSS_PROJECT_LIMITS,
  invalidateCrossProject,
  type CrossProjectCheck,
} from "../../api/cross-project";
import { queryKeys } from "../../api/query-keys";
import { getRun, retryRunGateway, type RunStatus } from "../../api/runs";
import { ContextLink } from "../../app/context-navigation";
import { ErrorNotice } from "../../app/error-notice";
import { formatTimestamp } from "../../app/format";
import { RecordedTime } from "../../app/recorded-time";
import {
  checkStateLabel,
  itemNoun,
  reportStatusLabel,
  reviewKindLabel,
  reviewStateLabel,
} from "../../app/vocabulary";
import { useSession } from "../../auth/session";
import {
  DetailPane,
  EmptyState,
  IdChip,
  Kbd,
  ProgressSegments,
  StatusChip,
  StatusGlyph,
  TechnicalDetails,
  type StatusTone,
} from "../../ui";
import {
  ActionDecision,
  FindingDecision,
  FindingSummary,
  ReportDecision,
} from "../decisions";
import { decisionOutcome } from "../decisions/model";
import { firstParagraph, plainText } from "../decisions/text";
import { artifactDetailPath } from "../artifacts/paths";
import { describeStopReason } from "../projects/audits/stop-reason";
import { deriveRunTriage, formatRunDuration } from "../runs/triage";
import type { InboxData } from "./data";
import {
  refKey,
  type InboxRef,
  type InboxRow,
  type InboxRowOf,
  type InboxSectionId,
} from "./model";
import {
  checkKind,
  checkPath,
  checkProgress,
  checkTypeLabel,
  excerpt,
  followUpPath,
  issuePath,
  itemLabel,
  pendingReviewsPath,
  PROGRESS_LABELS,
  primaryOutput,
  projectPath,
  recoveryReason,
  reportPath,
  reviewSubjectItem,
  reviewTitle,
  runPath,
  runStateLabel,
  SECTIONS,
} from "./present";

/** What every detail view needs from the page. */
export interface DetailContext {
  data: InboxData;
  /** The selected item's previous and next rows in the list (J and K). */
  previous: InboxRow | undefined;
  next: InboxRow | undefined;
  /** The decision after the selected one, for the decision bar's Next. */
  nextDecision: InboxRow | undefined;
  /** Selects a row (or nothing); keyboard moves replace the history entry. */
  onSelect: (
    row: InboxRow | undefined,
    options?: { replace?: boolean },
  ) => void;
  /** A decision was recorded on the item with this key. */
  onDecided: (key: string, outcome: string) => void;
  /** The item to move focus to once it shows, after a decision. */
  focusKey: string | undefined;
  /** "Decision recorded: …" while it is announced. */
  recorded: string;
}

function OpenIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M13.5 4.5h6v6M19.5 4.5L11 13M17.5 14v4.5a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1H10" />
    </svg>
  );
}

function Chevron({ direction }: { direction: "up" | "down" }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={direction === "up" ? "M6 14.5l6-6 6 6" : "M6 9.5l6 6 6-6"} />
    </svg>
  );
}

/** A link to an item's own page; going back returns to this selection. */
function OpenLink({ to, children }: { to: string; children: ReactNode }) {
  return (
    <ContextLink returnLabel="Inbox" to={to} className="inbox-open">
      {children}
      <OpenIcon />
    </ContextLink>
  );
}

/**
 * The top bar of the detail pane (V3B home mockup): where the item is and
 * its position, the link to its own page, and previous / next.
 */
function DetailBar({
  where,
  label,
  actions,
  context,
  focus,
}: {
  where: string;
  label: string;
  actions?: ReactNode;
  context: DetailContext;
  /** Move focus here once shown: the item arrived after a decision. */
  focus: boolean;
}) {
  const place = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    if (focus) place.current?.focus();
  }, [focus]);
  const { previous, next, onSelect } = context;
  return (
    <div className="inbox-detail-bar">
      <p className="inbox-detail-where" ref={place} tabIndex={-1}>
        <strong>{where}</strong> <span>{label}</span>
      </p>
      <div className="inbox-detail-actions">
        {actions}
        <button
          type="button"
          className="ui-btn inbox-step"
          data-size="sm"
          aria-label="Previous item"
          aria-keyshortcuts="K"
          disabled={previous === undefined}
          onClick={() => onSelect(previous, { replace: true })}
        >
          <Chevron direction="up" />
        </button>
        <button
          type="button"
          className="ui-btn inbox-step"
          data-size="sm"
          aria-label="Next item"
          aria-keyshortcuts="J"
          disabled={next === undefined}
          onClick={() => onSelect(next, { replace: true })}
        >
          <Chevron direction="down" />
        </button>
      </div>
    </div>
  );
}

/** "Decision recorded: …" after the previous item left the list. */
function Recorded({ text }: { text: string }) {
  if (text === "") return null;
  // The page-level status region announces it; this copy is for the eye.
  return (
    <p className="inbox-recorded" aria-hidden="true">
      <StatusGlyph tone="success" />
      {text}
    </p>
  );
}

/** Heading block of a detail: state chip, title and a meta line. */
function Head({
  tone,
  state,
  title,
  meta,
}: {
  tone: StatusTone;
  state: string;
  title: ReactNode;
  meta?: ReactNode;
}) {
  return (
    <header className="inbox-head">
      <StatusChip tone={tone}>{state}</StatusChip>
      <h2 className="inbox-title">{title}</h2>
      {meta === undefined ? null : (
        <div className="inbox-head-meta">{meta}</div>
      )}
    </header>
  );
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  const id = useId();
  return (
    <section className="inbox-block" aria-labelledby={id}>
      <h3 id={id} className="inbox-block-title">
        {title}
      </h3>
      {children}
    </section>
  );
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div>
      <dt>{term}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function sectionTitle(row: InboxRow | undefined, fallback: string): string {
  return row === undefined ? fallback : SECTIONS[row.section].title;
}

// ---------- Possible issue ----------

function IssueDetail({
  row,
  context,
}: {
  row: InboxRowOf<"issue">;
  context: DetailContext;
}) {
  const { project, audit, finding } = row.issue;
  const issues = context.data.model.decide.filter(
    (candidate) => candidate.type === "issue",
  );
  const position = issues.findIndex((candidate) => candidate.key === row.key);
  const { nextDecision, onSelect, onDecided } = context;
  return (
    <DetailPane
      header={
        <DetailBar
          where={SECTIONS.decide.title}
          label={
            position < 0
              ? "Possible issue"
              : `Possible issue ${position + 1} of ${issues.length}`
          }
          actions={
            <OpenLink to={issuePath(audit.auditId, finding.findingId)}>
              Open full review
            </OpenLink>
          }
          context={context}
          focus={context.focusKey === row.key}
        />
      }
      footer={
        <FindingDecision
          auditId={audit.auditId}
          finding={finding}
          next={
            nextDecision === undefined
              ? undefined
              : {
                  label: "Next decision",
                  onNext: () => onSelect(nextDecision, { replace: true }),
                }
          }
          onDecided={(result) =>
            onDecided(row.key, decisionOutcome(result.decision).label)
          }
        />
      }
    >
      <Recorded text={context.recorded} />
      <p className="inbox-context">
        From{" "}
        <Link to={checkPath(project.projectId, audit.auditId)}>
          {checkTypeLabel(audit)}
        </Link>{" "}
        on <Link to={projectPath(project.projectId)}>{project.name}</Link>
      </p>
      <FindingSummary
        auditId={audit.auditId}
        finding={finding}
        variant="compact"
      />
    </DetailPane>
  );
}

// ---------- Approval, applicability, report acceptance ----------

function ReportExcerpt({
  auditId,
  report,
}: {
  auditId: string;
  report: AuditReport | undefined;
}) {
  const summary =
    report?.summary === undefined
      ? ""
      : excerpt(plainText(firstParagraph(report.summary)), 600);
  return (
    <>
      {summary === "" ? null : <p className="inbox-excerpt">{summary}</p>}
      <p className="inbox-quiet-note">
        {report?.status === "proposed"
          ? "This report is proposed and not accepted yet. "
          : ""}
        A report is not a security or compliance certification.{" "}
        <ContextLink returnLabel="Inbox" to={reportPath(auditId)}>
          Open report
        </ContextLink>
      </p>
    </>
  );
}

function ReviewDetail({
  row,
  context,
}: {
  row: InboxRowOf<"review">;
  context: DetailContext;
}) {
  const api = usePublicAPI();
  const { project, audit, review } = row.decision;
  const acceptance = review.kind === "report-acceptance";
  // The same read as ReportDecision, so both share one cached report.
  const report = useQuery({
    queryKey: queryKeys.audits.report(audit.auditId),
    queryFn: () => getAuditReport(api, audit.auditId),
    enabled: acceptance,
  });
  const item = reviewSubjectItem(row.decision, context.data.items.byCheck);
  const kindLabel = reviewKindLabel(review.kind);
  const state = reviewStateLabel(review.state);
  const decided = context.onDecided;
  return (
    <DetailPane
      header={
        <DetailBar
          where={SECTIONS.decide.title}
          label={kindLabel}
          actions={
            acceptance ? (
              <OpenLink to={reportPath(audit.auditId)}>Open report</OpenLink>
            ) : (
              <OpenLink
                to={pendingReviewsPath(project.projectId, audit.auditId)}
              >
                Open in check
              </OpenLink>
            )
          }
          context={context}
          focus={context.focusKey === row.key}
        />
      }
      footer={
        acceptance ? (
          <ReportDecision
            auditId={audit.auditId}
            review={review}
            report={report.data}
            onDecided={(result) =>
              decided(row.key, decisionOutcome(result.decision).label)
            }
          />
        ) : (
          <ActionDecision
            auditId={audit.auditId}
            review={review}
            onDecided={(result) =>
              decided(row.key, decisionOutcome(result.decision).label)
            }
          />
        )
      }
    >
      <Recorded text={context.recorded} />
      <Head
        tone={state.tone}
        state={state.label}
        title={reviewTitle(row.decision, item, kindLabel)}
      />
      <dl className="inbox-facts">
        <Fact term="Check">
          <Link to={checkPath(project.projectId, audit.auditId)}>
            {checkTypeLabel(audit)}
          </Link>
        </Fact>
        <Fact term="Project">
          <Link to={projectPath(project.projectId)}>{project.name}</Link>
        </Fact>
        <Fact term="Subject">
          {acceptance
            ? "The check's proposed report"
            : item === undefined
              ? "A work item of this check"
              : itemLabel(item)}
        </Fact>
        <Fact term="Requested">
          <RecordedTime value={review.createdAt} />
        </Fact>
        {review.expiresAt === undefined ? null : (
          <Fact term="Expires">
            <RecordedTime value={review.expiresAt} />
          </Fact>
        )}
      </dl>
      {acceptance ? (
        <Block title="Proposed report">
          {report.isPending ? (
            <p className="inbox-quiet-note">Loading the report…</p>
          ) : report.isError ? (
            <ErrorNotice
              error={report.error}
              context="The report could not be loaded."
              onRetry={() => void report.refetch()}
            />
          ) : (
            <ReportExcerpt auditId={audit.auditId} report={report.data} />
          )}
        </Block>
      ) : null}
      <TechnicalDetails description="Request and subject identifiers.">
        <dl className="inbox-facts" data-size="sm">
          <Fact term="Request ID">
            <IdChip value={review.requestId} label="request ID" />
          </Fact>
          <Fact term="Subject ID">
            <IdChip value={review.subjectId} label="subject ID" />
          </Fact>
          <Fact term="Check ID">
            <IdChip value={audit.auditId} label="check ID" />
          </Fact>
          <Fact term="Subject revision">{review.subjectRevision}</Fact>
        </dl>
      </TechnicalDetails>
    </DetailPane>
  );
}

// ---------- Run ----------

/** Polls a waiting Run, whose recovery can change; other states settle. */
function runRefetch(query: Query<RunStatus, Error, RunStatus, QueryKey>) {
  return query.state.data?.state === "waiting"
    ? CROSS_PROJECT_LIMITS.pollMs
    : false;
}

function RunCause({ run }: { run: RunStatus }) {
  const triage = deriveRunTriage(run);
  const issue = triage.issue;
  if (run.state !== "failed" && run.state !== "cancelled") return null;
  return (
    <Block
      title={
        issue?.source === "cancellation" ? "Why it stopped" : "Primary cause"
      }
    >
      <p className="inbox-cause">
        {issue === undefined
          ? "No failure reason was reported. Open the Run to inspect the latest attempt."
          : issue.message}
      </p>
      {issue?.participant === undefined ? null : (
        <p className="inbox-quiet-note">
          Reported by {issue.participant}
          {issue.logicalAgent === undefined ? "" : ` ${issue.logicalAgent}`}
        </p>
      )}
    </Block>
  );
}

function RunRecovery({ run }: { run: RunStatus }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const retry = useMutation({
    mutationFn: () => retryRunGateway(api, run.runId),
    // Refetch rather than assume (S06:225-230): the Run says when it runs.
    onSettled: () =>
      Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.runs.all }),
        invalidateCrossProject(queryClient),
      ]),
  });
  const recovery = run.recovery;
  if (recovery === undefined && !retry.isSuccess) return null;
  return (
    <Block title="Model connection">
      {recovery === undefined ? null : (
        <>
          <p className="inbox-cause">{recoveryReason(recovery.code)}</p>
          <p className="inbox-quiet-note">
            {recovery.requiresRetry
              ? "Automatic recovery has stopped. Restore the model, then retry the connection. Completed work is kept."
              : recovery.nextRetryAt === undefined
                ? "Waiting for automatic recovery. Completed work is kept."
                : `Next automatic check: ${formatTimestamp(recovery.nextRetryAt)}. Completed work is kept.`}
          </p>
        </>
      )}
      {recovery?.requiresRetry === true ? (
        <div className="inbox-actions">
          <button
            type="button"
            className="ui-btn"
            data-variant="primary"
            data-size="sm"
            disabled={retry.isPending}
            onClick={() => retry.mutate()}
          >
            {retry.isPending ? "Enabling retry…" : "Retry model connection"}
          </button>
        </div>
      ) : null}
      {retry.isSuccess ? (
        <p className="inbox-quiet-note" role="status">
          Retry requested. The Run continues once the model answers.
        </p>
      ) : null}
      {retry.error === null ? null : <ErrorNotice error={retry.error} />}
    </Block>
  );
}

function RunPrimaryOutput({ run, data }: { run: RunStatus; data: InboxData }) {
  const primary = primaryOutput(run, run, false, data.workflowOutputs);
  return (
    <Block title="Primary result">
      {primary.state === "present" ? (
        <p>
          <ContextLink
            returnLabel="Inbox"
            to={artifactDetailPath(
              { kind: "run", id: run.runId },
              primary.artifact,
            )}
          >
            Open {primary.slot}
          </ContextLink>
          , the output this Workflow marks as its primary result.
        </p>
      ) : (
        <p>
          {primary.state === "missing"
            ? `The primary result ${primary.slot} was not produced. Open the Run to see every output.`
            : primary.state === "none"
              ? "This Workflow declares no primary result. Open the Run to see every output."
              : primary.state === "loading"
                ? "Reading the Workflow's declared outputs…"
                : "The declared outputs could not be read. Open the Run to see every output."}
        </p>
      )}
      <p className="inbox-quiet-note">
        A finished Run alone does not establish the quality of its result.
      </p>
    </Block>
  );
}

function RunDetail({
  runId,
  row,
  context,
}: {
  runId: string;
  row: InboxRowOf<"run"> | undefined;
  context: DetailContext;
}) {
  const api = usePublicAPI();
  const { session } = useSession();
  const canOperate =
    session?.principal.capabilities.includes("operations") === true;
  const query = useQuery({
    queryKey: queryKeys.runs.detail(runId),
    queryFn: () => getRun(api, runId),
    refetchInterval: runRefetch,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const label =
    row?.reason === "model"
      ? "Waiting for the model"
      : row?.reason === "failed"
        ? "Failed Run"
        : row?.reason === "finished"
          ? "Finished Run"
          : "Run";
  const run = query.data;
  const project =
    run?.projectId === undefined
      ? undefined
      : context.data.projectNames.get(run.projectId);
  const triage = run === undefined ? undefined : deriveRunTriage(run);
  const state = run === undefined ? undefined : runStateLabel(run.state);
  return (
    <DetailPane
      header={
        <DetailBar
          where={sectionTitle(row, "Runs")}
          label={label}
          actions={<OpenLink to={runPath(runId)}>Open Run</OpenLink>}
          context={context}
          focus={context.focusKey === refKey({ kind: "run", runId })}
        />
      }
    >
      <Recorded text={context.recorded} />
      {run === undefined || triage === undefined || state === undefined ? (
        query.isError ? (
          <ErrorNotice
            error={query.error}
            context="The Run could not be loaded."
            onRetry={() => void query.refetch()}
          />
        ) : (
          <p className="inbox-quiet-note" role="status">
            Loading the Run…
          </p>
        )
      ) : (
        <>
          <Head
            tone={state.tone}
            state={state.label}
            title={run.workflow}
            meta={
              <>
                <IdChip value={run.runId} label="Run ID" />
                {run.projectId === undefined ? null : (
                  <Link to={projectPath(run.projectId)}>
                    {project ?? "Project"}
                  </Link>
                )}
                {run.finishedAt === undefined ? null : (
                  <span>
                    Ended <RecordedTime value={run.finishedAt} />
                  </span>
                )}
              </>
            }
          />
          <RunCause run={run} />
          <RunRecovery run={run} />
          {run.state === "succeeded" ? (
            // A finished Run's next step is its result.
            <RunPrimaryOutput run={run} data={context.data} />
          ) : (
            <Block title="Next step">
              <p>
                <strong>{triage.guidance.title}.</strong>{" "}
                {triage.guidance.message}
              </p>
              {triage.guidance.operationsPath === undefined ||
              !canOperate ? null : (
                <p>
                  <Link to={triage.guidance.operationsPath}>
                    Open Operations diagnostics
                  </Link>
                </p>
              )}
              {run.state === "failed" ? (
                <p className="inbox-quiet-note">
                  Continue from failed stage and Configure another Run are
                  separate actions on the{" "}
                  <ContextLink returnLabel="Inbox" to={runPath(run.runId)}>
                    Run page
                  </ContextLink>
                  .
                </p>
              ) : null}
            </Block>
          )}
          <TechnicalDetails description="Stage, attempts and the cause's code.">
            <dl className="inbox-facts" data-size="sm">
              <Fact term="Stage">{triage.stage ?? "—"}</Fact>
              <Fact term="Attempts">{triage.attemptCount}</Fact>
              <Fact term="Duration">
                {formatRunDuration(triage.durationMs)}
              </Fact>
              {triage.issue === undefined ? null : (
                <>
                  <Fact term="Cause code">
                    <code>{triage.issue.code}</code>
                  </Fact>
                  <Fact term="Retryable">
                    {triage.issue.retryable === undefined
                      ? "Unknown"
                      : triage.issue.retryable
                        ? "Yes"
                        : "No"}
                  </Fact>
                </>
              )}
            </dl>
          </TechnicalDetails>
        </>
      )}
    </DetailPane>
  );
}

// ---------- Check ----------

function pinnedStaleTime(
  query: Query<AuditWorkspace, Error, AuditWorkspace, QueryKey>,
) {
  return query.state.data === undefined ? 0 : ("static" as const);
}

function CheckDetail({
  check,
  row,
  context,
}: {
  check: CrossProjectCheck;
  row: InboxRowOf<"check"> | undefined;
  context: DetailContext;
}) {
  const api = usePublicAPI();
  const { project, audit } = check;
  // The list's read when the check is counted there; otherwise read here.
  const workspace = useQuery({
    queryKey: queryKeys.inbox.workspace(audit.auditId, audit.revision),
    queryFn: () => getAuditWorkspace(api, audit.auditId),
    staleTime: pinnedStaleTime,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const kind = checkKind(check);
  const progress =
    workspace.data === undefined
      ? undefined
      : checkProgress(workspace.data, kind);
  const state = checkStateLabel(audit.state);
  const stop = describeStopReason(audit);
  const objective = audit.scope.objective?.trim() ?? "";
  return (
    <DetailPane
      header={
        <DetailBar
          where={sectionTitle(row, "Checks")}
          label="Check"
          actions={
            <OpenLink to={checkPath(project.projectId, audit.auditId)}>
              Open check
            </OpenLink>
          }
          context={context}
          focus={
            context.focusKey ===
            refKey({ kind: "check", auditId: audit.auditId })
          }
        />
      }
    >
      <Recorded text={context.recorded} />
      <Head
        tone={state.tone}
        state={state.label}
        title={checkTypeLabel(audit)}
        meta={
          <>
            <Link to={projectPath(project.projectId)}>{project.name}</Link>
            <IdChip value={audit.auditId} label="check ID" />
            <span>
              {audit.startedAt === undefined ? "Created " : "Started "}
              <RecordedTime value={audit.startedAt ?? audit.createdAt} />
            </span>
          </>
        }
      />
      {objective === "" ? null : (
        <Block title="Objective">
          <p className="inbox-excerpt">{excerpt(objective, 400)}</p>
        </Block>
      )}
      {stop === null ? null : (
        <p className="inbox-stop" data-tone={stop.tone}>
          <strong>{stop.label ?? "Stopped"}.</strong>{" "}
          {stop.message.trim() === "" ? null : stop.message}
        </p>
      )}
      <Block title="Progress">
        {progress === undefined ? (
          workspace.isError ? (
            <ErrorNotice
              error={workspace.error}
              context="Progress could not be loaded."
              onRetry={() => void workspace.refetch()}
            />
          ) : (
            <p className="inbox-quiet-note">Loading progress…</p>
          )
        ) : (
          <>
            <p className="inbox-progress-summary">{progress.summary}</p>
            {progress.total === 0 ? null : (
              <ProgressSegments
                label={progress.label}
                segments={progress.segments}
              />
            )}
            <dl className="inbox-facts" data-size="sm">
              <Fact term={PROGRESS_LABELS.done}>{progress.done}</Fact>
              <Fact term={PROGRESS_LABELS.issue}>{progress.issues}</Fact>
              <Fact term={PROGRESS_LABELS.followUp}>{progress.gaps}</Fact>
              <Fact term={PROGRESS_LABELS.unchecked}>{progress.unchecked}</Fact>
              {workspace.data === undefined ? null : (
                <>
                  <Fact term="Possible issues to review">
                    {workspace.data.unreviewedFindings}
                  </Fact>
                  <Fact term="Decisions waiting">
                    {workspace.data.pendingReviews}
                  </Fact>
                </>
              )}
            </dl>
          </>
        )}
      </Block>
      <div className="inbox-actions">
        <ContextLink
          returnLabel="Inbox"
          to={checkPath(project.projectId, audit.auditId)}
          className="ui-btn"
          data-variant="primary"
          data-size="sm"
        >
          Open check
        </ContextLink>
        {progress === undefined || progress.gaps === 0 ? null : (
          <ContextLink
            returnLabel="Inbox"
            to={followUpPath(project.projectId, audit.auditId)}
            className="ui-btn"
            data-size="sm"
          >
            Show {itemNoun(kind, 2)} that need follow-up
          </ContextLink>
        )}
      </div>
      {audit.state === "paused" ? (
        <p className="inbox-quiet-note">
          Continue, Stop and Delete are on the check page.
        </p>
      ) : null}
      <TechnicalDetails description="Revision, round and Runs of this check.">
        <dl className="inbox-facts" data-size="sm">
          <Fact term="Revision">{audit.revision}</Fact>
          <Fact term="Round">
            {audit.currentRoundId === undefined ? (
              "None yet"
            ) : (
              <IdChip value={audit.currentRoundId} label="round ID" />
            )}
          </Fact>
          <Fact term="Runs not finished">{audit.outstandingRunCount}</Fact>
          {workspace.data === undefined ? null : (
            <Fact term="Counted at">
              {formatTimestamp(workspace.data.asOf)}
            </Fact>
          )}
        </dl>
      </TechnicalDetails>
    </DetailPane>
  );
}

// ---------- Report ----------

function ReportDetail({
  row,
  context,
}: {
  row: InboxRowOf<"report">;
  context: DetailContext;
}) {
  const { project, audit, report } = row.report;
  const status = reportStatusLabel(report.status);
  return (
    <DetailPane
      header={
        <DetailBar
          where={SECTIONS.ready.title}
          label="Report"
          actions={
            <OpenLink to={reportPath(audit.auditId)}>Open report</OpenLink>
          }
          context={context}
          focus={context.focusKey === row.key}
        />
      }
    >
      <Recorded text={context.recorded} />
      <Head
        tone={status.tone}
        state={status.label}
        title={`${project.name} report`}
        meta={
          <>
            <Link to={checkPath(project.projectId, audit.auditId)}>
              {checkTypeLabel(audit)}
            </Link>
            <IdChip value={audit.auditId} label="check ID" />
            <span>
              Finished{" "}
              <RecordedTime value={audit.finishedAt ?? audit.updatedAt} />
            </span>
          </>
        }
      />
      <Block title="Summary">
        <ReportExcerpt auditId={audit.auditId} report={report} />
      </Block>
    </DetailPane>
  );
}

// ---------- Nothing selected, or not listed ----------

const OVERVIEW_ORDER: readonly InboxSectionId[] = [
  "decide",
  "unblock",
  "ready",
  "running",
];

const OVERVIEW_TONES: Readonly<Record<InboxSectionId, StatusTone>> = {
  decide: "review",
  unblock: "blocked",
  ready: "done",
  running: "progress",
};

export function Overview({
  context,
  counts,
}: {
  context: DetailContext;
  counts: Record<InboxSectionId, string>;
}) {
  const { model } = context.data;
  const heading = useRef<HTMLHeadingElement>(null);
  const focus = context.focusKey === "";
  useEffect(() => {
    if (focus) heading.current?.focus();
  }, [focus]);
  return (
    <DetailPane>
      <Recorded text={context.recorded} />
      <header className="inbox-head">
        <h2 className="inbox-title" ref={heading} tabIndex={-1}>
          Overview
        </h2>
        <p className="inbox-head-meta">
          What needs you across your projects. Choose an item in the list, or
          press <Kbd>J</Kbd> to start with the first one.
        </p>
      </header>
      <ul className="inbox-overview" role="list">
        {OVERVIEW_ORDER.map((id) => {
          const first = model[id][0];
          const content = (
            <>
              <StatusGlyph tone={OVERVIEW_TONES[id]} />
              <span className="inbox-overview-title">{SECTIONS[id].title}</span>
              <span className="inbox-overview-count">{counts[id]}</span>
            </>
          );
          return (
            <li key={id}>
              {first === undefined ? (
                <div className="inbox-overview-item">{content}</div>
              ) : (
                <button
                  type="button"
                  className="inbox-overview-item"
                  onClick={() => context.onSelect(first)}
                >
                  {content}
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </DetailPane>
  );
}

/** The page of an item that is not listed, from its reference alone. */
function referencePage(ref: InboxRef): { to: string; label: string } {
  switch (ref.kind) {
    case "issue":
      return {
        to: issuePath(ref.auditId, ref.findingId),
        label: "Open full review",
      };
    case "report":
      return { to: reportPath(ref.auditId), label: "Open report" };
    case "review":
    case "check":
      return {
        to: `/checks?check=${encodeURIComponent(ref.auditId)}`,
        label: "Open the check",
      };
    case "run":
      return { to: runPath(ref.runId), label: "Open Run" };
  }
}

function NotListed({
  selected,
  loading,
}: {
  selected: InboxRef;
  loading: boolean;
}) {
  const page = referencePage(selected);
  return (
    <DetailPane>
      {loading ? (
        <p className="inbox-quiet-note" role="status">
          Loading…
        </p>
      ) : (
        <EmptyState
          title="This item is no longer in your Inbox"
          action={
            <ContextLink returnLabel="Inbox" to={page.to}>
              {page.label}
            </ContextLink>
          }
        >
          It may have been decided, finished or removed since it was listed.
        </EmptyState>
      )}
    </DetailPane>
  );
}

/** The detail pane for the selected item. */
export function InboxDetail({
  selected,
  row,
  loading,
  context,
  counts,
}: {
  selected: InboxRef | undefined;
  /** The selected item's first row in the list, when it is listed. */
  row: InboxRow | undefined;
  /** The lists that could hold the selected item are still loading. */
  loading: boolean;
  context: DetailContext;
  counts: Record<InboxSectionId, string>;
}) {
  if (selected === undefined)
    return <Overview context={context} counts={counts} />;
  if (selected.kind === "run")
    return (
      <RunDetail
        key={selected.runId}
        runId={selected.runId}
        row={row?.type === "run" ? row : undefined}
        context={context}
      />
    );
  if (selected.kind === "check") {
    const check =
      row?.type === "check"
        ? row.check
        : context.data.checks.checks.find(
            (candidate) => candidate.audit.auditId === selected.auditId,
          );
    if (check !== undefined)
      return (
        <CheckDetail
          key={check.audit.auditId}
          check={check}
          row={row?.type === "check" ? row : undefined}
          context={context}
        />
      );
  }
  if (row?.type === "issue")
    return <IssueDetail key={row.key} row={row} context={context} />;
  if (row?.type === "review")
    return <ReviewDetail key={row.key} row={row} context={context} />;
  if (row?.type === "report")
    return <ReportDetail key={row.key} row={row} context={context} />;
  return <NotListed selected={selected} loading={loading} />;
}
