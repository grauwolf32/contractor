import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";

import { auditPollInterval, getAudit, type Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import {
  INBOX_CHECK_STATES,
  type CrossProjectCheck,
} from "../../../api/cross-project";
import { PublicAPIError } from "../../../api/error";
import { getProject } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp } from "../../../app/format";
import { RecordedTime } from "../../../app/recorded-time";
import {
  checkItemKind,
  checkStateLabel,
  itemNoun,
} from "../../../app/vocabulary";
import {
  DetailHeader,
  DetailPane,
  EmptyState,
  IdChip,
  ProgressSegments,
  StatusChip,
  TechnicalDetails,
} from "../../../ui";
import { useCheckWorkspace } from "../../projects/audits/check-data";
import { timeLimitText } from "../../projects/audits/check-format";
import { checkPath, sectionPath } from "../../projects/audits/check-links";
import {
  countLegend,
  countSegments,
  doneSummary,
  legendSentence,
  workCounts,
} from "../../projects/audits/check-model";
import { AuditControls } from "../../projects/audits/controls";
import { auditProfileLabel } from "../../projects/audits/labels";
import {
  ProgressLegend,
  StopReasonBanner,
} from "../../projects/audits/progress";

function plural(count: number, singular: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? singular : many}`;
}

function Summary({
  audit,
  projectName,
  fresh,
  readError,
  onRetry,
}: {
  audit: Audit;
  projectName: string | undefined;
  /** The check was read on its own, not only listed: controls may act. */
  fresh: boolean;
  /** Reading the check on its own failed. */
  readError: Error | null;
  onRetry: () => void;
}) {
  const workspace = useCheckWorkspace(audit, audit.state !== "draft");
  const state = checkStateLabel(audit.state);
  const type = auditProfileLabel(audit);
  const root = checkPath(audit.projectId, audit.auditId);
  const kind = checkItemKind(audit);
  const snapshot = workspace.data;
  const counts = snapshot === undefined ? undefined : workCounts(snapshot);
  const legend = counts === undefined ? [] : countLegend(counts);
  const shown = legend.filter((entry) => entry.count > 0);
  const pending = snapshot?.pendingReviews ?? 0;
  const unreviewed = snapshot?.unreviewedFindings ?? 0;
  const timeLimit = timeLimitText(audit);
  const coverage = sectionPath(audit.projectId, audit.auditId, "coverage");
  const decisions = INBOX_CHECK_STATES.includes(audit.state)
    ? "/"
    : `${sectionPath(audit.projectId, audit.auditId, "reviews")}?state=pending`;
  return (
    <DetailPane
      header={
        <DetailHeader
          title={type}
          status={
            <StatusChip tone={state.tone} size="sm">
              {state.label}
            </StatusChip>
          }
          meta={
            <>
              <Link to={`/projects/${encodeURIComponent(audit.projectId)}`}>
                {projectName ?? audit.projectId}
              </Link>
              <span>
                Updated <RecordedTime value={audit.updatedAt} />
              </span>
              <span className="checks-progress-id">
                <span>Check ID</span>
                <IdChip value={audit.auditId} label="check ID" />
              </span>
            </>
          }
          actions={
            <Link className="ui-btn" data-variant="primary" to={root}>
              Open check
            </Link>
          }
        />
      }
    >
      {audit.scope.objective ? (
        <p className="checks-summary-objective">{audit.scope.objective}</p>
      ) : null}
      {audit.state === "draft" ? (
        <p className="checks-quiet">
          This check is a draft: it has no {itemNoun(kind, 2)} yet. Start it to
          create them, or delete it.
        </p>
      ) : counts === undefined ? (
        workspace.error === null ? (
          <p className="checks-quiet" role="status">
            Loading progress…
          </p>
        ) : (
          <div className="checks-notice" data-tone="warning" role="status">
            <p>The progress of this check could not be loaded.</p>
            <button
              className="ui-btn"
              data-size="xs"
              type="button"
              onClick={() => void workspace.refetch()}
            >
              Try again
            </button>
          </div>
        )
      ) : (
        <section className="checks-summary-progress" aria-label="Progress">
          <p className="checks-progress-count">
            {counts.total === 0
              ? `No ${itemNoun(kind, 2)} yet`
              : doneSummary(snapshot!.completedChecks, counts.total)}
          </p>
          {counts.total === 0 ? null : (
            <ProgressSegments
              segments={countSegments(legend)}
              label={`${doneSummary(snapshot!.completedChecks, counts.total)}: ${legendSentence(shown)}`}
            />
          )}
          <ProgressLegend
            entries={counts.total === 0 ? [] : legend}
            // The counts pin the list they open to their revision (S19).
            linkFor={(group) =>
              `${coverage}?${new URLSearchParams({
                ...(group === "all" ? {} : { result: group }),
                auditRevision: String(snapshot!.auditRevision),
              }).toString()}`
            }
          />
        </section>
      )}
      <StopReasonBanner audit={audit} />
      {pending > 0 || unreviewed > 0 ? (
        <p className="checks-progress-waiting">
          {pending > 0 ? (
            <Link to={decisions}>
              {plural(
                pending,
                "decision waiting for you",
                "decisions waiting for you",
              )}
            </Link>
          ) : null}
          {unreviewed > 0 ? (
            <Link
              to={`/issues?${new URLSearchParams({ project: audit.projectId, state: "proposed" }).toString()}`}
            >
              {plural(
                unreviewed,
                "possible issue to review",
                "possible issues to review",
              )}
            </Link>
          ) : null}
        </p>
      ) : null}
      {timeLimit === undefined ? null : (
        <p className="checks-quiet">{timeLimit}</p>
      )}
      {fresh ? (
        <AuditControls audit={audit} projectName={projectName} menu />
      ) : readError === null ? (
        <p className="checks-quiet" role="status">
          Loading the controls…
        </p>
      ) : (
        <ErrorNotice
          error={readError}
          context="The check could not be read again; its controls are unavailable."
          onRetry={onRetry}
        />
      )}
      <TechnicalDetails description="Check type version, runs and dates.">
        <dl className="checks-facts">
          <div>
            <dt>Check type</dt>
            <dd>
              <IdChip
                value={`${audit.profile.name}@${audit.profile.version}`}
                display={`${audit.profile.name}@${audit.profile.version}`}
                label="check type version"
              />
            </dd>
          </div>
          <div>
            <dt>Runs</dt>
            <dd>
              {audit.submittedRunCount} submitted · {audit.outstandingRunCount}{" "}
              in progress
            </dd>
          </div>
          <div>
            <dt>Created</dt>
            <dd>{formatTimestamp(audit.createdAt)}</dd>
          </div>
          {audit.startedAt === undefined ? null : (
            <div>
              <dt>Started</dt>
              <dd>{formatTimestamp(audit.startedAt)}</dd>
            </div>
          )}
          {audit.finishedAt === undefined ? null : (
            <div>
              <dt>Ended</dt>
              <dd>{formatTimestamp(audit.finishedAt)}</dd>
            </div>
          )}
          <div>
            <dt>Revision</dt>
            <dd>{audit.revision}</dd>
          </div>
        </dl>
      </TechnicalDetails>
    </DetailPane>
  );
}

/**
 * The selected check of the Checks list: what it is, how far it got, what
 * waits for the user, its time limit and lifecycle controls, and the way
 * into the check page. Read on its own (and polled while it can change), so
 * a deep link works for checks beyond the list's bounds too.
 */
export function CheckSummary({
  auditId,
  listed,
}: {
  auditId: string;
  /** The check as the list has it, shown while the check loads. */
  listed: CrossProjectCheck | undefined;
}) {
  const api = usePublicAPI();
  const audit = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    queryFn: () => getAudit(api, auditId),
    refetchInterval: (query) =>
      auditPollInterval(
        query.state.data === undefined ? [] : [query.state.data],
        5_000,
      ),
    refetchOnReconnect: true,
    retry: (count, error) =>
      !(error instanceof PublicAPIError && error.status === 404) && count < 2,
  });
  const current = audit.data ?? listed?.audit;
  const project = useQuery({
    queryKey: queryKeys.projects.detail(current?.projectId ?? ""),
    queryFn: () => getProject(api, current!.projectId),
    enabled: listed === undefined && current !== undefined,
  });
  if (audit.error instanceof PublicAPIError && audit.error.status === 404)
    return (
      <DetailPane>
        <EmptyState title="This check no longer exists">
          It was deleted, or it belongs to someone else.
        </EmptyState>
      </DetailPane>
    );
  if (current === undefined)
    return (
      <DetailPane>
        {audit.error === null ? (
          <p className="checks-quiet" role="status">
            Loading check…
          </p>
        ) : (
          <ErrorNotice
            error={audit.error}
            onRetry={() => void audit.refetch()}
            retryPending={audit.isFetching}
          />
        )}
      </DetailPane>
    );
  return (
    <Summary
      audit={current}
      projectName={listed?.project.name ?? project.data?.name}
      fresh={audit.data !== undefined}
      readError={audit.error}
      onRetry={() => void audit.refetch()}
    />
  );
}
