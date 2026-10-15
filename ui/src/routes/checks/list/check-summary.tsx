import { useQuery, type UseQueryResult } from "@tanstack/react-query";
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
import { formatTimestamp, plural } from "../../../app/format";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { RecordedTime } from "../../../app/recorded-time";
import {
  checkItemKind,
  checkStateLabel,
  itemNoun,
} from "../../../app/vocabulary";
import {
  ButtonLink,
  DetailHeader,
  DetailPane,
  EmptyState,
  Facts,
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

function Summary({
  audit,
  projectName,
  listed,
  read,
}: {
  audit: Audit;
  projectName: string | undefined;
  /**
   * The list holds the check, so it is within the bounds the Inbox reads
   * too (the first projects, their newest checks).
   */
  listed: boolean;
  /** The check read on its own: only then may its controls act. */
  read: UseQueryResult<Audit>;
}) {
  const workspace = useCheckWorkspace(audit, audit.state !== "draft");
  const state = checkStateLabel(audit.state);
  const type = auditProfileLabel(audit);
  const root = checkPath(audit.projectId, audit.auditId);
  const kind = checkItemKind(audit);
  const pending = workspace.data?.pendingReviews ?? 0;
  const unreviewed = workspace.data?.unreviewedFindings ?? 0;
  const timeLimit = timeLimitText(audit);
  const coverage = sectionPath(audit.projectId, audit.auditId, "coverage");
  // The Inbox shows the decisions of running and waiting checks it reads; a
  // check beyond its bounds (a deep link) is decided on its own page.
  const decisions =
    listed && INBOX_CHECK_STATES.includes(audit.state)
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
            <ButtonLink variant="primary" to={root}>
              Open check
            </ButtonLink>
          }
        />
      }
    >
      {audit.scope.objective ? (
        <p className="checks-summary-objective">{audit.scope.objective}</p>
      ) : null}
      {audit.state === "draft" ? (
        <p className="ui-caption">
          This check is a draft: it has no {itemNoun(kind, 2)} yet. Start it to
          create them, or delete it.
        </p>
      ) : (
        <QueryView
          query={workspace}
          loading={
            <p className="ui-caption" role="status">
              Loading progress…
            </p>
          }
          errorContext="The progress of this check could not be loaded."
        >
          {(snapshot) => {
            const counts = workCounts(snapshot);
            const legend = countLegend(counts);
            const done = doneSummary(snapshot.completedChecks, counts.total);
            return (
              <section
                className="checks-summary-progress"
                aria-label="Progress"
              >
                <p className="checks-progress-count">
                  {counts.total === 0 ? `No ${itemNoun(kind, 2)} yet` : done}
                </p>
                {counts.total === 0 ? null : (
                  <ProgressSegments
                    segments={countSegments(legend)}
                    label={`${done}: ${legendSentence(legend.filter((entry) => entry.count > 0))}`}
                  />
                )}
                <ProgressLegend
                  entries={counts.total === 0 ? [] : legend}
                  // The counts pin the list they open to their revision (S19).
                  linkFor={(group) =>
                    `${coverage}?${new URLSearchParams({
                      ...(group === "all" ? {} : { result: group }),
                      auditRevision: String(snapshot.auditRevision),
                    }).toString()}`
                  }
                />
              </section>
            );
          }}
        </QueryView>
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
        <p className="ui-caption">{timeLimit}</p>
      )}
      <QueryView
        query={read}
        loading={
          <p className="ui-caption" role="status">
            Loading the controls…
          </p>
        }
        errorContext="The check could not be read again; its controls are unavailable."
      >
        {() => <AuditControls audit={audit} projectName={projectName} menu />}
      </QueryView>
      <TechnicalDetails description="Check type version, runs and dates.">
        <Facts
          size="sm"
          items={[
            {
              term: "Check type",
              value: (
                <IdChip
                  value={`${audit.profile.name}@${audit.profile.version}`}
                  display={`${audit.profile.name}@${audit.profile.version}`}
                  label="check type version"
                />
              ),
            },
            {
              term: "Runs",
              value: `${audit.submittedRunCount} submitted · ${audit.outstandingRunCount} in progress`,
            },
            { term: "Created", value: formatTimestamp(audit.createdAt) },
            {
              term: "Started",
              value: audit.startedAt && formatTimestamp(audit.startedAt),
            },
            {
              term: "Ended",
              value: audit.finishedAt && formatTimestamp(audit.finishedAt),
            },
            { term: "Revision", value: audit.revision },
          ]}
        />
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
    refetchInterval: pollInterval<Audit>({
      interval: 5_000,
      settled: (data) => auditPollInterval([data]) === false,
    }),
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
        <QueryView
          query={audit}
          loading={
            <p className="ui-caption" role="status">
              Loading check…
            </p>
          }
        >
          {() => null}
        </QueryView>
      </DetailPane>
    );
  return (
    <Summary
      audit={current}
      projectName={listed?.project.name ?? project.data?.name}
      listed={listed !== undefined}
      read={audit}
    />
  );
}
