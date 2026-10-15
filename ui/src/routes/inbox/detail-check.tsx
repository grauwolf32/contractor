import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";
import { Link } from "react-router";

import { getAuditWorkspace, type AuditWorkspace } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import type { CrossProjectCheck } from "../../api/cross-project";
import { queryKeys } from "../../api/query-keys";
import { formatTimestamp } from "../../app/format";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { checkStateLabel, itemNoun } from "../../app/vocabulary";
import {
  ButtonLink,
  DetailPane,
  Facts,
  IdChip,
  ProgressSegments,
  TechnicalDetails,
} from "../../ui";
import { checkPath } from "../projects/audits/check-links";
import { describeStopReason } from "../projects/audits/stop-reason";
import { newestCached, pinned } from "./data";
import {
  Block,
  DetailBar,
  Head,
  OpenLink,
  Recorded,
  type DetailContext,
} from "./detail-parts";
import { refKey, type InboxRowOf } from "./model";
import {
  checkKind,
  checkProgress,
  checkTypeLabel,
  excerpt,
  followUpPath,
  PROGRESS_LABELS,
  projectPath,
  sectionTitle,
} from "./present";

/**
 * A check of the Inbox (stuck, running or finished): its progress from the
 * workspace counters and the way to its page.
 */
export function CheckDetail({
  check,
  row,
  context,
}: {
  check: CrossProjectCheck;
  row: InboxRowOf<"check"> | undefined;
  context: DetailContext;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const { project, audit } = check;
  const { auditId, revision } = audit;
  // While a new revision loads, the counts shown last (or the newest cached
  // ones, e.g. read by the list) stay, as they do in the list.
  const placeholder = useCallback(
    (previous: AuditWorkspace | undefined) =>
      previous ??
      newestCached<AuditWorkspace>(
        queryClient,
        queryKeys.inbox.workspace(auditId, revision).slice(0, -1),
      ),
    [auditId, queryClient, revision],
  );
  // The list's read when the check is counted there; otherwise read here.
  const workspace = useQuery({
    queryKey: queryKeys.inbox.workspace(auditId, revision),
    queryFn: () => getAuditWorkspace(api, auditId),
    placeholderData: placeholder,
    ...pinned<AuditWorkspace>(),
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
        <QueryView
          query={workspace}
          errorContext="Progress could not be loaded."
          loading={<p className="ui-caption">Loading progress…</p>}
        >
          {(counts) => {
            const shown = checkProgress(counts, kind);
            return (
              <>
                <p className="inbox-progress-summary">{shown.summary}</p>
                {shown.total === 0 ? null : (
                  <ProgressSegments
                    label={shown.label}
                    segments={shown.segments}
                  />
                )}
                <Facts
                  size="sm"
                  items={[
                    { term: PROGRESS_LABELS.done, value: shown.done },
                    { term: PROGRESS_LABELS.issue, value: shown.issues },
                    { term: PROGRESS_LABELS.followUp, value: shown.gaps },
                    { term: PROGRESS_LABELS.unchecked, value: shown.unchecked },
                    {
                      term: "Possible issues to review",
                      value: counts.unreviewedFindings,
                    },
                    { term: "Decisions waiting", value: counts.pendingReviews },
                  ]}
                />
              </>
            );
          }}
        </QueryView>
      </Block>
      <div className="inbox-actions">
        <ButtonLink
          returnLabel="Inbox"
          to={checkPath(project.projectId, audit.auditId)}
          variant="primary"
          size="sm"
        >
          Open check
        </ButtonLink>
        {progress === undefined || progress.gaps === 0 ? null : (
          <ButtonLink
            returnLabel="Inbox"
            to={followUpPath(project.projectId, audit.auditId)}
            size="sm"
          >
            Show {itemNoun(kind, 2)} that need follow-up
          </ButtonLink>
        )}
      </div>
      {audit.state === "paused" ? (
        <p className="ui-caption">
          Continue, Stop and Delete are on the check page.
        </p>
      ) : null}
      <TechnicalDetails description="Revision, round and Runs of this check.">
        <Facts
          size="sm"
          items={[
            { term: "Revision", value: audit.revision },
            {
              term: "Round",
              value:
                audit.currentRoundId === undefined ? (
                  "None yet"
                ) : (
                  <IdChip value={audit.currentRoundId} label="round ID" />
                ),
            },
            { term: "Runs not finished", value: audit.outstandingRunCount },
            {
              term: "Counted at",
              value:
                workspace.data === undefined
                  ? undefined
                  : formatTimestamp(workspace.data.asOf),
            },
          ]}
        />
      </TechnicalDetails>
    </DetailPane>
  );
}
