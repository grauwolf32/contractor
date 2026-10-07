import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "react-router";

import { auditPollInterval, listProjectAudits } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { CursorControls } from "../../../app/cursor-controls";
import { useURLCursorStack } from "../../../app/pagination";
import { QueryView } from "../../../app/query-view";
import { RecordedTime } from "../../../app/recorded-time";
import { RefreshButton } from "../../../app/refresh-button";
import { checkStateLabel } from "../../../app/vocabulary";
import {
  EmptyState,
  IdChip,
  ListRow,
  ListSection,
  StatusChip,
  StatusGlyph,
  TechnicalDetails,
} from "../../../ui";
import { ProjectSectionActions } from "../navigation";
import { checkPath } from "./check-links";
import { AuditControls } from "./controls";
import { auditProfileLabel } from "./labels";
import { describeStopReason } from "./stop-reason";

import "./checks.css";

/**
 * The project's Checks tab, /projects/:projectId/audits: the project's
 * checks newest first in Server pages (the cursors live in the URL), polled
 * while one of them can change, each with compact lifecycle controls. New
 * checks start at /checks/new?project=<projectId>.
 */
export function ProjectAuditWorkspace({
  projectId,
  projectName,
}: {
  projectId: string;
  projectName?: string;
}) {
  const api = usePublicAPI();
  const location = useLocation();
  const pages = useURLCursorStack({
    param: "auditCursor",
    navigateOptions: { state: location.state },
  });
  const cursor = pages.cursor;
  const audits = useQuery({
    queryKey: queryKeys.projects.audits.list(projectId, cursor),
    queryFn: () =>
      listProjectAudits(api, {
        projectId,
        ...(cursor === undefined ? {} : { cursor }),
      }),
    refetchOnReconnect: true,
    refetchInterval: (query) =>
      auditPollInterval(query.state.data?.items ?? [], 5_000),
  });
  const start = `/checks/new?${new URLSearchParams({ project: projectId }).toString()}`;
  return (
    <div className="checks-project">
      <ProjectSectionActions>
        <RefreshButton
          isFetching={audits.isFetching}
          onRefresh={() => void audits.refetch()}
        />
        <Link
          className="ui-btn"
          data-variant="primary"
          data-size="sm"
          to={start}
        >
          Start a check
        </Link>
      </ProjectSectionActions>
      <QueryView
        query={audits}
        loading={
          <p className="checks-quiet" role="status">
            Loading checks…
          </p>
        }
        onRetry={() => void audits.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <EmptyState
            title="No checks yet"
            action={
              <Link className="ui-btn" data-size="sm" to={start}>
                Start a check
              </Link>
            }
          >
            A check reads this project&apos;s materials and reports what it
            finds, item by item.
          </EmptyState>
        }
      >
        {(data) => (
          <ListSection>
            {data.items.map((audit) => {
              const root = checkPath(projectId, audit.auditId);
              const state = checkStateLabel(audit.state);
              const stop = describeStopReason(audit);
              const type = auditProfileLabel(audit);
              return (
                <ListRow
                  key={audit.auditId}
                  id={`check-row-${audit.auditId}`}
                  to={root}
                  glyph={<StatusGlyph tone={state.tone} />}
                  title={type}
                  meta={[
                    <StatusChip key="state" tone={state.tone} size="sm">
                      {state.label}
                    </StatusChip>,
                    audit.state === "draft"
                      ? "Not started"
                      : `Runs: ${audit.submittedRunCount.toLocaleString("en-US")} submitted, ${audit.outstandingRunCount.toLocaleString("en-US")} in progress`,
                    <span key="updated">
                      Updated <RecordedTime value={audit.updatedAt} />
                    </span>,
                  ]}
                >
                  {audit.scope.objective ? (
                    <p className="checks-row-objective">
                      {audit.scope.objective}
                    </p>
                  ) : null}
                  {stop === null ? null : (
                    <p
                      className="checks-row-stop"
                      data-tone={stop.tone === "error" ? "blocked" : undefined}
                    >
                      {stop.sentence}
                    </p>
                  )}
                  <div className="checks-row-actions">
                    {audit.state === "waiting_review" ? (
                      <ContextLink
                        className="ui-btn"
                        data-size="sm"
                        returnLabel="Project checks"
                        to={`${root}/reviews?state=pending`}
                      >
                        Review decisions
                      </ContextLink>
                    ) : null}
                    <AuditControls
                      audit={audit}
                      projectName={projectName}
                      compact
                    />
                  </div>
                  <TechnicalDetails summary="Check identity">
                    <p className="checks-identity">
                      <IdChip
                        value={audit.auditId}
                        display={audit.auditId}
                        label={`check ID of ${type}`}
                      />
                      <IdChip
                        value={`${audit.profile.name}@${audit.profile.version}`}
                        display={`${audit.profile.name}@${audit.profile.version}`}
                        label={`check type version of ${type}`}
                      />
                    </p>
                  </TechnicalDetails>
                </ListRow>
              );
            })}
          </ListSection>
        )}
      </QueryView>
      <CursorControls
        label="Project check pages"
        {...pages.controls(audits.data?.page)}
      />
    </div>
  );
}
