import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import { RESOURCE_ID } from "../../api/contract";
import {
  cancelScanSweep,
  getScanSweep,
  listScanSweepMembers,
  listScanSweeps,
  purgeScanSweep,
  sweepKeys,
  sweepSettled,
  type ScanSweep,
} from "../../api/scan-sweeps";
import { CursorControls } from "../../app/cursor-controls";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import { formatShortDateTime } from "../../app/format";
import { useGuardedNavigate } from "../../app/page-location";
import { useURLCursorStack } from "../../app/pagination";
import { pollInterval } from "../../app/polling";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { readableValue } from "../../app/vocabulary";
import {
  AlertDialog,
  Button,
  ButtonLink,
  DetailHeader,
  DetailPane,
  EmptyState,
  Facts,
  ListPane,
  ListRow,
  ListSection,
  PaneLayout,
  StatusChip,
  TechnicalDetails,
} from "../../ui";
import "./scans.css";

const COUNT_LABELS: Record<keyof ScanSweep["counts"], string> = {
  total: "Projects",
  pending: "Waiting to start",
  admitted: "Scans started",
  skipped: "Skipped",
  failed: "Could not start",
  cancelled: "Cancelled before start",
  unsettled: "Still in progress",
};

export function RuleSweepsRoute() {
  useDocumentTitle("Rule sweeps");
  const api = usePublicAPI();
  const { sweepId } = useParams();
  const pages = useURLCursorStack({ param: "sweepsCursor" });
  const list = useQuery({
    queryKey: sweepKeys.list(pages.cursor),
    queryFn: () => listScanSweeps(api, pages.cursor),
  });
  return (
    <PaneLayout
      listLabel="Rule sweeps"
      detailLabel="Sweep progress"
      showDetail={sweepId !== undefined}
      backLink={{ to: "/checks/sweeps", label: "Back to sweeps" }}
      list={
        <ListPane
          title="Rule sweeps"
          subtitle="One rule across a captured Project scope"
          actions={
            <ButtonLink variant="primary" size="sm" to="/checks/sweeps/new">
              Start sweep
            </ButtonLink>
          }
          toolbar={
            <div className="ui-toolbar">
              <ButtonLink to="/checks" size="sm">
                Checks
              </ButtonLink>
              <ButtonLink to="/checks/web/new" size="sm">
                Web scan
              </ButtonLink>
              <RefreshButton
                onRefresh={() => void list.refetch()}
                isFetching={list.isFetching}
              />
            </div>
          }
        >
          <QueryView
            query={list}
            isEmpty={(value) => value.items.length === 0}
            empty={<EmptyState title="No rule sweeps yet" />}
          >
            {(value) => (
              <ListSection>
                {value.items.map((sweep) => (
                  <ListRow
                    key={sweep.sweepId}
                    title={`Sweep · ${formatShortDateTime(sweep.createdAt)}`}
                    meta={`${readableValue(sweep.state)} · ${sweep.counts.total} Projects · ${sweep.counts.unsettled} unsettled`}
                    selected={sweep.sweepId === sweepId}
                    to={`/checks/sweeps/${encodeURIComponent(sweep.sweepId)}`}
                  />
                ))}
              </ListSection>
            )}
          </QueryView>
          <CursorControls
            label="Sweep pages"
            {...pages.controls(list.data?.page)}
          />
        </ListPane>
      }
      detail={
        sweepId === undefined ? (
          <DetailPane>
            <EmptyState title="Choose a sweep">
              See admission outcomes and open each Project's scan result.
            </EmptyState>
          </DetailPane>
        ) : !RESOURCE_ID.test(sweepId) ? (
          <ErrorNotice error={new Error("Sweep ID is invalid")} />
        ) : (
          <SweepDetail key={sweepId} id={sweepId} />
        )
      }
    />
  );
}
function SweepDetail({ id }: { id: string }) {
  const api = usePublicAPI();
  const cache = useQueryClient();
  const navigate = useGuardedNavigate();
  const pages = useURLCursorStack({ param: "membersCursor" });
  const [confirm, setConfirm] = useState(false);
  const sweep = useQuery({
    queryKey: sweepKeys.detail(id),
    queryFn: ({ signal }) => getScanSweep(api, id, signal),
    refetchInterval: pollInterval<ScanSweep>({
      interval: 2000,
      settled: sweepSettled,
    }),
  });
  const settled = sweep.data !== undefined && sweepSettled(sweep.data);
  const members = useQuery({
    queryKey: [...sweepKeys.members(id), pages.cursor ?? null],
    queryFn: ({ signal }) =>
      listScanSweepMembers(api, id, pages.cursor, signal),
    refetchInterval: pollInterval<
      Awaited<ReturnType<typeof listScanSweepMembers>>
    >({ interval: 2000, settled: () => settled }),
  });
  useEffect(() => {
    if (settled) {
      void cache.invalidateQueries({ queryKey: sweepKeys.members(id) });
      void cache.invalidateQueries({ queryKey: ["scan-sweeps", "list"] });
    }
  }, [cache, id, settled]);
  const refresh = async () => {
    await Promise.all([
      cache.invalidateQueries({ queryKey: sweepKeys.detail(id) }),
      cache.invalidateQueries({ queryKey: sweepKeys.members(id) }),
      cache.invalidateQueries({ queryKey: ["scan-sweeps", "list"] }),
    ]);
  };
  const cancel = useMutation({
    mutationFn: () => cancelScanSweep(api, id),
    onSuccess: (value) => cache.setQueryData(sweepKeys.detail(id), value),
    onSettled: refresh,
  });
  const purge = useMutation({
    mutationFn: () => purgeScanSweep(api, id),
    onSuccess: async () => {
      await cache.invalidateQueries({ queryKey: sweepKeys.all });
      await navigate("/checks/sweeps");
    },
    onError: () => void refresh(),
  });
  return (
    <DetailPane>
      <QueryView query={sweep}>
        {(value) => (
          <>
            <DetailHeader
              title="Rule sweep"
              status={
                <StatusChip
                  tone={
                    value.state === "active"
                      ? "info"
                      : value.state === "cancelling"
                        ? "warning"
                        : "neutral"
                  }
                >
                  {readableValue(value.state)}
                </StatusChip>
              }
              actions={
                <div className="ui-toolbar">
                  <Button
                    size="sm"
                    disabled={value.state !== "active" || purge.isPending}
                    pending={cancel.isPending}
                    onClick={() => cancel.mutate()}
                  >
                    Cancel sweep
                  </Button>
                  {settled ? (
                    <Button
                      size="sm"
                      variant="danger"
                      disabled={cancel.isPending}
                      onClick={() => setConfirm(true)}
                    >
                      Delete sweep
                    </Button>
                  ) : null}
                  <RefreshButton
                    onRefresh={() => void refresh()}
                    isFetching={sweep.isFetching || members.isFetching}
                  />
                </div>
              }
            />
            <div className="scan-detail ui-stack">
              <Facts
                items={Object.entries(value.counts).map(([name, count]) => ({
                  term: COUNT_LABELS[name as keyof ScanSweep["counts"]],
                  value: count,
                }))}
              />
              <p className="ui-caption">
                Scans started counts the checks that were launched. Their final
                states and results appear below.
              </p>
              {cancel.error === null ? null : (
                <ErrorNotice error={cancel.error} />
              )}
              <QueryView
                query={members}
                isEmpty={(page) => page.items.length === 0}
                empty={<EmptyState title="No members on this page" />}
              >
                {(page) => (
                  <ListSection title="Project scans">
                    {page.items.map((member) => (
                      <ListRow
                        key={member.ordinal}
                        title={
                          <Link
                            to={`/projects/${encodeURIComponent(member.projectId)}`}
                          >
                            {member.projectId}
                          </Link>
                        }
                        meta={
                          <>
                            <span>{readableValue(member.state)}</span>
                            {member.runState === null ? null : (
                              <span>
                                {" "}
                                · Run {readableValue(member.runState)}
                              </span>
                            )}
                            {member.reason === "" ? null : (
                              <span> · {readableValue(member.reason)}</span>
                            )}
                          </>
                        }
                      >
                        <p className="scan-target ui-caption">
                          {member.target || "No target captured"}
                        </p>
                        {member.runId === null ? null : (
                          <Link
                            to={`/runs/${encodeURIComponent(member.runId)}`}
                          >
                            Open scan and results
                          </Link>
                        )}
                      </ListRow>
                    ))}
                  </ListSection>
                )}
              </QueryView>
              <CursorControls
                label="Member pages"
                {...pages.controls(members.data?.page)}
              />
              <TechnicalDetails>
                <p>
                  Sweep: <code>{value.sweepId}</code>
                </p>
                <p>
                  Workflow: <code>{value.workflow}</code>
                </p>
                <p>
                  Snapshot: <code>{value.snapshotDigest}</code>
                </p>
              </TechnicalDetails>
            </div>
            {confirm ? (
              <AlertDialog
                title="Delete this sweep?"
                confirmLabel="Delete sweep"
                pending={purge.isPending}
                onConfirm={() => purge.mutate()}
                onCancel={() => setConfirm(false)}
                error={
                  purge.error === null ? undefined : (
                    <ErrorNotice error={purge.error} />
                  )
                }
              >
                <p>
                  Removes the sweep's metadata and retained input binding. Its
                  Project scans, reports and your original rulepack stay
                  available.
                </p>
              </AlertDialog>
            ) : null}
          </>
        )}
      </QueryView>
    </DetailPane>
  );
}
