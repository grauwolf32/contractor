import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router";
import type { components } from "../../../api/generated/public";
import { auditPollInterval, type Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { requireData } from "../../../api/error";
import { MAX_EXPORT_SELECTION } from "../../../api/finding-exports";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls } from "../../../app/cursor-controls";
import { useURLCursorStack } from "../../../app/pagination";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { EmptyState, ListSection } from "../../../ui";
import { IssueRow } from "../../issues/issue-row";
import { auditProfileLabel } from "../../../modules/checks/labels";
import { useAuditProjectionRefresh } from "../../../modules/checks/projection-refresh";
import { FindingExportPanel } from "./finding-export";
import { useFindingExport } from "./finding-export-state";

/** The check-filtered view reads the same canonical Project Finding entities. */
export function AuditFindings({ audit }: { audit: Audit }) {
  return <AuditFindingList key={audit.auditId} audit={audit} />;
}

function AuditFindingList({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const pages = useURLCursorStack({ param: "findingsCursor" });
  const exporter = useFindingExport(audit.auditId);
  const key = [
    ...queryKeys.audits.allFindings(audit.auditId),
    "research",
    pages.cursor,
    audit.revision,
  ];
  const query = useQuery({
    queryKey: key,
    queryFn: async ({ signal }) =>
      requireData(
        await api.request((client) =>
          client.GET("/v1/audits/{auditId}/findings", {
            signal,
            params: {
              path: { auditId: audit.auditId },
              query: {
                limit: 50,
                // A continuation already carries its exact Audit revision.
                ...(pages.cursor
                  ? { cursor: pages.cursor }
                  : { auditRevision: audit.revision }),
              },
            },
          }),
        ),
      ),
    refetchInterval: pollInterval<components["schemas"]["AuditFindingPage"]>({
      interval: 5000,
      settled: () => auditPollInterval([audit]) === false,
    }),
  });
  useAuditProjectionRefresh(audit, key);
  const stale =
    query.error !== null ||
    (query.data !== undefined && query.data.auditRevision !== audit.revision) ||
    (query.data?.items ?? []).some((finding) => {
      const selection = exporter.selected.get(finding.findingId);
      return selection !== undefined && selection.revision !== finding.revision;
    });
  function refresh() {
    pages.reset();
    void client.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
      exact: true,
    });
    void client.resetQueries({
      queryKey: queryKeys.audits.allFindings(audit.auditId),
    });
  }
  return (
    <section className="research-section">
      <header className="research-section-head">
        <div>
          <p className="eyebrow">Canonical Project findings</p>
          <h2>Possible issues</h2>
        </div>
        <Link to={`/projects/${encodeURIComponent(audit.projectId)}/findings`}>
          All Project issues →
        </Link>
      </header>
      <FindingExportPanel
        state={exporter}
        visible={query.data?.items ?? []}
        readPending={query.isPending}
        stale={stale}
        onRefresh={refresh}
      />
      <QueryView query={query}>
        {(data) => (
          <>
            {data.items.length === 0 ? (
              <EmptyState title="No possible issues recorded">
                Collected work can still contain observations, refutations and
                gaps.
              </EmptyState>
            ) : (
              <ListSection>
                {data.items.map((finding) => (
                  <IssueRow
                    key={finding.findingId}
                    finding={finding}
                    trailing={
                      exporter.open ? (
                        <input
                          type="checkbox"
                          aria-label={`Select ${finding.title} for export`}
                          checked={exporter.selected.has(finding.findingId)}
                          disabled={
                            exporter.busy ||
                            stale ||
                            exporter.conflict ||
                            (!exporter.selected.has(finding.findingId) &&
                              exporter.selected.size >= MAX_EXPORT_SELECTION)
                          }
                          onChange={() => exporter.toggle(finding)}
                        />
                      ) : undefined
                    }
                    to={`/issues/${encodeURIComponent(finding.projectId)}/${encodeURIComponent(finding.findingId)}`}
                    severityBadges
                    context={auditProfileLabel(audit)}
                  />
                ))}
              </ListSection>
            )}
            <CursorControls
              label="Possible issue pages"
              {...pages.controls(data.page)}
            />
          </>
        )}
      </QueryView>
    </section>
  );
}
