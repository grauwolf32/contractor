import { pollInterval } from "../../../app/polling";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useId } from "react";
import { Link } from "react-router";

import {
  getAuditFinding,
  listAuditFindings,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { EmptyState, FilterChips, ListSection } from "../../../ui";
import { IssueRow } from "../../issues/issue-row";
import { issueHref, type StateFilter } from "../../issues/links";
import { AuditFindingCard } from "./finding-card";
import {
  DECISION_OPTIONS,
  SEVERITY_OPTIONS,
  STATE_OPTIONS,
  isFindingSeverity,
  isFindingState,
  isVerdictFilter,
} from "./finding-options";
import { AuditQueueError, AuditQueuePage } from "./queue";
import { useAuditQueue } from "./queue-state";
import { AuditAnchor } from "./shared";
import { FindingExportPanel } from "./finding-export";
import { useFindingExport } from "./finding-export-state";
import { MAX_EXPORT_SELECTION } from "../../../api/finding-exports";

import "../../issues/issues.css";

const STATE_CHIPS: readonly { value: StateFilter; label: string }[] = [
  ...STATE_OPTIONS.map((option) => ({
    value: option.value as StateFilter,
    label: option.label,
  })),
  { value: "all", label: "All" },
];

function SectionHead({
  audit,
  headingId,
}: {
  audit: Audit;
  headingId: string;
}) {
  return (
    <header className="issues-section-head">
      <h2 id={headingId} className="issues-section-title">
        Possible issues
      </h2>
      <div className="issues-section-links">
        <ContextLink
          returnLabel="Check possible issues"
          to={`/projects/${encodeURIComponent(audit.projectId)}/findings`}
        >
          All possible issues in this project
        </ContextLink>
      </div>
    </header>
  );
}

/**
 * One possible issue in full inside its check (`?finding=`, optionally with
 * the `?review=` it was opened for). A review that no longer matches the
 * possible issue's revision blocks deciding until the context is refreshed.
 */
function ExactFinding({
  audit,
  findingId,
}: {
  audit: Audit;
  findingId: string;
}) {
  const api = usePublicAPI();
  const finding = useQuery({
    queryKey: ["project-findings", audit.projectId, findingId],
    queryFn: () => getAuditFinding(api, audit.auditId, findingId),
    retry: false,
    refetchInterval: pollInterval({ interval: 20_000 }),
  });
  if (finding.error !== null)
    return <p role="alert">{finding.error.message}</p>;
  if (finding.data === undefined)
    return <p role="status">Loading possible issue…</p>;
  return (
    <>
      <p>
        <Link
          to={issueHref(finding.data, {
            state: "all",
            project: audit.projectId,
          })}
        >
          Open in Issues
        </Link>
      </p>
      <AuditFindingCard finding={finding.data} />
    </>
  );
}

export function AuditFindings({ audit }: { audit: Audit }) {
  return <AuditFindingList key={audit.auditId} audit={audit} />;
}

function AuditFindingList({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const headingId = useId();
  const decisionId = useId();
  const severityId = useId();
  const queue = useAuditQueue();
  const exporter = useFindingExport(audit.auditId);
  const rawState = queue.params.get("state") ?? "";
  const rawVerdict = queue.params.get("verdict") ?? "";
  const rawSeverity = queue.params.get("severity") ?? "";
  const state = isFindingState(rawState) ? rawState : undefined;
  const verdict = isVerdictFilter(rawVerdict) ? rawVerdict : undefined;
  const severity = isFindingSeverity(rawSeverity) ? rawSeverity : undefined;
  const exactId = queue.params.get("finding");
  const findings = useQuery({
    queryKey: [
      ...queryKeys.audits.allFindings(audit.auditId),
      queue.params.toString(),
    ],
    queryFn: () =>
      listAuditFindings(api, audit.auditId, {
        ...queue.request,
        ...(state === undefined ? {} : { state }),
        ...(verdict === undefined ? {} : { verdict }),
        ...(severity === undefined ? {} : { severity }),
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
    return <ExactFinding audit={audit} findingId={exactId} />;
  return (
    <section className="issues-section" aria-labelledby={headingId}>
      <AuditAnchor ready={findings.data !== undefined} />
      <SectionHead audit={audit} headingId={headingId} />
      <div className="issues-section-filters">
        <FilterChips
          label="Filter by state"
          options={STATE_CHIPS}
          value={state ?? "all"}
          onChange={(value) => queue.change("state", value)}
        />
        <div className="issues-filter-row">
          <span className="issues-select-field">
            <label htmlFor={decisionId} className="issues-field-label">
              Decision
            </label>
            <select
              id={decisionId}
              className="issues-select"
              value={verdict ?? ""}
              onChange={(event) => queue.change("verdict", event.target.value)}
            >
              {DECISION_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </span>
          <span className="issues-select-field">
            <label htmlFor={severityId} className="issues-field-label">
              Severity
            </label>
            <select
              id={severityId}
              className="issues-select"
              aria-describedby={`${severityId}-hint`}
              value={severity ?? ""}
              onChange={(event) => queue.change("severity", event.target.value)}
            >
              {SEVERITY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </span>
        </div>
        <p id={`${severityId}-hint`} className="ui-caption">
          Severity is the analyst&apos;s rating. The AI&apos;s suggestion stays
          with each possible issue and is never filtered on.
        </p>
      </div>
      <FindingExportPanel
        state={exporter}
        visible={findings.data?.items ?? []}
        stale={
          findings.error !== null ||
          (findings.data?.auditRevision !== undefined &&
            findings.data.auditRevision !== audit.revision)
        }
        onRefresh={refresh}
      />
      {findings.isPending ? (
        <p className="ui-caption" role="status">
          Loading possible issues…
        </p>
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
            <div className="issues-embedded-empty">
              <EmptyState title="No matching possible issues">
                <p>
                  A successful Run alone does not create or confirm a possible
                  issue.
                </p>
              </EmptyState>
            </div>
          ) : (
            <div className="issues-embedded-list">
              <ListSection>
                {findings.data.items.map((finding) => (
                  <IssueRow
                    key={finding.findingId}
                    id={`finding-${audit.auditId}-${finding.findingId}`}
                    finding={finding}
                    trailing={
                      exporter.open ? (
                        <input
                          type="checkbox"
                          aria-label={`Select ${finding.title} for export`}
                          checked={exporter.selected.has(finding.findingId)}
                          disabled={
                            exporter.busy ||
                            (!exporter.selected.has(finding.findingId) &&
                              exporter.selected.size >= MAX_EXPORT_SELECTION)
                          }
                          onChange={() => exporter.toggle(finding)}
                        />
                      ) : undefined
                    }
                    to={issueHref(finding, {
                      state: state ?? "all",
                      project: audit.projectId,
                      severity,
                    })}
                  />
                ))}
              </ListSection>
            </div>
          )}
        </>
      )}
    </section>
  );
}
