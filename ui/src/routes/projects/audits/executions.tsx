import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { Link, useLocation } from "react-router";

import { collectAuditPages } from "../../../api/audit-collections";
import {
  auditNeedsPolling,
  listAuditItems,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { ErrorNotice } from "../../artifacts/common";
import { StateBadge } from "../../runs/components";
import { auditCheckTitle } from "./check-title";
import { useAuditCoverage } from "./coverage-data";
import { useAuditProjectionRefresh } from "./projection-refresh";
import { AuditAnchor, ExactArtifactLink } from "./shared";

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

export function AuditChecks({
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

export function AuditRuns({
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
