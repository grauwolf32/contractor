import { useMemo } from "react";
import { Link } from "react-router";

import { type Audit, type AuditItem } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { ContextLink } from "../../../app/context-navigation";
import { ErrorNotice } from "../../artifacts/common";
import { StateBadge } from "../../runs/components";
import { auditCheckTitle } from "./check-title";
import { useAuditCoverage } from "./coverage-data";
import { useAuditItems } from "./items-data";
import { LoadMoreControl } from "./load-more";
import { ExactArtifactLink } from "./shared";

/**
 * Attempts, produced artifacts and exact identity of one check, as read from
 * the item collection. Rendered inside the opened Coverage row.
 */
export function AuditItemDetails({
  audit,
  item,
}: {
  audit: Audit;
  item: AuditItem;
}) {
  return (
    <div className="audit-item-details" id={`check-${item.itemId}-attempts`}>
      <p className="audit-item-status">
        <StateBadge state={item.state} />
        <span className="muted-copy">
          {item.workflowRole} · {item.finalDisposition ?? "Pending assessment"}{" "}
          · {item.attempts.length} attempts
        </span>
      </p>
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
              {attempt.result === undefined ? null : (
                <ExactArtifactLink
                  projectId={audit.projectId}
                  artifact={attempt.result}
                  label="Produced result"
                />
              )}
            </li>
          ))}
        </ol>
      )}
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
                {item.origin.standard.scheme}@{item.origin.standard.version}/
                {item.origin.standard.mappingKey}
              </code>
            </dd>
          </div>
        )}
      </dl>
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
      items.items.flatMap((item) =>
        item.attempts.map((attempt) => ({ item, attempt })),
      ),
    [items.items],
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
                      to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/coverage#check-${encodeURIComponent(item.itemId)}`}
                      title={item.subjectKey}
                    >
                      {auditCheckTitle(
                        coverage.items.find(
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
      <LoadMoreControl
        shown={items.items.length}
        noun="checks"
        truncated={items.truncated}
        loading={items.isLoadingMore}
        error={items.moreError}
        onLoadMore={items.loadMore}
      />
    </section>
  );
}
