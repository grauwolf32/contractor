import { useMemo } from "react";
import { Link } from "react-router";

import type { Audit, AuditItem } from "../../../api/audits";
import { ContextLink } from "../../../app/context-navigation";
import { ErrorNotice } from "../../../app/error-notice";
import { compactId } from "../../../app/format";
import { StaleDataWarning } from "../../../app/query-view";
import { capitalize, itemNoun, type ItemKind } from "../../../app/vocabulary";
import { TechnicalDetails } from "../../../ui";
import { auditCheckTitle } from "./check-title";
import type { CheckEntry } from "./check-model";
import { itemHash, sectionPath } from "./check-links";
import type { AuditCollectionQuery } from "./collections";
import { LoadMoreControl } from "./load-more";
import { ExactArtifactLink } from "./shared";

type AuditItemAttempt = AuditItem["attempts"][number];

function RunLink({ attempt }: { attempt: AuditItemAttempt }) {
  if (attempt.runId === undefined) return <>No run</>;
  return (
    <ContextLink
      returnLabel="Check"
      to={`/runs/${encodeURIComponent(attempt.runId)}`}
      title={attempt.runId}
    >
      {attempt.runDeleted ? "Deleted run provenance" : attempt.runId}
    </ContextLink>
  );
}

/**
 * Attempts, produced results and the exact identity of one item, for admins
 * and debugging: the item's technical details on the check page.
 */
export function ItemTechnicalDetails({
  audit,
  entry,
}: {
  audit: Audit;
  entry: CheckEntry;
}) {
  const { item, row } = entry;
  return (
    <TechnicalDetails description="Attempts, task package, results and identifiers.">
      <div className="checks-item-tech" id={`check-${row.itemId}-attempts`}>
        {item === undefined ? (
          <p className="checks-quiet">
            Attempts are not loaded for this {itemNoun(entry.kind, 1)} yet.
          </p>
        ) : item.attempts.length === 0 ? (
          <p className="checks-quiet">No run submitted.</p>
        ) : (
          <div className="checks-table-scroll">
            <table className="checks-table">
              <caption className="ui-visually-hidden">Attempts</caption>
              <thead>
                <tr>
                  <th scope="col">Attempt</th>
                  <th scope="col">State</th>
                  <th scope="col">Outcome</th>
                  <th scope="col">Collection</th>
                  <th scope="col">Run</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <tbody>
                {item.attempts.map((attempt) => (
                  <tr key={attempt.executionItemId}>
                    <td data-label="Attempt">{attempt.itemAttempt}</td>
                    <td data-label="State">{attempt.state}</td>
                    <td data-label="Outcome">
                      {attempt.terminalOutcome ?? "not finished"}
                    </td>
                    <td data-label="Collection">
                      {attempt.collectionDisposition ?? "not collected"}
                    </td>
                    <td data-label="Run">
                      <RunLink attempt={attempt} />
                    </td>
                    <td data-label="Result">
                      {attempt.result === undefined ? (
                        "None"
                      ) : (
                        <ExactArtifactLink
                          projectId={audit.projectId}
                          artifact={attempt.result}
                          label="Produced result"
                        />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {item === undefined ? null : (
          <div className="checks-artifact-list">
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
        )}
        <dl className="checks-facts">
          <div>
            <dt>Item key</dt>
            <dd>
              <code className="checks-mono">{row.itemKey}</code>
            </dd>
          </div>
          <div>
            <dt>Subject</dt>
            <dd>
              <code className="checks-mono">{row.subjectKey}</code>
            </dd>
          </div>
          {item === undefined ? null : (
            <>
              <div>
                <dt>Item state</dt>
                <dd>{item.state}</dd>
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
                    <code className="checks-mono">
                      {item.origin.standard.scheme}@
                      {item.origin.standard.version}/
                      {item.origin.standard.mappingKey}
                    </code>
                  </dd>
                </div>
              )}
            </>
          )}
          <div>
            <dt>Check ID</dt>
            <dd>
              <code className="checks-mono">{audit.auditId}</code>
            </dd>
          </div>
        </dl>
        {row.details?.taskDocument === undefined ? null : (
          <details className="checks-disclosure">
            <summary>Task document</summary>
            <pre className="checks-code">
              {JSON.stringify(row.details.taskDocument, null, 2)}
            </pre>
          </details>
        )}
      </div>
    </TechnicalDetails>
  );
}

/**
 * The runs the check started, one row per attempt, with links to each run
 * and to its item. Generic Runs stay authoritative for execution.
 */
export function AuditRuns({
  audit,
  kind,
  items,
  entries,
}: {
  audit: Audit;
  kind: ItemKind;
  items: AuditCollectionQuery<AuditItem>;
  entries: readonly CheckEntry[];
}) {
  const attempts = useMemo(
    () =>
      items.items.flatMap((item) =>
        item.attempts.map((attempt) => ({ item, attempt })),
      ),
    [items.items],
  );
  const rows = useMemo(
    () => new Map(entries.map((entry) => [entry.row.itemId, entry.row])),
    [entries],
  );
  const noun = capitalize(itemNoun(kind, 1));
  if (items.isPending)
    return (
      <p className="checks-quiet" role="status">
        Loading runs…
      </p>
    );
  if (items.error !== null && items.items.length === 0)
    return <ErrorNotice error={items.error} />;
  return (
    <div className="checks-runs">
      <p className="checks-runs-intro">
        <Link to="/runs">Global Runs →</Link>
      </p>
      {items.error === null ? null : (
        <StaleDataWarning
          error={items.error}
          onRetry={() => void items.refetch()}
          retryPending={items.isFetching}
        />
      )}
      {attempts.length === 0 ? (
        <p className="checks-quiet">No runs submitted yet.</p>
      ) : (
        <div className="checks-table-scroll">
          <table className="checks-table">
            <thead>
              <tr>
                <th scope="col">{noun}</th>
                <th scope="col">Attempt</th>
                <th scope="col">Run</th>
                <th scope="col">Technical outcome</th>
                <th scope="col">Collection</th>
              </tr>
            </thead>
            <tbody>
              {attempts.map(({ item, attempt }) => (
                <tr key={attempt.executionItemId}>
                  <td data-label={noun}>
                    <ContextLink
                      returnLabel="Check runs"
                      to={`${sectionPath(audit.projectId, audit.auditId, "coverage")}${itemHash(item.itemId)}`}
                      title={item.subjectKey}
                    >
                      {auditCheckTitle(rows.get(item.itemId) ?? item)}
                    </ContextLink>
                  </td>
                  <td data-label="Attempt">{attempt.itemAttempt}</td>
                  <td data-label="Run">
                    {attempt.runId === undefined ? (
                      "—"
                    ) : (
                      <ContextLink
                        returnLabel="Check"
                        to={`/runs/${encodeURIComponent(attempt.runId)}`}
                        title={attempt.runId}
                        aria-label={attempt.runId}
                      >
                        {compactId(attempt.runId)}
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
        noun={itemNoun(kind, 2)}
        truncated={items.truncated}
        loading={items.isLoadingMore}
        error={items.moreError}
        onLoadMore={items.loadMore}
      />
    </div>
  );
}
