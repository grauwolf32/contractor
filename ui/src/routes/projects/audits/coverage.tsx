import { ASSESSMENTS, GROUPS } from "./assessments";
import { useAuditCoverage } from "./coverage-data";
import { auditCheckTitle } from "./check-title";
import { useMemo, useState } from "react";
import { useLocation, useSearchParams } from "react-router";

import {
  type Audit,
  type AuditCoverageRow,
  type AuditItem,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { ErrorNotice } from "../../artifacts/common";
import type { AuditCollectionQuery } from "./collections";
import { AuditItemDetails } from "./executions";
import { useAuditItems } from "./items-data";
import { LoadMoreControl } from "./load-more";
import { AuditAnchor, AuditMarkdown } from "./shared";
import { RefreshButton } from "../../../app/refresh-button";

const CHECK_HASH_PREFIX = "#check-";

function hashedCheck(hash: string): string | undefined {
  if (!hash.startsWith(CHECK_HASH_PREFIX)) return undefined;
  try {
    return decodeURIComponent(hash.slice(CHECK_HASH_PREFIX.length));
  } catch {
    return undefined;
  }
}

/** Search text of a check's attempts (outcomes, states, Runs). */
function attemptSearchText(item: AuditItem | undefined): string[] {
  if (item === undefined) return [];
  return [
    item.state,
    item.workflowRole,
    item.finalDisposition,
    ...item.attempts.flatMap((attempt) => [
      attempt.state,
      attempt.terminalOutcome,
      attempt.collectionDisposition,
      attempt.runId,
    ]),
  ].filter((value): value is string => typeof value === "string");
}

const EVIDENCE_LABELS: Record<string, string> = {
  artifact: "Artifact",
  observation: "Observation",
  "tool-result": "Tool result",
  "manual-attestation": "Manual review",
  "runtime-metric": "Runtime measurement",
  implementation: "Implementation",
  tests: "Tests",
  "source-analysis": "Source analysis",
  "configuration-review": "Configuration review",
  "documentation-review": "Documentation review",
  "active-test": "Active test",
  "manual-review": "Manual review",
};

function readable(value: string) {
  return EVIDENCE_LABELS[value] ?? value;
}

function firstLine(text: string, limit = 200): string {
  const line =
    text
      .split("\n")
      .map((candidate) => candidate.replace(/^[#>*\-\s]+/u, "").trim())
      .find((candidate) => candidate !== "") ?? "";
  return line.length > limit ? `${line.slice(0, limit)}…` : line;
}

function CheckAttempts({
  audit,
  item,
  items,
}: {
  audit: Audit;
  item: AuditItem | undefined;
  items: AuditCollectionQuery<AuditItem>;
}) {
  if (item !== undefined) return <AuditItemDetails audit={audit} item={item} />;
  if (items.isPending)
    return (
      <p className="loading-copy" role="status">
        Loading attempts…
      </p>
    );
  if (items.error !== null) return <ErrorNotice error={items.error} />;
  if (items.truncated || items.isLoadingMore || items.moreError !== null)
    return (
      <LoadMoreControl
        shown={items.items.length}
        noun="checks"
        truncated={items.truncated}
        loading={items.isLoadingMore}
        error={items.moreError}
        onLoadMore={items.loadMore}
        label="Load more checks"
      />
    );
  return (
    <p className="muted-copy">No attempts are recorded for this check yet.</p>
  );
}

function CheckRow({
  audit,
  row,
  item,
  items,
  open,
  onToggle,
}: {
  audit: Audit;
  row: AuditCoverageRow;
  item: AuditItem | undefined;
  items: AuditCollectionQuery<AuditItem>;
  open: boolean;
  onToggle: (open: boolean) => void;
}) {
  const assessment = ASSESSMENTS[row.coverage.status];
  const details = row.details;
  const conclusion = details?.resultSummary || row.coverage.rationale;
  const completed = new Set(row.coverage.completed);
  const evidence = [
    ...new Set([...row.coverage.requested, ...row.coverage.completed]),
  ];
  const excerpt = firstLine(
    conclusion ?? details?.objective ?? assessment.description,
  );
  return (
    <article
      className={`audit-result-card audit-check-row audit-result-${assessment.group}`}
      aria-label={row.subjectKey}
      id={`check-${row.itemId}`}
    >
      <details
        className="audit-result-reading"
        open={open}
        onToggle={(event) => onToggle(event.currentTarget.open)}
      >
        <summary className="audit-check-summary">
          <span
            className="audit-check-ordinal"
            title={`Check ${row.ordinal + 1}`}
          >
            {row.ordinal + 1}
          </span>
          <h4 className="audit-check-title" title={row.subjectKey}>
            {auditCheckTitle(row)}
          </h4>
          <span
            className={`audit-assessment audit-assessment-${assessment.group}`}
            title={assessment.description}
          >
            {assessment.label}
          </span>
          <span className="audit-result-excerpt" title={excerpt}>
            {excerpt}
          </span>
        </summary>
        <div className="audit-check-body">
          <div className="audit-result-columns">
            <section>
              <h5>Task given to the model</h5>
              {details?.objective ? (
                <AuditMarkdown source={details.objective} />
              ) : (
                <p className="muted-copy">
                  Task text is unavailable for this check.
                </p>
              )}
              {details?.methods.length ? (
                <p className="audit-check-method">
                  Method: {details.methods.map(readable).join(", ")}
                </p>
              ) : null}
            </section>
            <section>
              <h5>Result</h5>
              <p className="audit-assessment-description">
                {assessment.description}
              </p>
              {conclusion ? (
                <AuditMarkdown source={conclusion} />
              ) : row.result ? (
                <p className="muted-copy">
                  No written conclusion is available.
                </p>
              ) : null}
              {row.coverage.rationale &&
              row.coverage.rationale !== conclusion ? (
                <AuditMarkdown source={row.coverage.rationale} />
              ) : null}
              {row.coverage.gaps.length ? (
                <div className="audit-evidence-gaps">
                  <strong>Limitations & missing evidence</strong>
                  <ul>
                    {row.coverage.gaps.map((gap, index) => (
                      <li key={index}>{gap}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </section>
          </div>
          <div
            className="audit-evidence-checklist"
            aria-label="Evidence coverage"
          >
            {evidence.length ? (
              evidence.map((value) => (
                <span
                  key={value}
                  className={
                    completed.has(value)
                      ? "audit-evidence-complete"
                      : "audit-evidence-missing"
                  }
                >
                  <span aria-hidden="true">
                    {completed.has(value) ? "✓" : "○"}
                  </span>{" "}
                  {readable(value)}{" "}
                  <small>
                    · {completed.has(value) ? "collected" : "missing"}
                  </small>
                </span>
              ))
            ) : (
              <span className="muted-copy">
                No evidence requirements listed.
              </span>
            )}
          </div>
          {details?.evidence.length ? (
            <section>
              <h5>Evidence supporting this result</h5>
              <ul className="audit-evidence-list">
                {details.evidence.map((entry) => (
                  <li key={entry.id}>
                    <strong>{readable(entry.kind)}</strong>
                    <AuditMarkdown source={entry.summary} />
                    <small>{entry.id}</small>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
          {details?.taskDocument ? (
            <details className="audit-record-details">
              <summary>Task document</summary>
              <pre className="audit-task-document">
                {JSON.stringify(details.taskDocument, null, 2)}
              </pre>
            </details>
          ) : null}
          <section className="audit-check-attempts" aria-label="Attempts">
            <h5>Attempts</h5>
            {open ? (
              <CheckAttempts audit={audit} item={item} items={items} />
            ) : null}
          </section>
          <div className="audit-check-links">
            <small className="muted-copy">Check ID: {row.itemKey}</small>
          </div>
        </div>
      </details>
    </article>
  );
}

export function AuditCoverage({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const { hash } = useLocation();
  const [params, setParams] = useSearchParams();
  const search = params.get("q") ?? "";
  const group =
    GROUPS.find((candidate) => candidate.id === params.get("result"))?.id ??
    "all";
  const coverage = useAuditCoverage(audit);
  const rows = coverage.items;
  // Rows opened by the reader (or addressed by a `#check-<itemId>` link).
  const [opened, setOpened] = useState<ReadonlySet<string>>(() => {
    const target = hashedCheck(hash);
    return new Set(target === undefined ? [] : [target]);
  });
  // A later `#check-<itemId>` link opens its row once (state adjusted during
  // render, so the closed row is not forced open again on re-render).
  const [seenHash, setSeenHash] = useState(hash);
  if (hash !== seenHash) {
    setSeenHash(hash);
    const target = hashedCheck(hash);
    if (target !== undefined && !opened.has(target))
      setOpened(new Set([...opened, target]));
  }
  // Attempts come from the item collection; read it only once a row is open
  // or a search needs attempt outcomes, under the same cap and polling rules.
  const items = useAuditItems(
    audit,
    api,
    opened.size > 0 || search.trim() !== "",
  );
  const itemsById = useMemo(
    () => new Map(items.items.map((item) => [item.itemId, item])),
    [items.items],
  );
  const filtered = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase();
    return rows.filter(
      (row) =>
        (group === "all" || ASSESSMENTS[row.coverage.status].group === group) &&
        [
          row.subjectKey,
          row.itemKey,
          row.details?.objective,
          row.details?.resultSummary,
          row.coverage.rationale,
          ...row.coverage.gaps,
          ...row.coverage.requested,
          ...(row.details?.evidence.map((entry) => entry.summary) ?? []),
          ...attemptSearchText(itemsById.get(row.itemId)),
        ]
          .filter(Boolean)
          .join(" ")
          .toLocaleLowerCase()
          .includes(needle),
    );
  }, [rows, group, search, itemsById]);

  function filter(key: string, value: string) {
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        if (value === "" || value === "all") next.delete(key);
        else next.set(key, value);
        return next;
      },
      { replace: true, preventScrollReset: true },
    );
  }

  if (coverage.isPending)
    return (
      <p className="loading-copy" role="status">
        Loading checks and results…
      </p>
    );
  if (coverage.error !== null)
    return (
      <div className="panel audit-section-panel">
        <ErrorNotice error={coverage.error} />
        <button
          className="secondary-button"
          disabled={coverage.isFetching}
          onClick={() => {
            setParams(
              (previous) => {
                const next = new URLSearchParams(previous);
                next.delete("auditRevision");
                return next;
              },
              { replace: true },
            );
            void coverage.refetch();
          }}
        >
          Retry loading coverage
        </button>
      </div>
    );
  if (!rows.length)
    return (
      <section className="empty-state panel">
        <h3>No checks yet</h3>
        <p>
          {audit.state === "draft"
            ? "Start the audit to create its tasks and track the results here."
            : "No checks are available for the current round yet."}
        </p>
      </section>
    );

  return (
    <section
      className="audit-coverage-workspace"
      aria-label="Coverage and results"
    >
      <AuditAnchor ready={!coverage.isPending} />
      <div className="section-heading">
        <div>
          <p className="eyebrow">Current round · Tasks and outcomes</p>
          <h3>Coverage and results</h3>
          <p className="muted-copy">
            Open a check to read the task, the conclusion, the evidence and its
            attempts.
          </p>
        </div>
        <RefreshButton
          isFetching={coverage.isFetching}
          onRefresh={() => {
            setParams(
              (previous) => {
                const next = new URLSearchParams(previous);
                next.delete("auditRevision");
                return next;
              },
              { replace: true },
            );
            void coverage.refetch();
          }}
        />
      </div>
      <div
        className="audit-coverage-summary"
        aria-label="Filter checks by result"
      >
        {GROUPS.map((candidate) => (
          <button
            type="button"
            key={candidate.id}
            className={`audit-summary-stat audit-summary-${candidate.id}`}
            aria-pressed={group === candidate.id}
            onClick={() => filter("result", candidate.id)}
          >
            <strong>
              {candidate.id === "all"
                ? rows.length
                : rows.filter(
                    (row) =>
                      ASSESSMENTS[row.coverage.status].group === candidate.id,
                  ).length}
            </strong>
            <span>{candidate.label}</span>
          </button>
        ))}
      </div>
      <p className="audit-coverage-note">
        These results describe individual checks. An audit can finish with
        issues or incomplete checks; excluded checks do not count as passed.
      </p>
      <div className="audit-coverage-toolbar">
        <label>
          Search checks
          <input
            type="search"
            placeholder="Task, result, evidence, attempt outcome or check ID…"
            value={search}
            onChange={(event) => filter("q", event.currentTarget.value)}
          />
        </label>
        <p role="status">
          Showing {filtered.length} of {coverage.truncated ? "≥" : ""}
          {rows.length} checks
        </p>
        {search || group !== "all" ? (
          <button
            className="secondary-button"
            onClick={() =>
              setParams(
                (previous) => {
                  const next = new URLSearchParams(previous);
                  next.delete("q");
                  next.delete("result");
                  return next;
                },
                { replace: true, preventScrollReset: true },
              )
            }
          >
            Clear filters
          </button>
        ) : null}
      </div>
      {filtered.length ? (
        <div className="audit-check-rows">
          {filtered.map((row) => (
            <CheckRow
              key={row.itemId}
              audit={audit}
              row={row}
              item={itemsById.get(row.itemId)}
              items={items}
              open={opened.has(row.itemId)}
              onToggle={(isOpen) =>
                setOpened((current) => {
                  if (current.has(row.itemId) === isOpen) return current;
                  const next = new Set(current);
                  if (isOpen) next.add(row.itemId);
                  else next.delete(row.itemId);
                  return next;
                })
              }
            />
          ))}
        </div>
      ) : (
        <div className="compact-empty panel">
          <strong>No checks match these filters.</strong>
          <p>Try another search or clear the filters to see every check.</p>
        </div>
      )}
      <LoadMoreControl
        shown={rows.length}
        noun="checks"
        truncated={coverage.truncated}
        loading={coverage.isLoadingMore}
        error={coverage.moreError}
        onLoadMore={coverage.loadMore}
      />
    </section>
  );
}
