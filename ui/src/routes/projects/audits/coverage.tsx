import { ContextLink } from "../../../app/context-navigation";
import { ASSESSMENTS, GROUPS } from "./assessments";
import { useAuditCoverage } from "./coverage-data";
import { auditCheckTitle } from "./check-title";
import { useMemo } from "react";
import { useSearchParams } from "react-router";

import { type Audit, type AuditCoverageRow } from "../../../api/audits";
import { ErrorNotice } from "../../artifacts/common";
import { AuditMarkdown } from "./shared";
import { RefreshButton } from "../../../app/refresh-button";

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

function CheckRow({ audit, row }: { audit: Audit; row: AuditCoverageRow }) {
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
    >
      <details className="audit-result-reading">
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
              <summary>Exact task document</summary>
              <pre className="audit-task-document">
                {JSON.stringify(details.taskDocument, null, 2)}
              </pre>
            </details>
          ) : null}
          <div className="audit-check-links">
            <ContextLink
              returnLabel="Coverage and results"
              to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/checks#check-${row.itemId}`}
            >
              View attempts & execution details →
            </ContextLink>
            <small className="muted-copy">Check ID: {row.itemKey}</small>
          </div>
        </div>
      </details>
    </article>
  );
}

export function AuditCoverage({ audit }: { audit: Audit }) {
  const [params, setParams] = useSearchParams();
  const search = params.get("q") ?? "";
  const group =
    GROUPS.find((candidate) => candidate.id === params.get("result"))?.id ??
    "all";
  const coverage = useAuditCoverage(audit);
  const rows = useMemo(() => coverage.data ?? [], [coverage.data]);
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
        ]
          .filter(Boolean)
          .join(" ")
          .toLocaleLowerCase()
          .includes(needle),
    );
  }, [rows, group, search]);

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
      <div className="section-heading">
        <div>
          <p className="eyebrow">Current round · Tasks and outcomes</p>
          <h3>Coverage and results</h3>
          <p className="muted-copy">
            Open a check to read the task, the conclusion and the evidence.
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
            placeholder="Task, result, evidence or check ID…"
            value={search}
            onChange={(event) => filter("q", event.currentTarget.value)}
          />
        </label>
        <p role="status">
          Showing {filtered.length} of {rows.length} checks
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
            <CheckRow key={row.itemId} audit={audit} row={row} />
          ))}
        </div>
      ) : (
        <div className="compact-empty panel">
          <strong>No checks match these filters.</strong>
          <p>Try another search or clear the filters to see every check.</p>
        </div>
      )}
    </section>
  );
}
