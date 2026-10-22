import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  getAudit,
  listAuditFindingProvenance,
  type Audit,
  type AuditFinding,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import {
  listFindingCandidates,
  listFindingMembers,
  type FindingPreview,
} from "../../api/finding-correlations";
import { queryKeys } from "../../api/query-keys";
import { ContextLink } from "../../app/context-navigation";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import {
  findingStateLabel,
  severityLabel,
  verdictLabel,
  checkStateLabel,
} from "../../app/vocabulary";
import { Field, StatusChip, TechnicalDetails } from "../../ui";
import { auditProfileLabel } from "../projects/audits/labels";
import { CorrelationPages } from "./correlation-pages";
import { issueEvidencePath, issuePath } from "./links";

const unavailable = {
  not_recorded: "Matching metadata was not recorded for this observation.",
  no_qualified_key:
    "This observation has no qualified issue key. Possible matches may still be available.",
  primary_key_ambiguous:
    "More than one issue key applies. Automatic equality is unavailable; possible matches remain separate.",
};

/** Exact equality has no winning observation and does not decide vulnerability truth. */
export function FindingDuplicates({
  audit,
  finding,
}: {
  audit: Audit;
  finding: AuditFinding;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details
      className="issues-duplicates"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>Duplicates</summary>
      {open ? (
        <DuplicateContents
          key={`${audit.revision}:${finding.revision}:${finding.correlation?.projectRevision ?? 0}`}
          audit={audit}
          finding={finding}
        />
      ) : null}
    </details>
  );
}

function DuplicateContents({
  audit,
  finding,
}: {
  audit: Audit;
  finding: AuditFinding;
}) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const mark = finding.correlation;
  return (
    <div className="issues-duplicates-body">
      <p className="ui-caption">
        Each observation keeps its own check, evidence and analyst rating.
        Matching does not confirm an issue.
      </p>
      {mark?.availability !== "available" ? (
        <p>{unavailable[mark?.reason ?? "not_recorded"]}</p>
      ) : (
        <>
          <p>
            Exact matches in this project. All observations keep their own
            results.
          </p>
          <CorrelationPages
            queryKey={queryKeys.findingCorrelations.members(
              audit.projectId,
              mark.groupKey,
              mark.projectRevision,
            )}
            load={(cursor, signal) =>
              listFindingMembers(
                api,
                audit.projectId,
                mark.groupKey,
                cursor === undefined ? {} : { cursor },
                signal,
              )
            }
            onRestart={() =>
              void client.invalidateQueries({
                queryKey: queryKeys.audits.detail(audit.auditId),
              })
            }
            moreLabel="Load more observations"
          >
            {(pages) => {
              const items = pages.flatMap((page) => page.items);
              const partial = pages.at(-1)!.page.hasMore;
              return (
                <>
                  <p className="ui-caption">
                    Showing {items.length} retained observations
                    {partial ? " · partial list" : " · complete list"}.
                    Observation counts do not measure unique or confirmed
                    issues.
                  </p>
                  <ul className="issues-duplicates-list">
                    {items.map((item) => (
                      <li key={`${item.auditId}:${item.findingId}`}>
                        <Observation
                          item={item}
                          finding={finding}
                          projectId={audit.projectId}
                        />
                      </li>
                    ))}
                  </ul>
                </>
              );
            }}
          </CorrelationPages>
        </>
      )}
      <TechnicalDetails description="The Server's matching basis and exact source metadata.">
        <pre className="issues-duplicates-code">
          {JSON.stringify(
            {
              correlation: mark ?? null,
              fingerprint: finding.fingerprint ?? null,
              anchors: finding.anchors ?? null,
            },
            null,
            2,
          )}
        </pre>
      </TechnicalDetails>
      <PossibleMatches audit={audit} finding={finding} />
    </div>
  );
}

function Observation({
  item,
  finding,
  projectId,
}: {
  item: FindingPreview;
  finding: AuditFinding;
  projectId: string;
}) {
  const current =
    item.auditId === finding.auditId && item.findingId === finding.findingId;
  const state = findingStateLabel(item.state);
  return (
    <article className="issues-duplicate-observation">
      <p className="issues-duplicates-head">
        <ContextLink
          to={issuePath(item.auditId, item.findingId)}
          returnLabel="Related observations"
        >
          {item.title}
          {item.titleTruncated ? "…" : ""}
        </ContextLink>
        {current ? (
          <StatusChip tone="neutral" size="sm">
            This observation
          </StatusChip>
        ) : null}
        <StatusChip tone={state.tone} size="sm">
          {state.label}
        </StatusChip>
      </p>
      <div className="issues-duplicate-rating">
        <span>
          Analyst:{" "}
          <strong>
            {item.analystVerdict === undefined
              ? "Not reviewed"
              : verdictLabel(item.analystVerdict).label}
          </strong>
        </span>
        <span>
          Severity:{" "}
          <strong>
            {item.analystSeverity === undefined
              ? "Not set"
              : severityLabel(item.analystSeverity)}
          </strong>
        </span>
      </div>
      <p className="ui-caption">
        AI suggestion:{" "}
        {item.severitySuggestion === ""
          ? "None"
          : severityLabel(item.severitySuggestion)}{" "}
        · Verification: {item.semanticAssessment ?? "Not verified yet"}
      </p>
      <div className="issues-duplicates-actions">
        <ContextLink
          to={issueEvidencePath(item.auditId, item.findingId)}
          returnLabel="Related observations"
        >
          Evidence
        </ContextLink>
        <span className="ui-caption">
          {item.auditId === finding.auditId
            ? "In this check"
            : "In another check"}{" "}
          ·{" "}
          {item.origin.receiptId === finding.firstProposal.receiptId
            ? "Same source proposal as this finding"
            : "Different source proposal"}
        </span>
      </div>
      <ObservationContext item={item} projectId={projectId} />
    </article>
  );
}

function ObservationContext({
  item,
  projectId,
}: {
  item: FindingPreview;
  projectId: string;
}) {
  const api = usePublicAPI();
  const [open, setOpen] = useState(false);
  const check = useQuery({
    queryKey: queryKeys.audits.detail(item.auditId),
    queryFn: ({ signal }) => getAudit(api, item.auditId, signal),
    enabled: open,
  });
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Check context</summary>
      {open ? (
        <>
          <ContextLink
            to={`/projects/${encodeURIComponent(projectId)}/audits/${encodeURIComponent(item.auditId)}`}
            returnLabel="Related observations"
          >
            Open check
          </ContextLink>
          <QueryView
            query={check}
            errorContext="Check context could not be loaded."
          >
            {(audit) => (
              <>
                <p>
                  {auditProfileLabel(audit)} ·{" "}
                  {checkStateLabel(audit.state).label} · Created{" "}
                  <RecordedTime value={audit.createdAt} />
                </p>
                <p className="ui-caption">
                  Coverage has not been compared. A missing match does not mean
                  an issue is new or fixed, or that a check is clean.
                </p>
                <TechnicalDetails description="Exact check, baseline and source configuration.">
                  <pre className="issues-duplicates-code">
                    {JSON.stringify(
                      {
                        auditId: audit.auditId,
                        revision: audit.revision,
                        profile: audit.profile,
                        baseline: audit.baseline ?? null,
                        inputs: audit.inputs,
                        origin: item.origin,
                        findingRevision: item.revision,
                        assessmentId: item.assessmentId,
                        decisionId: item.decisionId,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </TechnicalDetails>
              </>
            )}
          </QueryView>
        </>
      ) : null}
    </details>
  );
}

function PossibleMatches({
  audit,
  finding,
}: {
  audit: Audit;
  finding: AuditFinding;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Possible matches</summary>
      {open ? <CandidateContents audit={audit} finding={finding} /> : null}
    </details>
  );
}

function CandidateContents({
  audit,
  finding,
}: {
  audit: Audit;
  finding: AuditFinding;
}) {
  const api = usePublicAPI();
  const [receipt, setReceipt] = useState(finding.firstProposal.receiptId);
  const [scope, setScope] = useState<"project" | "audit">("project");
  return (
    <div className="issues-duplicates-body">
      <p className="ui-caption">
        Possible matches are suggestions, separate from automatic duplicates.
      </p>
      <CorrelationPages
        queryKey={[
          ...queryKeys.audits.provenance(
            audit.auditId,
            finding.findingId,
            audit.revision,
            finding.revision,
          ),
          "correlation-source-picker",
        ]}
        load={(cursor, signal) =>
          listAuditFindingProvenance(
            api,
            audit.auditId,
            finding.findingId,
            {
              auditRevision: audit.revision,
              findingRevision: finding.revision,
              ...(cursor === undefined ? {} : { cursor }),
            },
            signal,
          )
        }
        refreshable={false}
        moreLabel="Load more source proposals"
        onRestart={() => setReceipt(finding.firstProposal.receiptId)}
      >
        {(pages) => {
          const records = pages
            .flatMap((page) => page.items)
            .filter(
              (record) =>
                record.relation === "contributing" &&
                record.receiptId !== finding.firstProposal.receiptId,
            );
          const receipts = [
            ...new Map(
              records.map((record) => [record.receiptId, record]),
            ).values(),
          ];
          return (
            <>
              <details className="issues-duplicate-options">
                <summary>Search options</summary>
                <div className="issues-duplicates-body">
                  <Field label="Search scope">
                    {(control) => (
                      <select
                        {...control}
                        value={scope}
                        onChange={(event) =>
                          setScope(event.target.value as typeof scope)
                        }
                      >
                        <option value="project">
                          All checks in this project
                        </option>
                        <option value="audit">This check</option>
                      </select>
                    )}
                  </Field>
                  <Field
                    label="Source proposal"
                    hint={
                      pages.at(-1)!.page.hasMore
                        ? "More source proposals may be available on later pages."
                        : "All retained source choices loaded."
                    }
                  >
                    {(control) => (
                      <select
                        {...control}
                        value={receipt}
                        onChange={(event) => setReceipt(event.target.value)}
                      >
                        <option value={finding.firstProposal.receiptId}>
                          Original proposal
                        </option>
                        {receipts.map((record, index) => (
                          <option
                            key={record.receiptId}
                            value={record.receiptId}
                          >
                            Supporting proposal {index + 1} ·{" "}
                            {record.origin.workflow.name}
                          </option>
                        ))}
                      </select>
                    )}
                  </Field>
                </div>
              </details>
              <CandidateResults
                audit={audit}
                finding={finding}
                receipt={receipt}
                scope={scope}
              />
            </>
          );
        }}
      </CorrelationPages>
    </div>
  );
}

function CandidateResults({
  audit,
  finding,
  receipt,
  scope,
}: {
  audit: Audit;
  finding: AuditFinding;
  receipt: string;
  scope: "project" | "audit";
}) {
  const api = usePublicAPI();
  return (
    <CorrelationPages
      key={`${receipt}:${scope}`}
      queryKey={queryKeys.findingCorrelations.candidates(
        audit.auditId,
        finding.findingId,
        receipt,
        scope,
        audit.revision,
      )}
      load={(cursor, signal) =>
        listFindingCandidates(
          api,
          audit.auditId,
          finding.findingId,
          {
            sourceReceiptId: receipt,
            scope,
            ...(cursor === undefined ? {} : { cursor }),
          },
          signal,
        )
      }
      moreLabel="Continue searching"
    >
      {(pages) => {
        const all = pages.flatMap((page) => page.items);
        const possible = all.filter((item) => !item.automaticallyCorrelated);
        const exact = all.length - possible.length;
        const last = pages.at(-1)!;
        return (
          <>
            {last.source.hintsAvailability === "unavailable" ? (
              <p>
                {last.source.hintsReason === "discarded"
                  ? "Matching metadata was discarded with the source result."
                  : "Matching metadata was not recorded for this source result."}{" "}
                No match is not proof of uniqueness.
              </p>
            ) : null}
            {!last.source.familiesAvailable ? (
              <p className="ui-caption">Issue-family hints are unavailable.</p>
            ) : null}
            <p className="ui-caption">
              {possible.length} possible matches loaded
              {last.page.hasMore
                ? " · partial search"
                : " · search complete"}. {exact} automatic matches encountered;
              browse them in Duplicates above.
            </p>
            {last.workLimited && last.page.hasMore ? (
              <p className="ui-caption">
                This page reached its search-work limit. Continue searching even
                if it contains no matches.
              </p>
            ) : null}
            <ul className="issues-duplicates-list">
              {possible.map((candidate) => (
                <li
                  key={`${candidate.finding.auditId}:${candidate.finding.findingId}`}
                >
                  <StatusChip tone="neutral" size="sm">
                    Possible match
                  </StatusChip>
                  <Observation
                    item={candidate.finding}
                    finding={finding}
                    projectId={audit.projectId}
                  />
                  <p>
                    {candidate.reasons
                      .map((reason) =>
                        reason.kind === "structural-site"
                          ? "Same structural site; issue context may differ."
                          : reason.kind === "selected-producer-equality"
                            ? "Selected producer key matches; the primary issue groups differ."
                            : "Same subject and issue family; issue equality is unproven.",
                      )
                      .join(" ")}
                  </p>
                  {candidate.reasonsTruncated ? (
                    <p className="ui-caption">
                      Only some matching reasons are shown.
                    </p>
                  ) : null}
                  <TechnicalDetails description="Exact matching reasons and limitations.">
                    <pre className="issues-duplicates-code">
                      {JSON.stringify(candidate.reasons, null, 2)}
                    </pre>
                  </TechnicalDetails>
                </li>
              ))}
            </ul>
          </>
        );
      }}
    </CorrelationPages>
  );
}
