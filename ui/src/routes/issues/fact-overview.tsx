import { useState } from "react";
import type {
  ProjectFinding,
  ProjectFindingEditResult,
} from "../../api/audits";
import { TechnicalDetails } from "../../ui";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { DecisionMarkdown } from "../decisions/markdown";
import { FindingDecision } from "../decisions";
import { SourceEvidence } from "./evidence-panel";
import { useFindingSource } from "./source-data";
import { IssueRecordedDecision } from "./overview";

export function FactOverview({
  finding,
  onDecided,
}: {
  finding: ProjectFinding;
  onDecided?: ((result: ProjectFindingEditResult) => void) | undefined;
}) {
  const source = useFindingSource(finding);
  const [assess, setAssess] = useState(false);
  const claim = finding.claim;
  const limitations = [
    ...(claim?.limitations ?? []),
    ...(claim?.preconditions ?? []),
  ].filter((s) => s.trim());
  return (
    <>
      <section className="fact-observation" aria-label="Reported observation">
        <h3 className="issues-section-label">Observation</h3>
        {claim?.description ? (
          <DecisionMarkdown source={claim.description} />
        ) : (
          <p className="ui-caption">No description was recorded.</p>
        )}
      </section>
      <section className="fact-source" aria-label="Original source">
        <h3 className="issues-section-label">Original source</h3>
        <QueryView
          query={source}
          loading={<p role="status">Loading source…</p>}
          errorContext="The original source could not be loaded."
          onRetry={() => void source.refetch()}
        >
          {(value) => (
            <>
              <p className="fact-context">
                {value.origin.workflow.name}@{value.origin.workflow.version} ·{" "}
                <RecordedTime value={value.createdAt} />
              </p>
              {value.document.title !== finding.title && (
                <p>{value.document.title}</p>
              )}
              <SourceEvidence source={value} showEmptyHttp={false} />
            </>
          )}
        </QueryView>
      </section>
      {limitations.length > 0 && (
        <section className="fact-limitations">
          <h3 className="issues-section-label">Limitations and conditions</h3>
          <ul>
            {limitations.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </section>
      )}
      {claim?.standard_refs.length ? (
        <section className="fact-limitations">
          <h3 className="issues-section-label">Standard references</h3>
          <ul>
            {claim.standard_refs.map((ref, i) => (
              <li key={i}>
                {ref.requirement_id} · {ref.scheme}@{ref.version}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      <IssueRecordedDecision finding={finding} />
      <details
        className="fact-assessment"
        onToggle={(event) => setAssess(event.currentTarget.open)}
      >
        <summary>Analyst assessment</summary>
        <FindingDecision
          finding={finding}
          presentation="review"
          shortcuts={assess}
          onDecided={onDecided}
        />
      </details>
      <TechnicalDetails>
        <p>
          Project {finding.projectId} · Finding {finding.findingId} · Revision{" "}
          {finding.revision}
        </p>
      </TechnicalDetails>
    </>
  );
}
