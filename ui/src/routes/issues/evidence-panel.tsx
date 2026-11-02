import {
  type ProjectFinding,
  type FindingProposalReceipt,
} from "../../api/audits";
import { useFindingSource } from "./source-data";
import { QueryView } from "../../app/query-view";
import { useId } from "react";

import {
  CapturedExchange,
  LocationList,
} from "../projects/audits/finding-locations";
import { FindingEvidenceFile } from "./evidence-file";
import { Icon } from "../../app/icon";
import { canShowEvidenceSnippet, isCredentialHeader } from "./evidence";

import "./issues.css";

/**
 * Everything the possible issue rests on: authored locations, the captured
 * HTTP exchange (credential-looking header values masked until shown) and
 * the evidence files the check retained.
 */
export function EvidencePanel({ finding }: { finding: ProjectFinding }) {
  const source = useFindingSource(finding);
  return (
    <QueryView
      query={source}
      loading={<p role="status">Loading retained evidence…</p>}
      errorContext="The source could not be loaded."
    >
      {(value) => <SourceEvidence source={value} />}
    </QueryView>
  );
}
export function SourceEvidence({ source }: { source: FindingProposalReceipt }) {
  const id = useId();
  const document = source.document;
  const locations = document.locations ?? [];
  const exchange = document.http_exchange;
  const files = source.projectEvidence ?? [];
  const masked =
    exchange?.attempts.some((attempt) =>
      [...attempt.headers, ...(attempt.response_headers ?? [])].some((header) =>
        isCredentialHeader(header.name),
      ),
    ) ?? false;
  if (locations.length === 0 && exchange === undefined && files.length === 0)
    return (
      <p className="ui-caption">
        This possible issue names no locations and has no captured evidence.
      </p>
    );
  return (
    <div className="issues-evidence-workbench">
      {locations.length === 0 ? null : (
        <section
          className="issues-evidence-section"
          aria-labelledby={`${id}-locations`}
        >
          <h3 id={`${id}-locations`} className="issues-heading">
            Locations
          </h3>
          <LocationList locations={locations} />
        </section>
      )}
      {exchange === undefined ? null : (
        <section
          className="issues-evidence-section"
          aria-labelledby={`${id}-http`}
        >
          <h3 id={`${id}-http`} className="issues-heading">
            Captured HTTP exchange
          </h3>
          {masked ? (
            <p className="ui-caption">
              Values that look like credentials are hidden. Show them one at a
              time.
            </p>
          ) : null}
          <CapturedExchange exchange={exchange} collapsible={false} />
        </section>
      )}
      {files.length === 0 ? null : (
        <section
          className="issues-evidence-section"
          aria-labelledby={`${id}-files`}
        >
          <h3 id={`${id}-files`} className="issues-heading">
            Retained evidence
          </h3>
          {files.map((artifact, index) => (
            <FindingEvidenceFile
              key={`${artifact.ref.namespace}/${artifact.ref.name}@${artifact.ref.revision}`}
              source={source}
              index={index}
              snippet={index < 2 && canShowEvidenceSnippet(artifact)}
            />
          ))}
        </section>
      )}
      {exchange === undefined ? (
        <p className="issues-no-http">
          <Icon name="question-circle" size={14} />
          No HTTP exchanges were captured for this issue.
        </p>
      ) : null}
    </div>
  );
}
