import { useId } from "react";

import type { AuditFinding } from "../../api/audits";
import { formatBytes } from "../../app/format";
import {
  CapturedExchange,
  LocationList,
} from "../projects/audits/finding-locations";
import { isCredentialHeader } from "./evidence";

import "./issues.css";

function EvidenceFiles({ finding }: { finding: AuditFinding }) {
  const document = finding.firstProposal.document;
  const responseBody = document.http_exchange?.response_body_evidence_id;
  return (
    <ul className="issues-file-list">
      {finding.firstProposal.evidence.map((artifact, index) => {
        const evidenceId = document.evidence_ids[index];
        return (
          <li
            key={`${artifact.ref.namespace}/${artifact.ref.name}@${artifact.ref.revision}`}
            title={`Revision ${artifact.ref.revision} · ${artifact.digest}`}
          >
            <span className="issues-mono">
              {evidenceId ?? `evidence ${index + 1}`}
            </span>
            <span className="issues-file-name">
              {artifact.ref.namespace}/{artifact.ref.name}
            </span>
            <span className="issues-quiet">
              {evidenceId !== undefined && evidenceId === responseBody
                ? "Response body · "
                : ""}
              {artifact.mediaType} · {formatBytes(artifact.sizeBytes)}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Everything the possible issue rests on: authored locations, the captured
 * HTTP exchange (credential-looking header values masked until shown) and
 * the evidence files the check retained.
 */
export function EvidencePanel({ finding }: { finding: AuditFinding }) {
  const id = useId();
  const document = finding.firstProposal.document;
  const locations = document.locations ?? [];
  const exchange = document.http_exchange;
  const files = finding.firstProposal.evidence;
  const masked =
    exchange?.attempts.some((attempt) =>
      [...attempt.headers, ...(attempt.response_headers ?? [])].some((header) =>
        isCredentialHeader(header.name),
      ),
    ) ?? false;
  if (locations.length === 0 && exchange === undefined && files.length === 0)
    return (
      <p className="issues-quiet">
        This possible issue names no locations and has no captured evidence.
      </p>
    );
  return (
    <>
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
            <p className="issues-quiet">
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
          <EvidenceFiles finding={finding} />
        </section>
      )}
    </>
  );
}
