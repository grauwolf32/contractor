import type { Audit } from "../../../api/audits";
import { ContextLink } from "../../../app/context-navigation";
import { Facts, StatusChip, TechnicalDetails } from "../../../ui";
import { exactArtifactLink } from "./artifact-links";
import type { CheckEntry } from "./check-model";
import { AuditMaterial } from "./material";

function ScanSource({
  audit,
  artifact,
  label,
}: {
  audit: Audit;
  artifact: Audit["inputs"][string];
  label: string;
}) {
  for (const role of Object.values(audit.preparation?.roles ?? {})) {
    for (const [name, output] of Object.entries(role.outputs)) {
      if (
        output.artifact.digest === artifact.digest &&
        output.artifact.ref.namespace === artifact.ref.namespace &&
        output.artifact.ref.name === artifact.ref.name &&
        output.artifact.ref.revision === artifact.ref.revision
      ) {
        return (
          <AuditMaterial
            auditId={audit.auditId}
            artifact={artifact}
            receipt={`prepare:${output.executionId}:${name}`}
            label={label}
          />
        );
      }
    }
  }
  return (
    <ContextLink
      returnLabel="Endpoint details"
      to={exactArtifactLink(audit.projectId, artifact)}
    >
      {label}
    </ContextLink>
  );
}

/** B: exact inputs, scan coverage and the saved report stay beside the endpoint. */
export function ScanInputs({
  audit,
  entry,
}: {
  audit: Audit;
  entry: CheckEntry;
}) {
  const scan = entry.row.details?.scan;
  if (scan === undefined) return null;
  const result = entry.row.result;
  const accepted = entry.item?.acceptedResult;
  const receipt =
    result !== undefined &&
    accepted?.digest === result.digest &&
    accepted.ref.revision === result.ref.revision &&
    entry.item?.lastExecutionItemId !== undefined
      ? `result/${entry.item.lastExecutionItemId}`
      : undefined;
  return (
    <section
      className="checks-scan-inputs"
      aria-label="Scan inputs and evidence"
    >
      <div className="ui-toolbar" data-justify="between">
        <h3>
          {scan.scanner === "nuclei"
            ? "Nuclei · URL template check"
            : "SQLMap · Request check"}
        </h3>
        <StatusChip tone={scan.runnable ? "info" : "blocked"}>
          {scan.runnable ? "Input prepared" : "Input incomplete"}
        </StatusChip>
      </div>
      <p>
        {scan.scanner === "nuclei"
          ? "Checks templates against the fixed URL. It does not replay the OpenAPI method, request body or authenticated session."
          : "Checks the prepared request and selected test parameters. A finished scan leaves security coverage inconclusive until its observations are verified."}
      </p>
      <ol className="checks-input-chain">
        <li>
          <ScanSource audit={audit} artifact={scan.source} label="API spec" />
          <span className="ui-caption">Exact source of this endpoint</span>
        </li>
        <li>
          <ScanSource
            audit={audit}
            artifact={scan.settings}
            label="Scan settings"
          />
          <span className="ui-caption">Pinned operation bindings</span>
        </li>
        <li>
          <strong>
            {scan.scanner === "nuclei" ? "Fixed URL" : "Prepared request"}
          </strong>
          <span className="ui-caption">
            {scan.scanner === "nuclei"
              ? (scan.targetUrl ?? "Unavailable")
              : scan.testParameters.join(", ") || "No test parameters"}
          </span>
        </li>
        <li>
          {receipt === undefined || result === undefined ? (
            <span className="ui-caption">No loaded accepted report</span>
          ) : (
            <>
              <AuditMaterial
                auditId={audit.auditId}
                artifact={result}
                receipt={receipt}
                label="Saved result package"
              />
              {entry.row.details?.evidence
                .filter((evidence) => evidence.contentMemberId !== undefined)
                .map((evidence) => (
                  <AuditMaterial
                    key={evidence.id}
                    auditId={audit.auditId}
                    artifact={result}
                    receipt={receipt}
                    member={evidence.contentMemberId}
                    label={
                      evidence.kind === "scanner-report"
                        ? "Scanner report"
                        : evidence.summary
                    }
                  />
                ))}
            </>
          )}
        </li>
      </ol>
      {scan.gaps.length === 0 ? null : (
        <ul className="checks-string-list">
          {scan.gaps.map((gap) => (
            <li key={gap}>{gap}</li>
          ))}
        </ul>
      )}
      <TechnicalDetails summary="Exact scan provenance">
        <Facts
          size="sm"
          items={[
            { term: "Operation", value: <code>{scan.operation}</code> },
            {
              term: "Preparation",
              value: <code>{scan.preparationDigest}</code>,
            },
            {
              term: "Request",
              value:
                scan.requestDigest === undefined ? (
                  "URL mode"
                ) : (
                  <code>{scan.requestDigest}</code>
                ),
            },
          ]}
        />
      </TechnicalDetails>
    </section>
  );
}
