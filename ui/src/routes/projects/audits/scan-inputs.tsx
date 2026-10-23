import type { Audit } from "../../../api/audits";
import { ContextLink } from "../../../app/context-navigation";
import { AuditScanInputsWidget } from "../../../widgets/audit-scans/view";
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
    <AuditScanInputsWidget
      scan={scan}
      source={
        <ScanSource audit={audit} artifact={scan.source} label="API spec" />
      }
      settings={
        <ScanSource
          audit={audit}
          artifact={scan.settings}
          label="Scan settings"
        />
      }
      result={
        receipt === undefined || result === undefined ? (
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
        )
      }
    />
  );
}
