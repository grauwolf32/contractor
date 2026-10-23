import type { ReactNode } from "react";
import type { AuditCoverageRow } from "../../api/audits";
import { Facts, StatusChip, TechnicalDetails } from "../../ui";
import { auditScanWidget } from "./registry";

type Scan = NonNullable<NonNullable<AuditCoverageRow["details"]>["scan"]>;
/** Domain presentation only. The Audit host owns evidence links and receipts. */
export function AuditScanInputsWidget({
  scan,
  source,
  settings,
  result,
}: {
  scan: Scan;
  source: ReactNode;
  settings: ReactNode;
  result: ReactNode;
}) {
  const widget = auditScanWidget(scan.scanner);
  return (
    <section
      className="checks-scan-inputs"
      aria-label="Scan inputs and evidence"
    >
      <div className="ui-toolbar" data-justify="between">
        <h3>{widget?.title ?? "Scanner inputs"}</h3>
        <StatusChip tone={scan.runnable ? "info" : "blocked"}>
          {scan.runnable ? "Input prepared" : "Input incomplete"}
        </StatusChip>
      </div>
      <p>
        {widget?.description ??
          "No specialized view is registered for this scanner. Inspect the exact inputs and saved evidence below."}
      </p>
      <ol className="checks-input-chain">
        <li>
          {source}
          <span className="ui-caption">Exact source of this endpoint</span>
        </li>
        <li>
          {settings}
          <span className="ui-caption">Pinned operation bindings</span>
        </li>
        <li>
          <strong>{widget?.targetLabel ?? "Prepared target"}</strong>
          <span className="ui-caption">
            {widget?.targetValue(scan) ?? "No specialized target display"}
          </span>
        </li>
        <li>{result}</li>
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
