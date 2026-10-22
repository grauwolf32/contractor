import type { AuditFinding } from "../../api/audits";
import { StatusChip } from "../../ui";

/** System equality is independent of every human verdict and severity. */
export function CorrelationMark({ finding }: { finding: AuditFinding }) {
  const mark = finding.correlation;
  if (mark?.availability !== "available" || !mark.hasOtherObservations)
    return null;
  return (
    <StatusChip tone="neutral" size="sm">
      Automatic match
    </StatusChip>
  );
}
