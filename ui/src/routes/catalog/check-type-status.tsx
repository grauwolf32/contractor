import type { AuditProfile } from "../../api/audits";
import { StatusChip } from "../../ui";
import { compatibilityReasonText } from "./audit-preset-data";

/** "Available" or "Unavailable on this server", as a chip with its glyph. */
export function CheckTypeAvailability({
  profile,
  size = "md",
}: {
  profile: AuditProfile;
  size?: "sm" | "md";
}) {
  return profile.serverCompatible ? (
    <StatusChip tone="success" size={size}>
      Available
    </StatusChip>
  ) : (
    <StatusChip tone="blocked" size={size}>
      Unavailable on this server
    </StatusChip>
  );
}

/** Why the server cannot run a check type: one sentence per reason code. */
export function CompatibilityReasons({
  reasons,
}: {
  reasons: readonly string[];
}) {
  if (reasons.length === 0) return null;
  return (
    <ul className="library-reasons">
      {reasons.map((reason) => (
        <li key={reason}>
          <span>{compatibilityReasonText(reason)}</span> <code>{reason}</code>
        </li>
      ))}
    </ul>
  );
}
