import type { AuditProfile } from "../../api/audits";
import { StatusChip } from "../../ui";

/** "Available" or "Unavailable on this server", as a chip with its glyph. */
export function CheckTypeAvailability({
  profile,
  size = "md",
}: {
  profile: AuditProfile;
  size?: "sm" | "md";
}) {
  return (
    <StatusChip tone="neutral" size={size}>
      Published @{profile.ref.version}
    </StatusChip>
  );
}
