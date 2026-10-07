import { StatusChip } from "../../../ui";

/** A managed credential exists only while it is usable: it reads "active". */
export function ActiveChip() {
  return (
    <StatusChip tone="done" size="sm">
      <span className="ops-state-word">active</span>
    </StatusChip>
  );
}
