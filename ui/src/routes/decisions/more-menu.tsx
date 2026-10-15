import { useId, useState } from "react";

import { Icon } from "../../app/icon";
import { Button } from "../../ui";

export interface MoreDecision {
  id: string;
  label: string;
  onSelect: () => void;
}

/**
 * The decision bar's "More" control: a disclosure that shows the decisions
 * without their own button (Duplicate…, Reopen) right after the toggle. They
 * wrap with the verdicts on narrow panes instead of overlapping the content.
 */
export function MoreDecisions({ items }: { items: readonly MoreDecision[] }) {
  const [open, setOpen] = useState(false);
  const groupId = useId();
  if (items.length === 0) return null;
  return (
    <div className="decisions-more">
      <Button
        aria-expanded={open}
        aria-controls={groupId}
        aria-label="More decisions"
        onClick={() => setOpen((value) => !value)}
      >
        <span>More</span>
        <Icon
          name="chevron-right"
          size={14}
          className="decisions-more-chevron"
        />
      </Button>
      <div
        id={groupId}
        role="group"
        aria-label="More decisions"
        className="decisions-more-items"
        hidden={!open}
      >
        {items.map((item) => (
          <Button
            key={item.id}
            onClick={() => {
              setOpen(false);
              item.onSelect();
            }}
          >
            {item.label}
          </Button>
        ))}
      </div>
    </div>
  );
}
