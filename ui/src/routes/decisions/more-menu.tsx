import { useId, useState } from "react";

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
      <button
        type="button"
        className="ui-btn"
        aria-expanded={open}
        aria-controls={groupId}
        aria-label="More decisions"
        onClick={() => setOpen((value) => !value)}
      >
        <span>More</span>
        <svg
          className="decisions-more-chevron"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          focusable="false"
        >
          <path d="M9.5 6l6 6-6 6" />
        </svg>
      </button>
      <div
        id={groupId}
        role="group"
        aria-label="More decisions"
        className="decisions-more-items"
        hidden={!open}
      >
        {items.map((item) => (
          <button
            key={item.id}
            type="button"
            className="ui-btn"
            onClick={() => {
              setOpen(false);
              item.onSelect();
            }}
          >
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}
