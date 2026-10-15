import { useId, useState, type ReactNode } from "react";

import "./materials.css";

/**
 * Tabs over panels that stay mounted (Rendered / Source of a preview,
 * Request / Response of an HTTP exchange). The arrow keys move to the
 * previous or next tab and wrap, Home and End go to the first and last;
 * only the chosen tab is in the tab order. `note` sits at the end of the
 * tab row.
 */
export function PanelTabs<T extends string>({
  label,
  tabs,
  note,
}: {
  label: string;
  tabs: readonly { id: T; label: ReactNode; panel: ReactNode }[];
  note?: ReactNode;
}) {
  const id = useId();
  const [chosen, setChosen] = useState<T | undefined>(tabs[0]?.id);
  const tabId = (tab: T) => `${id}-${tab}-tab`;
  const panelId = (tab: T) => `${id}-${tab}-panel`;
  function choose(index: number): void {
    const tab = tabs.at(index % tabs.length);
    if (tab === undefined) return;
    setChosen(tab.id);
    document.getElementById(tabId(tab.id))?.focus();
  }
  const current = Math.max(
    0,
    tabs.findIndex((tab) => tab.id === chosen),
  );
  return (
    <>
      <div
        aria-label={label}
        className="materials-tabs"
        role="tablist"
        onKeyDown={(event) => {
          const target = {
            ArrowLeft: current - 1,
            ArrowRight: current + 1,
            Home: 0,
            End: tabs.length - 1,
          }[event.key];
          if (target === undefined) return;
          event.preventDefault();
          choose(target);
        }}
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            aria-controls={panelId(tab.id)}
            aria-selected={tab.id === chosen}
            tabIndex={tab.id === chosen ? 0 : -1}
            className="materials-tab"
            id={tabId(tab.id)}
            role="tab"
            type="button"
            onClick={() => setChosen(tab.id)}
          >
            {tab.label}
          </button>
        ))}
        {note === undefined ? null : (
          <span className="materials-tabs-note">{note}</span>
        )}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          aria-labelledby={tabId(tab.id)}
          hidden={tab.id !== chosen}
          id={panelId(tab.id)}
          role="tabpanel"
          tabIndex={0}
        >
          {tab.panel}
        </div>
      ))}
    </>
  );
}
