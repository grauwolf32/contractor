import {
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";

export interface TabItem<T extends string> {
  id: T;
  label: string;
  /** Shown after the label in monospace; part of the tab's name. */
  count?: number | undefined;
}

/** Id of a tab's button, for `aria-labelledby` on its panel. */
function tabId(base: string, id: string): string {
  return `${base}-tab-${id}`;
}

/** Id of a tab's panel. */
function panelId(base: string, id: string): string {
  return `${base}-panel-${id}`;
}

/**
 * A tab list (WAI-ARIA tabs pattern): one tab in the tab order, ← / → move
 * and select with wrap-around, Home and End jump to the ends. Only the
 * selected tab controls a rendered panel.
 */
export function Tabs<T extends string>({
  label,
  items,
  selected,
  onSelect,
  idBase,
}: {
  label: string;
  items: readonly TabItem<T>[];
  selected: T;
  onSelect: (id: T) => void;
  idBase: string;
}) {
  const buttons = useRef(new Map<T, HTMLButtonElement>());

  function onKeyDown(event: ReactKeyboardEvent<HTMLButtonElement>, at: number) {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const last = items.length - 1;
    let target: number | undefined;
    switch (event.key) {
      case "ArrowRight":
        target = at === last ? 0 : at + 1;
        break;
      case "ArrowLeft":
        target = at === 0 ? last : at - 1;
        break;
      case "Home":
        target = 0;
        break;
      case "End":
        target = last;
        break;
      default:
        return;
    }
    const item = items[target];
    if (item === undefined) return;
    event.preventDefault();
    onSelect(item.id);
    buttons.current.get(item.id)?.focus();
  }

  return (
    <div role="tablist" aria-label={label} className="issues-tabs">
      {items.map((item, at) => {
        const active = item.id === selected;
        return (
          <button
            key={item.id}
            ref={(node) => {
              if (node === null) buttons.current.delete(item.id);
              else buttons.current.set(item.id, node);
            }}
            type="button"
            role="tab"
            id={tabId(idBase, item.id)}
            className="issues-tab"
            aria-selected={active}
            aria-controls={active ? panelId(idBase, item.id) : undefined}
            tabIndex={active ? 0 : -1}
            onClick={() => onSelect(item.id)}
            onKeyDown={(event) => onKeyDown(event, at)}
          >
            {item.label}
            {item.count === undefined ? null : (
              <>
                {" "}
                <span className="issues-tab-count">{item.count}</span>
              </>
            )}
          </button>
        );
      })}
    </div>
  );
}

/** The selected tab's panel, named by its tab. */
export function TabPanel({
  idBase,
  id,
  children,
}: {
  idBase: string;
  id: string;
  children: ReactNode;
}) {
  return (
    <div
      role="tabpanel"
      id={panelId(idBase, id)}
      aria-labelledby={tabId(idBase, id)}
      className="issues-tabpanel"
      tabIndex={0}
    >
      {children}
    </div>
  );
}
