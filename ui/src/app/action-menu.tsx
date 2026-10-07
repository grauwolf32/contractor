import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

import "./action-menu.css";
import { placeMenu } from "./menu-placement";

/**
 * Browsers with the Popover API show the open panel in the top layer; others
 * (and the test environment) keep it as a fixed-position element.
 */
const POPOVER_SUPPORTED =
  typeof HTMLElement !== "undefined" &&
  typeof HTMLElement.prototype.showPopover === "function";

/** Native disclosure: regular Tab order, with dismissal and focus restoration. */
export function ActionMenu({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  const disclosure = useRef<HTMLDetailsElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);

  // Positioning writes straight to the panel's style: it synchronises the DOM
  // with the trigger's geometry and needs no React state.
  const place = useCallback(() => {
    const element = disclosure.current;
    const summary = element?.querySelector("summary");
    const panel = menu.current;
    if (!element?.open || !summary || !panel) return;
    const placement = placeMenu(
      summary.getBoundingClientRect(),
      { width: panel.offsetWidth, height: panel.offsetHeight },
      { width: window.innerWidth, height: window.innerHeight },
    );
    panel.style.top = `${placement.top}px`;
    panel.style.left = `${placement.left}px`;
    panel.style.maxHeight = `${placement.maxHeight}px`;
    panel.dataset.placed = "";
  }, []);

  useEffect(() => {
    function dismiss(event: PointerEvent | KeyboardEvent) {
      const element = disclosure.current;
      if (!element?.open || element.closest("[inert]")) return;
      if (event instanceof KeyboardEvent) {
        if (event.key !== "Escape") return;
        event.preventDefault();
        element.open = false;
        element.querySelector("summary")?.focus();
      } else if (
        event.target instanceof Node &&
        !element.contains(event.target)
      ) {
        element.open = false;
      }
    }
    document.addEventListener("keydown", dismiss);
    document.addEventListener("pointerdown", dismiss);
    return () => {
      document.removeEventListener("keydown", dismiss);
      document.removeEventListener("pointerdown", dismiss);
    };
  }, []);

  // Show the panel in the top layer (Popover API) so no ancestor's overflow,
  // containment or transform can clip or offset it, then measure and follow
  // scrolling and resizing while it stays open. The panel keeps its DOM place,
  // so Tab still moves from the trigger into the menu.
  useLayoutEffect(() => {
    if (!open) return undefined;
    const panel = menu.current;
    if (
      panel &&
      typeof panel.showPopover === "function" &&
      !panel.matches(":popover-open")
    ) {
      panel.showPopover();
    }
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, place]);

  return (
    <details
      ref={disclosure}
      className="additional-actions project-actions-menu"
      onToggle={(event) => {
        const isOpen = event.currentTarget.open;
        if (!isOpen && menu.current) {
          const panel = menu.current;
          if (
            typeof panel.hidePopover === "function" &&
            panel.matches(":popover-open")
          ) {
            panel.hidePopover();
          }
          // Forget the last placement so the next opening measures afresh.
          delete menu.current.dataset.placed;
          menu.current.style.removeProperty("top");
          menu.current.style.removeProperty("left");
          menu.current.style.removeProperty("max-height");
        }
        setOpen(isOpen);
      }}
    >
      <summary aria-label={label} title={label}>
        ⋯
      </summary>
      <div
        ref={menu}
        className="additional-actions-menu"
        popover={POPOVER_SUPPORTED ? "manual" : undefined}
      >
        {children}
      </div>
    </details>
  );
}
