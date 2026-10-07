import { useEffect, useState, type RefObject } from "react";

/**
 * Whether the element behind `ref` scrolls on its own while `watch` is true:
 * its content is taller than its box, as for a recorded decision capped where
 * the pane footer is pinned (ui.css `.ui-footer-record`). A box that scrolls
 * without a control of its own needs a tab stop, or the keyboard can never
 * reach the rest of it.
 *
 * Checked when watching starts and whenever the box or its content resizes:
 * the viewport (and with it the cap), a reason that finishes loading, a
 * narrow screen where nothing is capped. False where ResizeObserver is
 * missing.
 */
export function useScrollsOnItsOwn(
  ref: RefObject<HTMLElement | null>,
  watch: boolean,
): boolean {
  const [scrolls, setScrolls] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!watch || element === null || typeof ResizeObserver === "undefined")
      return undefined;
    // Observing reports the current sizes first, so this checks at once.
    const observer = new ResizeObserver(() => {
      // A pixel of slack: rounding alone never makes a box scroll.
      setScrolls(element.scrollHeight - element.clientHeight > 1);
    });
    observer.observe(element);
    // The content can grow while the box already stands at its cap.
    for (const child of Array.from(element.children)) observer.observe(child);
    return () => observer.disconnect();
  }, [ref, watch]);
  return watch && scrolls;
}
