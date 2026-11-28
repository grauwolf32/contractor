import {
  useLayoutEffect,
  useRef,
  type MouseEventHandler,
  type RefObject,
} from "react";
import { useLocation } from "react-router";

// Bounded, tab-local reading positions. Material bytes and form data never live here.
const positions = new Map<string, number>();
const MAX_POSITIONS = 32;

function scrollOwner(region: RefObject<HTMLElement | null>) {
  const pane = region.current?.closest<HTMLElement>(".ui-panes-detail");
  return pane != null &&
    pane.scrollHeight > pane.clientHeight &&
    /auto|scroll/.test(getComputedStyle(pane).overflowY)
    ? pane
    : document.scrollingElement;
}

/** Restore a material list after its separate reader page has unmounted it. */
export function useMaterialListScroll(
  region: RefObject<HTMLElement | null>,
  ready: boolean,
): MouseEventHandler<HTMLElement> {
  const location = useLocation();
  const key = location.pathname + location.search;
  const restored = useRef(false);
  useLayoutEffect(() => {
    const top = positions.get(key);
    if (!ready || restored.current || top === undefined) return;
    // The Project workspace also initializes its pane after mount.
    const frame = window.requestAnimationFrame(() => {
      const owner = scrollOwner(region);
      if (owner !== null) owner.scrollTop = top;
      restored.current = true;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [key, ready, region]);
  return (event) => {
    if (
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      !(event.target instanceof Element) ||
      event.target.closest(".materials-row-link") === null
    )
      return;
    positions.delete(key);
    positions.set(key, scrollOwner(region)?.scrollTop ?? 0);
    if (positions.size > MAX_POSITIONS)
      positions.delete(positions.keys().next().value!);
  };
}
