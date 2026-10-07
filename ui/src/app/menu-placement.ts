/** Space kept between the menu and the viewport edges, in pixels. */
const VIEWPORT_MARGIN = 8;
/** Space between the trigger and the menu, in pixels. */
const TRIGGER_GAP = 6;

export interface MenuPlacement {
  top: number;
  left: number;
  maxHeight: number;
}

/**
 * Places the menu next to its trigger in viewport coordinates, so scrolling
 * panes and overflow never clip it. A trigger in the left half of the
 * viewport opens the menu to the right (left edges aligned); one in the right
 * half opens it to the left (right edges aligned). The menu opens downwards
 * unless only the space above fits. Both are clamped to the viewport.
 */
export function placeMenu(
  trigger: DOMRect,
  menu: { width: number; height: number },
  viewport: { width: number; height: number },
): MenuPlacement {
  const maxLeft = Math.max(
    VIEWPORT_MARGIN,
    viewport.width - menu.width - VIEWPORT_MARGIN,
  );
  const opensRight = trigger.left + trigger.width / 2 < viewport.width / 2;
  const preferred = opensRight ? trigger.left : trigger.right - menu.width;
  const left = Math.min(Math.max(preferred, VIEWPORT_MARGIN), maxLeft);

  const below = trigger.bottom + TRIGGER_GAP;
  const spaceBelow = viewport.height - below - VIEWPORT_MARGIN;
  const above = trigger.top - TRIGGER_GAP - menu.height;
  const top =
    menu.height > spaceBelow && above >= VIEWPORT_MARGIN ? above : below;
  return {
    top,
    left,
    maxHeight: Math.max(
      viewport.height - top - VIEWPORT_MARGIN,
      VIEWPORT_MARGIN * 4,
    ),
  };
}
