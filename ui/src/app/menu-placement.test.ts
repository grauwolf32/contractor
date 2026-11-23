import { describe, expect, it } from "vitest";

import { placeMenu } from "./menu-placement";

const viewport = { width: 1440, height: 900 };
const menu = { width: 192, height: 120 };

function rect(left: number, top: number, width = 32, height = 32): DOMRect {
  return {
    left,
    top,
    width,
    height,
    right: left + width,
    bottom: top + height,
    x: left,
    y: top,
    toJSON: () => ({}),
  } as DOMRect;
}

describe("placeMenu", () => {
  it("opens to the left from a trigger in the right half", () => {
    const placement = placeMenu(rect(1300, 80), menu, viewport);
    expect(placement.left).toBe(1332 - 192);
    expect(placement.top).toBe(80 + 32 + 6);
  });

  it("opens to the right from a trigger in the left half", () => {
    expect(placeMenu(rect(100, 80), menu, viewport).left).toBe(100);
    // A trigger at the left edge of a detail pane, as on /checks.
    expect(placeMenu(rect(466, 375), menu, viewport).left).toBe(466);
  });

  it("stays inside the viewport on narrow screens", () => {
    const placement = placeMenu(rect(4, 80), menu, { width: 180, height: 600 });
    expect(placement.left).toBe(8);
  });

  it("opens upwards only when the space below is too small and above fits", () => {
    expect(placeMenu(rect(600, 820), menu, viewport).top).toBe(820 - 6 - 120);
    expect(
      placeMenu(rect(600, 60), { width: 192, height: 880 }, viewport).top,
    ).toBe(60 + 32 + 6);
  });

  it("limits the height to the space left below the menu", () => {
    const placement = placeMenu(rect(600, 300), menu, viewport);
    expect(placement.maxHeight).toBe(900 - (300 + 32 + 6) - 8);
  });
});
