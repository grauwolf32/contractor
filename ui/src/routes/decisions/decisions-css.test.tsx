import { readFileSync } from "node:fs";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

// The rules of decisions.css that go with the pinned pane footer's cap
// (ui.css .ui-footer-record, tested in src/ui/ui-css.test.tsx), read from
// the sheet as jsdom parses it: jsdom evaluates no media queries.
let sheet: HTMLStyleElement;
beforeAll(() => {
  sheet = document.createElement("style");
  sheet.textContent = readFileSync(
    "src/routes/decisions/decisions.css",
    "utf8",
  );
  document.head.append(sheet);
});
afterAll(() => sheet.remove());

const PINNED = "(min-width: 821px) and (min-height: 560px)";

/** The declarations of `selector`, at the top level or in @media `media`. */
function rule(selector: string, media?: string): CSSStyleDeclaration {
  const top = Array.from(sheet.sheet?.cssRules ?? []);
  const rules =
    media === undefined
      ? top
      : top.flatMap((candidate) =>
          candidate instanceof CSSMediaRule &&
          candidate.media.mediaText === media
            ? Array.from(candidate.cssRules)
            : [],
        );
  for (const candidate of rules) {
    if (
      candidate instanceof CSSStyleRule &&
      candidate.selectorText === selector
    )
      return candidate.style;
  }
  throw new Error(
    `no ${selector} rule${media === undefined ? "" : ` in @media ${media}`}`,
  );
}

describe("decisions.css", () => {
  it("keeps a capped current decision's actions in view while its reason scrolls", () => {
    const actions = rule(
      ".ui-detail-pane-footer .ui-footer-record .decisions-current-actions",
      PINNED,
    );
    expect(actions.position).toBe("sticky");
    expect(actions.bottom).toBe("0px");
    expect(actions.zIndex).toBe("1");
    expect(actions.getPropertyValue("background")).toBe("var(--surface)");
    // The row carries the section's gap and bottom padding as its own, so it
    // covers what scrolls under it and sits where it did when all fits.
    expect(actions.getPropertyValue("padding-block")).toBe("0.5rem 0.75rem");
    const section = rule(
      ".ui-detail-pane-footer .ui-footer-record > .decisions-current",
      PINNED,
    );
    expect(section.gap).toBe("0.125rem");
    expect(section.paddingBottom).toBe("0px");
  });

  it("rings a scrolling decision record from inside, where no pane clips it", () => {
    const ring = rule('.decisions-request[tabindex="0"]:focus-visible');
    expect(ring.outline).toBe("2px solid var(--focus)");
    expect(ring.outlineOffset).toBe("-2px");
  });
});
