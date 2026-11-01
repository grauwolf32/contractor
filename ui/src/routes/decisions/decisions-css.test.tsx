import { readFileSync } from "node:fs";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

// The focus ring for an item/report approval record inside a scrolling footer.
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
  it("rings a scrolling decision record from inside, where no pane clips it", () => {
    const ring = rule('.decisions-request[tabindex="0"]:focus-visible');
    expect(ring.outline).toBe("2px solid var(--focus)");
    expect(ring.outlineOffset).toBe("-2px");
  });
});
