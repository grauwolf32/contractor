import { readFileSync } from "node:fs";
import { render } from "@testing-library/react";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { IdChip } from "./chips";
import { TechnicalDetails } from "./content";
import { ListPane } from "./panes";

// The rules of ui.css that the components rely on, applied by jsdom (which
// cascades stylesheets but evaluates no media queries).
let sheet: HTMLStyleElement;
beforeAll(() => {
  sheet = document.createElement("style");
  sheet.textContent = readFileSync("src/ui/ui.css", "utf8");
  document.head.append(sheet);
});
afterAll(() => sheet.remove());

const PINNED = "(min-width: 821px) and (min-height: 560px)";

/** The declarations of `selector` inside the @media rule for `media`. */
function mediaRule(media: string, selector: string): CSSStyleDeclaration {
  for (const rule of Array.from(sheet.sheet?.cssRules ?? [])) {
    if (!(rule instanceof CSSMediaRule) || rule.media.mediaText !== media)
      continue;
    for (const inner of Array.from(rule.cssRules)) {
      if (inner instanceof CSSStyleRule && inner.selectorText === selector)
        return inner.style;
    }
  }
  throw new Error(`no ${selector} rule in @media ${media}`);
}

describe("ui.css", () => {
  it("lets the list pane's title controls shrink with the pane", () => {
    const { container } = render(
      <ListPane
        title="Possible issues"
        actions={
          <span>
            <select aria-label="Project">
              <option>A project with a very long name</option>
            </select>
          </span>
        }
      >
        <p>Rows</p>
      </ListPane>,
    );
    const actions = container.querySelector<HTMLElement>(
      ".ui-list-pane-actions",
    );
    if (actions === null) throw new Error("no actions");
    const style = getComputedStyle(actions);
    expect(style.flexShrink).toBe("1");
    expect(style.minWidth).toBe("0px");
    expect(style.maxWidth).toBe("100%");
    for (const control of [
      actions.firstElementChild,
      actions.querySelector("select"),
    ]) {
      if (!(control instanceof HTMLElement)) throw new Error("no control");
      expect(getComputedStyle(control).minWidth).toBe("0px");
      expect(getComputedStyle(control).maxWidth).toBe("100%");
    }
  });

  it("turns only the chevron of the open disclosure itself", () => {
    const { container } = render(
      <TechnicalDetails summary="Outer" defaultOpen>
        <TechnicalDetails summary="Inner">
          <p>Digest</p>
        </TechnicalDetails>
      </TechnicalDetails>,
    );
    const [outer, inner] = Array.from(
      container.querySelectorAll(".ui-tech-chevron"),
    );
    if (outer === undefined || inner === undefined)
      throw new Error("no chevrons");
    expect(getComputedStyle(outer).transform).toBe("rotate(90deg)");
    expect(getComputedStyle(inner).transform).not.toBe("rotate(90deg)");
  });

  it("wraps a long identifier only in the wrap variant", () => {
    const value = "stage_execution_0f3c2a1b5d6e4f708a9b1c2d3e4f9e7d";
    const { container, rerender } = render(
      <IdChip value={value} display={value} label="stage execution ID" wrap />,
    );
    const shown = () => {
      const element = container.querySelector(".ui-id-chip-value");
      if (element === null) throw new Error("no value");
      return getComputedStyle(element);
    };
    expect(shown().whiteSpace).toBe("normal");
    expect(shown().wordBreak).toBe("break-all");
    expect(shown().textOverflow).toBe("clip");
    rerender(
      <IdChip value={value} display={value} label="stage execution ID" />,
    );
    expect(shown().whiteSpace).toBe("nowrap");
    expect(shown().textOverflow).toBe("ellipsis");
  });

  it("caps a recorded decision where the pane footer is pinned", () => {
    const footer = mediaRule(PINNED, ".ui-detail-pane-footer");
    expect(footer.position).toBe("sticky");
    const record = mediaRule(
      PINNED,
      ".ui-detail-pane-footer .ui-footer-record",
    );
    expect(record.maxHeight).toBe("min(45vh, 24rem)");
    expect(record.overflowY).toBe("auto");
    expect(record.borderTop).toMatch(/^1px solid/);
  });
});
