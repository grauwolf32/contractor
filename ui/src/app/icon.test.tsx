import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Icon } from "./icon";

describe("Icon", () => {
  it("is decorative unless it is given a label", () => {
    const { container } = render(<Icon name="delete" />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(svg).toHaveAttribute("width", "20");
    expect(svg).toHaveClass("app-icon");
    expect(svg?.querySelector("path")).not.toBeNull();
  });

  it("is a named image with a label, at the size and weight asked for", () => {
    render(
      <Icon
        name="chevron-right"
        label="Open"
        size={13}
        strokeWidth={2}
        className="runs-section-chevron"
      />,
    );
    const svg = screen.getByRole("img", { name: "Open" });
    expect(svg).not.toHaveAttribute("aria-hidden");
    expect(svg).toHaveAttribute("width", "13");
    expect(svg).toHaveAttribute("stroke-width", "2");
    expect(svg).toHaveClass("app-icon", "runs-section-chevron");
  });
});
